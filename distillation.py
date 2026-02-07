"""
DMD distillation: train a one-step generator G_θ(z, c) and student velocity v_φ(x_t, t, c)
by alternating (1) regression of G to clean_prediction, (2) training v_φ from G,
(3) IKL update of G using score matching.

Options: (alignment a, augment g); Mgen, Mvel, Iter.
"""

import torch
import torch.nn as nn

from score import ScoreModel


def train_generator_regression(
    G,
    clean_model,
    num_steps=1000,
    batch_size=128,
    C=6,
    N=2,
    lr=1e-3,
    device=None,
    verbose=True,
):
    """
    Train G_θ(z, c) to match the pretrained clean-prediction model at t=1:
    Loss = ||G_θ(z, c) - x(z, 1, c)||^2 with z ~ N(0,I), c ~ Uniform(0, C-1).

    Args:
        G: OneStepGenerator (or any module with forward(z, c) -> x).
        clean_model: CleanPredModel wrapping teacher velocity; clean_model(z, t=1, c) = x*(z,1,c).
        num_steps: number of regression steps (e.g. 1000).
        batch_size: K per step.
        C: number of classes (labels 0..C-1).
        N: ambient dimension.
        lr: learning rate.
        device: torch device.
        verbose: print loss every 100 steps.

    Returns:
        List of losses.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = G.to(device).train()
    clean_model.eval()
    opt = torch.optim.Adam(G.parameters(), lr=lr)
    losses = []

    for step in range(num_steps):
        z = torch.randn(batch_size, N, device=device)
        c = torch.randint(0, C, (batch_size,), device=device, dtype=torch.long)
        with torch.no_grad():
            target = clean_model(z, 1.0, c)
        pred = G(z, c)
        loss = nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if verbose and (step + 1) % 100 == 0:
            print(f"  [G reg] step {step + 1}/{num_steps} loss {loss.item():.6f}")
    return losses


def train_velocity_from_generator(
    v_phi,
    G_theta,
    Mvel,
    batch_size,
    C,
    N,
    drop_label_prob=0.0,
    lr=1e-4,
    device=None,
    verbose=True,
):
    """
    Train student velocity v_φ(x_t, t, c) for Mvel steps using fresh samples from G_θ.

    Each step:
        z ~ N(0, I_N), c ~ p(C),  x0 = G_θ(z, c)  [no grad through G_θ]
        ε ~ N(0, I_N)  (independent)
        x_t = (1-t) x0 + t ε,  target = ε - x0
        loss = ||v_φ(x_t, t, c) - target||^2

    G_θ is frozen; fresh z and ε are drawn every step (no pre-cached data source).

    Args:
        v_phi: VelocityMLP (student) with forward(x_t, t, c) -> v.
        G_theta: OneStepGenerator; kept frozen (eval mode, no grad).
        Mvel: number of velocity training steps.
        batch_size: K per step.
        C, N: number of classes, ambient dim.
        drop_label_prob: p for dropping c to empty label.
        lr, device, verbose: training options.

    Returns:
        List of losses.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_theta.eval()
    v_phi = v_phi.to(device).train()
    opt = torch.optim.Adam(v_phi.parameters(), lr=lr)
    num_classes = v_phi.num_classes
    losses = []

    for step in range(Mvel):
        z = torch.randn(batch_size, N, device=device)
        c = torch.randint(0, C, (batch_size,), device=device, dtype=torch.long)
        with torch.no_grad():
            x0 = G_theta(z, c)
        eps = torch.randn(batch_size, N, device=device)

        # Drop label with prob p -> set c = empty (num_classes)
        if drop_label_prob > 0:
            mask = torch.rand(batch_size, device=device) < drop_label_prob
            c = c.clone()
            c[mask] = num_classes

        t = torch.rand(batch_size, device=device)
        x_t = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * eps
        target = eps - x0

        v = v_phi(x_t, t, c)
        loss = nn.functional.mse_loss(v, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if verbose and (step + 1) % 100 == 0:
            print(f"    [v_phi] step {step + 1}/{Mvel} loss {loss.item():.6f}")

    return losses


def train_generator_ikl(
    G_theta,
    score_model_phi,
    score_model_teacher,
    Mgen,
    batch_size,
    C,
    N,
    a=1.0,
    g=1.0,
    lr=1e-4,
    reweight=False,
    device=None,
    verbose=True,
):
    """
    Update G_θ for Mgen steps by directly injecting the IKL gradient into θ:

    ∇_θ L_IKL = E[ (a(s_φ(x_t,t,c) - s(x_t,t,c)) + (1-g)(s(x_t,t,c) - s(x_t,t,∅)))
                     · (1-t) · ∂G_θ/∂θ ]
    where:
        x_0 = G_θ(z, c),  ε ~ N(0,I),  x_t = (1-t)*x_0 + t*ε,  t ~ U[0.01, 0.99].

    This is NOT a loss — it is the gradient itself.  We compute
        coeff = (a(s_φ - s) + (1-g)(s - s_∅)) · (1-t)       (shape (B, N), detached)
    then call  x_0.backward(gradient=coeff / B)  so that the accumulated param
    gradient equals  (1/B) Σ_i coeff_i · ∂G_θ(z_i,c_i)/∂θ,
    and opt.step() performs  θ ← θ − η · ∇_θ L_IKL.

    Args:
        G_theta: OneStepGenerator (trainable).
        score_model_phi: ScoreModel(v_phi) — student score.
        score_model_teacher: ScoreModel(teacher velocity) — teacher score.
        Mgen: number of generator steps.
        batch_size: K per step.
        C, N: classes, dim.
        a, g: alignment and guidance (e.g. a=1, g=1).
        lr, device, verbose: training options.
        reweight: whether to use time-reweighting. If False, use exact derivation; if True, use * t**2/(1-t) as reweight (based on original DMD)

    Returns:
        List of gradient-norm values (for logging).
    """
    device = device or next(G_theta.parameters()).device
    G_theta = G_theta.to(device).train()
    score_model_phi.eval()
    score_model_teacher.eval()
    empty_label = score_model_teacher.velocity_model.num_classes
    opt = torch.optim.Adam(G_theta.parameters(), lr=lr)
    grad_norms = []

    for step in range(Mgen):
        z = torch.randn(batch_size, N, device=device)
        eps = torch.randn(batch_size, N, device=device)
        c = torch.randint(0, C, (batch_size,), device=device, dtype=torch.long)
        t = torch.rand(batch_size, device=device) * 0.96 + 0.02  # U[0.02, 0.98]
        t_col = t.unsqueeze(-1)  # (B, 1)

        # 1. Forward G_theta once (keep graph for backward later)
        x_0 = G_theta(z, c) 
        
        # 2. Compute coeff using detached x_0
        with torch.no_grad():
            x_0_detached = x_0.detach()  
            x_t = (1 - t_col) * x_0_detached + t_col * eps
            
            s_phi = score_model_phi(x_t, t, c)
            s = score_model_teacher(x_t, t, c)
            c_empty = torch.full((batch_size,), empty_label, dtype=torch.long, device=device)
            s_empty = score_model_teacher(x_t, t, c_empty)
            if reweight:
                coeff = (a * (s_phi - s) + (1 - g) * (s - s_empty)) * t_col**2 
            else:
                coeff = (a * (s_phi - s) + (1 - g) * (s - s_empty)) * (1 - t_col)

        # 3. Backward directly
        opt.zero_grad()
        x_0.backward(gradient=coeff / batch_size) 
        torch.nn.utils.clip_grad_norm_(G_theta.parameters(), max_norm=1.0) # Gradient clip
        opt.step()


        # Log gradient norm for monitoring
        gn = sum(p.grad.norm().item() ** 2 for p in G_theta.parameters() if p.grad is not None) ** 0.5
        grad_norms.append(gn)
        if verbose and (step + 1) % 50 == 0:
            print(f"  [G IKL] step {step + 1}/{Mgen}  grad_norm {gn:.6f}")
    return grad_norms


def dmd_distillation(
    G_theta,
    v_phi,
    clean_model,
    score_model_teacher,
    C,
    N,
    Mgen=100,
    Mvel=100,
    Iter=100,
    reg_steps=1000,
    batch_size=128,
    a=1.0,
    g=1.0,
    drop_label_prob=0.0,
    lr_G=1e-4,
    lr_v=1e-4,
    device=None,
    verbose=True,
):
    """
    DMD iterative distillation: alternate training v_φ from G_θ and G_θ via IKL.

    1) Train G_θ regression: ||G_θ(z,c) - x(z,1,c)||^2 for reg_steps.
    2) For iter = 1..Iter:
       - Train v_φ for Mvel steps from G_θ (fresh z, independent ε each step).
       - Train G_θ for Mgen steps (IKL) using s_φ and teacher score.

    Args:
        G_theta: OneStepGenerator.
        v_phi: VelocityMLP (student velocity).
        clean_model: CleanPredModel(teacher velocity).
        score_model_teacher: ScoreModel(teacher velocity).
        C, N: number of classes, ambient dimension.
        Mgen: generator steps per iteration.
        Mvel: velocity steps per iteration.
        Iter: number of (v_φ, G_θ) alternations.
        reg_steps: initial G regression steps.
        batch_size: batch size for all steps.
        a, g: IKL alignment and guidance (e.g. a=1, g=1).
        drop_label_prob: for v_φ training.
        lr_G, lr_v: learning rates for G and v.
        device, verbose: options.

    Returns:
        G_theta, v_phi (trained in place), and dict of loss lists for logging.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_theta = G_theta.to(device)
    v_phi = v_phi.to(device)
    clean_model = clean_model.to(device)
    score_model_teacher = score_model_teacher.to(device)

    history = {"reg": [], "vel": [], "ikl": []}

    if verbose:
        print("Phase 0: Train G_θ regression to clean_prediction(z, 1, c)")
    history["reg"] = train_generator_regression(
        G_theta, clean_model, num_steps=reg_steps, batch_size=batch_size,
        C=C, N=N, lr=lr_G, device=device, verbose=verbose,
    )

    for it in range(Iter):
        if verbose:
            print(f"\n--- Iteration {it + 1}/{Iter} ---")
            print(f"Phase 1: Train v_φ for {Mvel} steps (fresh samples from G_θ)")
        vel_losses = train_velocity_from_generator(
            v_phi, G_theta, Mvel=Mvel, batch_size=batch_size, C=C, N=N,
            drop_label_prob=drop_label_prob, lr=lr_v,
            device=device, verbose=verbose,
        )
        history["vel"].extend(vel_losses)

        score_model_phi = ScoreModel(v_phi)
        if verbose:
            print(f"Phase 2: Train G_θ for {Mgen} steps (IKL)")
        ikl_losses = train_generator_ikl(
            G_theta, score_model_phi, score_model_teacher,
            Mgen=Mgen, batch_size=batch_size, C=C, N=N, a=a, g=g,
            lr=lr_G, device=device, verbose=verbose,
        )
        history["ikl"].extend(ikl_losses)

    return G_theta, v_phi, history
