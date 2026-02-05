"""
DMD distillation: train a one-step generator G_θ(z, c) and student velocity v_φ(x_t, t, c)
by alternating (1) regression of G to clean_prediction, (2) training v_φ from G (pairing or not),
(3) IKL update of G using score matching.

Options: PAIRING (alignment, augment) = (a=1, g=1); Mgen, Mvel, Iter.
"""

import torch
import torch.nn as nn

from score import ScoreModel
from training import train_flow_matching


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


def _make_data_source_from_G(G, C, N, pairing, device):
    """Return a callable(K, device) -> (x0, c, eps) for flow-matching.
    pairing: if True, eps = z (coupling (z, G(z,c))); else eps ~ N(0,I) independent.
    """
    def data_source(K, device=None):
        dev = device or next(G.parameters()).device
        z = torch.randn(K, N, device=dev)
        c = torch.randint(0, C, (K,), device=dev, dtype=torch.long)
        x0 = G(z, c)
        if pairing:
            eps = z
        else:
            eps = torch.randn_like(z, device=dev)
        return x0, c, eps
    return data_source


def train_velocity_from_generator(
    v_phi,
    G_theta,
    Mvel,
    batch_size,
    C,
    N,
    pairing=True,
    drop_label_prob=0.0,
    lr=1e-4,
    device=None,
    verbose=True,
):
    """
    Train student velocity v_φ(x_t, t, c) for Mvel steps using samples from G_θ.

    If pairing: (z, G_θ(z,c)) coupling — sample z ~ N(0,I), c ~ p(C), x0 = G_θ(z,c), use eps = z.
    If not pairing: x0 = G_θ(z,c), eps ~ N(0,I) independent.

    Args:
        v_phi: VelocityMLP (student) with forward(x_t, t, c) -> v.
        G_theta: OneStepGenerator; kept frozen.
        Mvel: number of velocity training steps.
        batch_size: K per step.
        C, N: number of classes, ambient dim.
        pairing: use (z, G(z,c)) coupling (eps=z) else independent eps.
        drop_label_prob: p for dropping c to empty label.
        lr, device, verbose: training options.

    Returns:
        List of losses.
    """
    G_theta.eval()
    data_source = _make_data_source_from_G(G_theta, C, N, pairing, device)
    return train_flow_matching(
        v_phi,
        data_source,
        num_steps=Mvel,
        batch_size=batch_size,
        drop_label_prob=drop_label_prob,
        lr=lr,
        device=device,
        verbose=verbose,
    )


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
    device=None,
    verbose=True,
):
    """
    Update G_θ for Mgen steps using approximate IKL gradient:

    ∇_θ L_IKL ∝ E[ (a(s_φ - s) + (1-g)(s - s_∅)) · (1-t) · ∂G_θ/∂θ ]
    with x_t = G_θ(z, c), t ~ U[0.01, 0.99], z ~ N(0,I), c ~ p(C).

    We minimize L = -(coeff.detach() * G_θ(z,c)).sum() so that ∂L/∂θ = -coeff·∂G_θ/∂θ,
    i.e. we ascend in the desired direction.

    Args:
        G_theta: OneStepGenerator (trainable).
        score_model_phi: ScoreModel(v_phi) — student score.
        score_model_teacher: ScoreModel(teacher velocity) — teacher score.
        Mgen: number of generator steps.
        batch_size: K per step.
        C, N: classes, dim.
        a, g: alignment and guidance (e.g. a=1, g=1).
        lr, device, verbose: training options.

    Returns:
        List of losses.
    """
    device = device or next(G_theta.parameters()).device
    G_theta = G_theta.to(device).train()
    score_model_phi.eval()
    score_model_teacher.eval()
    empty_label = score_model_teacher.velocity_model.num_classes
    opt = torch.optim.Adam(G_theta.parameters(), lr=lr)
    losses = []

    for step in range(Mgen):
        z = torch.randn(batch_size, N, device=device)
        c = torch.randint(0, C, (batch_size,), device=device, dtype=torch.long)
        t = torch.rand(batch_size, device=device) * 0.98 + 0.01  # U[0.01, 0.99]

        x_t = G_theta(z, c)

        with torch.no_grad():
            s_phi = score_model_phi(x_t, t, c)
            s = score_model_teacher(x_t, t, c)
            c_empty = torch.full((batch_size,), empty_label, dtype=torch.long, device=device)
            s_empty = score_model_teacher(x_t, t, c_empty)
        # coeff shape (B, N)
        coeff = (a * (s_phi - s) + (1 - g) * (s - s_empty)) * (1 - t).unsqueeze(-1)
        loss = -(coeff.detach() * G_theta(z, c)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if verbose and (step + 1) % 50 == 0:
            print(f"  [G IKL] step {step + 1}/{Mgen} loss {loss.item():.6f}")
    return losses


def dmd_distillation(
    G_theta,
    v_phi,
    clean_model,
    score_model_teacher,
    C,
    N,
    pairing=True,
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
       - Train v_φ for Mvel steps from G_θ (pairing or not).
       - Train G_θ for Mgen steps (IKL) using s_φ and teacher score.

    Args:
        G_theta: OneStepGenerator.
        v_phi: VelocityMLP (student velocity).
        clean_model: CleanPredModel(teacher velocity).
        score_model_teacher: ScoreModel(teacher velocity).
        C, N: number of classes, ambient dimension.
        pairing: if True, use (z, G(z,c)) coupling when training v_φ; else independent eps.
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

    # Train G_0 to match backward clean_prediction...
    if verbose:
        print("Phase 0: Train G_θ regression to clean_prediction(z, 1, c)")
    history["reg"] = train_generator_regression(
        G_theta, clean_model, num_steps=reg_steps, batch_size=batch_size,
        C=C, N=N, lr=lr_G, device=device, verbose=verbose,
    )

    for it in range(Iter):
        if verbose:
            print(f"\n--- Iteration {it + 1}/{Iter} ---")
        if verbose:
            print("Phase 1: Train v_φ for Mvel steps (from G_θ, pairing={})".format(pairing))
        vel_losses = train_velocity_from_generator(
            v_phi, G_theta, Mvel=Mvel, batch_size=batch_size, C=C, N=N,
            pairing=pairing, drop_label_prob=drop_label_prob, lr=lr_v,
            device=device, verbose=verbose,
        )
        history["vel"].extend(vel_losses)

        score_model_phi = ScoreModel(v_phi)
        if verbose:
            print("Phase 2: Train G_θ for Mgen steps (IKL)")
        ikl_losses = train_generator_ikl(
            G_theta, score_model_phi, score_model_teacher,
            Mgen=Mgen, batch_size=batch_size, C=C, N=N, a=a, g=g,
            lr=lr_G, device=device, verbose=verbose,
        )
        history["ikl"].extend(ikl_losses)

    return G_theta, v_phi, history
