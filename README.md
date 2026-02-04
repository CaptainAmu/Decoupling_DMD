# DMD: Flow Matching with Class Conditioning

Flow-matching velocity model v_θ(x_t, t, c) with simple/pair data sources, training, ODE inference, and score/clean-prediction definitions.

## Layout

- **`data/sampling.py`**
  - **`simple_sample(C, N, dims=None)`**  
    Callable that samples from C classes of blobs in R^N. Each class has an isotropic Gaussian; `dims` is a length-C list (default `[N]*C`) controlling per-class scale.  
    Call with `K` to get `(x, c)` with `x` shape `(K, N)` and `c` in `{0, ..., C-1}`.

  - **`pair_sample(f, N)`**  
    Callable that, when called with `K`, samples `z ~ N(0, I_N)` and returns `(x0, c, z) = (f(z)[0], f(z)[1], z)`. Use the returned `z` as ε in the pair-sample training objective.

- **`models/velocity.py`**
  - **`VelocityMLP(num_classes, dim, ...)`**  
    Predicts instantaneous velocity `v_θ(x_t, t, c)`. Uses a class embedding for `num_classes + 1` classes (0,…,C−1 data + C = ∅), a time embedding, concat with `x_t`, and an MLP.  
    `forward(x_t, t, c)` returns velocity of shape `(B, N)`.

- **`training.py`**
  - **`train_flow_matching(model, data_source, num_steps, batch_size, drop_label_prob=p, ...)`**  
    Trains `v_θ` with the flow-matching loss.  
    - If `data_source(K, device)` returns **2** values `(x0, c)`, it uses the simple-sample objective (draws ε ~ N(0,I) internally).  
    - If it returns **3** values `(x0, c, epsilon)`, it uses the pair-sample objective with that ε.  
    Labels are replaced by the empty class with probability `drop_label_prob`.

- **`inference.py`**
  - **`ode_backward(v_model, T, K, c, device=None)`**  
    Integrates the ODE backward from t=1 (noise) to t≈0 with T steps and step size 1/T:  
    `x_{t - 1/T} = x_t - v_θ(x_t, t, c) * (1/T)`.  
    Returns approximate samples of shape `(K, N)`.

- **`score.py`**
  - **`clean_pred(v_model, x_t, t, c)`**  
    Clean-prediction: `x(x_t, t, c) := x_t - t * v(x_t, t, c)`.

  - **`score_s(v_model, x_t, t, c, g=1.0, empty_label=None)`**  
    Score model  
    `s = (t-1)/t * v(∅) - 1/t * x_t + g * ( (t-1)/t * (v(c) - v(∅)) )`  
    with ∅ = empty class index (`num_classes` by default). `g=0` recovers the unconditional score.

## Usage sketch

```python
from DMD import simple_sample, pair_sample, VelocityMLP, train_flow_matching, ode_backward, clean_pred, score_s

# Data
C, N = 5, 32
sampler = simple_sample(C, N, dims=[N]*C)
x0, c = sampler(128)  # 128 samples

# Model and training
model = VelocityMLP(num_classes=C, dim=N, hidden_dim=256, num_layers=4)
train_flow_matching(model, sampler, num_steps=5000, batch_size=128, drop_label_prob=0.1)

# Inference
c_gen = torch.randint(0, C, (64,))
x0_gen = ode_backward(model, T=50, K=64, c=c_gen)

# Score and clean prediction (fixed pretrained v)
x_clean = clean_pred(model, x_t, t, c)
s = score_s(model, x_t, t, c, g=1.0)
```
