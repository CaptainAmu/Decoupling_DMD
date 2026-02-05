# DMD: Flow Matching with Class Conditioning

Flow-matching velocity model v_θ(x_t, t, c) with simple/pair data sources, training, ODE inference, and score/clean-prediction definitions.

## Layout

- **`data/sampling.py`**
  - **`simple_sample(C, N, dims=None)`**  
    Callable that samples from **C classes of blobs in R^N**, where each class lives on a `dims[c]`‑dimensional Gaussian subspace embedded in R^N.  
    - `dims` is a length‑C list (default `[N]*C`), with `1 <= dims[c] <= N`.  
    - Example: `dims = [2,2,2,2,2,1]` gives 6 classes, 5 are 2‑D blobs, 1 is 1‑D.  
    - Class means are placed on a large sphere in R^N to keep blobs well separated.  
    Call with:
      - `sampler(K, device=None)` → random class per point (uniform over `{0, …, C-1}`), returns `(x, c)` with `x` shape `(K, N)`, `c` in `{0, ..., C-1}`.  
      - `sampler(K, device=None, class_idx=k)` with `k in {0, …, C-1}` → all points from class `k`.  
      - `sampler(K, device=None, class_idx=C)` or `class_idx=None` → each point gets a random class in `{0, …, C-1}`.  
      - `sampler(K, device=None, class_idx=vec)` where `vec` is length‑K → per‑sample class control.

  - **`pair_sample(f, N)`**  
    Callable that, when called with `K`, samples `z ~ N(0, I_N)` and returns `(x0, c, z)` where `(x0, c) = f(z)`. Use the returned `z` as ε in the pair‑sample training objective.

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
  - **`ode_backward(v_model, T, K, c, device=None, guidance=1.0)`**  
    Integrates the ODE backward from t=1 (noise) to t≈0 with T steps and step size 1/T:  
    `x_{t - 1/T} = x_t - v_θ(x_t, t, c) * (1/T)`.  
    When `guidance != 1.0`, it instead uses
    \[
    v_{\text{guided}}(x_t,t,c,g) = v(x_t,t,\emptyset) + g \cdot (v(x_t,t,c) - v(x_t,t,\emptyset))
    \]
    as the velocity (class-conditional guidance).  
    Returns approximate samples of shape `(K, N)`.

  - **`compare_inference(v_model, T, K, c, sampler, device=None, guidance=1.0)`**  
    Helper that runs `ode_backward` and the true sampler side‑by‑side to compare learned vs true distributions for given class control `c`.  
    - `c` can be:
      - `int in {0, …, C-1}`: all samples are from that class.  
      - `int == C`: all samples are uniformly drawn over `{0, …, C-1}`.  
      - a length‑K tensor / list of ints: per‑sample class indices in `{0, …, C-1}` (or all C for random).

- **`score.py`**
  - **`clean_pred(v_model, x_t, t, c)`**  
    Clean-prediction: `x(x_t, t, c) := x_t - t * v(x_t, t, c)`.

  - **`score_s(v_model, x_t, t, c, g=1.0, empty_label=None)`**  
    Score model  
    `s = (t-1)/t * v(∅) - 1/t * x_t + g * ( (t-1)/t * (v(c) - v(∅)) )`  
    with ∅ = empty class index (`num_classes` by default). `g=0` recovers the unconditional score.

## Usage sketch

```python
from DMD import simple_sample, pair_sample, VelocityMLP, train_flow_matching, ode_backward, CleanPredModel, ScoreModel

# Data
C, N = 5, 32
sampler = simple_sample(C, N, dims=[N]*C)
x0, c = sampler(128)  # 128 samples (random classes 0..C-1)

# Model and training
model = VelocityMLP(num_classes=C, dim=N, hidden_dim=256, num_layers=4)
train_flow_matching(model, sampler, num_steps=5000, batch_size=128, drop_label_prob=0.1)

# Inference (e.g. all samples from class 2 with guidance g=3.0)
c_gen = 2
x0_gen = ode_backward(model, T=50, K=64, c=c_gen, guidance=3.0)

# Score and clean prediction (fixed pretrained v)
clean_model = CleanPredModel(model)
score_model = ScoreModel(model)

x_clean = clean_model(x_t, t, c)
s = score_model(x_t, t, c, g=1.0)
```
