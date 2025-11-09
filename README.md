## Technical Implementation of Shortcut Model (per official repo)

We implement a function that provides training targets (which differ across methods such as flow matching, shortcut models, consistency models, etc.):

`(x_t, v_t, t, k) = get_targets(method = flow_matching / shortcut / ...)`

- `k` is the level-code parameter used to encode `dt` (the step size); used only in certain methods.
- The DiT model takes inputs `(x_t, t, y)` (and optionally `k`) and outputs `v_hat`.
- Loss: mean squared error between target and prediction:

$$
\text{loss} = \mathrm{MSE}(v_t, \hat{v}_t)
$$

---

## Flow Matching

### Setup

Given a data sample $$x_1$$ and Gaussian noise $$x_0 \sim \mathcal{N}(0, I)$$, define the linear flow:

$$
x_t = (1 - t)\,x_0 + t\,x_1,\quad t \in [0, 1].
$$

Velocity field (constant in \(t\)):

$$
v_t = \frac{\partial x_t}{\partial t} = x_1 - x_0.
$$

**Added endpoint guard** (avoid degeneracy at \(t = 1\)):

$$
x_t = (1 - (1-\varepsilon)t)\,x_0 + t\,x_1,\quad \varepsilon \approx 10^{-5},
$$

which gives

$$
v_t = x_1 - (1-\varepsilon)x_0.
$$

### Batch Construction

For a mini-batch of size \(B\):

1. **Label dropout (CFG training)**  
   With probability `class_dropout_prob`, replace label \(y\) by the unconditional id `num_classes`.  
   This enables classifier-free guidance at inference.

2. **Sample times** 
   The code includes two time sampling methods:
   
   (a) via a Kumaraswamy($$\rho=2$$) transform (Beta(2,2)-like), then clamp to \([0.02, 0.98]\):

   - Sample $$\(u \sim \mathcal{U}(0,1)\)$$
   - Set $$\(t = \big(1 - (1-u)^{1/\rho}\big)^{1/\rho}\)$$

   (b) A uniform bin sampling - $${0,1/T,2/T, ..., 127/T}$$, where `T` is the `denoise_timesteps` parameter. 

4. **Noise & flow pairs**

   - Sample $$x_0 \sim \mathcal{N}(0, I)$$
   - Set $$x_t = (1-t)x_0 + t x_1$$
   - Set $$v_t = x_1 - x_0$$

5. **Level code (sentinel)**

   - Let `T = denoise_timesteps`, $$K = log_2 T$$
   - Attach constant level code $$k = K$$ (ignored in pure FM; keeps interface compatible)

### Training Objective

DiT model $$f_\theta(x, t, k, y)$$ predicts velocity:

$$
\mathcal{L}_{\text{FM}}
= \mathbb{E}\left[ \| f_\theta(x_t, t, k{=}K, y_{\text{eff}}) - v_t \|_2^2 \right]
$$



where $$y_{\text{eff}}\$$ are labels after dropout.

### Inference (Uniform N Steps)

Use Euler with uniform steps \(\Delta t = 1/T\):

$$
x \leftarrow x + \Delta t \cdot f_\theta(x, t, k, y),
$$

with \(t\) advanced on a fixed grid (e.g. midpoints).

### Notes

- Beta(2,2)-style sampling and clamping away from endpoints improve stability.
- Sentinel level code $$\(k = K\)$$ keeps compatibility with $$\(k\)$$-conditioned shortcut models.

---

## Shortcut Models

We follow the official paper and JAX repo, with added flexibility.

### Notation

- $$T$$: (power-of-two) number of denoising bins  
- $$K = log_2 T$$  
- Dyadic level $$k$$ encodes step size: $$dt = 2^{-k}$$

- Time $$t \in [0,1]$$ and level $$k$$ have separate embeddings.  
- $$x_t = (1 - (1-\varepsilon)t) x_0 + t x_1.$$

---

## Shortcut Model Training Procedure
The code has comments based on the following 1-5.


### 1. Sample Step Size `dt` (Bootstrap Slice)

For a bootstrap sub-batch:

- Sample levels $$k \in \{0, \dots, K-1\}$$.  
- If batch size is smaller than number of levels, pad with \(k = 0\).

**Outputs:**

- Student level code: $$k$$ 
- Teacher level code: $$k + 1$$ 
- Step sizes: $$dt = 2^{-k}$$ and $$dt/2$$

---

### 2. Sample Start Time `t` (Bootstrap Slice)

Given chosen $$dt$$:

- Uniform sample:
  $$t \in \{0, dt, 2dt, \dots, 1 - dt\}$$
- Construct $$x_t$$ on the linear path (with $$\varepsilon$$).

---

### 3. Bootstrap “Shortcut” Teacher (Local Target)

Use a two-call Heun (trapezoid) estimate at the half-step level:

1. **Predictor**

   $$v_{b1} = f_{\text{teacher}}(x_t, t, k+1, y)$$

2. **Half update**

   $$x_{t_2} = \mathrm{clip}\big(x_t + \tfrac{dt}{2}\,v_{b1}, [-4, 4]\big)$$ ; 
   $$t_2 = t + \tfrac{dt}{2}$$

3. **Corrector**

   $$v_{b2} = f_{\text{teacher}}(x_{t_2}, t_2, k+1, y)$$

4. **Local target**

   $$v_{text{target}} = \tfrac12\,(v_{b1} + v_{b2})$$

Student prediction at full level:

$$v_{\text{pred}} = f_\theta(x_t, t, k, y)$$

Loss:

$$\big\| v_{\text{pred}} - v_{\text{target}} \big\|_2^2.$$

---

### 4. Flow-Matching Targets (Global Supervision)

For the remaining batch items:

1. Sample $$t in {0, 1/T, \dots, T-1/T}$$.  
2. Build $$x_t$$ as above.  
3. Set FM target:

   $$v_t = x_1 - (1-\varepsilon)x_0$$

4. Apply label dropout for CFG.  
5. Attach sentinel level code \(k = K\).

Train with:

$$
\big\|| f_\theta(x_t, t, \text{level}{=}k, y) - v_t \big\||_2^2.
$$

---

## Merge, Loss, and Diagnostics

- Let per-rank batch size be $$B$$.  
- Let

  $$B_{\text{boot}} = \left\lfloor \frac{B}{\texttt{bootstrap\_every}} \right\rfloor.$$

- Use:
  - $$B_{\text{boot}}$$ samples for shortcut bootstrap.  
  - $$B - B_{\text{boot}}$$ samples for FM.

Concatenate both subsets and average/sum their losses.

---

## Control Flags

- **`bootstrap_every`**  
  Fraction of the mini-batch used for bootstrap.  
  Example: `4` → \(1/4\) bootstrap, \(3/4\) FM.

- **`bootstrap_dt_bias`**  
  (Bins scheme only)  
  - `0`: uniform over levels $$k$$.  
  - `>0`: bias towards coarser steps (small $$k$$, large `dt`), still covering finer levels.

- **`bootstrap_cfg`**  
  Enables CFG inside teacher:

  $$v_{\text{cfg}} = v_{\text{uncond}} + s\,(v_{\text{cond}} - v_{\text{uncond}})$$

  Student matches this guided target.

- **`bootstrap_ema`**  
  - On: teacher uses EMA weights (recommended).  
  - Off: teacher uses live weights (cheaper, noisier).

---

## Practical Defaults

Recommended settings:

- `bootstrap_every` = 4–8  
- `bootstrap_dt_bias` = 0 initially  
- `bootstrap_cfg` = 1 (if using CFG at inference)  
- `bootstrap_ema` = on  

Implementation notes:

- Normalize level code by \(K\) inside the `dt`-embedder.  
- Keep \(t\) and normalized level code as floats.  
- Labels are integer class ids.

---

## Sampling in Shortcut Models (Official Implementation)

Let:

- $$M = 128$$ (total discretization steps)  
- $$log_2(M) = 7$$

The actual scheme:

- Step sizes: $$d \in {1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64}$$

- Smallest step is \(1/64\), not \(1/128\), because self-consistency uses two steps of size \(d/2\).  
- For each $$d$$, start times: $$t \in \{0, d, 2d, \dots, 1-d\}$$

- With `bootstrap_dt_bias = 0`, each dyadic step size appears with equal frequency → uniform over log step sizes.

This yields structured coverage across scales, but the distribution is **fixed** and **not adaptive** to model difficulty.

Our adaptive method modifies the \((t, d)\) sampling distribution based on model error.

---

