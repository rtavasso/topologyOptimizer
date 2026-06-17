# Topology-Aware Prediction of Neural Network Training Dynamics

## Implementation and Research Specification

**Working title:** TAD — Topology-Aware Dynamics  
**Primary research question:** Does network topology predict systematic rotation, scaling, coordination, and functional evolution of future updates beyond what tuned momentum, online subspace tracking, and current-gradient methods already capture—and can that residual predictability improve training after full compute amortization?

**Status:** Implementation-ready research specification  
**Intended reader:** An engineering LLM starting from an empty repository  
**Primary framework:** PyTorch  
**Initial scope:** Synthetic regression tasks, deep linear networks, and an early two-layer nonlinear falsification test  
**Expansion scope:** Residual MLPs and a small autoregressive transformer

---

## 1. Executive summary

A neural network can be viewed as a topology of interacting linear maps, interleaved in realistic models with nonlinearities, normalization, residual connections, and attention. Standard optimizers use current stochastic gradients plus simple coordinatewise or tensorwise history. They do not explicitly model how the network's maps evolve as a coupled dynamical system, how their functional action changes on the occupied activation subspaces, or whether future update geometry can be predicted beyond tuned momentum and online subspace tracking.

This project tests whether that residual temporal structure exists and whether it is useful.

The first experimental system is a deep linear network trained on controlled synthetic teacher-generated data:

\[
f_\theta(x)=W_LW_{L-1}\cdots W_1x.
\]

A small nonlinear two-layer MLP is introduced immediately after the initial linear sanity experiment, rather than being deferred until the end. This prevents analytic coupling in deep linear networks from being mistaken for evidence that the same predictability survives activation gating.

During ordinary training, the system records parameters, gradients, optimizer state, activations, error signals, map products, Gram matrices, singular subspaces, and functional probe actions. A separate **dynamics predictor** is trained offline to predict future functional states, gradients, updates, and subspaces at horizons from one to one hundred steps.

The central comparison is not against persistence. It is against strong history-only estimators:

- tuned exponential moving averages and momentum;
- linear autoregressive and state-space models;
- previous-subspace and EMA-subspace estimators;
- online low-rank subspace tracking;
- periodic SVD refresh using current gradients;
- analytic topology baselines in the deep-linear setting.

The core scientific target is the residual:

\[
R_{t+h}=Y_{t+h}-\widehat Y^{\text{strong baseline}}_{t+h}.
\]

The project asks whether network topology and functional state predict this residual. This prevents a learned model from receiving credit for merely rediscovering momentum, Adam, or a periodic SVD.

The work is divided into three tiers:

1. **Scientific identifiability:** Is future functional/update structure predictable beyond strong online baselines?
2. **Optimizer oracle:** Can that information construct better local updates, even with expensive candidate evaluation?
3. **Deployable optimization:** Can the method improve validation loss per fully amortized FLOP, wall-clock time on fixed hardware, or optimizer memory?

Candidate-update search is initially treated as an oracle experiment, not as an immediately efficient optimizer. It establishes whether better updates exist in the proposal family. Only after that signal exists should a learned scorer or proposal model amortize the additional forward passes.

The primary scientific outcome is a precise map of what is predictable, at which horizons, in which representations, and under which data regimes. The primary engineering outcome is an optimizer augmentation that improves validation performance after accounting for trajectory generation, predictor training, online inference, and reuse across target runs.

---

## 2. Core hypotheses

Implement every experiment so these hypotheses can be accepted or rejected independently.

### H1 — Functional dynamics are more predictable than parameter dynamics

The action of a layer on the occupied activation subspace is more predictable than its individual entries:

\[
W_{t+h}P
\]

should be easier to predict than:

\[
W_{t+h},
\]

where \(P\) is a fixed or activation-derived probe matrix.

### H2 — Future dynamics contain predictable state evolution plus batch innovation

For a future gradient or update target \(Y_{t+h}\):

\[
Y_{t+h}
=
\underbrace{\mathbb E[Y_{t+h}\mid \mathcal H_t,\mathcal T_t]}_{\text{history- and topology-predictable dynamics}}
+
\underbrace{\xi_{t+h}}_{\text{batch/data innovation}},
\]

where \(\mathcal H_t\) is optimization history and \(\mathcal T_t\) is topology-aware functional state.

The predictable term is not assumed to be merely the population gradient. Even in deterministic full-batch descent, gradients rotate and rescale as parameters move through curvature:

\[
g_{t+1}\approx g_t-\eta H_tg_t.
\]

The project tests whether topology-aware state helps predict this evolution beyond momentum and linear state-space baselines.

### H3 — Gradient/update dynamics contain a slow subspace and fast coordinates

For layer \(\ell\):

\[
G_t^\ell
=
U_t^\ell C_t^\ell(V_t^\ell)^\top+E_t^\ell,
\]

where \(U_t^\ell,V_t^\ell\) may evolve more slowly than the batch-conditioned coefficients \(C_t^\ell\). The current or predicted subspace should capture nontrivial energy in future gradients:

\[
R_{t+h\mid t}^{\ell}
=
\frac{
\|(U_t^\ell)^\top G_{t+h}^{\ell}V_t^\ell\|_F^2
}{
\|G_{t+h}^{\ell}\|_F^2
}.
\]

A learned subspace predictor is useful only if it beats previous-subspace, EMA-subspace, online subspace tracking, and periodic-SVD baselines.

### H4 — Network topology contributes information beyond layer-local history

A predictor that models all maps and their connectivity jointly should outperform independent per-layer predictors after controlling for exact analytic coupling in deep linear networks.

For a two-layer network:

\[
G_{W_1}=W_2^\top G_M,
\qquad
G_{W_2}=G_MW_1^\top.
\]

These contractions are supplied as an explicit analytic baseline. A topology-aware learner receives credit only for predictive improvement beyond that baseline.

### H5 — Internal factorization geometry matters beyond the end-to-end map

The product:

\[
M_t=W_{L,t}\cdots W_{1,t}
\]

is not sufficient because functionally equivalent factorizations can have different optimization dynamics. Gram matrices, activation covariances, error covariances, prefix/suffix products, and balance statistics should add predictive value.

### H6 — Predictability varies by horizon, phase, optimizer, and data regime

Predictability should vary across:

- initialization and warmup;
- early representation formation;
- middle training;
- late convergence;
- distribution shifts;
- rare or corrupted batches;
- optimizer family.

The model must not report one aggregate predictability number.

### H7 — Predictability is not merely optimizer imitation

Predicting Adam's applied update from \((g_t,m_t,v_t,\theta_t)\) is trivial. Useful targets must include future gradients from unseen batches, future functional states, residuals beyond strong momentum/subspace baselines, and candidate-update outcomes measured by held-out loss.

### H8 — Residual predictability can improve optimization

The useful signal may appear as:

- predicted rotation of the gradient subspace;
- cross-layer coordinated scaling;
- future update magnitude;
- curvature-sensitive turning;
- a functional target state;
- uncertainty indicating when prediction should not be trusted.

At least one integration mode should improve validation loss per compute, time to target loss on fixed hardware, or memory.

### H9 — Transfer determines practical value

A run-specific predictor may be scientifically informative but is unlikely to be economically useful. A deployable method must amortize predictor construction over multiple target runs and retain value across held-out seeds, teachers, data regimes, and preferably architectures.

---

## 3. Non-goals

The initial implementation must not attempt all of the following:

- frontier-scale LLM pretraining;
- replacement of backpropagation from the first experiment;
- full dense trajectory storage for billion-parameter models;
- claiming that LoRA proves all training is globally low-dimensional;
- relying solely on training-batch loss when evaluating proposed updates;
- treating a candidate-search oracle as an efficient optimizer before amortization;
- claiming novelty for effects explained by tuned EMA, momentum, analytic linear coupling, or periodic SVD;
- using a diffusion model before deterministic and low-rank baselines are exhausted;
- optimizing only prediction MSE without testing downstream training utility.

---

## 4. Mathematical setup

### 4.1 Deep linear student

For depth \(L\), define:

\[
h_0=x,
\]

\[
h_\ell=W_\ell h_{\ell-1},
\qquad \ell=1,\dots,L,
\]

\[
\hat y=h_L.
\]

The end-to-end map is:

\[
M_\theta=W_LW_{L-1}\cdots W_1.
\]

The base loss is mean squared error:

\[
\mathcal L_B(\theta)
=
\frac{1}{|B|}
\sum_{(x,y)\in B}
\frac{1}{d_y}
\|\hat y-y\|_2^2.
\]

Optional regularization:

\[
\mathcal L_{\text{reg}}
=
\lambda_W\sum_\ell \|W_\ell\|_F^2.
\]

### 4.2 Teacher process

Generate labels from a teacher:

\[
y=M_*(z)x+\epsilon,
\]

where \(z\) is an optional latent regime and:

\[
\epsilon\sim\mathcal N(0,\sigma_\epsilon^2I).
\]

The teacher can be:

- fixed;
- slowly drifting;
- piecewise stationary;
- sampled per run;
- low rank;
- full rank;
- spectrally ill-conditioned;
- factored through a teacher network with its own depth.

### 4.3 State evolution is nontrivial even without batch noise

For deterministic full-batch gradient descent:

\[
\theta_{t+1}=\theta_t-\eta g_t,
\]

and therefore:

\[
g_{t+1}=\nabla L(\theta_t-\eta g_t)
\approx g_t-\eta H_tg_t.
\]

The temporal prediction problem therefore contains both:

1. deterministic state evolution through curvature and network factorization;
2. stochastic innovation from the future data batch.

Every analysis must distinguish these sources by including full-batch or very-large-batch controls in addition to stochastic minibatch training.

### 4.4 Exact two-layer product dynamics

For:

\[
M=W_2W_1,
\]

and end-to-end gradient \(G_M\), the layer gradients are:

\[
G_2=G_MW_1^\top,
\]

\[
G_1=W_2^\top G_M.
\]

Under SGD and ignoring second-order update terms:

\[
\Delta M
\approx
-\eta
\left(
G_MW_1^\top W_1
+
W_2W_2^\top G_M
\right).
\]

This exact topology-dependent structure is a required validation target for the implementation. Numerical gradients and logged updates must agree with these formulas within tolerance.

---

## 5. Synthetic data-generating processes

The repository must expose all generators through configuration. Each run must be exactly reproducible from a seed.

### 5.1 Base Gaussian matrix regression

Sample:

\[
x\sim\mathcal N(0,\Sigma_x),
\]

\[
y=M_*x+\epsilon.
\]

Configuration:

```yaml
data:
  type: gaussian_matrix_regression
  input_dim: 64
  output_dim: 32
  train_size: 1000000
  validation_size: 10000
  batch_size: 256
  input_covariance:
    type: identity
  noise_std: 0.01
```

### 5.2 Input covariance families

Support:

1. **Identity**
   \[
   \Sigma_x=I.
   \]

2. **Power-law spectrum**
   \[
   \lambda_i\propto i^{-\alpha}.
   \]

3. **Spiked covariance**
   \[
   \Sigma_x=I+\sum_{j=1}^{r_s}\gamma_ju_ju_j^\top.
   \]

4. **Random rotated diagonal**
   \[
   \Sigma_x=Q\Lambda Q^\top.
   \]

5. **Low-dimensional manifold plus noise**
   \[
   x=Az+\xi,
   \quad z\in\mathbb R^r,
   \quad r\ll d_x.
   \]

6. **Mixture of covariances**
   \[
   x\sim\sum_k\pi_k\mathcal N(0,\Sigma_k).
   \]

7. **Time-varying covariance**
   \[
   \Sigma_x(t).
   \]

The generator must allow smoothly rotating principal subspaces and abrupt switches.

### 5.3 Teacher matrix families

Support:

1. random Gaussian;
2. orthogonal/semi-orthogonal;
3. prescribed rank;
4. prescribed singular-value spectrum;
5. block diagonal;
6. sparse;
7. sum of low-rank experts;
8. compositional teacher:
   \[
   M_*=T_LT_{L-1}\cdots T_1;
   \]
9. regime-conditioned teacher:
   \[
   M_*(z)=M_0+\sum_j z_j\Delta M_j;
   \]
10. slowly drifting teacher:
    \[
    M_*(t+1)=\operatorname{normalize}(M_*(t)+\delta_t).
    \]

### 5.4 Batch heterogeneity generator

Each batch may contain a mixture of:

- common/easy samples;
- rare/high-leverage samples;
- noisy samples;
- contradictory samples;
- samples from different teacher regimes;
- outliers.

This generator is required because temporal predictability may depend strongly on whether batches are homogeneous or contain surprise events.

Example:

```yaml
data:
  batch_mixture:
    components:
      - name: common
        probability: 0.90
        covariance: common_cov
        teacher: base_teacher
        noise_std: 0.01
      - name: rare
        probability: 0.08
        covariance: rare_cov
        teacher: rare_teacher
        noise_std: 0.01
      - name: corrupted
        probability: 0.02
        covariance: common_cov
        teacher: random_labels
        noise_std: 1.0
```

### 5.5 Curriculum and distribution shift

Required schedules:

- stationary;
- easy-to-hard;
- hard-to-easy;
- gradual covariance rotation;
- abrupt domain switch;
- periodic regime alternation;
- increasing noise;
- decreasing noise;
- changing mixture proportions.

Every logged training step must include the ground-truth data regime.

### 5.6 Finite dataset and streaming dataset modes

Support:

- finite fixed train set;
- deterministic streaming generator keyed by step and seed;
- resampling every epoch;
- exact replay of the batch sequence.

Exact replay is mandatory so candidate updates can be tested against identical future batches.

---

## 6. Student network families

### Phase 1A: deep linear networks

Required configurations:

- depths: 1, 2, 4, 8;
- square and rectangular layers;
- bottleneck and expansion topologies;
- residual linear network:
  \[
  h_{\ell+1}=h_\ell+W_\ell h_\ell;
  \]
- branched linear DAG.

Initial implementation target:

```yaml
model:
  type: deep_linear
  dimensions: [64, 128, 32]
  bias: false
  initialization: balanced_svd
```

Initialization modes:

- Gaussian;
- Xavier;
- orthogonal;
- identity plus noise;
- balanced factorization of an initial end-to-end map;
- deliberately unbalanced factorization.

Deep linear networks are an analytic laboratory and implementation sanity check. Their success is not treated as evidence that equivalent structure exists in nonlinear networks.

### Phase 1B: immediate nonlinear falsification test

After the first depth-2 linear experiment, implement:

\[
f(x)=W_2\phi(W_1x)
\]

with ReLU and GELU variants. Use the same synthetic teacher and covariance controls where possible.

This experiment must occur before a large deep-linear sweep. It tests whether predictability and topology benefit survive activation gating rather than merely reproducing closed-form linear contractions.

Track:

- activation masks or local Jacobians;
- activation covariance;
- error covariance;
- effective local map:
  \[
  J_t(x)=W_2D_t(x)W_1;
  \]
- probe-averaged Jacobian actions.

### Phase 2: residual MLPs

Add:

- 2–8 layers;
- ReLU, GELU, and SiLU;
- LayerNorm or RMSNorm;
- residual connections.

### Phase 3: small transformer

Only begin after the linear and immediate nonlinear falsification experiments are complete.

Suggested scale:

- 2–6 layers;
- hidden size 128–512;
- synthetic sequence tasks or a small text corpus;
- explicit logging of \(W_Q,W_K,W_V,W_O\) and MLP matrices;
- topology-derived composites:
  \[
  W_QW_K^\top,\qquad W_VW_O.
  \]

---

## 7. Required optimizer and dynamics baselines

Implement all baselines through common interfaces.

### 7.1 Training optimizers

- SGD;
- SGD with tuned momentum;
- Nesterov momentum;
- Adam;
- AdamW;
- Muon for matrix parameters, where practical;
- optional Shampoo/SOAP experiments after core milestones.

Every optimizer must expose a serializable state snapshot.

### 7.2 Direction-prediction baselines

Required:

- last gradient/update;
- constant velocity;
- tuned gradient EMA across a beta grid;
- actual optimizer momentum;
- linear autoregression;
- vector autoregression in a sketch space;
- DMD/Koopman-style linear latent transition;
- Kalman/state-space estimator where dimensions permit.

### 7.3 Subspace-prediction baselines

Required:

- previous gradient/update SVD subspace;
- SVD of EMA gradient;
- exponentially weighted covariance eigenspace;
- online PCA/Oja subspace tracker;
- periodic SVD refresh using real gradients;
- GaLore-style periodic projection mechanism, separated from its memory-saving claim.

### 7.4 Topology baselines

For deep linear networks:

- exact layer-gradient contractions from the end-to-end gradient;
- prediction from the end-to-end map only;
- prediction from map plus Gram matrices;
- independent layers;
- true graph;
- shuffled graph.

A learned predictor must beat the appropriate strong baseline, not merely persistence.

---

## 8. Trajectory logging

### 8.1 Design principles and logging modes

The logging representation and predictor representation must be co-designed before trajectory generation. The system supports two explicit modes:

**Discovery mode:** full tensors every step for small models.

**Scaling mode:** fixed sketches, probe actions, covariance summaries, spectral summaries, and selected exact blocks.

Sketch matrices and probe sets are generated once per run, versioned, and stored with the dataset.

The logging system must:

- never mutate training behavior;
- support exact replay;
- use chunked storage;
- support per-step and periodic fields;
- include schema versioning;
- validate tensor shapes and checksums;
- permit partial loading by run, step range, layer, and field.

Recommended storage:

- Zarr or HDF5 for tensor arrays;
- Parquet for scalar/tabular metadata;
- YAML/JSON for run configuration;
- safetensors for occasional full checkpoints.

### 8.2 Required per-step global fields

```text
run_id
seed
step
epoch
wall_time
optimizer_type
learning_rate
batch_size
data_regime_id
train_loss_before
train_loss_after_optional
validation_loss_periodic
gradient_global_norm
update_global_norm
parameter_global_norm
```

### 8.3 Required per-layer fields

For every linear map \(W_\ell\):

```text
W_t                         optional every step, mandatory periodically
G_t                         full for small models
optimizer_momentum_t
optimizer_second_moment_t   when applicable
DeltaW_t
activation_mean
activation_covariance
output_covariance
backprop_error_mean
backprop_error_covariance
cross_covariance_error_input
weight_frobenius_norm
gradient_frobenius_norm
update_frobenius_norm
gradient_momentum_cosine
top_singular_values_W
top_singular_values_G
top_singular_values_DeltaW
top_left_singular_vectors_G
top_right_singular_vectors_G
top_left_singular_vectors_DeltaW
top_right_singular_vectors_DeltaW
probe_actions_WP
probe_actions_GP
```

For small Phase 1 models, log full tensors every step. This is intentional. Do not prematurely optimize storage.

### 8.4 Network-level topology fields

For sequential networks:

\[
M_t=W_L\cdots W_1.
\]

Log:

- end-to-end map \(M_t\);
- end-to-end gradient where available;
- every prefix product:
  \[
  P_{\ell,t}=W_\ell\cdots W_1;
  \]
- every suffix product:
  \[
  S_{\ell,t}=W_L\cdots W_\ell;
  \]
- neighboring balance errors:
  \[
  \mathcal B_{\ell,t}
  =
  \|W_{\ell+1}^\top W_{\ell+1}
    -
    W_\ell W_\ell^\top\|_F;
  \]
- condition numbers;
- stable ranks;
- effective ranks.

For DAGs, record graph edges and path products.

### 8.5 Probe sets

Create fixed probe matrices at run initialization:

1. random Gaussian probes;
2. standard basis subset;
3. principal components of initial input distribution;
4. persistent calibration activations;
5. teacher singular vectors.

For each map:

\[
Z_{\ell,t}=W_{\ell,t}P_\ell.
\]

For each prefix/composed map:

\[
Z^{\text{prefix}}_{\ell,t}=P_{\ell,t}P_0.
\]

Probe sets must remain fixed within a run so temporal differences are meaningful.

### 8.6 Logging cadence

Default Phase 1:

- scalar fields: every step;
- full \(W,G,\Delta W\): every step;
- SVD fields: every 5 steps;
- validation evaluation: every 25 steps;
- full checkpoint: every 100 steps.

Expose all cadences in config.

---

## 9. Derived analysis dataset

Implement a preprocessing command that transforms raw trajectories into supervised windows.

For history length \(H\) and prediction horizon \(h\):

\[
X_t=
\{\mathcal S_{t-H+1},\dots,\mathcal S_t\},
\]

\[
Y_t=\mathcal T_{t+h}.
\]

Supported horizons:

\[
h\in\{1,2,4,8,16,32,64,100\}.
\]

Targets:

- `next_weight`;
- `weight_delta`;
- `future_weight_delta`;
- `next_gradient`;
- `future_gradient`;
- `next_update`;
- `future_end_to_end_map`;
- `next_probe_action`;
- `future_probe_action`;
- `gradient_subspace`;
- `update_subspace`;
- `subspace_energy_capture`;
- `future_validation_loss`;
- `candidate_update_score`;
- `residual_beyond_ema`;
- `residual_beyond_online_subspace`;
- `future_subspace_rotation`;
- `future_update_scale`.

Prevent leakage:

- split by entire trajectory/run, not random windows;
- maintain separate teacher matrices and seeds across splits;
- include out-of-distribution teacher spectra in test;
- optionally hold out network widths/depths.

---

## 10. Representations to compare

No single representation should be assumed best. Implement explicit ablations.

### R0 — Raw parameter representation

\[
\{W_{\ell,t},G_{\ell,t},M_{\ell,t}\}.
\]

### R1 — Delta representation

\[
\{\Delta W_{\ell,t},G_{\ell,t}\}.
\]

### R2 — Probe-action representation

\[
\{W_{\ell,t}P_\ell,G_{\ell,t}P_\ell\}.
\]

### R3 — Spectral representation

\[
\{U_r,\Sigma_r,V_r\}
\]

for weights, gradients, and updates.

Use subspace projectors or sign-aligned bases to avoid arbitrary SVD sign flips.

### R4 — Covariance/functional representation

\[
\{
C_{xx},
C_{yy},
C_{\delta\delta},
C_{\delta x}
\}.
\]

### R5 — Topology-aware product representation

\[
\{
W_\ell,
P_\ell,
S_\ell,
M,
W_\ell^\top W_\ell,
W_\ell W_\ell^\top
\}.
\]

### R6 — Whitened operator representation

\[
A_{\ell,t}
=
C_{yy,\ell,t}^{-1/2}
W_{\ell,t}
C_{xx,\ell,t}^{1/2}.
\]

Use regularized inverse square roots:

\[
(C+\epsilon I)^{-1/2}.
\]

### R7 — Hybrid latent representation

An encoder learns a compact state from R2–R6.

---

## 11. Dynamics predictor models

Implement models in increasing complexity.

### 11.1 Strong baseline predictors

Required:

1. persistence;
2. constant velocity;
3. tuned EMA/momentum extrapolation;
4. linear autoregression;
5. ridge regression on flattened or sketched states;
6. vector autoregression;
7. DMD/Koopman-style transition;
8. online subspace tracker;
9. periodic-SVD future-subspace baseline;
10. analytic contraction baseline for deep linear networks.

Neural predictors must be evaluated both on the raw target and on the residual beyond the strongest applicable baseline:

\[
R_{t+h}
=
Y_{t+h}-\widehat Y^{\text{baseline}}_{t+h}.
\]

### 11.2 Independent per-layer sequence predictor

Use an MLP, GRU, or small Transformer over each layer’s history independently.

Input:

\[
S^\ell_{t-H+1:t}.
\]

Output:

\[
\widehat{\Delta W}^\ell_{t+h}
\]

or compressed target.

### 11.3 Topology-aware graph dynamics model

Represent the network as a directed graph:

- nodes: tensors/maps and optional activation spaces;
- edges: composition relationships;
- node features: map state, gradient state, covariance state, optimizer state;
- edge features: dimensions, adjacency type, residual/branch relation.

Use message passing over the graph at each historical time, followed by temporal modeling.

Suggested architecture:

1. per-node state encoder;
2. \(K_g\) graph message-passing layers;
3. temporal Transformer/GRU across \(H\) states;
4. per-node decoder;
5. optional global end-to-end decoder.

For edge \(\ell\rightarrow j\):

\[
m_{\ell\to j}
=
\phi_e(z_\ell,z_j,e_{\ell j}),
\]

\[
z_j'
=
\phi_v
\left(
z_j,
\sum_{\ell\in\mathcal N(j)}m_{\ell\to j}
\right).
\]

The architecture must support variable depth and dimensions through shared modules plus dimensionality-independent sketches.

### 11.4 Slow-subspace / fast-coordinate model

Explicitly model:

\[
G_t^\ell
\approx
U_t^\ell C_t^\ell(V_t^\ell)^\top.
\]

Predict:

\[
\widehat U_{t+h}^\ell,
\quad
\widehat V_{t+h}^\ell.
\]

Option A: also predict \(C\).

Option B: at online use time, derive \(C\) from a fresh gradient:

\[
C_t=(\widehat U_t)^\top G_t\widehat V_t.
\]

This is a high-priority model because it can remain useful even if exact future gradients are unpredictable.

### 11.5 Probabilistic model

Only after deterministic baselines.

Implement conditional Gaussian or mixture-density prediction over latent deltas:

\[
p(z_{t+h}\mid z_{t-H+1:t}).
\]

A diffusion model is optional and only justified if:

- the conditional target distribution is demonstrably multimodal;
- deterministic models have high irreducible residual;
- sampled candidates improve update selection.

If implemented, diffusion should operate in a compressed update latent, not directly over every parameter.

---

## 12. Predictor training objectives

Use multiple losses.

### 12.1 Parameter loss

\[
\mathcal L_W
=
\sum_\ell
\frac{
\|\widehat W_{\ell,t+h}-W_{\ell,t+h}\|_F^2
}{
\|W_{\ell,t+h}\|_F^2+\epsilon
}.
\]

### 12.2 Delta loss

\[
\mathcal L_\Delta
=
\sum_\ell
\frac{
\|\widehat{\Delta W}_{\ell,t:h}
-\Delta W_{\ell,t:h}\|_F^2
}{
\|\Delta W_{\ell,t:h}\|_F^2+\epsilon
}.
\]

### 12.3 Direction loss

\[
\mathcal L_{\cos}
=
1-
\cos(
\widehat{\Delta W},
\Delta W
).
\]

### 12.4 Probe-action loss

\[
\mathcal L_{\text{probe}}
=
\sum_\ell
\frac{
\|
\widehat W_{\ell,t+h}P_\ell
-
W_{\ell,t+h}P_\ell
\|_F^2
}{
\|W_{\ell,t+h}P_\ell\|_F^2+\epsilon
}.
\]

### 12.5 End-to-end map loss

\[
\mathcal L_M
=
\frac{
\|\widehat M_{t+h}-M_{t+h}\|_F^2
}{
\|M_{t+h}\|_F^2+\epsilon
}.
\]

### 12.6 Functional output loss

On fixed calibration inputs \(X_c\):

\[
\mathcal L_{\text{function}}
=
\frac{
\|\widehat f_{t+h}(X_c)-f_{t+h}(X_c)\|_F^2
}{
\|f_{t+h}(X_c)\|_F^2+\epsilon
}.
\]

### 12.7 Subspace loss

Use projection-matrix distance:

\[
\mathcal L_U
=
\|
\widehat U\widehat U^\top-UU^\top
\|_F^2.
\]

Similarly for \(V\).

### 12.8 Spectral loss

\[
\mathcal L_\sigma
=
\|
\log(\widehat\sigma+\epsilon)
-
\log(\sigma+\epsilon)
\|_2^2.
\]

### 12.9 Multi-step rollout loss

For autoregressive rollout:

\[
\mathcal L_{\text{rollout}}
=
\sum_{k=1}^{K}
\gamma^{k-1}
d(\widehat S_{t+k},S_{t+k}).
\]

### 12.10 Residualized prediction loss

For a strong baseline \(b\):

\[
R_{t+h}=Y_{t+h}-\widehat Y^{(b)}_{t+h}.
\]

Train either:

\[
\widehat R_{t+h}=F_\phi(S_{\le t})
\]

and reconstruct:

\[
\widehat Y_{t+h}=\widehat Y^{(b)}_{t+h}+\widehat R_{t+h},
\]

or condition the model explicitly on the baseline prediction. Report incremental \(R^2\), action error, and optimization value beyond the baseline.

### 12.11 Combined objective

Start with:

\[
\mathcal L
=
\lambda_\Delta\mathcal L_\Delta
+
\lambda_{\text{probe}}\mathcal L_{\text{probe}}
+
\lambda_M\mathcal L_M
+
\lambda_U\mathcal L_U
+
\lambda_\sigma\mathcal L_\sigma.
\]

All weights configurable. Log each component separately.

---

## 13. Measurements of temporal structure

These analyses are mandatory before attempting optimizer integration.

### 13.1 Autocorrelation

For scalar and vector features:

\[
\rho(h)
=
\operatorname{corr}(s_t,s_{t+h}).
\]

Measure for:

- layer gradient norms;
- update norms;
- singular values;
- effective rank;
- Gram matrices;
- probe actions;
- end-to-end map delta.

### 13.2 Gradient and update cosine similarity

\[
c_G(h)
=
\cos(G_t,G_{t+h}),
\]

\[
c_\Delta(h)
=
\cos(\Delta W_t,\Delta W_{t+h}).
\]

### 13.3 Subspace overlap

For rank \(r\):

\[
O_U(t,t+h)
=
\frac{1}{r}
\|U_t^\top U_{t+h}\|_F^2,
\]

\[
O_V(t,t+h)
=
\frac{1}{r}
\|V_t^\top V_{t+h}\|_F^2.
\]

### 13.4 Future energy captured by current subspace

\[
R_{t+h\mid t}
=
\frac{
\|U_t^\top G_{t+h}V_t\|_F^2
}{
\|G_{t+h}\|_F^2
}.
\]

Compute for gradient and update subspaces, ranks:

\[
r\in\{1,2,4,8,16,32\}.
\]

### 13.5 Effective and stable rank

Effective rank:

\[
r_{\text{eff}}
=
\exp
\left(
-\sum_i p_i\log p_i
\right),
\quad
p_i=\frac{\sigma_i}{\sum_j\sigma_j}.
\]

Stable rank:

\[
r_{\text{stable}}
=
\frac{\|A\|_F^2}{\|A\|_2^2}.
\]

### 13.6 Predictable variance

For target \(Y\) and prediction \(\hat Y\):

\[
R^2
=
1-
\frac{
\sum\|Y-\hat Y\|^2
}{
\sum\|Y-\bar Y\|^2
}.
\]

Report against persistence and constant-velocity baselines.

### 13.7 Functional predictability

Measure:

\[
E_{\text{action}}
=
\mathbb E_{x\sim p_{\text{eval}}}
\|
(\widehat W-W)x
\|^2.
\]

For the whole network:

\[
E_f
=
\mathbb E_x
\|
\widehat f(x)-f(x)
\|^2.
\]

### 13.8 Innovation decomposition

Define predicted drift:

\[
D_t=\hat{\Delta W}_t,
\]

and innovation:

\[
I_t=\Delta W_t-D_t.
\]

Measure:

- energy ratio \(\|D_t\|^2/\|\Delta W_t\|^2\);
- cosine of drift with true update;
- correlation of innovation magnitude with rare/noisy batches;
- downstream value of drift-only versus innovation-only updates.

### 13.9 Phase-conditioned analysis

Bucket steps by normalized training progress and by loss:

- 0–5%;
- 5–20%;
- 20–60%;
- 60–90%;
- 90–100%.

Report every temporal metric by phase.

### 13.10 Incremental predictability beyond strong baselines

For baseline \(b\), report:

\[
\Delta R^2_b
=
R^2_{\text{TAD}}-R^2_b.
\]

Also report reduction in functional action error, subspace angle error, and future-energy-capture error relative to tuned EMA and online subspace tracking.

### 13.11 Balancedness and analytic invariants

For adjacent deep-linear layers, track the appropriate Gram-difference invariant under the configured gradient-flow assumptions. Use it as:

- a logger/autograd correctness check;
- a predictable slow quantity;
- an intervention measurement under finite-step SGD, AdamW, and candidate updates.

Do not state that gradient descent restores arbitrary imbalance. Report preservation or drift relative to initialization.

### 13.12 Topology contribution

Compare:

1. independent layer model;
2. product-only model;
3. topology-aware model;
4. topology-aware model without Gram matrices;
5. topology-aware model without activations/errors;
6. topology-aware model with shuffled graph edges.

Topology is useful only if the real graph beats the shuffled or independent baselines.

---

## 14. From prediction to optimization

Online work is divided into oracle experiments and deployable experiments.

### 14.1 Tier II oracle: candidate update selection

Generate candidates from one real gradient:

\[
\Delta_1=\Delta_{\text{tuned baseline}},
\]

\[
\Delta_2=\widehat\Delta_{\text{predictor}},
\]

\[
\Delta_3=\alpha\Delta_1+(1-\alpha)\Delta_2,
\]

plus scale, projection, and topology-coupled variants.

Evaluate candidates on a separate held-out selection microbatch:

\[
k^*=\arg\min_kL_{B_{\text{select}}}(\theta+\Delta_k).
\]

This is initially a scientific oracle. Its purpose is to answer whether the predictor exposes better local moves, not whether the method already wins on loss per FLOP.

Record:

- winner frequency;
- held-out loss improvement over tuned baseline;
- same-batch versus held-out selection gap;
- candidate diversity;
- oracle upper bound as proposal count increases;
- forward-pass cost.

### 14.2 Predicted subspace optimizer

Predict \(U,V\), compute a fresh gradient \(G\), and form:

\[
G_{\text{proj}}=UU^\top GVV^\top.
\]

Compare against:

- full baseline optimizer;
- previous-SVD subspace;
- EMA-gradient SVD;
- online PCA/Oja tracking;
- periodic real-gradient SVD refresh;
- GaLore-style projection mechanism;
- learned predicted subspace;
- projected plus residual:
  \[
  G'=G_{\text{proj}}+\lambda(G-G_{\text{proj}}).
  \]

The learned method succeeds only if it improves optimization or reduces required refresh/backward cost beyond these cheap estimators.

### 14.3 Topology-coupled update proposals

Construct proposals that coordinate adjacent maps using predicted shared modes, scale ratios, or functional target deltas. Compare against independent layerwise application of the same per-layer proposals.

For deep linear networks, include analytically derived coupled proposals so the learner is not credited for rediscovering known contractions.

### 14.4 Lookahead learning-rate and trust selection

Use predicted future state, loss, or uncertainty to choose among learning-rate multipliers and whether to trust the predictor:

\[
\eta_k\in\{0.25,0.5,1,2,4\}\eta_{\text{base}}.
\]

A confidence gate may fall back to the baseline optimizer during regime shifts or high predicted innovation.

### 14.5 Tier III: intermittent backward passes

Compute a true gradient every \(K\) steps and use predicted or extrapolated updates between:

\[
K\in\{2,4,8,16\}.
\]

Compare at equal:

- optimizer steps;
- examples;
- backward-pass count;
- FLOPs;
- wall-clock on fixed hardware;
- fully amortized cost.

### 14.6 Long-horizon target steering

Predict:

\[
\widehat W_{t+H}
\quad\text{or}\quad
\widehat M_{t+H},
\]

then define a cautious target-directed component:

\[
\Delta W_{\text{target}}
=
\gamma\frac{\widehat W_{t+H}-W_t}{H}.
\]

Blend with a fresh baseline update and gate by prediction uncertainty. Treat as speculative.

### 14.7 Learned proposal scorer and amortization

Offline, create candidate updates and record held-out outcomes:

\[
(S_t,\Delta_k)\mapsto L_{\text{eval}}(\theta_t+\Delta_k).
\]

Train a scorer to rank candidates and reduce actual forward evaluations. Only this amortized form is considered a candidate deployable optimizer.

### 14.8 Compute accounting

Report separately:

\[
C_{\text{marginal}}
=
C_{\text{online inference and extra evaluation}},
\]

and:

\[
C_{\text{amortized/run}}
=
\frac{C_{\text{trajectory generation}}+C_{\text{predictor training}}}{N_{\text{target runs}}}
+C_{\text{marginal}}.
\]

Show break-even target-run count.

---

## 15. Evaluation protocol

### 15.1 Data splits

Split by complete runs:

- train trajectories: 70%;
- validation trajectories: 15%;
- test trajectories: 15%.

Hold out combinations of:

- teacher matrix;
- seed;
- covariance;
- noise level;
- optimizer;
- depth;
- width;
- curriculum.

### 15.2 Offline metrics

Report:

- normalized MSE;
- cosine similarity;
- \(R^2\);
- subspace overlap;
- future energy capture;
- probe-action error;
- end-to-end functional error;
- rollout error by horizon;
- calibration of probabilistic predictions.

### 15.3 Online metrics

Primary:

\[
\text{validation loss versus total estimated FLOPs}.
\]

Also:

- validation loss versus step;
- validation loss versus examples;
- wall-clock time to fixed loss on identical fixed hardware;
- final validation loss at fixed compute;
- optimizer memory;
- additional storage, predictor-training cost, inference overhead, and break-even reuse count;
- training stability and divergence rate.

### 15.4 Statistical rigor

For every key result:

- minimum 5 random seeds;
- report mean, standard deviation, and confidence interval;
- paired tests when runs share batch sequences;
- use exact same data stream for optimizer comparisons;
- retain failed/diverged runs in reporting;
- predefine primary metric before large sweeps.

---

## 16. Experiment matrix

### E0 — Infrastructure and analytic validation

- one- and two-layer linear regression;
- analytic gradients and product dynamics;
- balancedness/invariant checks;
- exact replay;
- temporary candidate update restoration.

### E1 — Minimal linear crux

Depth-2 stationary linear network. Test:

1. future direction beyond tuned EMA;
2. future subspace beyond online tracking and periodic SVD;
3. topology beyond analytic contractions;
4. functional probe prediction versus raw parameter prediction;
5. transfer to one held-out teacher.

Do not run a large sweep until this report is complete.

### E2 — Immediate nonlinear falsification

Two-layer ReLU/GELU network under the same controlled task. Repeat E1's core comparisons using local Jacobian and probe-action representations.

### E3 — Full-batch versus stochastic dynamics

Compare full-batch, very-large-batch, and minibatch training to separate deterministic state evolution from batch innovation.

### E4 — Depth and topology

Compare depth 1, 2, 4, 8 and branched/residual networks.

### E5 — Distribution drift and surprise

Stationary, gradual drift, abrupt switches, rare modes, mixture batches, and corrupted samples.

### E6 — Optimizer transfer

Train on SGD/momentum trajectories and test on AdamW/Muon where representations permit, and vice versa.

### E7 — Width/depth/teacher transfer

Hold out teachers, widths, depths, and one topology.

### E8 — Oracle proposal selection

Measure whether predicted information can construct better held-out local updates, ignoring deployability initially but accounting for oracle cost.

### E9 — Predicted-subspace optimizer

Compare against all strong online subspace estimators.

### E10 — Intermittent backward and amortized scorer

Only after E8 or E9 establishes meaningful signal.

### E11 — Residual MLP and transformer extension

Proceed only after the immediate nonlinear test shows nontrivial residual predictability.

---

## 17. Success and failure criteria

### Stage A — residual temporal structure exists

At horizon \(h=1\), the topology-aware model must beat the strongest tuned history-only baseline on at least two functionally meaningful metrics:

- probe-action error;
- end-to-end functional error;
- future-update cosine;
- subspace angle;
- future-energy capture.

At \(h\ge16\), it must retain statistically significant advantage on at least one functional metric.

### Stage B — topology adds information

The true topology must outperform:

- independent layer models;
- shuffled topology;
- product-only state;
- analytic deep-linear contractions where applicable.

### Stage C — signal survives nonlinearity

The immediate two-layer ReLU/GELU experiment must show at least one residual predictive advantage beyond momentum or online subspace tracking. Failure here does not invalidate all linear dynamics analysis, but blocks broad claims about realistic networks.

### Stage D — oracle optimization value

A predictor-derived proposal family must beat a tuned optimizer update on held-out selection loss often enough and by enough magnitude to justify amortization work.

### Stage E — deployable optimization value

At least one amortized method must improve:

- validation loss per fully amortized FLOP;
- time to target loss on fixed hardware;
- final validation loss at fixed compute;
- or optimizer memory at matched quality.

### Stage F — transfer

The method must retain measurable value on held-out teachers and seeds. Practical optimizer claims additionally require reuse across multiple target runs and a reported break-even count.

### Negative-result value

The project remains useful if it establishes any of the following:

- topology adds no information beyond tuned momentum;
- subspaces persist but are no more predictable than online tracking;
- analytic linear coupling does not survive nonlinear gating;
- topology improves prediction but not update quality;
- oracle updates help but cannot be amortized;
- prediction only works run-specifically;
- long-horizon forecasts collapse under distribution shift.

---

## 18. Repository structure

```text
topology-aware-dynamics/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data/
│   ├── model/
│   ├── optimizer/
│   ├── logging/
│   ├── predictor/
│   └── experiments/
├── src/tad/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── base.py
│   │   ├── gaussian.py
│   │   ├── covariance.py
│   │   ├── teachers.py
│   │   ├── mixtures.py
│   │   └── schedules.py
│   ├── models/
│   │   ├── deep_linear.py
│   │   ├── linear_dag.py
│   │   ├── mlp.py
│   │   └── transformer.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── optimizers.py
│   │   ├── hooks.py
│   │   ├── replay.py
│   │   └── candidate_updates.py
│   ├── logging/
│   │   ├── schema.py
│   │   ├── writer.py
│   │   ├── reader.py
│   │   ├── probes.py
│   │   └── spectral.py
│   ├── topology/
│   │   ├── graph.py
│   │   ├── products.py
│   │   ├── invariants.py
│   │   └── representations.py
│   ├── datasets/
│   │   ├── trajectory_windows.py
│   │   └── preprocessing.py
│   ├── predictors/
│   │   ├── baselines.py
│   │   ├── layer_rnn.py
│   │   ├── graph_dynamics.py
│   │   ├── subspace_model.py
│   │   └── probabilistic.py
│   ├── losses/
│   │   └── dynamics_losses.py
│   ├── evaluation/
│   │   ├── offline.py
│   │   ├── temporal_structure.py
│   │   ├── online_optimizer.py
│   │   └── reports.py
│   └── utils/
│       ├── seeds.py
│       ├── linalg.py
│       └── profiling.py
├── scripts/
│   ├── generate_trajectories.py
│   ├── build_dynamics_dataset.py
│   ├── train_predictor.py
│   ├── evaluate_predictor.py
│   ├── run_online_optimizer.py
│   └── make_report.py
├── tests/
│   ├── test_gradients.py
│   ├── test_product_dynamics.py
│   ├── test_replay.py
│   ├── test_logging_schema.py
│   ├── test_subspace_metrics.py
│   ├── test_candidate_updates.py
│   └── test_determinism.py
└── artifacts/
    ├── trajectories/
    ├── processed/
    ├── checkpoints/
    └── reports/
```

---

## 19. Command-line interface

Required commands:

```bash
tad generate-trajectories --config configs/experiments/e1.yaml
tad build-dataset --config configs/experiments/e1_predict.yaml
tad train-predictor --config configs/experiments/e1_predict.yaml
tad evaluate-predictor --checkpoint path/to/checkpoint
tad analyze-temporal-structure --run-dir path/to/run
tad run-online-optimizer --config configs/experiments/e7.yaml
tad make-report --experiment-dir path/to/experiment
```

Every command must save:

- resolved config;
- git commit;
- environment information;
- random seeds;
- metrics;
- logs;
- artifacts.

---

## 20. Testing and correctness requirements

### 20.1 Analytic gradient test

For a two-layer linear network, verify:

\[
G_2=G_MW_1^\top,
\qquad
G_1=W_2^\top G_M.
\]

Compare to PyTorch autograd.

### 20.2 Product update test

Verify exact:

\[
M_{t+1}-M_t
=
\Delta W_2W_1
+
W_2\Delta W_1
+
\Delta W_2\Delta W_1.
\]

### 20.3 Replay determinism

A run replayed from seed and config must reproduce:

- batch tensors;
- losses;
- gradients;
- updates;
- parameters.

Use strict tolerances on CPU and documented tolerances on GPU.

### 20.4 Candidate update isolation

Temporary application of candidate \(\Delta\) must restore:

- model parameters;
- optimizer state;
- RNG state;
- gradients.

### 20.5 SVD sign/subspace invariance

Metrics must not change under singular-vector sign flips or rotations within degenerate subspaces.

### 20.6 Balancedness and invariant test

Under configurations where the deep-linear gradient-flow invariant applies, verify numerical preservation relative to initialization. Document finite-step and optimizer-induced drift rather than assuming restoration.

### 20.7 Strong-baseline reproducibility

Verify tuned EMA, online subspace tracking, and periodic-SVD baselines on synthetic trajectories with known dynamics.

### 20.8 No train/test leakage

Unit test that no `run_id`, teacher id, or seed is shared across trajectory splits.

---

## 21. Initial default experiment

Use this as the first end-to-end implementation target.

```yaml
experiment:
  name: e1_two_layer_stationary
  seeds: [0, 1, 2, 3, 4]

data:
  type: gaussian_matrix_regression
  input_dim: 64
  output_dim: 32
  batch_size: 256
  steps: 5000
  input_covariance:
    type: power_law
    alpha: 1.0
  teacher:
    type: prescribed_spectrum
    rank: 16
    spectrum: geometric
    condition_number: 50
  noise_std: 0.01
  validation_size: 4096

model:
  type: deep_linear
  dimensions: [64, 128, 32]
  bias: false
  initialization: xavier

optimizer:
  type: adamw
  learning_rate: 0.001
  betas: [0.9, 0.999]
  weight_decay: 0.0

logging:
  full_tensor_every: 1
  svd_every: 5
  validation_every: 25
  checkpoint_every: 100
  svd_rank: 16
  random_probe_count: 16
  activation_probe_count: 16

dynamics_dataset:
  history_lengths: [1, 4, 16, 32]
  horizons: [1, 2, 4, 8, 16, 32, 64, 100]

predictors:
  - persistence
  - constant_velocity
  - ridge
  - tuned_ema
  - vector_autoregression
  - online_subspace_tracker
  - periodic_svd
  - layer_gru
  - topology_graph_gru
  - residualized_topology_graph_gru
  - slow_subspace_fast_coordinate
```

---

## 22. Required first report

The report generated after E1 must contain:

1. training curves;
2. gradient/update norm trajectories;
3. singular-value trajectories by layer;
4. effective-rank trajectories;
5. gradient and update cosine versus horizon;
6. subspace overlap versus horizon;
7. future gradient energy captured by current subspace;
8. probe-action autocorrelation;
9. prediction metrics for all baselines;
10. topology-aware versus independent, analytic, EMA, online-tracker, and periodic-SVD baselines;
11. horizon-conditioned performance;
12. phase-conditioned performance;
13. qualitative plots of true and predicted singular trajectories;
14. residual predictability beyond the strongest baseline;
15. one-step candidate-update oracle comparisons on held-out microbatches;
16. marginal and amortized compute/storage accounting;
17. explicit conclusion for each hypothesis.

---

## 23. Implementation order

### Milestone 1 — Correct synthetic training and logging

- generators;
- deep linear model;
- deterministic replay;
- dense discovery logging;
- analytic and invariant tests.

### Milestone 2 — Minimal linear crux

- strong EMA/state-space/subspace baselines;
- probe-action and topology representations;
- residualized prediction;
- held-out teacher transfer.

### Milestone 3 — Immediate nonlinear falsification

- two-layer ReLU/GELU model;
- local Jacobian/effective-map features;
- repeat core residual-predictability tests.

### Milestone 4 — Broader temporal analysis

- data drift;
- batch heterogeneity;
- depth and graph topology;
- optimizer transfer.

### Milestone 5 — Oracle optimization value

- candidate proposal families;
- held-out forward selection;
- predicted-subspace integration.

### Milestone 6 — Amortized deployable methods

- learned proposal scorer;
- confidence gating;
- intermittent backward passes;
- full amortized compute accounting.

### Milestone 7 — Residual MLP and transformer extension

Proceed only if Milestones 2–3 show residual predictability beyond strong baselines.

---

## 24. Interpretation rules

1. LoRA demonstrates low-rank adaptation capacity, not a fixed global training subspace.
2. High prediction accuracy does not imply optimizer usefulness.
3. Predicting Adam's update from Adam state is not a result.
4. Same-batch loss reduction is not enough; use held-out selection batches.
5. Candidate search is an oracle until its cost is amortized.
6. A stable subspace may be useful even when exact gradients are unpredictable.
7. A learned subspace must beat EMA, online tracking, and periodic real-gradient SVD.
8. Functional behavior matters more than raw matrix-entry error.
9. Topology must beat shuffled topology and analytic linear baselines.
10. Deep-linear alignment or balance results must not be generalized without checking their assumptions.
11. Gradient flow preserves relevant balance differences under specific conditions; it does not generally restore arbitrary imbalance.
12. Full-batch controls are required to separate state evolution from batch noise.
13. Long-horizon predictions must be evaluated both directly and autoregressively.
14. Distribution shifts are first-class tests.
15. Practical claims require fully amortized compute and break-even reuse counts.
16. Negative results must be retained rather than tuned away.
17. Effective-rank convention must be explicit: default spectral entropy uses normalized singular-value energy \(p_i=\sigma_i^2/\sum_j\sigma_j^2\).

---

## 25. Literature context

The project is motivated by several adjacent findings, without assuming that any of them already answer the research question.

- LoRA hypothesizes and demonstrates that many downstream weight changes can be constrained to low rank, providing evidence for low-dimensional adaptation structure:  
  https://arxiv.org/abs/2106.09685

- GaLore exploits slowly changing low-rank gradient structure to reduce optimizer-state memory during full-parameter training:  
  https://arxiv.org/abs/2403.03507

- Learned-optimizer work shows that update rules can be meta-learned, while also highlighting stability and generalization difficulties:  
  https://arxiv.org/abs/1703.00441  
  https://arxiv.org/abs/2312.07174

- Deep linear networks provide an analytically tractable setting for studying gradient-flow and matrix-factorization dynamics:  
  https://epubs.siam.org/doi/abs/10.1137/24M1715519

- Symmetries and conserved quantities matter because parameterizations with different matrices can represent equivalent functions:  
  https://openreview.net/forum?id=9ZpciCOunFb

This project differs by focusing specifically on **supervised prediction of topology-aware map dynamics across optimization time** and evaluating whether those predictions improve actual training.

---

## 26. Final research question

The repository should ultimately make it possible to answer:

> Given recent optimization history, activation/error geometry, optimizer state, data-regime statistics, and the topology of interacting linear maps, can we predict systematic rotation, scaling, coordination, and functional evolution of future updates beyond tuned momentum and online subspace tracking at horizons from one to one hundred steps—and can that residual prediction be converted into better training after full amortized compute accounting?

The project should prefer a precise negative answer over a vague positive one.
