<h1 align="center">The Reciprocity Gradient</h1>

<p align="center">
  Official simulation and experiment code for <strong>"The Reciprocity Gradient"</strong> (NeurIPS 2026).
</p>

<p align="center">
  <strong>Authors:</strong><br>
  Yue Lin, Pascal Poupart, Shuhui Zhu, Dan Qiao, Wenhao Li, Yuan Liu, Hongyuan Zha, Baoxiang Wang
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.08323"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.08323-b31b1b.svg"></a>
  <a href="https://neurips.cc/virtual/2026/loc/sydney/poster/153284"><img alt="NeurIPS 2026" src="https://img.shields.io/badge/NeurIPS%202026-Poster-4b44ce.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

## Overview

In reputation-mediated multi-agent interactions, every action and evaluative signal emitted by a learning agent reshapes third-party reputations along branching chains before feeding back into its own future rewards (*the influence attribution problem*).

**Reciprocity Gradient (RG)** backpropagates reward gradients analytically through differentiable private estimators of opponents' policies learned from public observations, jointly optimizing actions ($\pi^i$) and evaluative signals ($\varphi^i$) without intrinsic rewards or reward shaping.

<details>
<summary><a name="repository-structure"></a><strong>Repository Structure</strong></summary>

```text
.
├── agent.py                                    # Transformer-based learner (separate action π and signal φ nets)
├── env.py                                      # Single-episode continuous random-matching donation game
├── env_batched.py                              # Batched differentiable donation game & payoff utilities
├── requirements.txt
├── LICENSE
└── experiments/
    ├── opponent_modeling_utils.py              # Differentiable opponent surrogates (MLP / FactoredMLP / Amortized)
    ├── checkpoint_utils.py                     # Atomic JSON checkpointing helpers
    ├── paper_style.py                          # Shared plotting style used by training scripts
    ├── population_warm_up/                     # Stationary reputation warm-up & precomputed steady_state_*.pt
    ├── exp1_backward_graph/                    # Backward computation graph & finite-difference check
    ├── exp7_joint_opponent_modeling/           # [Main / Flagship] Joint π + φ with learned opponent models (Sec. 5, Fig. 3)
    ├── exp5e_explore_pretrain/                 # [Main] Action-only π with learned opponent models (Sec. 5, B.1)
    ├── exp6_signal_opponent_modeling/          # [Main] Signal-only φ with learned opponent models (Sec. 5, B.2)
    ├── exp9_scalability/                       # [Main] Population scaling N=5..30 via parameter sharing (Sec. 5, D)
    ├── exp2_action_optimization/               # [Oracle Ref] Action-only vs. true opponents (App. H, J.2)
    ├── exp3_signal_optimization/               # [Oracle Ref] Signal-only vs. true opponents (App. H, J.2)
    └── exp4_joint_gt/                          # [Oracle Ref] Joint π + φ vs. true opponents (App. J.2)
```

</details>

## Installation

Python 3.10+ (validated on Python 3.12 with PyTorch 2.x on CPU) is recommended:

```bash
pip install -r requirements.txt
```

<details>
<summary>To render computation graphs in <code>experiments/exp1_backward_graph/backward_graph.py</code>, also install <code>torchviz</code> and system <code>graphviz</code>:</summary>

```bash
pip install torchviz
# macOS: brew install graphviz
# Ubuntu/Debian: sudo apt-get install graphviz
```

</details>

**Quick smoke test** (run from the repository root):

```bash
python experiments/exp9_scalability/train_expD.py --smoke
```

## Running Experiments

> Run all scripts from the repository root directory.

### 1. Prerequisites & Mechanism Verification

#### Stationary Reputation Warm-Up (`experiments/population_warm_up/`):

Precomputed stationary distributions (`steady_state_{L3,L6,identity,mixed_hard,mixed_soft}.pt`) are **already included in this repository**. They are only needed by **Scalability (`exp9_scalability`)** and steady-state initialization runs in `exp2`/`exp3`; you do **not** need to run warm-up before running the main Learned OM experiments (`exp7`, `exp5e`, `exp6`).

To regenerate them from scratch:

```bash
python experiments/population_warm_up/warm_up.py
python experiments/population_warm_up/warm_up_mixed.py
```

#### Backward Computation Graph & Finite-Difference Verification (Section 4 / Appendix D):

Standalone 3-agent sanity check verifying that one `torch.autograd.grad` pass matches finite differences across agent boundaries (no prerequisites):

```bash
python experiments/exp1_backward_graph/backward_graph.py
```

### 2. Main Experiments (Section 5 & Figure 3)

These are the primary experiments of the paper using **Learned Opponent Modeling** (purely observational, no oracle access to opponent parameters).

#### (a) Flagship Main Experiment — Joint Action + Signal Optimization (Section 5, Figure 3, Setting B.3)

> **Prerequisites:** **None** (fully self-contained; uses `all_uniform` initialization and runs real-env collection, surrogate fitting, and joint RG optimization end-to-end).

```bash
# Main flagship setting: HybridCoop + AllDefector (T_outer=200, asymmetric LR)
python experiments/exp7_joint_opponent_modeling/train_exp7_joint.py --opp-set HybridCoopAllD
```

```bash
# Run all 4 opponent sets in Table 12 (HybridCoopAllD, L3, L6, ProudCoopAllD)
python experiments/exp7_joint_opponent_modeling/train_exp7_joint.py
```

```bash
# Multi-seed (seeds 0..4) policy-evolution snapshots used in Figure 3 (T_outer=125)
python experiments/exp7_joint_opponent_modeling/run_seed_snapshots_multi.py
```

#### (b) Single-Pathway Main Experiments — Action-Only (B.1) & Signal-Only (B.2)

> **Prerequisites:** **None** (fully self-contained; in `train_exp5e.py`, Phase 0 random-policy exploration pretraining runs automatically inside the script before policy training).

```bash
# B.1: Action-only vs. L6 (exploration-based pretraining + frozen FactoredMLP surrogates)
python experiments/exp5e_explore_pretrain/train_exp5e.py
python experiments/exp5e_explore_pretrain/run_seed_snapshot_5e_multi.py
```

```bash
# B.2: Signal-only vs. ProudCoop + AllDefector (c=5, online-refitted surrogates)
python experiments/exp6_signal_opponent_modeling/train_exp6_signal_only.py    # 15 outer iters (MLP/TF/Factored)
python experiments/exp6_signal_opponent_modeling/run_seed_snapshot_6_multi.py # 125 outer iters (MLP, LR=8e-3)
```

#### (c) Population Scalability via Parameter Sharing (Section 5, $N \in \{5, 10, 15, 20, 25, 30\}$)

> **Prerequisites:** Loads `experiments/population_warm_up/steady_state_{L3,L6}.pt` (**already included** in the repo). Run `train_exp9_n10_stable.py` across $N$ **before** calling `aggregate_stable.py` to summarize the results.

```bash
# Step 1: Run stabilized amortized-estimator sweep for N in {5, 10, 15, 20, 25, 30}
for N in 5 10 15 20 25 30; do
    python experiments/exp9_scalability/train_exp9_n10_stable.py --N $N --pathway all
done
```

```bash
# Step 2: Aggregate per-N JSON outputs into summary tables
python experiments/exp9_scalability/aggregate_stable.py
```

```bash
# Additional per-opponent vs. amortized comparison sweeps
python experiments/exp9_scalability/train_exp9_scalability.py
python experiments/exp9_scalability/train_expD.py
```

### 3. Oracle Reference Benchmarks (Appendices B, H, J.2)

These scripts backpropagate directly through the true opponent functions (white-box oracle access) to establish upper-bound references for the main experiments above.

> **Prerequisites:** Only `train_vs_leading_eight.py` (in `exp2`/`exp3`) loads the included `steady_state_*.pt` files; all other scripts below require no prior step.

```bash
# A.1: Action-only Oracle (vs. Identity and Leading Eight L3/L6)
python experiments/exp2_action_optimization/train_vs_identity.py
python experiments/exp2_action_optimization/train_vs_leading_eight.py
```

```bash
# A.2: Signal-only Oracle (vs. Identity, Leading Eight, and ProudCoop + AllDefector)
python experiments/exp3_signal_optimization/train_vs_identity.py
python experiments/exp3_signal_optimization/train_vs_leading_eight.py
python experiments/exp3_signal_optimization/train_vs_mixed_exp3h1.py          # c=1 (Table 7, ref 2.25)
python experiments/exp3_signal_optimization/train_vs_mixed_exp3h1_diag_c5.py  # c=5 diagnostic (ref 1.25)
```

```bash
# A.3: Joint (Action + Signal) Oracle
# HybridCoop + AllDefector asymmetric-LR protocol (Figure 11 / Table 13, 25k epochs, pi-LR=3e-5, phi-LR=3e-3)
python experiments/exp4_joint_gt/sweep_exp4_joint.py \
    --config PP_v2_100x_budget --seed 0 \
    --out experiments/exp4_joint_gt/pp_v2_seed0.json
# L3 / L6 / ProudCoop + AllDefector baseline joint Oracle runs
python experiments/exp4_joint_gt/train_exp4_joint_gt.py
```

## Citation

**Paper (arXiv):** <https://arxiv.org/abs/2605.08323><br>
**NeurIPS 2026 Poster Page:** <https://neurips.cc/virtual/2026/loc/sydney/poster/153284>

If you find this codebase useful in your research, please cite:

```bibtex
@inproceedings{lin2026reciprocity,
  title={The Reciprocity Gradient},
  author={Lin, Yue and Poupart, Pascal and Zhu, Shuhui and Qiao, Dan and Li, Wenhao and Liu, Yuan and Zha, Hongyuan and Wang, Baoxiang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://arxiv.org/abs/2605.08323}
}
```

## License

This project is released under the [MIT License](LICENSE).
