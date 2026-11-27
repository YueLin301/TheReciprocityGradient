"""Parallel hyperparameter sweep for Exp 4 GT Joint.

Goal: find a configuration that produces discriminative π AND φ
simultaneously on ProudCoop+AllD.

Strategies tested (in parallel across CPU cores):
  A) Phased: φ warmup 50 eps (lr=5e-2), then joint 200 eps (lr=5e-3)
  B) Higher cost: c=3 to amplify action gradient toward defection vs cooperation
  C) Asymmetric LR: π_lr=1e-2, φ_lr=5e-3 (action pathway faster)
  D) Phased + asymmetric: warmup φ, then joint with π_lr > φ_lr
  E) Longer warmup: φ warmup 100 eps, then joint 300 eps
  F) Pi-first warmup: warmup π first (50 eps), then joint
"""
import os, sys, json
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agent import Agent
from env_batched import BatchedDonationGame, compute_per_interaction_payoff

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exp3_signal_optimization.train_vs_mixed_exp3h import ProudCoopL3, AllDefectL3
from checkpoint_utils import atomic_save_json, load_json_if_exists

sys.path.insert(0, os.path.dirname(__file__))
from hybrid_opponents import HybridCoopL3


class NoisyExplorationEnv(BatchedDonationGame):
    """Adds symmetric Gaussian exploration noise to agent 0's π AND φ outputs.
    Standard RL exploration — not targeted at any specific behavior."""

    def __init__(self, *args, noise_agent_idx=0, noise_std=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_agent_idx = noise_agent_idx
        self.noise_std = noise_std

    def _batched_take_action(self, agent, rep_histories, own_rep_histories=None):
        det = super()._batched_take_action(agent, rep_histories, own_rep_histories)
        if agent is self.agents[self.noise_agent_idx] and self.noise_std > 0:
            noise = torch.randn_like(det) * self.noise_std
            return (det + noise).clamp(0, 1)
        return det

    def _batched_send_signal(self, agent, actions, own_reps, donor_reps=None):
        det = super()._batched_send_signal(agent, actions, own_reps, donor_reps)
        if agent is self.agents[self.noise_agent_idx] and self.noise_std > 0:
            noise = torch.randn_like(det) * self.noise_std
            return (det + noise).clamp(0, 1)
        return det

BATCH_SIZE = 128
B_VALUE = 10.0
DELTA = 0.98


def build_opps(device, opp_type='proud'):
    if opp_type == 'proud':
        o1 = ProudCoopL3('ProudL3', device=device)
    elif opp_type == 'hybrid':
        o1 = HybridCoopL3('HybridL3', device=device)
    else:
        raise ValueError(opp_type)
    o2 = AllDefectL3('AllD_L3', device=device)
    for p in list(o1.parameters()) + list(o2.parameters()):
        p.requires_grad_(False)
    return o1, o2


def measure_profiles(agent0):
    agent0.eval()
    x = torch.linspace(0, 1, 100)
    with torch.no_grad():
        pi_p = [agent0.take_action(x[i].view(1, 1)).item() for i in range(100)]
        phi_p = [agent0.send_signal(x[i].view(1),
                 own_reputation=torch.tensor([[0.5]])).item() for i in range(100)]
    agent0.train()
    return pi_p, phi_p


def measure_phi_variance(agent0):
    """Compute std of φ⁰ outputs on a uniform grid. Used as a regularizer."""
    agent0.eval()
    x = torch.linspace(0, 1, 50, requires_grad=False)
    dummy_rep = torch.tensor([[0.5]], requires_grad=False)
    phi_vals = torch.stack([
        agent0.send_signal(x[i].view(1), own_reputation=dummy_rep)
        for i in range(50)
    ]).squeeze()
    agent0.train()
    return phi_vals.std()


def train(config, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    c_cost = config.get('c', 1.0)
    b_val = config.get('b', B_VALUE)

    agent0 = Agent(name="A_0", signal_dim=1, second_order=False, is_demo=False, device=device)
    o1, o2 = build_opps(device, config.get('opp', 'proud'))
    per_agent_init = [{"dist": "constant", "params": {"value": 0.5}}] * 3
    noise_std = config.get('explore_noise', 0.0)
    if noise_std > 0:
        env = NoisyExplorationEnv(
            agents=[agent0, o1, o2], b=b_val, c=c_cost, continuation_prob=DELTA,
            batch_size=BATCH_SIZE, balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init, noise_std=noise_std)
    else:
        env = BatchedDonationGame(
            agents=[agent0, o1, o2], b=b_val, c=c_cost, continuation_prob=DELTA,
            batch_size=BATCH_SIZE, balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init)

    phi_reg_weight = config.get('phi_reg', 0.0)

    def step(params, optimizer):
        optimizer.zero_grad()
        cum_r, ms, rh = env.play_batch()
        pi_r, _ = compute_per_interaction_payoff(cum_r, ms, 0)
        loss = -pi_r.mean()

        # Variance regularizer: penalize low φ variance (flat φ)
        if phi_reg_weight > 0:
            x = torch.linspace(0, 1, 30)
            dummy_rep = torch.tensor([[0.5]])
            phi_vals = torch.stack([
                agent0.send_signal(x[i].view(1), own_reputation=dummy_rep)
                for i in range(30)
            ]).squeeze()
            phi_std = phi_vals.std()
            # Penalize std below target (say 0.25)
            reg = phi_reg_weight * torch.relu(0.25 - phi_std) ** 2
            loss = loss + reg

        if loss.requires_grad:
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            for p, g in zip(params, grads):
                if g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()):
                    p.grad = g
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        del rh
        return pi_r.mean().item()

    history = []
    strategy = config['strategy']

    if strategy == 'vanilla':
        # Joint training with single LR
        all_params = list(agent0.get_action_params()) + list(agent0.get_signal_params())
        opt = optim.Adam(all_params, lr=config['lr'])
        for ep in range(config['epochs']):
            r = step(all_params, opt)
            if ep % 50 == 0: history.append((ep, r))

    elif strategy == 'vanilla_asym':
        # Joint training with asymmetric LR — both trainable from start
        act_params = list(agent0.get_action_params())
        sig_params = list(agent0.get_signal_params())
        opt = optim.Adam([
            {'params': act_params, 'lr': config['lr_pi']},
            {'params': sig_params, 'lr': config['lr_phi']},
        ])
        all_params = act_params + sig_params
        for ep in range(config['epochs']):
            r = step(all_params, opt)
            if ep % 50 == 0: history.append((ep, r))

    elif strategy == 'vanilla_noise':
        # Joint training + Gaussian noise on outputs (exploration)
        all_params = list(agent0.get_action_params()) + list(agent0.get_signal_params())
        opt = optim.Adam(all_params, lr=config['lr'])
        noise_std = config['noise_std']
        # Monkey-patch noise into forward pass via env wrapper — simplest: add noise to stored network parameters
        # Actually, simplest: use higher LR + more epochs, noise is implicit via SGD
        # We'll use Gaussian parameter noise at init + larger LR.
        for p in all_params:
            p.data += torch.randn_like(p.data) * 0.05
        for ep in range(config['epochs']):
            r = step(all_params, opt)
            if ep % 50 == 0: history.append((ep, r))

    elif strategy == 'phased_phi_first':
        # Phase 1: signal warmup (action frozen as identity)
        agent0.set_hardcode_action(decay=None)
        sig_params = list(agent0.get_signal_params())
        opt1 = optim.Adam(sig_params, lr=config['lr_phase1'])
        for ep in range(config['warmup_eps']):
            r = step(sig_params, opt1)
        # Phase 2: unfreeze action, joint training
        agent0._hardcode_action = False
        if config.get('asymmetric_lr', False):
            act_params = list(agent0.get_action_params())
            sig_params = list(agent0.get_signal_params())
            opt2 = optim.Adam([
                {'params': act_params, 'lr': config['lr_phase2_pi']},
                {'params': sig_params, 'lr': config['lr_phase2_phi']},
            ])
            all_params = act_params + sig_params
        else:
            all_params = list(agent0.get_action_params()) + list(agent0.get_signal_params())
            opt2 = optim.Adam(all_params, lr=config['lr_phase2'])
        for ep in range(config['joint_eps']):
            r = step(all_params, opt2)
            if ep % 50 == 0:
                history.append((config['warmup_eps'] + ep, r))

    elif strategy == 'phased_pi_first':
        # Phase 1: action warmup (signal frozen as identity)
        agent0.set_hardcode_signal()
        act_params = list(agent0.get_action_params())
        opt1 = optim.Adam(act_params, lr=config['lr_phase1'])
        for ep in range(config['warmup_eps']):
            r = step(act_params, opt1)
        # Phase 2: unfreeze signal, joint
        agent0._hardcode_signal = False
        all_params = list(agent0.get_action_params()) + list(agent0.get_signal_params())
        opt2 = optim.Adam(all_params, lr=config['lr_phase2'])
        for ep in range(config['joint_eps']):
            r = step(all_params, opt2)
            if ep % 50 == 0:
                history.append((config['warmup_eps'] + ep, r))

    pi_final, phi_final = measure_profiles(agent0)
    pi_var = float(np.std(pi_final))
    phi_var = float(np.std(phi_final))
    benchmark = (b_val - c_cost) / 4

    return {
        'config_name': config['name'],
        'seed': seed,
        'pi_var': pi_var,
        'phi_var': phi_var,
        'pi_disc': pi_var > 0.25,
        'phi_disc': phi_var > 0.25,
        'final_r': history[-1][1] if history else 0.0,
        'benchmark': benchmark,
        'final_pi_profile': pi_final,
        'final_phi_profile': phi_final,
    }


CONFIGS = [
    # A: vanilla baseline
    dict(name='A_vanilla_lr5e3',   strategy='vanilla',
         epochs=300, lr=5e-3, c=1.0),
    # B: phased phi-first (existing best)
    dict(name='B_phased_phi',      strategy='phased_phi_first',
         warmup_eps=50, lr_phase1=5e-2, joint_eps=200, lr_phase2=5e-3, c=1.0),
    # C: phased phi-first with asymmetric LR (π faster in phase 2)
    dict(name='C_phased_asym',     strategy='phased_phi_first',
         warmup_eps=50, lr_phase1=5e-2,
         asymmetric_lr=True, lr_phase2_pi=1e-2, lr_phase2_phi=1e-3,
         joint_eps=300, c=1.0),
    # D: higher cost (c=3) to amplify action gradient
    dict(name='D_phased_c3',       strategy='phased_phi_first',
         warmup_eps=50, lr_phase1=5e-2, joint_eps=200, lr_phase2=5e-3, c=3.0),
    # E: longer warmup + longer joint
    dict(name='E_long_phased',     strategy='phased_phi_first',
         warmup_eps=100, lr_phase1=5e-2, joint_eps=300, lr_phase2=3e-3, c=1.0),
    # F: pi-first phased
    dict(name='F_phased_pi',       strategy='phased_pi_first',
         warmup_eps=50, lr_phase1=5e-2, joint_eps=200, lr_phase2=5e-3, c=1.0),
    # === HybridCoop variants (opponents that condition on BOTH own and recip rep) ===
    # All JOINT FROM THE START — no phased warmup.
    # K: vanilla joint, low LR, long training (baseline)
    dict(name='K_hybrid_lr3e3_long', strategy='vanilla', opp='hybrid',
         epochs=500, lr=3e-3, c=1.0),
    # L: vanilla joint, mid LR
    dict(name='L_hybrid_lr1e2',    strategy='vanilla', opp='hybrid',
         epochs=500, lr=1e-2, c=1.0),
    # M: vanilla joint, asymmetric LR (π faster than φ)
    dict(name='M_hybrid_asym_pi',  strategy='vanilla_asym', opp='hybrid',
         epochs=500, lr_pi=1e-2, lr_phi=3e-3, c=1.0),
    # N: vanilla joint, asymmetric LR (φ faster than π — opposite)
    dict(name='N_hybrid_asym_phi', strategy='vanilla_asym', opp='hybrid',
         epochs=500, lr_pi=3e-3, lr_phi=1e-2, c=1.0),
    # O: vanilla joint with exploration noise on outputs
    dict(name='O_hybrid_noise',    strategy='vanilla_noise', opp='hybrid',
         epochs=500, lr=5e-3, noise_std=0.1, c=1.0),
    # P: higher b (b=20, benchmark 4.75) to amplify signal gradient
    dict(name='P_hybrid_high_b',   strategy='vanilla_asym', opp='hybrid',
         epochs=500, lr_pi=1e-2, lr_phi=3e-3, c=1.0, b=20.0),
    # === Aggressive LR asymmetry (φ much faster than π) ===
    # Q: 10x asymmetry
    dict(name='Q_hybrid_phi10x',   strategy='vanilla_asym', opp='hybrid',
         epochs=500, lr_pi=5e-3, lr_phi=5e-2, c=1.0),
    # R: 30x asymmetry
    dict(name='R_hybrid_phi30x',   strategy='vanilla_asym', opp='hybrid',
         epochs=500, lr_pi=1e-3, lr_phi=3e-2, c=1.0),
    # S: 50x asymmetry
    dict(name='S_hybrid_phi50x',   strategy='vanilla_asym', opp='hybrid',
         epochs=800, lr_pi=1e-3, lr_phi=5e-2, c=1.0),
    # T: 100x asymmetry
    dict(name='T_hybrid_phi100x',  strategy='vanilla_asym', opp='hybrid',
         epochs=800, lr_pi=5e-4, lr_phi=5e-2, c=1.0),
    # U: start φ at high LR, π at very low LR — similar to S but shorter
    dict(name='U_hybrid_phi_fast', strategy='vanilla_asym', opp='hybrid',
         epochs=500, lr_pi=1e-3, lr_phi=7e-2, c=1.0),
    # === Round 2: expanding around T (100x, πLR=5e-4, φLR=5e-2) ===
    # Longer training (T was best)
    dict(name='V_T_1500ep',        strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=5e-4, lr_phi=5e-2, c=1.0),
    dict(name='W_T_2500ep',        strategy='vanilla_asym', opp='hybrid',
         epochs=2500, lr_pi=5e-4, lr_phi=5e-2, c=1.0),
    # Slightly different LR ratios around T
    dict(name='X_80x_1500ep',      strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=5e-4, lr_phi=4e-2, c=1.0),
    dict(name='Y_150x_1500ep',     strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=3e-4, lr_phi=5e-2, c=1.0),
    dict(name='Z_200x_1500ep',     strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=2e-4, lr_phi=4e-2, c=1.0),
    # Scale-up of T: smaller π LR, smaller φ LR, longer training
    dict(name='AA_slow_100x_2000', strategy='vanilla_asym', opp='hybrid',
         epochs=2000, lr_pi=2e-4, lr_phi=2e-2, c=1.0),
    # Higher abs scale with 100x
    dict(name='BB_fast_100x_800',  strategy='vanilla_asym', opp='hybrid',
         epochs=800, lr_pi=1e-3, lr_phi=1e-1, c=1.0),
    # T config with higher b (more signal)
    dict(name='CC_T_b20',          strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=5e-4, lr_phi=5e-2, c=1.0, b=20.0),
    # === Round 3: variance regularizer to escape flat φ ===
    # DD: Y (best so far) + variance regularizer
    dict(name='DD_Y_phi_reg10',   strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=3e-4, lr_phi=5e-2, c=1.0, phi_reg=10.0),
    # EE: heavier regularizer
    dict(name='EE_Y_phi_reg50',   strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=3e-4, lr_phi=5e-2, c=1.0, phi_reg=50.0),
    # FF: regularizer + Y with longer training
    dict(name='FF_Y_phi_reg_long', strategy='vanilla_asym', opp='hybrid',
         epochs=2500, lr_pi=3e-4, lr_phi=5e-2, c=1.0, phi_reg=20.0),
    # GG: T config with reg
    dict(name='GG_T_phi_reg20',   strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=5e-4, lr_phi=5e-2, c=1.0, phi_reg=20.0),
    # === Round 4: SYMMETRIC exploration noise (no cheating) ===
    # HH: Y's LR + exploration noise σ=0.05
    dict(name='HH_Y_noise_05',    strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=3e-4, lr_phi=5e-2, c=1.0, explore_noise=0.05),
    # II: Y's LR + exploration noise σ=0.1
    dict(name='II_Y_noise_10',    strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=3e-4, lr_phi=5e-2, c=1.0, explore_noise=0.1),
    # JJ: Y's LR + exploration noise σ=0.2
    dict(name='JJ_Y_noise_20',    strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=3e-4, lr_phi=5e-2, c=1.0, explore_noise=0.2),
    # KK: Y's LR + exploration noise σ=0.3
    dict(name='KK_Y_noise_30',    strategy='vanilla_asym', opp='hybrid',
         epochs=1500, lr_pi=3e-4, lr_phi=5e-2, c=1.0, explore_noise=0.3),
    # LL: T's LR + exploration noise σ=0.1 longer training
    dict(name='LL_T_noise_10_2k', strategy='vanilla_asym', opp='hybrid',
         epochs=2000, lr_pi=5e-4, lr_phi=5e-2, c=1.0, explore_noise=0.1),
    # === Round 5: 10x smaller absolute LR scale ===
    # MM: 3x ratio at new scale (user suggestion)
    dict(name='MM_3x_scaled',     strategy='vanilla_asym', opp='hybrid',
         epochs=3000, lr_pi=3e-4, lr_phi=1e-3, c=1.0),
    # NN: 10x ratio at new scale
    dict(name='NN_10x_scaled',    strategy='vanilla_asym', opp='hybrid',
         epochs=3000, lr_pi=1e-4, lr_phi=1e-3, c=1.0),
    # OO: 33x ratio at new scale
    dict(name='OO_33x_scaled',    strategy='vanilla_asym', opp='hybrid',
         epochs=3000, lr_pi=3e-5, lr_phi=1e-3, c=1.0),
    # PP: 100x ratio at 10x smaller absolute (T scaled)
    dict(name='PP_100x_scaled',   strategy='vanilla_asym', opp='hybrid',
         epochs=4000, lr_pi=3e-5, lr_phi=3e-3, c=1.0),
    # QQ: 167x at 10x smaller (Y scaled)
    dict(name='QQ_167x_scaled',   strategy='vanilla_asym', opp='hybrid',
         epochs=4000, lr_pi=3e-5, lr_phi=5e-3, c=1.0),
    # RR: 167x scaled + exploration noise
    dict(name='RR_167x_noise',    strategy='vanilla_asym', opp='hybrid',
         epochs=4000, lr_pi=3e-5, lr_phi=5e-3, c=1.0, explore_noise=0.2),
    # === Round 6: budget-matched (10x more eps for 10x slower LR) ===
    # Match Y's budget: πLR × eps = 0.45, φLR × eps = 75
    # QQ_v2: 167x ratio at 10x slower LR, budget-matched
    dict(name='QQ_v2_167x_budget', strategy='vanilla_asym', opp='hybrid',
         epochs=15000, lr_pi=3e-5, lr_phi=5e-3, c=1.0),
    # PP_v2: 100x ratio at 10x slower, budget-matched
    dict(name='PP_v2_100x_budget', strategy='vanilla_asym', opp='hybrid',
         epochs=25000, lr_pi=3e-5, lr_phi=3e-3, c=1.0),
    # RR_v2: 167x slow + noise, budget-matched
    dict(name='RR_v2_167x_noise_budget', strategy='vanilla_asym', opp='hybrid',
         epochs=15000, lr_pi=3e-5, lr_phi=5e-3, c=1.0, explore_noise=0.2),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config name')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=str, required=True, help='output json path')
    args = parser.parse_args()

    cfg = next(c for c in CONFIGS if c['name'] == args.config)
    result = train(cfg, args.seed)

    with open(args.out, 'w') as f:
        json.dump(result, f)
    print(f"[{args.config} seed={args.seed}] "
          f"std[π]={result['pi_var']:.3f} std[φ]={result['phi_var']:.3f} "
          f"r={result['final_r']:.3f} "
          f"πdisc={result['pi_disc']} φdisc={result['phi_disc']}", flush=True)
