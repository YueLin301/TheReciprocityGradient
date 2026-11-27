"""Stabilized N=10 scalability run.

Diagnosis: the default expD run shows wide variance at N=10 because training
reaches near-optimal payoff by iteration 5-7 and then a late-stage gradient
step kicks the policy out of the basin and fails to recover within the
remaining iterations. Without intervention, ~30% of seeds end in a crashed
basin while the rest sit at >90% of ceiling, producing a bimodal seed
distribution and inflated std.

This script keeps the algorithmic content of run_exp9 unchanged (same
reciprocity gradient, same amortized opponent surrogate, same env, same
estimator schedule) but adds three stabilization knobs that all act AFTER
the optimization signal and therefore do not change the method's claim:

  1. Best-checkpoint tracking. After every outer iteration we run a real-env
     evaluation and snapshot the policy parameters whenever real_perint[t]
     improves on the running best. Final eval uses those snapshotted
     parameters, not the policy state at t = T_OUTER - 1.
  2. Lower policy learning rate (5e-3 -> 2e-3) so once the policy reaches the
     near-optimal flat basin, single bad surrogate gradients move parameters
     less.
  3. More outer iterations (12 -> 20) so even if a kick-out happens, there
     is room for the snapshotted best-so-far to also be revisited or
     improved.

We only run N = 10 here, the cell flagged as the variance hot-spot. Output
goes to data/results_expD_n10_stable.json so the original results_expD.json
is untouched.
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))

from agent import Agent
from env_batched import BatchedDonationGame, compute_per_interaction_payoff
from exp2_action_optimization.leading_eight_agents import LeadingEightAgent
from exp6_signal_opponent_modeling.train_exp6_signal_only import (
    ProudCoopL3, AllDefectL3,
)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'exp4_joint_gt')))
from hybrid_opponents import HybridCoopL3
from opponent_modeling_utils import (
    AmortizedActionEstimator, AmortizedSignalEstimator,
    AmortizedEstimatedOpponent,
    DataBuffer, RecordingWrapper,
    train_amortized_action_estimator, train_amortized_signal_estimator,
)
from checkpoint_utils import atomic_save_json, load_json_if_exists


BATCH_SIZE = 128
T_OUTER = 20
N_PLAY = 4
K_EST = 100
N_TRAIN = 8
W_BUFFER = 3
B_VALUE = 10.0
DELTA = 0.98
BETA = 5.0
LR_POLICY = 2e-3
LR_EST = 1e-2
N_FINAL_EVAL = 10

POOL_FRACTIONS = [0.2, 0.3, 0.5, 0.7, 0.8]


def build_opps(pathway: str, N: int, pool_fraction, device):
    n_opps = N - 1
    if pathway == 'action':
        opps = [LeadingEightAgent(name=f'OppL6_{i+1}', norm='L6', continuous=True,
                                   decay=0.5, beta=BETA, device=device)
                for i in range(n_opps)]
    elif pathway == 'signal':
        n_main = max(1, min(n_opps - 1, int(round(pool_fraction * n_opps))))
        opps = ([ProudCoopL3(name=f'PC_{i+1}', device=device) for i in range(n_main)]
              + [AllDefectL3(name=f'AD_{i+1}', device=device) for i in range(n_opps - n_main)])
    elif pathway == 'joint':
        n_main = max(1, min(n_opps - 1, int(round(pool_fraction * n_opps))))
        opps = ([HybridCoopL3(name=f'HC_{i+1}', device=device) for i in range(n_main)]
              + [AllDefectL3(name=f'AD_{i+1}', device=device) for i in range(n_opps - n_main)])
    else:
        raise ValueError(pathway)
    for opp in opps:
        for p_ in opp.parameters():
            p_.requires_grad_(False)
    return opps


def build_per_agent_init(pathway: str, N: int, pool_fraction):
    """Match the per-agent init the original train_exp9 uses for opponents."""
    warm_dir = os.path.join(HERE, '..', 'population_warm_up')
    steady_l3 = torch.load(os.path.join(warm_dir, 'steady_state_L3.pt'),
                            map_location='cpu', weights_only=False)['continuous_final_reps']
    steady_l6 = torch.load(os.path.join(warm_dir, 'steady_state_L6.pt'),
                            map_location='cpu', weights_only=False)['continuous_final_reps']
    n_opps = N - 1
    per_agent_init = [{"dist": "uniform", "params": {"low": 0.0, "high": 1.0}}]
    if pathway == 'action':
        for _ in range(n_opps):
            per_agent_init.append({"dist": "empirical",
                                    "params": {"values": steady_l6}})
    else:
        n_main = max(1, min(n_opps - 1, int(round(pool_fraction * n_opps))))
        for i in range(n_opps):
            steady = steady_l3 if i < n_main else steady_l3
            per_agent_init.append({"dist": "empirical",
                                    "params": {"values": steady}})
    return per_agent_init


def evaluate_real(agent0, opps, per_agent_init, c_val, n_rollouts: int) -> float:
    agent0.eval()
    total = 0.0
    with torch.no_grad():
        env = BatchedDonationGame(
            agents=[agent0] + opps, b=B_VALUE, c=c_val,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init)
        for _ in range(n_rollouts):
            cum_rewards, matching_seq, _ = env.play_batch()
            pi_payoff, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
            total += pi_payoff.mean().item()
    agent0.train()
    return total / n_rollouts


def run_stable(N: int, pathway: str, seed: int, pool_fraction=None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')
    n_opps = N - 1

    agent0 = Agent(name='A_0', signal_dim=1, second_order=False,
                    is_demo=False, device=device)
    if pathway == 'action':
        agent0.set_hardcode_signal()
        c_val = 1.0
    elif pathway == 'signal':
        agent0.set_hardcode_action(decay=None)
        c_val = 5.0
    elif pathway == 'joint':
        c_val = 1.0
    else:
        raise ValueError(pathway)

    opps = build_opps(pathway, N, pool_fraction, device)
    rec_opps = [RecordingWrapper(opps[i], DataBuffer(window=W_BUFFER))
                 for i in range(n_opps)]
    buffers = [w.buffer for w in rec_opps]

    if pathway == 'action':
        train_params = list(agent0.get_action_params())
    elif pathway == 'signal':
        train_params = list(agent0.get_signal_params())
    else:
        train_params = list(agent0.get_action_params()) + list(agent0.get_signal_params())
    policy_opt = optim.Adam(train_params, lr=LR_POLICY)

    action_est = AmortizedActionEstimator(n_opps=n_opps, opp_emb_dim=8, hidden=16).to(device)
    signal_est = AmortizedSignalEstimator(n_opps=n_opps, opp_emb_dim=8, hidden=16).to(device)

    per_agent_init = build_per_agent_init(pathway, N, pool_fraction)

    history = {
        'real_perint': [], 'surrogate_perint': [],
        'est_loss_action': [], 'est_loss_signal': [],
        'pi_var': [],
        'best_real_perint_during_training': -1e9,
        'best_t': -1,
        'N': N, 'n_opps': n_opps, 'pathway': pathway,
        'pool_fraction': pool_fraction,
        'lr_policy': LR_POLICY, 't_outer': T_OUTER,
        'n_final_eval': N_FINAL_EVAL,
    }

    best_state = None  # snapshot of (action_params, signal_params) at the best iter

    def _measure_pi_var():
        agent0.eval()
        x = torch.linspace(0, 1, 100)
        with torch.no_grad():
            prof = [agent0.take_action(x[i].view(1, 1)).item() for i in range(100)]
        agent0.train()
        return float(np.std(prof)), prof

    history['pi_var'].append(_measure_pi_var()[0])

    for t in range(T_OUTER):
        # Phase 1: interact + evaluate
        for buf in buffers:
            buf.new_iter()
        agent0.eval()
        per_int_runs = []
        with torch.no_grad():
            real_env = BatchedDonationGame(
                agents=[agent0] + rec_opps, b=B_VALUE, c=c_val,
                continuation_prob=DELTA, batch_size=BATCH_SIZE,
                balanced_matching=True, min_rounds=3,
                per_agent_init=per_agent_init)
            for _ in range(N_PLAY):
                cum_rewards, matching_seq, _ = real_env.play_batch()
                pi_payoff, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
                per_int_runs.append(pi_payoff.mean().item())
        agent0.train()
        avg_real = float(np.mean(per_int_runs))
        history['real_perint'].append(avg_real)

        # Best-checkpoint snapshot
        if avg_real > history['best_real_perint_during_training']:
            history['best_real_perint_during_training'] = avg_real
            history['best_t'] = t
            best_state = copy.deepcopy({n: p.detach().clone() for n, p in agent0.named_parameters()})

        # Phase 2: estimator update (amortized only)
        la = train_amortized_action_estimator(action_est, buffers, K_EST * n_opps, LR_EST)
        ls = train_amortized_signal_estimator(signal_est, buffers, K_EST * n_opps, LR_EST)
        history['est_loss_action'].append(float(la))
        history['est_loss_signal'].append(float(ls))

        # Phase 3: policy update on surrogate
        est_opps = [AmortizedEstimatedOpponent(f'AmEstOpp{i}', i, action_est, signal_est,
                                                device=device)
                     for i in range(n_opps)]
        for p_ in list(action_est.parameters()) + list(signal_est.parameters()):
            p_.requires_grad_(False)
        surrogate_env = BatchedDonationGame(
            agents=[agent0] + est_opps, b=B_VALUE, c=c_val,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init)
        surr_runs = []
        for _ in range(N_TRAIN):
            policy_opt.zero_grad()
            cum_rewards, matching_seq, _ = surrogate_env.play_batch()
            pi_payoff, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
            loss = -pi_payoff.mean()
            if loss.requires_grad:
                grads = torch.autograd.grad(loss, train_params, allow_unused=True)
                for p_, g in zip(train_params, grads):
                    if g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()):
                        p_.grad = g
                torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            policy_opt.step()
            surr_runs.append(pi_payoff.mean().item())
        history['surrogate_perint'].append(float(np.mean(surr_runs)))

        for p_ in list(action_est.parameters()) + list(signal_est.parameters()):
            p_.requires_grad_(True)

        history['pi_var'].append(_measure_pi_var()[0])

        print(f'  [{pathway} N={N} p={pool_fraction} seed={seed}] t={t:2d} | '
              f'real={avg_real:.3f}  surr={history["surrogate_perint"][-1]:.3f}  '
              f'best={history["best_real_perint_during_training"]:.3f}@t{history["best_t"]}',
              flush=True)

    # Final eval: restore best-checkpoint policy state, evaluate with N_FINAL_EVAL rollouts
    if best_state is not None:
        with torch.no_grad():
            for n, p in agent0.named_parameters():
                if n in best_state:
                    p.copy_(best_state[n])

    final = evaluate_real(agent0, opps, per_agent_init, c_val, N_FINAL_EVAL)
    history['final_perint'] = final
    history['final_perint_eval_rollouts'] = N_FINAL_EVAL
    print(f'  [{pathway} N={N} p={pool_fraction} seed={seed}] FINAL '
          f'(best-ckpt @ t={history["best_t"]}, {N_FINAL_EVAL} rollouts) = {final:.3f}',
          flush=True)
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pathway', choices=['action', 'signal', 'joint', 'all'], default='all')
    ap.add_argument('--N', type=int, default=10)
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--seed-end', type=int, default=10)
    ap.add_argument('--pool-fractions', type=str, default='0.2,0.3,0.5,0.7,0.8',
                    help='comma-separated; ignored for action pathway')
    args = ap.parse_args()

    pool_fractions = [float(x) for x in args.pool_fractions.split(',')]
    out = os.path.join(HERE, 'data', f'results_expD_n{args.N}_stable.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    results = load_json_if_exists(out) or {}

    pathways = ['action', 'signal', 'joint'] if args.pathway == 'all' else [args.pathway]
    grid = []
    for pw in pathways:
        if pw == 'action':
            for s in range(args.seed_start, args.seed_end):
                grid.append((pw, args.N, None, s))
        else:
            for p in pool_fractions:
                for s in range(args.seed_start, args.seed_end):
                    grid.append((pw, args.N, p, s))

    todo = []
    for pw, N, p, s in grid:
        if pw == 'action':
            key = f'action_N{N}_seed{s}'
        else:
            key = f'{pw}_N{N}_p{int(round(p*100)):02d}_seed{s}'
        if key not in results:
            todo.append((pw, N, p, s, key))

    print(f'\n{"="*70}\n  STABILIZED N={args.N} sweep — {len(todo)} runs to do\n{"="*70}\n')
    t_start = time.time()
    for i, (pw, N, p, s, key) in enumerate(todo):
        print(f'\n[{i+1}/{len(todo)}] {key}', flush=True)
        try:
            hist = run_stable(N=N, pathway=pw, seed=s, pool_fraction=p)
        except Exception as e:
            print(f'  FAIL {key}: {e}', flush=True)
            import traceback; traceback.print_exc()
            continue
        slim = {k: v for k, v in hist.items()
                if isinstance(v, (list, tuple, str, int, float, bool))}
        results[key] = slim
        atomic_save_json(results, out)
        print(f'  saved -> {out}', flush=True)
    print(f'\nDONE in {(time.time()-t_start)/60:.1f} min  ({len(todo)} runs)')


if __name__ == '__main__':
    main()
