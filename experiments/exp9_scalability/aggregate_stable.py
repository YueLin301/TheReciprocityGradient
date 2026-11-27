"""Aggregate the stabilized expD per-N JSONs into the LaTeX table format.

Reads results_expD_n{N}_stable.json for each N in NS, computes mean+/-std on
final_perint, and prints a LaTeX table block ready to drop into the
tab:evaluation_summary slot in paper/content.tex.
"""
import json
import os
import sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_stable_for_N(N: int) -> dict:
    p = os.path.join(HERE, 'data', f'results_expD_n{N}_stable.json')
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def aggregate(NS, p_values=(20, 30, 50, 70, 80)):
    table = {}  # (pathway, N, p) -> (mean, std, n)
    counts = {}
    for N in NS:
        d = load_stable_for_N(N)
        groups = defaultdict(list)
        for k, v in d.items():
            if 'final_perint' not in v:
                continue
            if k.startswith(f'action_N{N}_seed'):
                groups[('action', None)].append(v['final_perint'])
            elif k.startswith(f'signal_N{N}_p'):
                p = int(k.split('_p')[1].split('_')[0])
                groups[('signal', p)].append(v['final_perint'])
            elif k.startswith(f'joint_N{N}_p'):
                p = int(k.split('_p')[1].split('_')[0])
                groups[('joint', p)].append(v['final_perint'])
        for (pw, p), vals in groups.items():
            table[(pw, N, p)] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        counts[N] = len(d)
    return table, counts


def fmt(t, ceiling=None):
    if t is None:
        return '---'
    m, s, n = t
    pct = f' ({100*m/ceiling:.0f}\\%)' if ceiling else ''
    return f'${m:.2f}\\pm {s:.2f}${pct}  [n={n}]'


def emit_latex_block(table, NS):
    print()
    print('% ===== AUTO-GENERATED stable scalability rows =====')
    print()
    print(r'\begin{table}[ht]')
    print(r'\centering\small')
    print(r'\setlength{\tabcolsep}{4pt}')
    print(r'\renewcommand{\arraystretch}{0.95}')
    print(r'\begin{tabular}{l c c c c c}')
    print(r'\toprule')
    print(r'$N$ & $p{=}0.2$ & $p{=}0.3$ & $p{=}0.5$ & $p{=}0.7$ & $p{=}0.8$ \\')
    print(r'\midrule')

    print(r'\multicolumn{6}{l}{\emph{Joint vs HybridCoop+AllDefector (ceiling $2.25$)}} \\')
    for N in NS:
        row = [f'\\quad ${N}$']
        for p in (20, 30, 50, 70, 80):
            row.append(fmt(table.get(('joint', N, p)), ceiling=2.25))
        print(' & '.join(row) + r' \\')

    print(r'\midrule')
    print(r'\multicolumn{6}{l}{\emph{Signal vs ProudCoop+AllDefector, $c{=}5$ (ceiling $1.25$)}} \\')
    for N in NS:
        row = [f'\\quad ${N}$']
        for p in (20, 30, 50, 70, 80):
            row.append(fmt(table.get(('signal', N, p)), ceiling=1.25))
        print(' & '.join(row) + r' \\')

    print(r'\midrule')
    print(r'\multicolumn{6}{l}{\emph{Action vs L6 (no $p$ axis; ceiling $4.5$)}} \\')
    for N in NS:
        t = table.get(('action', N, None))
        if t is None:
            print(f'\\quad ${N}$ & ' + r' \multicolumn{5}{c}{---} \\')
        else:
            m, s, n = t
            pct = 100 * m / 4.5
            print(f'\\quad ${N}$ & ' + r'\multicolumn{5}{c}{' + f'${m:.2f}\\pm {s:.2f}$ ($n={n}$, ${pct:.0f}\\%$)' + r'} \\')

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\caption{\textbf{Stabilized scalability sweep.} Per-interaction payoff (mean $\pm$ std across 10 seeds, best-checkpoint reporting on real-env evaluation). Joint pool sweeps over HybridCoop fraction $p$; signal pool over ProudCoop fraction $p$ at $c=5$; action vs.\ L6.}')
    print(r'\label{tab:scalability_stable}')
    print(r'\end{table}')


def emit_summary(table, counts, NS):
    print('\n=== summary ===')
    for N in NS:
        print(f'N={N}: {counts.get(N, 0)}/110 entries done')
    print()
    print('cells (mean+/-std, n):')
    for N in NS:
        for cell, p_list in [('action', [None]), ('signal', [20, 30, 50, 70, 80]),
                              ('joint', [20, 30, 50, 70, 80])]:
            row = []
            for p in p_list:
                t = table.get((cell, N, p))
                if t is None:
                    row.append('---')
                else:
                    m, s, n = t
                    row.append(f'{m:+.3f}+/-{s:.3f}(n{n})')
            ptag = '' if cell == 'action' else f' p=20/30/50/70/80'
            print(f'  N={N:2d} {cell:8s}{ptag}: {row}')


if __name__ == '__main__':
    NS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [5, 10, 15, 20, 25, 30]
    table, counts = aggregate(NS)
    emit_summary(table, counts, NS)
    emit_latex_block(table, NS)
