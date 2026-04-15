#!/usr/bin/env python3
"""Analyze all benchmark rounds (R1-R4) and rank by composite score."""
import json, os, glob, sys

base = 'output/benchmarks/tosca_params/cat0_blue_high_res'
rows = []
for d in sorted(glob.glob(f'{base}/r*/')):
    name = os.path.basename(d.rstrip('/'))
    report = os.path.join(d, 'benchmark_report.json')
    args_file = os.path.join(d, 'benchmark_args.txt')
    if not os.path.isfile(report):
        continue
    with open(report) as f:
        r = json.load(f)
    args_str = ''
    if os.path.isfile(args_file):
        with open(args_file) as f:
            args_str = f.read().strip()
    params = {}
    toks = args_str.split()
    for i, t in enumerate(toks):
        if t.startswith('--') and i+1 < len(toks):
            params[t[2:]] = toks[i+1]
    
    psnr = r.get('psnr',{}).get('test_psnr', 0)
    chamfer = r.get('chamfer_distance', 999)
    g2s = r.get('gaussian_to_recon_mesh_surface', {})
    g2s_msq = g2s.get('mean_squared', 999)
    g2s_max = g2s.get('max', 999)
    g2s_mean = g2s.get('mean', 999)
    num_g = r.get('num_gaussians', 0)
    cc = r.get('connected_components',{}).get('num_components', 0)
    
    rows.append({
        'name': name, 'psnr': psnr, 'chamfer': chamfer,
        'g2s_msq': g2s_msq, 'g2s_max': g2s_max, 'g2s_mean': g2s_mean,
        'num_g': num_g, 'cc': cc, **params
    })

print(f'Total completed runs: {len(rows)}')
print()

# Score: lower is better. Target: PSNR~48.5, chamfer~0.101, g2s_msq~0.2
def score(r):
    psnr_err = max(0, 48.5 - r['psnr']) / 48.5
    chamfer_err = max(0, (r['chamfer'] - 0.101) / 0.101)
    msq_err = max(0, (r['g2s_msq'] - 0.2) / 0.2)
    max_penalty = max(0, r['g2s_max'] - 5) / 10  # moderate penalty
    return psnr_err + chamfer_err + msq_err + 0.3 * max_penalty
for r in rows:
    r['score'] = score(r)
rows.sort(key=lambda r: r['score'])

hdr = f"{'name':12s} {'PSNR':>7s} {'Chamf':>7s} {'g2s_msq':>8s} {'g2s_max':>8s} {'g2s_mn':>8s} {'num_g':>7s} {'CC':>3s} {'score':>6s} | {'bpsf':>6s} {'mop':>5s} {'lmvg':>5s} {'ldist':>6s} {'dgt':>8s} {'ldn':>5s} {'ldss':>5s} {'psa':>5s} {'pmst':>5s} {'pd':>5s}"
sep = '-' * len(hdr)

def show(r):
    return (f"{r['name']:12s} {r['psnr']:7.2f} {r['chamfer']:7.4f} {r['g2s_msq']:8.4f} "
            f"{r['g2s_max']:8.2f} {r['g2s_mean']:8.4f} {r['num_g']:7d} {r['cc']:3d} {r['score']:6.3f} | "
            f"{r.get('big_point_scale_factor','?'):>6s} {r.get('min_opacity_prune','?'):>5s} "
            f"{r.get('lambda_multi_view_geo','?'):>5s} {r.get('lambda_distortion','?'):>6s} "
            f"{r.get('densify_grad_threshold','?'):>8s} {r.get('lambda_depth_normal','?'):>5s} "
            f"{r.get('lambda_dssim','?'):>5s} {r.get('prune_scale_anisotropy','?'):>5s} "
            f"{r.get('prune_min_scale_threshold','?'):>5s} {r.get('percent_dense','?'):>5s}")

print("=== TOP 30 OVERALL ===")
print(hdr)
print(sep)
for r in rows[:30]:
    print(show(r))

print()
print("=== ALL R4 RUNS ===")
r4 = [r for r in rows if r['name'].startswith('r4_')]
r4.sort(key=lambda r: r['score'])
print(hdr)
print(sep)
for r in r4:
    print(show(r))

# Per-parameter analysis for R4
print()
print("=== R4 PER-PARAMETER WINNER (by score) ===")
param_names = ['big_point_scale_factor', 'min_opacity_prune', 'lambda_multi_view_geo',
               'lambda_distortion', 'densify_grad_threshold', 'lambda_depth_normal',
               'lambda_dssim', 'prune_scale_anisotropy', 'prune_min_scale_threshold', 'percent_dense']
for p in param_names:
    vals = {}
    for r in r4:
        v = r.get(p, '?')
        if v not in vals or r['score'] < vals[v]['score']:
            vals[v] = r
    if vals:
        print(f"\n  {p}:")
        for v in sorted(vals.keys()):
            r = vals[v]
            print(f"    {v:>8s} -> score={r['score']:.3f}  PSNR={r['psnr']:.2f}  chamfer={r['chamfer']:.4f}  g2s_msq={r['g2s_msq']:.4f}  g2s_max={r['g2s_max']:.2f}  ({r['name']})")
