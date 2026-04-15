"""Run benchmark evaluations from a JSON job list with configurable parallelism.

Usage:
    python eval_node.py --jobs eval_part1.json --workers 20
"""
import argparse, glob, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON = '/home/rotem.shezaf/miniconda3/envs/geo_splat/bin/python3'
EVAL_SCRIPT = 'scripts/tosca/benchmark/evaluate_benchmark.py'

def run_eval(args):
    d, gt, shape, sw = args['dir'], args['gt'], args['shape'], args['sw']
    # Skip if already done (race condition guard)
    if os.path.exists(f'{d}/benchmark_report.json'):
        return (shape, sw, 'SKIP')
    cmd = [PYTHON, EVAL_SCRIPT, '--output_dir', d, '--iteration', '45000']
    if gt:
        cmd += ['--gt_mesh', gt]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        if result.returncode == 0 and os.path.exists(f'{d}/benchmark_report.json'):
            return (shape, sw, 'OK')
        else:
            err = result.stderr[-300:] if result.stderr else 'no stderr'
            return (shape, sw, f'FAIL rc={result.returncode}: {err}')
    except subprocess.TimeoutExpired:
        return (shape, sw, 'TIMEOUT')
    except Exception as e:
        return (shape, sw, f'ERROR: {e}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', required=True, help='JSON file with job list')
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()

    with open(args.jobs) as f:
        jobs = json.load(f)

    # Filter out already-done
    jobs = [j for j in jobs if not os.path.exists(os.path.join(j['dir'], 'benchmark_report.json'))]
    print(f"Running {len(jobs)} evals with {args.workers} workers", flush=True)

    ok = skip = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_eval, job): job for job in jobs}
        for future in as_completed(futures):
            shape, sw, status = future.result()
            elapsed = time.time() - t0
            total = ok + skip + fail
            if status == 'OK':
                ok += 1
                print(f"  OK  [{total+1}/{len(jobs)} {elapsed:.0f}s]: {shape}/{sw}", flush=True)
            elif status == 'SKIP':
                skip += 1
                print(f"  SKIP[{total+1}/{len(jobs)} {elapsed:.0f}s]: {shape}/{sw}", flush=True)
            else:
                fail += 1
                print(f"  FAIL[{total+1}/{len(jobs)} {elapsed:.0f}s]: {shape}/{sw}: {status[:100]}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min): {ok} ok, {skip} skip, {fail} fail out of {len(jobs)}")

if __name__ == '__main__':
    main()
