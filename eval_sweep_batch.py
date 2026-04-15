import glob, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

base = 'TrainData/TOSCA/SyntheticColmapData/blue_texture'
processed = 'TrainData/TOSCA/processed'
eval_script = 'scripts/tosca/benchmark/evaluate_benchmark.py'
python3 = '/home/rotem.shezaf/miniconda3/envs/geo_splat/bin/python3'

sweep_dirs = sorted(glob.glob(f'{base}/*/high_res/light_0/sweep_sw*'))
jobs = []
for d in sweep_dirs:
    if not os.path.exists(os.path.join(d, 'recon.ply')):
        continue
    if os.path.exists(os.path.join(d, 'benchmark_report.json')):
        continue
    parts = d.split('/')
    shape = parts[4]
    gt_meshes = sorted(glob.glob(f'{processed}/{shape}/mesh_high_res_*.ply'))
    gt = gt_meshes[0] if gt_meshes else ''
    jobs.append((d, gt, shape, os.path.basename(d)))

print(f"Evaluating {len(jobs)} sweep runs with 2 parallel workers (timeout 900s)...")
sys.stdout.flush()

def run_eval(args):
    d, gt, shape, sw = args
    cmd = [python3, eval_script, '--output_dir', d, '--iteration', '45000']
    if gt:
        cmd += ['--gt_mesh', gt]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        if result.returncode == 0 and os.path.exists(f'{d}/benchmark_report.json'):
            return (shape, sw, 'OK')
        else:
            return (shape, sw, f'FAIL rc={result.returncode}: {result.stderr[-300:] if result.stderr else "no stderr"}')
    except subprocess.TimeoutExpired:
        return (shape, sw, 'TIMEOUT 900s')
    except Exception as e:
        return (shape, sw, f'ERROR: {str(e)[:200]}')

ok = 0
fail = 0
t0 = time.time()
with ProcessPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(run_eval, job): job for job in jobs}
    for future in as_completed(futures):
        shape, sw, status = future.result()
        elapsed_so_far = time.time() - t0
        if status == 'OK':
            ok += 1
            print(f"  OK [{ok+fail}/{len(jobs)} {elapsed_so_far:.0f}s]: {shape}/{sw}")
        else:
            fail += 1
            print(f"  FAIL [{ok+fail}/{len(jobs)} {elapsed_so_far:.0f}s]: {shape}/{sw}: {status[:100]}")
        sys.stdout.flush()

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min): {ok} ok, {fail} fail out of {len(jobs)}")
