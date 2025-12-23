import subprocess
import time
import shutil
import os
from pathlib import Path

def run_cmd(cmd, cwd=".", desc=""):
    print(f"Running: {desc}...")
    start_time = time.time()
    try:
        subprocess.run(
            cmd, 
            cwd=cwd, 
            shell=True, 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        return None
    duration = time.time() - start_time
    print(f"  -> Time: {duration:.2f}s")
    return duration

def setup_venv(path):
    if path.exists():
        shutil.rmtree(path)
    subprocess.run([os.sys.executable, "-m", "venv", str(path)], check=True)

def benchmark():
    base_dir = Path.cwd()
    venv_pip = base_dir / "bench_env_pip"
    venv_uv = base_dir / "bench_env_uv"
    
    results = {}
    
    print("=== Starting Benchmark: pip vs uv ===\n")

    # --- Scenario 1: pip (Cold) ---
    setup_venv(venv_pip)
    pip_cmd = f"source {venv_pip}/bin/activate && pip install .[all,dev,test] --no-cache-dir"
    results['pip_cold'] = run_cmd(pip_cmd, desc="pip install (Cold Cache)")
    
    # --- Scenario 2: pip (Warm) ---
    setup_venv(venv_pip) # Recreate env, but cache might be used if we didn't use --no-cache-dir. 
    # Standard pip warm run: pip usually caches wheels in ~/.cache/pip. 
    # To simulate warm cache, we just run install without --no-cache-dir
    pip_cmd_warm = f"source {venv_pip}/bin/activate && pip install .[all,dev,test]"
    results['pip_warm'] = run_cmd(pip_cmd_warm, desc="pip install (Warm Cache)")

    # --- Scenario 3: uv pip (Cold) ---
    setup_venv(venv_uv)
    # Ensure uv cache is ignored/cleared for this run? 
    # best approximation for per-command cold cache is --no-cache
    uv_cmd = f"source {venv_uv}/bin/activate && uv pip install .[all,dev,test] --no-cache"
    results['uv_cold'] = run_cmd(uv_cmd, desc="uv pip install (Cold Cache)")

    # --- Scenario 4: uv pip (Warm) ---
    setup_venv(venv_uv)
    # Warm run uses the default cache
    uv_cmd_warm = f"source {venv_uv}/bin/activate && uv pip install .[all,dev,test]"
    results['uv_warm'] = run_cmd(uv_cmd_warm, desc="uv pip install (Warm Cache)")
    
    # --- Scenario 5: uv sync (Cold-ish) ---
    # uv sync creates its own venv. We'll verify how fast it is vs pip install
    # We delete .venv first
    if (base_dir / ".venv").exists():
        shutil.rmtree(base_dir / ".venv")
    
    uv_sync_cmd = "uv sync --all-extras --no-cache"
    results['uv_sync_cold'] = run_cmd(uv_sync_cmd, desc="uv sync (Cold Cache)")

    # --- Scenario 6: uv sync (Warm) ---
    if (base_dir / ".venv").exists():
        shutil.rmtree(base_dir / ".venv")
        
    uv_sync_warm = "uv sync --all-extras"
    results['uv_sync_warm'] = run_cmd(uv_sync_warm, desc="uv sync (Warm Cache)")

    # Cleanup
    if venv_pip.exists(): shutil.rmtree(venv_pip)
    if venv_uv.exists(): shutil.rmtree(venv_uv)
    
    print("\n=== Benchmark Results ===")
    print(f"| Method | Scenario | Time (s) | Speedup vs pip |")
    print(f"|--------|----------|----------|----------------|")
    
    pip_c = results['pip_cold']
    for key, val in results.items():
        if val is None: val = 0
        scenario = key.replace('_', ' ').title()
        speedup = f"{pip_c / val:.1f}x" if val > 0 and pip_c else "-"
        print(f"| {key.split('_')[0]} | {scenario} | {val:.2f} | {speedup} |")

if __name__ == "__main__":
    benchmark()
