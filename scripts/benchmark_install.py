import subprocess
import time
import shutil
import os
import sys
from pathlib import Path
from typing import Optional, Dict

def run_cmd(cmd: str, cwd: Path = Path("."), desc: str = "") -> Optional[float]:
    """
    Execute a shell command and measure its execution time.
    
    Args:
        cmd: The shell command to execute.
        cwd: Current working directory for the command.
        desc: Description of the task for logging.
        
    Returns:
        Duration in seconds if successful, None if failed.
    """
    print(f"Running: {desc}...")
    start_time = time.time()
    try:
        subprocess.run(
            cmd, 
            cwd=cwd, 
            shell=True,  # Kept as True for complex chaining (source && pip)
            # Alternatively, we could call the venv binary directly, but 'source' is idiomatic for venvs.
            executable="/bin/bash" if sys.platform != "win32" else None,
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print(f"Error running command: {cmd}")
        return None
    
    duration = time.time() - start_time
    print(f"  -> Time: {duration:.2f}s")
    return duration

def setup_venv(path: Path) -> None:
    """Create a fresh virtual environment at the specified path."""
    if path.exists():
        shutil.rmtree(path)
    subprocess.run([sys.executable, "-m", "venv", str(path)], check=True)

def benchmark() -> None:
    """Run the complete pip vs uv benchmark suite."""
    base_dir = Path.cwd()
    venv_pip = base_dir / "bench_env_pip"
    venv_uv = base_dir / "bench_env_uv"
    
    results: Dict[str, Optional[float]] = {}
    
    print("=== Starting Benchmark: pip vs uv ===\n")

    # --- Scenario 1: pip (Cold) ---
    setup_venv(venv_pip)
    pip_cmd = f"source {venv_pip}/bin/activate && pip install .[all,dev,test] --no-cache-dir"
    results['pip_cold'] = run_cmd(pip_cmd, desc="pip install (Cold Cache)")
    
    # --- Scenario 2: pip (Warm) ---
    setup_venv(venv_pip) 
    # Standard pip warm run: pip usually caches wheels in ~/.cache/pip. 
    pip_cmd_warm = f"source {venv_pip}/bin/activate && pip install .[all,dev,test]"
    results['pip_warm'] = run_cmd(pip_cmd_warm, desc="pip install (Warm Cache)")

    # --- Scenario 3: uv pip (Cold) ---
    setup_venv(venv_uv)
    # --no-cache enforces a clean install for this command
    uv_cmd = f"source {venv_uv}/bin/activate && uv pip install .[all,dev,test] --no-cache"
    results['uv_cold'] = run_cmd(uv_cmd, desc="uv pip install (Cold Cache)")

    # --- Scenario 4: uv pip (Warm) ---
    setup_venv(venv_uv)
    # Warm run uses the default uv cache
    uv_cmd_warm = f"source {venv_uv}/bin/activate && uv pip install .[all,dev,test]"
    results['uv_warm'] = run_cmd(uv_cmd_warm, desc="uv pip install (Warm Cache)")
    
    # --- Scenario 5: uv sync (Cold-ish) ---
    # uv sync creates its own managed .venv
    uv_venv = base_dir / ".venv"
    if uv_venv.exists():
        shutil.rmtree(uv_venv)
    
    uv_sync_cmd = "uv sync --all-extras --no-cache"
    results['uv_sync_cold'] = run_cmd(uv_sync_cmd, desc="uv sync (Cold Cache)")

    # --- Scenario 6: uv sync (Warm) ---
    if uv_venv.exists():
        shutil.rmtree(uv_venv)
        
    uv_sync_warm = "uv sync --all-extras"
    results['uv_sync_warm'] = run_cmd(uv_sync_warm, desc="uv sync (Warm Cache)")

    # Cleanup
    if venv_pip.exists(): shutil.rmtree(venv_pip)
    if venv_uv.exists(): shutil.rmtree(venv_uv)
    
    print("\n=== Benchmark Results ===")
    print(f"| Method | Scenario | Time (s) | Speedup vs pip |")
    print(f"|--------|----------|----------|----------------|")
    
    pip_c = results.get('pip_cold')
    # Default to 0.0 avoids division error if benchmark failed
    baseline = pip_c if pip_c is not None else 0.0
    
    for key, val in results.items():
        if val is None: 
            val_display = "Failed"
            speedup = "-"
        else:
            val_display = f"{val:.2f}"
            speedup = f"{baseline / val:.1f}x" if val > 0 and baseline > 0 else "-"
            
        scenario = key.replace('_', ' ').title()
        method_name = key.split('_')[0]
        print(f"| {method_name} | {scenario} | {val_display} | {speedup} |")
        
if __name__ == "__main__":
    benchmark()
