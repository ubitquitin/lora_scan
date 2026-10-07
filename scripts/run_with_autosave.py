#!/usr/bin/env python
"""Run sweep with automatic Google Drive saves after each completed run."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def sync_to_drive(local_runs: Path, drive_dir: Path):
    """Sync local runs directory to Google Drive."""
    drive_runs = drive_dir / "runs"
    drive_runs.mkdir(parents=True, exist_ok=True)

    print("[AUTO-SAVE] Syncing results to Google Drive...")
    try:
        # Use rsync if available, otherwise fall back to shutil
        result = subprocess.run(
            ["rsync", "-a", "--delete", f"{local_runs}/", f"{drive_runs}/"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"[AUTO-SAVE] ✓ Results synced to {drive_runs}")
            return True
    except FileNotFoundError:
        # rsync not available, use Python's shutil
        pass

    # Fallback to shutil
    if drive_runs.exists():
        shutil.rmtree(drive_runs)
    shutil.copytree(local_runs, drive_runs)
    print(f"[AUTO-SAVE] ✓ Results synced to {drive_runs}")
    return True


def run_sweep_with_autosave(experiment_path: str, drive_dir: str, **kwargs):
    """Run sweep and sync to Google Drive after each completed run."""
    local_runs = Path("runs")
    drive_path = Path(drive_dir)

    # Restore previous results if they exist
    drive_runs = drive_path / "runs"
    if drive_runs.exists() and any(drive_runs.iterdir()):
        print(f"[AUTO-SAVE] Restoring previous results from {drive_runs}...")
        if local_runs.exists():
            shutil.rmtree(local_runs)
        shutil.copytree(drive_runs, local_runs)
        print("[AUTO-SAVE] ✓ Previous results restored")

    # Build command
    cmd = [sys.executable, "scripts/run.py", experiment_path]
    if kwargs.get("seeds"):
        cmd.extend(["--seeds"] + [str(s) for s in kwargs["seeds"]])
    if kwargs.get("only"):
        cmd.extend(["--only"] + kwargs["only"])
    if kwargs.get("force"):
        cmd.append("--force")

    # Run sweep with line-by-line output
    print(f"[AUTO-SAVE] Starting sweep with auto-save enabled")
    print(f"[AUTO-SAVE] Command: {' '.join(cmd)}\n")

    completed_runs = 0
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        for line in process.stdout:
            print(line, end='', flush=True)

            # Detect when a run completes
            if "Training completed in" in line or "======================================================================" in line:
                completed_runs += 1
                if completed_runs % 1 == 0:  # Sync after every run
                    sync_to_drive(local_runs, drive_path)

        process.wait()

        # Final sync
        print("\n[AUTO-SAVE] Final sync to Google Drive...")
        sync_to_drive(local_runs, drive_path)
        print("[AUTO-SAVE] ✓ All results saved to Google Drive")

        return process.returncode

    except KeyboardInterrupt:
        print("\n[AUTO-SAVE] Interrupted! Saving current progress...")
        process.terminate()
        sync_to_drive(local_runs, drive_path)
        print("[AUTO-SAVE] ✓ Progress saved to Google Drive")
        return 1


def main():
    ap = argparse.ArgumentParser(description="Run sweep with auto-save to Google Drive")
    ap.add_argument("experiment", help="Path to experiment config file")
    ap.add_argument("--drive-dir", required=True, help="Google Drive directory for results")
    ap.add_argument("--seeds", type=int, nargs="+", help="Override the seeds in the file")
    ap.add_argument("--only", nargs="+", help="Run only variants whose name contains one of these")
    ap.add_argument("--force", action="store_true", help="Re-run even if completed runs exist")
    args = ap.parse_args()

    sys.exit(run_sweep_with_autosave(
        args.experiment,
        args.drive_dir,
        seeds=args.seeds,
        only=args.only,
        force=args.force
    ))


if __name__ == "__main__":
    main()
