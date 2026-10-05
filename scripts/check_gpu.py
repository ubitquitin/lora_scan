#!/usr/bin/env python
"""Quick GPU availability check for lora-scan."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    print("Checking GPU availability for lora-scan...\n")

    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed. Install with: pip install torch")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n❌ No CUDA-capable GPU detected!")
        print("\nPossible reasons:")
        print("  1. No GPU present in this machine")
        print("  2. CUDA drivers not installed")
        print("  3. PyTorch installed without CUDA support")
        print("\nTo install PyTorch with CUDA:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return 1

    # GPU detected - show details
    from lorascan import train

    try:
        gpu_info = train.verify_gpu()
        print(f"\n✅ GPU Ready!")
        print(f"  Device: {gpu_info['device_name']}")
        print(f"  Memory: {gpu_info['memory_gb']}GB")
        print(f"  Precision: {'bf16 (best)' if gpu_info['supports_bf16'] else 'fp16 (good)'}")
        print(f"  Device ID: cuda:{gpu_info['device_id']}")

        # Memory recommendations
        mem = gpu_info['memory_gb']
        if mem < 12:
            print(f"\n⚠️  Warning: {mem}GB VRAM may be insufficient for default batch sizes")
            print("   Consider reducing batch sizes in configs/defaults.yaml")
        elif mem >= 24:
            print(f"\n✨ {mem}GB VRAM is excellent for this workload")
        else:
            print(f"\n✓ {mem}GB VRAM should work well with default settings")

        return 0

    except RuntimeError as e:
        print(f"\n❌ GPU check failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
