"""Utility functions for memory management and debugging."""
import gc


def cleanup_cuda():
    """Clean up CUDA memory - comprehensive cleanup matching mentor's approach."""
    gc.collect()  # Initial gc

    # 1. Explicitly free large CUDA tensors from Python heap
    try:
        import torch
        freed = 0
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda and obj.numel() > 1e6:
                    freed += obj.numel() * obj.element_size()
                    obj.data = torch.empty(0, device='cpu')
            except (ReferenceError, RuntimeError):
                pass

        # 2. Find and delete lingering tensors/models in global scope
        for name in list(globals().keys()):
            obj = globals()[name]
            if isinstance(obj, torch.Tensor) and obj.device.type == 'cuda':
                print(f'  Lingering tensor: {name} — {obj.shape} {obj.dtype}')
                del globals()[name]
            elif isinstance(obj, torch.nn.Module):
                try:
                    device = next(obj.parameters()).device.type
                    if device == 'cuda':
                        allocated = sum(p.numel() * p.element_size() for p in obj.parameters()) / 1e9
                        print(f'  Lingering module: {name} — {type(obj).__name__} ~{allocated:.2f} GB')
                        del globals()[name]
                except StopIteration:
                    pass

        # 3. gc + cuda cache clear
        gc.collect()
        gc.collect()  # 2nd pass for circular refs

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 4. Report
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'  Freed ~{freed/1e9:.2f} GB | Allocated: {allocated:.2f} / {total:.2f} GB | Reserved: {reserved:.2f} GB')

    except ImportError:
        # torch not available, just do gc
        pass


def collect_gc():
    """Simple garbage collection - alias for cleanup_cuda."""
    cleanup_cuda()
