import time
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

def calculate_inference_time(model: nn.Module,input_tensor: torch.Tensor,num_runs: int = 100,warmup_runs: int = 2) -> float:
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()

    timings = []
    with torch.no_grad():
        # Warm-up runs to stabilize measurements
        for _ in range(warmup_runs):
            _ = model(input_tensor)

        # Timed runs
        for _ in range(num_runs):
            # Synchronize CUDA device before starting the timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
    avg_time_ms = (sum(timings) / num_runs) * 1000
    return avg_time_ms


def calculate_gflops(model: nn.Module,input_tensor: torch.Tensor) -> float:
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()
    flop_analyzer = FlopCountAnalysis(model, input_tensor)
    total_flops = flop_analyzer.total()
    # Convert FLOPs to GFLOPs
    gflops = total_flops / 1e9
    return gflops
