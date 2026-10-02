"""OscProbGPU: CUDA port of selected OscProb neutrino oscillation calculators.

Models: Fast, NSI, NUNM, Sterile (3+1), Decay. Each model exposes OscProb-style
parameter setters and a batched API::

    import numpy as np
    import oscprobgpu as opg

    p = opg.Fast(devices=[0])              # devices=[] for the CPU backend
    p.set_grid(np.geomspace(1, 100, 200), np.linspace(-1, 0, 200))
    p.calculate()                          # asynchronous on GPU
    P = p.probs()                          # P[nubar, a, b, iC, iE] = P(a -> b)

Flavour indices: 0 = e, 1 = mu, 2 = tau, 3 = sterile.
"""

from ._oscprobgpu import (  # noqa: F401
    NSI,
    NUNM,
    Decay,
    Fast,
    PremModel,
    Sterile,
    cuda_device_count,
    has_cuda,
)

__all__ = ["Fast", "NSI", "NUNM", "Sterile", "Decay", "PremModel",
           "cuda_device_count", "has_cuda", "default_devices"]


def default_devices():
    """[0] if a CUDA device is available, else [] (CPU backend)."""
    return [0] if has_cuda and cuda_device_count() > 0 else []
