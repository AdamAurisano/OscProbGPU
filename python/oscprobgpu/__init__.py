"""OscProbGPU: CUDA port of selected OscProb neutrino oscillation calculators.

Models: Fast, NSI, NUNM, Sterile (3+1), Decay, LIV (SME), SNSI (scalar NSI),
Deco (decoherence), SiderealLIV (direction/time-dependent SME), OQS (open
quantum system). Each model exposes OscProb-style
parameter setters and a batched API::

    import numpy as np
    import oscprobgpu as opg

    p = opg.Fast(devices=[0])              # devices=[] for the CPU backend
    p.set_grid(np.geomspace(1, 100, 200), np.linspace(-1, 0, 200))
    p.calculate()                          # asynchronous on GPU
    P = p.probs()                          # P[nubar, a, b, iC, iE] = P(a -> b)

Flavour indices: 0 = e, 1 = mu, 2 = tau, 3 = sterile.

Exact gradients (off unless requested)::

    p.set_gradient_params()                # model defaults (see
                                           # default_gradient_params)
    p.set_gradient_params(["th23", "dm31", "zoa_1"])  # or by name
    p.calculate_gradient()
    G = p.grad()                           # G[nubar, p, a, b, iC, iE]
    g = p.weighted_gradient(w)             # sum(w * dP/dp), w like probs()

Bin averages (calculate_binned_gradient, binned_grad,
weighted_gradient_binned), event lists (prob_points_grad,
weighted_gradient_points, weighted_gradient_points_binned) and fixed paths
(prob_path_grad, avg_path_grad) have gradient variants too. Parameter names:
the class property parameter_names (model) and the instance property
earth_parameter_names (Z/A per Earth layer type). See
examples/gradient_fit.py for a binned likelihood fit with exact Jacobians.
"""

from ._oscprobgpu import (  # noqa: F401
    LIV,
    NSI,
    Deco,
    NUNM,
    OQS,
    SNSI,
    SiderealLIV,
    Decay,
    Fast,
    PremModel,
    Sterile,
    _dlpack,
    cuda_device_count,
    path_column_depth,
    path_transmission,
    has_cuda,
)

__all__ = ["Fast", "NSI", "NUNM", "Sterile", "Decay", "LIV", "SNSI", "Deco", "SiderealLIV", "OQS", "PremModel",
           "DeviceArray", "cuda_device_count", "has_cuda", "default_devices",
           "path_column_depth", "path_transmission"]


class DeviceArray:
    """Zero-copy view of a result in GPU memory (float64, C order).

    Exposes ``__dlpack__`` / ``__dlpack_device__`` and
    ``__cuda_array_interface__``, so ``cupy.asarray(a)``,
    ``torch.from_dlpack(a)`` or ``numba.cuda.as_cuda_array(a)`` use the memory
    without copying, and the propagators' weighted_gradient* methods accept it
    as weights. Valid until the propagator recomputes or is destroyed (the
    view keeps the propagator alive).
    """

    def __init__(self, owner, ptr, shape, device):
        self._owner, self._ptr = owner, ptr
        self.shape, self.device = tuple(shape), device
        self.dtype = "float64"

    @property
    def size(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def __dlpack__(self, stream=None, **kwargs):
        return _dlpack(self._ptr, list(self.shape), self.device, self._owner)

    def __dlpack_device__(self):
        return (2, self.device)  # kDLCUDA

    @property
    def __cuda_array_interface__(self):
        return {"shape": self.shape, "typestr": "<f8", "data": (self._ptr, True),
                "version": 3, "strides": None}

    def __repr__(self):
        return f"DeviceArray(shape={self.shape}, device={self.device})"


def _device_probs(self):
    """Grid probabilities on the GPU (shape of probs()) as a DeviceArray.
    Single-GPU propagators."""
    return DeviceArray(self, *self._device_view("probs"))


def _device_binned(self):
    """Bin averages on the GPU (shape of binned()) as a DeviceArray.
    Single-GPU propagators."""
    return DeviceArray(self, *self._device_view("binned"))


for _cls in (Fast, NSI, NUNM, Sterile, Decay, LIV, SNSI, Deco, SiderealLIV, OQS):
    _cls.device_probs = _device_probs
    _cls.device_binned = _device_binned


def default_devices():
    """[0] if a CUDA device is available, else [] (CPU backend)."""
    return [0] if has_cuda and cuda_device_count() > 0 else []
