"""OscProbGPU: CUDA port of selected OscProb neutrino oscillation calculators.

Models: Fast, NSI, NUNM, Sterile (3+1; 3+2 and 3+3 with nus=5, 6), Decay,
LIV (SME), SNSI (scalar NSI),
Deco (decoherence), SiderealLIV (direction/time-dependent SME), OQS (open
quantum system). Each model exposes OscProb-style
parameter setters and a batched API::

    import numpy as np
    import oscprobgpu as opg

    p = opg.Fast(devices=[0])              # devices=[] for the CPU backend
    p.set_grid(np.geomspace(1, 100, 200), np.linspace(-1, 0, 200))
    p.calculate()                          # asynchronous on GPU
    P = p.probs()                          # P[nubar, a, b, iC, iE] = P(a -> b)

Flavour indices: 0 = e, 1 = mu, 2 = tau, 3.. = sterile.

Sterile neutrinos, as OscProb's PMNS_Sterile(numNus)::

    p = opg.Sterile(nus=5)                 # 3+2: an opg.Sterile5 (nus=6: Sterile6)
    p.set_dm(5, 1.3); p.set_angle(2, 5, 0.1); p.set_delta(2, 5, 0.4)
    p.n_flavours                           # 5; probs() is [nubar, 5, 5, iC, iE]

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

Many parameter points at once (fits, scans): a point is the model's current
parameters with some model parameters replaced, given as names and an
(n_points, len(names)) array::

    A, dA = p.binned_batch(["th23", "dm31"], values, earths=None, gradient=True)
    # A[p, nubar, a, b, iCbin, iEbin]; calculate_binned_batch: into CUDA arrays

Analytic bin averages for fixed paths (Fast, NSI, NUNM, Sterile)::

    A = p.avg_path_analytic(edges, path, measure="invE")         # A[a, b, bin]
    h = p.analytic_batch(edges, path, nubar=False, measure="invE")
    A, dA = p.avg_path_analytic_batch(h, ["th24", "dm41"], values, gradient=True)
    g = p.avg_path_analytic_batch_weighted(h, ["th24", "dm41"], values, w)

A 3D Earth of depth x latitude x longitude bins (OscProb EarthModelBinned)
instead of PREM; paths depend on the azimuth too::

    e = opg.EarthModelBinned()             # data/earth_binned_default.txt
    e.set_det_pos(6368, 42.4, 13.5)        # radius km, latitude, longitude (deg)
    p = opg.Fast(earth=e)
    p.set_earth_azimuth(250)               # grids and bin averages (degrees)
    P = p.prob_points(E, cz, nubar, azimuth=az)   # per-event azimuths

Region indices play the role of PREM layer types for zoa_<i> and rho_<i>.

Long calls release the GIL, so other Python threads run meanwhile (one
propagator must not be used from two threads at once; separate propagators
may).
"""

import os as _os

# Installed wheels carry the Earth model tables in oscprobgpu/data; PremModel()
# finds them through OPG_DATA_DIR (a user setting wins). Builds from the source
# tree have no such directory and use the path compiled into the module.
_data = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data")
if _os.path.isfile(_os.path.join(_data, "prem_default.txt")):
    _os.environ.setdefault("OPG_DATA_DIR", _data)

from ._oscprobgpu import (  # noqa: E402,F401
    LIV,
    NSI,
    Deco,
    NUNM,
    OQS,
    SNSI,
    AnalyticAvgOptions,
    SiderealLIV,
    Decay,
    EarthModelBinned,
    __version__,
    Fast,
    PremModel,
    Sterile as _Sterile4,
    Sterile5,
    Sterile6,
    _cuda_alloc,
    _cuda_to_host,
    _dlpack,
    cuda_device_count,
    path_column_depth,
    path_transmission,
    has_cuda,
)

__all__ = ["__version__", "Fast", "NSI", "NUNM", "Sterile", "Sterile5", "Sterile6", "Decay",
           "LIV", "SNSI", "Deco", "SiderealLIV", "OQS", "PremModel", "EarthModelBinned",
           "AnalyticAvgOptions", "DeviceArray", "cuda_empty",
           "cuda_device_count", "has_cuda", "default_devices", "path_column_depth",
           "path_transmission"]


class DeviceArray:
    """Zero-copy view of GPU memory (float64, C order).

    Exposes ``__dlpack__`` / ``__dlpack_device__`` and
    ``__cuda_array_interface__``, so ``cupy.asarray(a)``,
    ``torch.from_dlpack(a)`` or ``numba.cuda.as_cuda_array(a)`` use the memory
    without copying, and the propagators accept it as CUDA weights or
    outputs. ``get()`` copies it to a numpy array.

    Views of a propagator's results (device_probs(), device_binned()) keep
    the propagator alive and are read-only. Their contents change when the
    propagator recomputes; after set_grid (probs), set_bins (binned) or
    set_earth (both) the buffer may be freed or reallocated, and any use of
    an older view raises RuntimeError (see ``valid``). Handing a view out
    (``__dlpack__`` whatever its ``stream`` argument,
    ``__cuda_array_interface__``, ``get()``) first waits for the propagator's
    pending work, so consumers on any stream see finished results.
    """

    def __init__(self, owner, ptr, shape, device, what=None, generation=None):
        self._owner, self._ptr = owner, ptr
        self.shape, self.device = tuple(shape), device
        self.dtype = "float64"
        self._what, self._gen = what, generation

    @property
    def size(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def valid(self):
        """False once the propagator may have freed or moved the buffer."""
        return self._what is None or self._owner._device_generation(self._what) == self._gen

    def _ready(self):
        if not self.valid:
            raise RuntimeError(f"DeviceArray: stale view of {self._what} (the propagator's "
                               "grid, bins or Earth changed since it was taken)")
        if self._what is not None:
            self._owner.wait()

    def __dlpack__(self, stream=None, **kwargs):
        self._ready()
        return _dlpack(self._ptr, list(self.shape), self.device, self._owner,
                       self._what is None)

    def __dlpack_device__(self):
        return (2, self.device)  # kDLCUDA

    @property
    def __cuda_array_interface__(self):
        self._ready()
        return {"shape": self.shape, "typestr": "<f8",
                "data": (self._ptr, self._what is not None),
                "version": 3, "strides": None}

    def get(self):
        """Copy to a numpy array (waits for the device)."""
        self._ready()
        return _cuda_to_host(self._ptr, list(self.shape), self.device)

    def __repr__(self):
        return f"DeviceArray(shape={self.shape}, device={self.device})"


def cuda_empty(shape, device=0):
    """Zero-filled float64 GPU array that owns its memory (a DeviceArray), e.g.
    for calculate_binned_batch or avg_path_analytic_batch_device outputs
    without CuPy or PyTorch. ``get()`` copies it to the host."""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    n = 1
    for s in shape:
        n *= s
    ptr, owner = _cuda_alloc(n, device)
    return DeviceArray(owner, ptr, shape, device)


def _view(self, what):
    ptr, shape, dev, gen = self._device_view(what)
    return DeviceArray(self, ptr, shape, dev, what, gen)


def _device_probs(self):
    """Grid probabilities on the GPU (shape of probs()) as a DeviceArray.
    Single-GPU propagators."""
    return _view(self, "probs")


def _device_binned(self):
    """Bin averages on the GPU (shape of binned()) as a DeviceArray.
    Single-GPU propagators."""
    return _view(self, "binned")


def _checks_device_arrays(f):
    """Wrap a method taking CUDA arrays: a stale DeviceArray argument raises
    RuntimeError (nanobind would report a failed conversion as TypeError)."""
    def g(self, *args, **kwargs):
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, DeviceArray):
                a._ready()
        return f(self, *args, **kwargs)
    g.__name__ = getattr(f, "__name__", "method")
    g.__doc__ = getattr(f, "__doc__", None)
    return g


_CUDA_ARG_METHODS = ("weighted_gradient", "weighted_gradient_binned", "calculate_binned_batch",
                     "avg_path_analytic_batch_device", "avg_path_analytic_batch_weighted")

_STERILE_N = {5: Sterile5, 6: Sterile6}


class Sterile(_Sterile4):
    """Oscillations with sterile neutrinos (OscProb's PMNS_Sterile(nus)).

    ``Sterile(earth=PremModel(), devices=[], threads=0, nus=4)``: nus = 4
    (3+1, the default) makes a Sterile; nus = 5 or 6 returns a Sterile5 or
    Sterile6 (3+2, 3+3), the classes compiled for those sizes, which take
    the same arguments and have the same methods with N = nus flavours.
    """

    def __new__(cls, *args, nus=4, **kwargs):
        if nus == 4:
            return super().__new__(cls)
        if nus in _STERILE_N:
            return _STERILE_N[nus](*args, **kwargs)
        raise ValueError(f"Sterile: nus = {nus} is not built (4, 5 or 6)")

    def __init__(self, *args, nus=4, **kwargs):
        super().__init__(*args, **kwargs)


for _cls in (Fast, NSI, NUNM, _Sterile4, Sterile5, Sterile6, Decay, LIV, SNSI, Deco,
             SiderealLIV, OQS):
    _cls.device_probs = _device_probs
    _cls.device_binned = _device_binned
    for _m in _CUDA_ARG_METHODS:
        if hasattr(_cls, _m):
            setattr(_cls, _m, _checks_device_arrays(getattr(_cls, _m)))


def default_devices():
    """[0] if a CUDA device is available, else [] (CPU backend)."""
    return [0] if has_cuda and cuda_device_count() > 0 else []
