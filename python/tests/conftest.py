import os

import numpy as np
import pytest

import oscprobgpu as opg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("OPG_TEST_DATA_DIR",
                      os.path.join(HERE, "..", "..", "tests", "data"))


def load(name):
    return np.load(os.path.join(DATA, name))


def test_devices():
    """Backends to test: CPU always, plus GPU device(s) when available."""
    env = os.environ.get("OPG_TEST_DEVICES", "0")
    gpu = [int(x) for x in env.split(",") if x != ""]
    backends = [[]]
    if opg.has_cuda and opg.cuda_device_count() >= len(gpu):
        backends.append(gpu)
    return backends


@pytest.fixture(params=test_devices(), ids=lambda d: "gpu" if d else "cpu")
def devices(request):
    return request.param


def set_nominal(p):
    """NuFIT 5.2 NO, as SetNominalPars in OscProb/test/Utils.h"""
    p.set_dm(2, 7.41e-5)
    p.set_dm(3, 2.507e-3)
    p.set_angle(1, 2, np.arcsin(np.sqrt(0.303)))
    p.set_angle(1, 3, np.arcsin(np.sqrt(0.02225)))
    p.set_angle(2, 3, np.arcsin(np.sqrt(0.451)))
    p.set_delta(1, 3, 232 * np.pi / 180)
    return p


TEST_PATH = np.array([[1000, 2, 0.5], [1000, 4, 0.5], [1000, 2, 0.5]], float)
VAC_PATH = np.array([[1300, 0, 0.5]], float)
