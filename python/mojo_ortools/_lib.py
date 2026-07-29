"""ctypes boundary for the Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_ORTOOLS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-ortools.so"
)

I = ctypes.c_int64
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1

_SIGNATURES = {
    "mot_propagate": ([I] * 9 + [I] * 4, I),
    "mot_validate_assignment": ([I] * 10, I),
    "mot_construct_routes": ([I] * 9 + [I] * 3, I),
    "mot_two_opt": ([I] * 5, I),
    "mot_relocate": ([I] * 9, I),
    "mot_route_cost": ([I] * 4, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "ortools.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    pixi = shutil.which("pixi")
    command = (
        [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
        if pixi
        else ["bash", os.path.join(ROOT, "build", "build.sh")]
    )
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIBRARY, name)
            function.argtypes = argtypes
            function.restype = restype
    return _LIBRARY


def i64(values, *, copy: bool = False) -> np.ndarray:
    """Return a C-contiguous int64 array without lossy coercion."""

    original = np.asarray(values)
    if original.size == 0:
        return np.empty(original.shape, dtype=np.int64, order="C")
    if original.dtype.kind not in "biu":
        raise TypeError("expected integer values")
    if original.dtype.kind == "u" and original.size and np.any(original > I64_MAX):
        raise OverflowError("value does not fit in int64")
    if original.dtype.kind == "O":
        # Object arrays are deliberately rejected above: accepting them would
        # make it easy for arbitrary Python objects to narrow silently.
        raise TypeError("expected integer values")
    return np.array(
        original,
        dtype=np.int64,
        order="C",
        copy=copy or not (
            original.dtype == np.int64
            and original.flags.c_contiguous
            and original.flags.aligned
            and original.flags.writeable
        ),
    )


def nonempty(values) -> np.ndarray:
    array = i64(values)
    return array if array.size else np.zeros(1, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    if array.dtype != np.int64 or not array.flags.c_contiguous or not array.flags.aligned:
        raise TypeError("FFI arrays must be aligned, C-contiguous int64 arrays")
    if not array.flags.writeable:
        raise TypeError("FFI arrays must be writable")
    if array.size == 0:
        raise ValueError("FFI arrays must use a nonempty sentinel")
    return int(array.ctypes.data)
