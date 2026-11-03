"""Build-time check that a GPU solver image can actually factorise, without a GPU.

Runs inside ``Dockerfile.gpu`` after the requirements are installed. Exit 0 means
the image has the stack a GPU solve needs; any other exit fails the BUILD, so a
broken image cannot reach Harbor and cannot be deployed to the A4000.

Why this file exists rather than two ``python -c`` lines in the Dockerfile: the
permission string is long enough to be unreadable when backslash-continued, and
Docker's line continuation cannot carry a multi-line Python literal. As a
checked-in script it is also runnable (and mutation-tested) outside a build.

Why an import check is not enough
---------------------------------
The solver imports cupy, cupyx.scipy.sparse and nvmath on its GPU path, and
``mse_backend._missing_gpu_package()`` names the first of those that is absent.
All three import cleanly WITHOUT cuDSS. cuDSS arrives only because
nvmath-python declares it under ``extra == "cu12"``:

    nvidia-cudss-cu12==0.7.*; extra == "cu12"

so ``nvmath-python==0.9.0`` on its own installs no cuDSS at all, and the first
``nvmath.sparse.advanced.DirectSolver`` factorisation is where that surfaces --
on the A4000, after build, push and deploy. That is the failure this probe
closes: it loads the cuDSS shared object itself.

``dlopen`` needs no GPU (it is the loader, not a kernel), so this still fails at
build time. Only a real factorisation needs the card, and that is the lead's
bounded-Job smoke on one A4000.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import sys
from pathlib import Path

_CUDSS_DIST = "nvidia-cudss-cu12"

_NO_PKG = (
    "no " + _CUDSS_DIST + " distribution: nvmath-python does not install cuDSS "
    "on its own (it is declared only under the cu12 extra). Pin it in "
    "requirements-gpu.txt."
)


def import_probe() -> str:
    """Import each GPU package the solver needs. Returns what it checked.

    This deliberately does NOT go through ``mse_backend._missing_gpu_package()``.
    That helper returns ``"the CUDA stack"`` as a LABEL for an error message when
    nothing is missing, so a caller cannot tell "all present" from "the stack is
    missing" -- a first version of this probe read the label as a failure and
    refused a correct image. The packages are imported here, one name at a time,
    from the same tuple the solver uses, so the two cannot drift.
    """
    sys.path.insert(0, "/opt/app")
    import mse_backend  # noqa: PLC0415 -- needs the path insert above

    for module, name in mse_backend._GPU_PACKAGES:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise AssertionError(
                f"this GPU image cannot run a GPU solve: {name!r} is not "
                f"importable ({exc})"
            ) from exc
    return ", ".join(name for _, name in mse_backend._GPU_PACKAGES)


def cudss_probe() -> str:
    """Load the cuDSS shared object, no GPU required. Returns the library name.

    Located through the ``nvidia-cudss-cu12`` DISTRIBUTION, not by importing a
    module: that wheel is data-only. Measured on 0.7.1.6, it installs

        nvidia/cu12/lib/libcudss.so.0        (no importable nvidia.cudss package)

    so ``importlib.util.find_spec("nvidia.cudss")`` fails on a correct image.
    An earlier version of this probe did exactly that and refused a good
    install -- a check whose constants were assumed rather than read off the
    thing it checks.
    """
    try:
        dist = importlib.metadata.distribution(_CUDSS_DIST)
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError(_NO_PKG) from None
    lib_root = Path(dist.locate_file("nvidia")) / "cu12" / "lib"
    libs = sorted(lib_root.glob("*cudss*.so*"))
    if not libs:
        raise AssertionError(f"{_CUDSS_DIST} is installed but ships no cuDSS library under {lib_root}")
    # Load the SONAME-pinned file (libcudss.so.0); loading raises OSError if a
    # dependency it needs is missing, which is the other half of the check.
    ctypes.CDLL(str(libs[0]))
    return libs[0].name
    ctypes.CDLL(libs[0])  # raises OSError if a dependency it needs is missing
    return libs[0].rsplit("/", 1)[-1]


def main() -> int:
    import_probe()
    print("GPU STACK IMPORTS OK")
    print("cuDSS LOADS OK: " + cudss_probe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
