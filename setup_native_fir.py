# setup_native_fir.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    # MSVC
    extra_compile_args = ["/O2", "/GL"]
    extra_link_args = ["/LTCG"]
else:
    # GCC / Clang
    extra_compile_args = ["-O3", "-march=native"]

ext_modules = [
    Extension(
        "src.app.native_fir.fir_decimator",  # ← パッケージ名込み
        [str(SRC_DIR / "app" / "native_fir" / "fir_decimator.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="native_fir",
    ext_modules=cythonize(
        ext_modules,
        annotate=False,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
        },
    ),
)
