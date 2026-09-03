from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(Extension(
        name="core",
        sources=["core.pyx"]
    )),
    include_dirs=[np.get_include()],
)