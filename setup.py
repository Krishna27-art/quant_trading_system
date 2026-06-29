import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        name="features.fast_features",
        sources=["features/fast_features.pyx"],
        include_dirs=[np.get_include(), "features/src"],
        language="c++",
    )
]

setup(
    name="fast_features",
    ext_modules=cythonize(extensions),
    packages=[],
)
