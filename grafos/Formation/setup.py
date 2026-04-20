# python setup.py build_ext --inplace

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# List all your individual model files here
extensions = [
    Extension("bb_model", ["bb_model.pyx"]),
    Extension("copying_model", ["copying_model.pyx"]),
    Extension("sbm_pa_model", ["sbm_pa_model.pyx"]),
    Extension("ergm_model", ["ergm_model.pyx"]),
    Extension("bter_model", ["bter_model.pyx"]),
    Extension("kronecker_model", ["kronecker_model.pyx"])
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()]
)