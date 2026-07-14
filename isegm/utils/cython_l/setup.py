from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Set the compiler
os.environ["CC"] = "winegcc"
os.environ["CXX"] = "wineg++"

setup(
    ext_modules=cythonize("_get_dist_maps.pyx"),
    compiler_directives={'language_level': "3"}
)