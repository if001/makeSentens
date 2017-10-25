"""
python setup.py build_ext --inplace

"""


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension( "cython_package", ["readfile_cython.pyx"] ),
    #Extension( "cython_package", ["readfile_cython.c"] ),
]

setup(
    name = "readfile cython",
    cmdclass = { "build_ext" : build_ext },
    ext_modules = ext_modules,
    include_dirs = [numpy.get_include()],
)

