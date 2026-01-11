import os

import numpy as np
import setuptools
from Cython.Build.Dependencies import cythonize
from Cython.Distutils import build_ext

# TODO - To compile use: python setup.py build_ext --inplace
# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
GEODESIC_NAME = "vtpdist"
COMPILER_DIRECTIVES = {"language_level": 3, }
INCLUDE_DIRS = [np.get_include()]  # NumPy dtypes
INSTALL_REQUIREMENTS = ["numpy", "scipy", "cython"]
DEFINED_MACROS = [
    ("NDEBUG", 1),  # Disable assertions; one is failing geodesic_mesh.h:405
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")  # Suppress NumPy deprecation warnings
]

if "COVERAGE" in os.environ:
    COMPILER_DIRECTIVES["linetrace"] = True
    DEFINED_MACROS.append(("CYTHON_TRACE_NOGIL", "1"))

ROOT_PYX = "src/cygeodesic.pyx"
# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

GEODESIC_MODULE = [
    setuptools.Extension(
        name=GEODESIC_NAME,  # Name of extension
        sources=[ROOT_PYX],  # Filename of Cython source
        language="c++",  # Cython create C++ source
        define_macros=DEFINED_MACROS,
        # extra_compile_args=["/openmp", ],
        # extra_compile_args=['-lomp', ],
        extra_compile_args=['-fopenmp', ],
        # extra_link_args=['-lomp'],
        extra_link_args=['-fopenmp'],
        # extra_link_args=["/openmp", "/fp:fast"],
        include_dirs=INCLUDE_DIRS,
    )
]


# For Linux:
extra_compile_args = ["--std=c++14", '-fpermissive']

extra_link_args = ["--std=c++14"]

# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


class new_build_ext(build_ext):
    def finalize_options(self):
        self.distribution.ext_modules = cythonize(
            self.distribution.ext_modules,
            compiler_directives=COMPILER_DIRECTIVES,
            annotate=False,
        )
        if not self.include_dirs:
            self.include_dirs = []
        elif isinstance(self.include_dirs, str):
            self.include_dirs = [self.include_dirs]
        self.include_dirs += INCLUDE_DIRS
        super().finalize_options()


setuptools.setup(
    ext_modules=GEODESIC_MODULE,
    include_dirs=INCLUDE_DIRS,
    cmdclass={"build_ext": new_build_ext},
    install_requires=INSTALL_REQUIREMENTS,
    description="Compute geodesic distances",
)
