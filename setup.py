import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

ext = [Extension("membershipdt._builder_cy", ["membershipdt/_builder_cy.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-O3"], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]
setup(ext_modules=cythonize(ext, language_level=3))
