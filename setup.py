import os
import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

march = os.environ.get("MEMBERSHIPDT_MARCH", "native")
arch_flags = [] if march in ("", "none", "portable") else [f"-march={march}", f"-mtune={march}"]
ext = [Extension("membershipdt._builder_cy", ["src/membershipdt/_builder_cy.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-O3", "-fopenmp", *arch_flags], extra_link_args=["-fopenmp"], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]
setup(ext_modules=cythonize(ext, language_level=3, compiler_directives={"language_level": 3}))
