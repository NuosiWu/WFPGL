try:
    from setuptools import setup
    from setuptools.extension import Extension

except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy as np

sourcefiles = ['_pglasso.pyx', '_pglasso.c', 'theta_update.pyx', '__init__.py']


setup(
    name = "pglasso",
    version = "2.0.0",
    author='Maxim Grechkin & Nuosi Wu',
    author_email='nuosiwu@gmail.com',
    url="https://github.com/NuosiWu/WFPGL",
    packages=['pglasso'],
    scripts=["pglasso/__init__.py"],
    description="Modified PGL for WFPGL implementation",
    license="MIT",
    ext_modules = cythonize( Extension("_pglasso", ["_pglasso.pyx"], include_dirs=[np.get_include()])),
    install_requires=[
        "cython >= 0.20.1",
        "numpy >= 1.8.0",
        "networkx >= 1.8.1"
    ]
)
