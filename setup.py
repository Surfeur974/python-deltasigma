#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup python-deltasigma (fork Surfeur974) + extensions Cython.

- Tous les deltasigma/*.pyx sont detectes et compiles automatiquement.
- Sans Cython : compile les .c livres a cote des .pyx s'ils existent.
- .pyx avec ``cimport numpy`` : headers NumPy ajoutes automatiquement.
- Echec de compilation tolere (repli Python), SAUF si DELTASIGMA_STRICT=1.

Build :   python setup.py build_ext --inplace
Debug :   DELTASIGMA_STRICT=1 python setup.py build_ext --inplace
"""
import glob
import os
import re
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

__version__ = "0.2.3"

HERE = os.path.abspath(os.path.dirname(__file__))
PACKAGE = "deltasigma"
STRICT = os.environ.get("DELTASIGMA_STRICT", "") not in ("", "0")

# Modules NON compiles : _simulateDSM_cblas exige une lib CBLAS externe
# (cblas.h + .lib), absente sous Windows. Le repli Python/autres backends
# prend le relais.
EXCLUDE = {"_simulateDSM_cblas"}

try:
    from Cython.Build import cythonize
    HAVE_CYTHON = True
except ImportError:
    cythonize = None
    HAVE_CYTHON = False

_CIMPORT_NUMPY = re.compile(r"^\s*(cimport\s+numpy|from\s+numpy\s+cimport)", re.M)


def read(fname):
    try:
        with open(os.path.join(HERE, fname)) as fp:
            return fp.read()
    except IOError:
        return ""


# --------------------------------------------------------------------------
# Extensions Cython
# --------------------------------------------------------------------------
def optimisation_flags():
    return ["-O3"], []


def discover_modules():
    pkg_dir = os.path.join(HERE, PACKAGE)
    pyx = {os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(pkg_dir, "*.pyx"))}
    pyx -= EXCLUDE
    if HAVE_CYTHON:
        return sorted(pyx)
    c_files = {os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(pkg_dir, "*.c"))}
    manquants = sorted(pyx - c_files)
    if manquants:
        print("ATTENTION : Cython absent, pas de .c pour : %s" % ", ".join(manquants))
        print("            -> pip install cython")
    return sorted(pyx & c_files)


def needs_numpy_headers(name):
    pyx = os.path.join(HERE, PACKAGE, name + ".pyx")
    if not os.path.exists(pyx):
        return False
    with open(pyx, encoding="utf-8") as f:
        return bool(_CIMPORT_NUMPY.search(f.read()))


def build_extensions():
    suffix = ".pyx" if HAVE_CYTHON else ".c"
    compile_args, link_args = optimisation_flags()
    extensions = []
    for name in discover_modules():
        include_dirs, macros = [], []
        if needs_numpy_headers(name):
            import numpy
            include_dirs.append(numpy.get_include())
            macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))
        extensions.append(
            Extension(
                "%s.%s" % (PACKAGE, name),
                sources=[os.path.join(PACKAGE, name + suffix)],
                include_dirs=include_dirs,
                define_macros=macros,
                extra_compile_args=compile_args,
                extra_link_args=link_args,
                optional=not STRICT,
            )
        )
    print("Extensions detectees : %s" % [e.name for e in extensions])
    if HAVE_CYTHON and extensions:
        # cythonize module par module : un .pyx casse n'empeche plus
        # les autres d'etre construits (sauf en mode STRICT).
        ok = []
        for ext in extensions:
            try:
                ok += cythonize(
                    [ext],
                    compiler_directives={
                        "language_level": 3,
                        "boundscheck": False,
                        "wraparound": False,
                        "cdivision": True,
                        "initializedcheck": False,
                    },
                )
            except Exception as exc:
                if STRICT:
                    raise
                print("*" * 70)
                print("Cython a echoue sur %s : %s" % (ext.name, exc))
                print("Module ignore, backend Python conserve.")
                print("*" * 70)
        extensions = ok
    return extensions


class ToleranteBuildExt(build_ext):
    """N'echoue pas sans compilateur (sauf DELTASIGMA_STRICT=1)."""

    def run(self):
        try:
            build_ext.run(self)
        except Exception as exc:
            if STRICT:
                raise
            self._avertir("(global)", exc)

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except Exception as exc:
            if STRICT:
                raise
            self._avertir(ext.name, exc)

    @staticmethod
    def _avertir(nom, exc):
        print()
        print("*" * 70)
        print("Extension NON construite : %s" % nom)
        print("  %s: %s" % (type(exc).__name__, exc))
        print("Backend Python conserve. Relance avec DELTASIGMA_STRICT=1")
        print("pour voir l'erreur complete du compilateur.")
        print("*" * 70)
        print()


# --------------------------------------------------------------------------
setup(
    name='deltasigma',
    version=__version__,
    packages=find_packages(),
    package_data={
        'deltasigma': ['tests/test_data/*.mat', 'tests/test_data/*.txt',
                       '*.pyx', '*.pxd', '*.c'],
    },
    ext_modules=build_extensions(),
    cmdclass={"build_ext": ToleranteBuildExt},
    install_requires=["numpy==2.5.4", "scipy==1.18.1", "matplotlib==3.11.2"],
    python_requires=">=3.8",
    zip_safe=False,
    include_package_data=True,
    author="Giuseppe Venturini and others (fork: Surfeur974)",
    author_email="giuseppe.g.venturini@ieee.org",
    description="a Python package to synthesize, simulate, scale and map "
                "to implementable topologies delta sigma modulators. "
                "Modernized fork of the Python Delta-Sigma Toolbox",
    long_description=''.join([read('pypi_description.rst'), '\n\n',
                              read('CHANGES.rst')]),
    license="BSD",
    keywords="delta sigma modulator simulator",
    url="https://github.com/Surfeur974/python-deltasigma",
    test_suite="deltasigma.tests",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
    ],
)