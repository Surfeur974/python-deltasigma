# -*- coding: utf-8 -*-
"""Remplacements modernes pour les tests de deltasigma.

- ``raises``       : remplace ``nose.tools.raises`` (nose ne marche plus
                     depuis Python 3.12, module ``imp`` supprime).
- ``pkg_resources``: remplace ``pkg_resources`` (retire de setuptools
                     recent) pour resource_filename / resource_stream /
                     resource_string.
"""
import functools
import importlib
import os
import types


def raises(*exceptions):
    """Decorateur : le test doit lever une des exceptions donnees."""
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except exceptions:
                return
            raise AssertionError(
                "%s n'a pas leve %s" % (func.__name__,
                                        " ou ".join(e.__name__ for e in exceptions)))
        return wrapper
    return decorate


def _base_dir(package):
    """Dossier de reference, meme semantique que pkg_resources."""
    if not isinstance(package, str):
        package = package.__name__
    mod = importlib.import_module(package)
    path = os.path.dirname(os.path.abspath(mod.__file__))
    return path


def resource_filename(package, resource_name):
    parts = resource_name.split("/")
    return os.path.join(_base_dir(package), *parts)


def resource_stream(package, resource_name):
    return open(resource_filename(package, resource_name), "rb")


def resource_string(package, resource_name):
    with resource_stream(package, resource_name) as f:
        return f.read()


def resource_exists(package, resource_name):
    return os.path.exists(resource_filename(package, resource_name))


pkg_resources = types.SimpleNamespace(
    resource_filename=resource_filename,
    resource_stream=resource_stream,
    resource_string=resource_string,
    resource_exists=resource_exists,
)
