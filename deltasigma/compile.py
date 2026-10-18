"""Point d'entree de compatibilite pour l'ancien script de build Cython.

Les extensions natives sont desormais compilees via setup.py. Ce module
n'importe plus Cython au chargement, afin que `import deltasigma.compile`
fonctionne meme sans Cython installe.
"""


def main():
    try:
        import Cython  # noqa: F401
    except ImportError:
        print("Cython absent : backend Python/NumPy utilise.")
        return False
    print("Utiliser : python setup.py build_ext --inplace")
    return True


if __name__ == "__main__":
    main()
