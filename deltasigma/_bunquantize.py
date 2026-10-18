# -*- coding: utf-8 -*-
# _bunquantize.py
# Module providing the bunquantize function
# Copyright 2013 Giuseppe Venturini
# This file is part of python-deltasigma.
#
# python-deltasigma is a 1:1 Python replacement of Richard Schreier's
# MATLAB delta sigma toolbox (aka "delsigma"), upon which it is heavily based.
# The delta sigma toolbox is (c) 2009, Richard Schreier.
#
# python-deltasigma is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LICENSE file for the licensing terms.

"""Module providing the bunquantize() function
"""

import numpy as np

from ._utils import carray


def bunquantize(q):
    """The value corresponding to a bidirectionally quantized quantity.

    ``q`` is a (2n, m) ndarray containing the powers of 2 and their signs
    for each quantized value.

    .. seealso::
        :func:`bquantize`

    """
    # Liste de matrices CSD (sortie de bquantize) : longueurs differentes
    # possibles, NumPy 2 refuse d'en faire un tableau -> une valeur par element.
    if isinstance(q, (list, tuple)) and len(q) and \
            all(np.ndim(qi) == 2 for qi in q):
        return carray([float(bunquantize(qi).sum()) for qi in q])
    q = np.atleast_2d(carray(q))
    # q est de forme (2n, m) : lignes paires = exposants, lignes impaires = signes.
    # Une matrice CSD unique (2, m) doit donner UNE valeur, pas m valeurs.
    if q.shape[0] % 2 != 0:
        raise ValueError('q doit avoir un nombre pair de lignes, recu %d'
                         % q.shape[0])
    y = []
    for start in range(0, q.shape[0], 2):
        # ligne paire = exposants de 2, ligne impaire = signes
        powers = q[start, :]
        signs = q[start + 1, :]
        y += [float(np.sum(signs*2.**powers))]
    return carray(y)
