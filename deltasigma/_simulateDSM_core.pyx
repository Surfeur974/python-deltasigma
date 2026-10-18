# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
"""Noyau compile de simulateDSM.

Points de conception importants :

* On n'utilise PAS ``cimport numpy``. Le noyau repose uniquement sur des
  memoryviews typees (protocole tampon standard de Python). Le fichier C
  genere ne reference donc aucune structure interne de NumPy, ce qui le
  rend compatible a la fois avec NumPy 1.x et NumPy 2.x. C'est essentiel :
  un ``.c`` genere avec ``cimport numpy`` contre NumPy 1.x ne compile plus
  avec NumPy 2.0 (la structure ``PyArray_Descr`` a change).

* La boucle principale s'execute ``nogil`` : aucun appel a l'interpreteur
  Python n'a lieu pendant la simulation.
"""

import numpy as np

from libc.math cimport floor, fabs


cdef inline double _quantize(double y, double levels) noexcept nogil:
    """Quantificateur delta-sigma, identique a ds_quantize().

    * nombre de niveaux pair  -> quantificateur mid-rise
    * nombre de niveaux impair -> quantificateur mid-tread
    """
    cdef double value
    cdef double limit = levels - 1.0

    if (<long> levels) % 2 == 0:
        value = 2.0*floor(0.5*y) + 1.0
    else:
        value = 2.0*floor(0.5*(y + 1.0))

    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def simulateDSM_core(double[:, ::1] u,
                     double[:, ::1] A,
                     double[:, ::1] B,
                     double[:, ::1] C,
                     double[:, ::1] D1,
                     double[::1] nlev,
                     double[::1] x0):
    """Simule un modulateur delta-sigma decrit par sa representation d'etat.

    Tous les tableaux doivent etre contigus en C et de type float64.
    """
    cdef Py_ssize_t nu = u.shape[0]
    cdef Py_ssize_t N = u.shape[1]
    cdef Py_ssize_t order = A.shape[0]
    cdef Py_ssize_t nq = C.shape[0]
    cdef Py_ssize_t i, j, k, q
    cdef double acc, value, magnitude

    v_arr = np.empty((nq, N), dtype=np.float64)
    y_arr = np.empty((nq, N), dtype=np.float64)
    xn_arr = np.empty((order, N), dtype=np.float64)
    xmax_arr = np.empty(order, dtype=np.float64)
    xnew_arr = np.empty(order, dtype=np.float64)

    cdef double[:, ::1] v = v_arr
    cdef double[:, ::1] y = y_arr
    cdef double[:, ::1] xn = xn_arr
    cdef double[::1] xmax = xmax_arr
    cdef double[::1] xnew = xnew_arr

    with nogil:
        for j in range(order):
            xmax[j] = fabs(x0[j])

        for i in range(N):
            # --- sortie du filtre de boucle, puis quantification ---
            for q in range(nq):
                acc = 0.0
                for j in range(order):
                    acc += C[q, j]*x0[j]
                for j in range(nu):
                    acc += D1[q, j]*u[j, i]
                y[q, i] = acc
                v[q, i] = _quantize(acc, nlev[q])

            # --- mise a jour de l'etat ---
            for j in range(order):
                acc = 0.0
                for k in range(order):
                    acc += A[j, k]*x0[k]
                for k in range(nu):
                    acc += B[j, k]*u[k, i]
                for q in range(nq):
                    acc += B[j, nu + q]*v[q, i]
                xnew[j] = acc

            # --- memorisation de l'etat et de son maximum ---
            for j in range(order):
                x0[j] = xnew[j]
                xn[j, i] = xnew[j]
                magnitude = fabs(xnew[j])
                if magnitude > xmax[j]:
                    xmax[j] = magnitude

    return (v_arr.squeeze(), xn_arr.squeeze(),
            xmax_arr.reshape((-1, 1)), y_arr.squeeze())
