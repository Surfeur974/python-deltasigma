# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
"""Noyau compile de simulateQDSM (modulateurs quadrature).

Comme pour ``_simulateDSM_core``, aucun ``cimport numpy`` n'est utilise :
seules des memoryviews typees sont employees, ce qui rend le fichier C
genere compatible NumPy 1.x et NumPy 2.x.
"""

import numpy as np

from libc.math cimport floor, fabs, sqrt


cdef inline double _quantize(double y, double levels) noexcept nogil:
    """Quantificateur scalaire, applique separement aux parties I et Q."""
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


def simulateQDSM_core(double complex[:, ::1] u,
                      double complex[:, ::1] A,
                      double complex[:, ::1] B,
                      double complex[:, ::1] C,
                      double complex[:, ::1] D1,
                      int order,
                      nlev,
                      int nq,
                      double complex[::1] x0):
    """Simule un modulateur delta-sigma quadrature.

    ``nlev`` peut etre un scalaire ou une sequence : il est converti en
    tableau float64 avant l'entree dans la section ``nogil``.
    """
    cdef Py_ssize_t nu = u.shape[0]
    cdef Py_ssize_t N = u.shape[1]
    cdef Py_ssize_t i, j, k, q
    cdef double complex acc
    cdef double magnitude, levels

    levels_arr = np.ascontiguousarray(
        np.real(np.atleast_1d(nlev)), dtype=np.float64
    )
    if levels_arr.shape[0] < nq:
        levels_arr = np.repeat(levels_arr[:1], nq)

    v_arr = np.empty((nq, N), dtype=np.complex128)
    y_arr = np.empty((nq, N), dtype=np.complex128)
    xn_arr = np.empty((order, N), dtype=np.complex128)
    xmax_arr = np.empty(order, dtype=np.float64)
    xnew_arr = np.empty(order, dtype=np.complex128)

    cdef double[::1] levels_view = levels_arr
    cdef double complex[:, ::1] v = v_arr
    cdef double complex[:, ::1] y = y_arr
    cdef double complex[:, ::1] xn = xn_arr
    cdef double[::1] xmax = xmax_arr
    cdef double complex[::1] xnew = xnew_arr

    with nogil:
        for j in range(order):
            xmax[j] = sqrt(x0[j].real*x0[j].real
                           + x0[j].imag*x0[j].imag)

        for i in range(N):
            for q in range(nq):
                acc = 0.0
                for j in range(order):
                    acc = acc + C[q, j]*x0[j]
                for j in range(nu):
                    acc = acc + D1[q, j]*u[j, i]
                y[q, i] = acc
                levels = levels_view[q]
                v[q, i] = (_quantize(acc.real, levels)
                           + 1j*_quantize(acc.imag, levels))

            for j in range(order):
                acc = 0.0
                for k in range(order):
                    acc = acc + A[j, k]*x0[k]
                for k in range(nu):
                    acc = acc + B[j, k]*u[k, i]
                for q in range(nq):
                    acc = acc + B[j, nu + q]*v[q, i]
                xnew[j] = acc

            for j in range(order):
                x0[j] = xnew[j]
                xn[j, i] = xnew[j]
                magnitude = sqrt(xnew[j].real*xnew[j].real
                                 + xnew[j].imag*xnew[j].imag)
                if magnitude > xmax[j]:
                    xmax[j] = magnitude

    return (v_arr.squeeze(), xn_arr.squeeze(),
            xmax_arr.reshape((-1, 1)), y_arr.squeeze())
