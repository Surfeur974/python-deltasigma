# -*- coding: utf-8 -*-
# _plotPZ.py
# Module providing the plotPZ function
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

"""Module providing plotPZ(), useful to plot the poles and zeros of a transfer function.
"""

import numpy as np
import matplotlib.pyplot as plt

from ._utils import _get_zpk



def plotPZ(H, color='b', markersize=5, showlist=False):
    """Plot the poles and zeros of a transfer function."""

    z, p, _ = _get_zpk(H)
    p = np.real_if_close(np.round(p, 5))
    z = np.real_if_close(np.round(z, 5))

    pole_fmt = {'marker': 'x', 'markersize': markersize}
    zero_fmt = {'marker': 'o', 'markersize': markersize}

    if isinstance(color, (list, tuple)):
        pole_fmt['color'] = color[0]
        zero_fmt['color'] = color[1]
    else:
        pole_fmt['color'] = color
        zero_fmt['color'] = color

    ax = plt.gca()
    ax.grid(True)

    # Poles and zeros
    ax.plot(p.real, p.imag, linestyle='None', **pole_fmt)
    if len(z) > 0:
        ax.plot(z.real, z.imag, linestyle='None', **zero_fmt)

    # Unit circle and axes
    circle = np.exp(2j * np.pi * np.linspace(0, 1, 400))
    ax.plot(circle.real, circle.imag, color='black', linewidth=1)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.axvline(0, color='black', linewidth=0.7)

    # Keep enough room for the complete unit circle.
    # The extra horizontal space accommodates the legend.
    margin = 1.1
    if showlist:
        xlim = (-margin, 1.8)
    else:
        xlim = (-margin, margin)

    ax.set_xlim(xlim)
    ax.set_ylim(-margin, margin)
    ax.set_aspect('equal', adjustable='box')

    if showlist:
        markers = []
        descr = []

        # List one pole/zero from each conjugate pair
        for pi in p[p.imag >= 0]:
            markers.append(
                plt.Line2D((), (), linestyle='None', **pole_fmt)
            )
            if np.isclose(pi.imag, 0, atol=1e-5):
                descr.append('%+.4f' % pi.real)
            else:
                descr.append('%+.4f+/-j%.4f' % (pi.real, pi.imag))

        for zi in z[z.imag >= 0]:
            markers.append(
                plt.Line2D((), (), linestyle='None', **zero_fmt)
            )
            if np.isclose(zi.imag, 0, atol=1e-5):
                descr.append('%+.4f' % zi.real)
            else:
                descr.append('%+.4f+/-j%.4f' % (zi.real, zi.imag))

        ax.legend(
            markers,
            descr,
            title="Poles (x) and zeros (o)",
            ncol=1,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            handlelength=0.55,
            prop={'size': 10}
        )

    ax.set_ylabel('Imag')
    ax.set_xlabel('Real')
