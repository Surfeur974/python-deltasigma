# -*- coding: utf-8 -*-
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
def _finite_xy(x,y):
    try: return np.isfinite(float(x)) and np.isfinite(float(y))
    except (TypeError,ValueError): return False
def figureMagic(xRange=None, dx=None, xLab=None, yRange=None, dy=None, yLab=None, size=None, name=None):
    fig=plt.gcf(); ax=plt.gca(); max_w,max_h,max_dpi,max_px=16.,10.,100.,2000.
    dpi=float(fig.get_dpi())
    if not np.isfinite(dpi) or dpi<=0: dpi=100.
    dpi=min(dpi,max_dpi); fig.set_dpi(dpi)
    if size is not None:
        size=np.asarray(size,dtype=float).ravel()
        if size.size!=2 or not np.all(np.isfinite(size)) or np.any(size<=0):
            warn("figureMagic(): size invalide, valeur par defaut (12, 6).")
            size=np.array([12.,6.])
        fig.set_size_inches(min(float(size[0]),max_w,max_px/dpi), min(float(size[1]),max_h,max_px/dpi), forward=True)
    for txt in list(ax.texts):
        x,y=txt.get_position()
        if not _finite_xy(x,y): txt.remove()
    if xRange is not None:
        xR=np.asarray(xRange,dtype=float).ravel()
        if xR.size==2 and np.all(np.isfinite(xR)) and xR[0]!=xR[1]: ax.set_xlim(float(xR[0]),float(xR[1]))
        else: warn(f"figureMagic(): xRange ignore ({xRange}).")
    if yRange is not None:
        yR=np.asarray(yRange,dtype=float).ravel()
        if yR.size==2 and np.all(np.isfinite(yR)) and yR[0]!=yR[1]: ax.set_ylim(float(yR[0]),float(yR[1]))
        else: warn(f"figureMagic(): yRange ignore ({yRange}).")
    if xRange is not None or yRange is not None: ax.set_autoscale_on(False); ax.set_aspect("auto", adjustable="box")
    if dx is not None:
        dx=float(dx)
        if np.isfinite(dx) and dx>0:
            x1,x2=ax.get_xlim(); step=dx if x2>=x1 else -dx; ticks=np.arange(x1,x2+0.5*step,step,dtype=float)
            if 0<ticks.size<=1000: ax.set_xticks(ticks)
    if dy is not None:
        dy=float(dy)
        if np.isfinite(dy) and dy>0:
            y1,y2=ax.get_ylim(); step=dy if y2>=y1 else -dy; ticks=np.arange(y1,y2+0.5*step,step,dtype=float)
            if 0<ticks.size<=1000: ax.set_yticks(ticks)
    if isinstance(xLab,str): ax.set_xlabel(xLab)
    if isinstance(yLab,str): ax.set_ylabel(yLab)
    if name is not None: ax.set_title(str(name), fontsize=14)
    ax.grid(True); w,h=fig.get_size_inches()
    if not np.all(np.isfinite([w,h])) or w<=0 or h<=0: fig.set_dpi(100.); fig.set_size_inches(12.,6.,forward=True); w,h=12.,6.
    px_w,px_h=w*fig.get_dpi(),h*fig.get_dpi()
    if px_w>max_px or px_h>max_px:
        scale=min(max_px/px_w,max_px/px_h); fig.set_size_inches(w*scale,h*scale,forward=True)
    return None
