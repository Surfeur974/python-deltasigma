# -*- coding: utf-8 -*-
import numpy as np
def peakSNR(snr, amp):
    snr=np.asarray(snr,dtype=float).ravel(); amp=np.asarray(amp,dtype=float).ravel()
    ok=np.isfinite(snr)&np.isfinite(amp); snr=snr[ok]; amp=amp[ok]
    if snr.size==0: return np.nan, np.nan
    o=np.argsort(amp); amp=amp[o]; snr=snr[o]
    i=int(np.argmax(snr)); peak_snr=float(snr[i]); peak_amp=float(amp[i])
    lo=max(i-1,0); hi=min(i+1,snr.size-1)
    if hi-lo>=2:
        xs=amp[lo:hi+1]; ys=snr[lo:hi+1]
        try:
            a,b,_=np.polyfit(xs,ys,2)
            if np.isfinite(a) and a<0:
                xv=-b/(2*a); yv=float(np.polyval([a,b,_],xv))
                if np.isfinite(xv) and np.isfinite(yv) and xs.min()<=xv<=xs.max() and yv>=peak_snr:
                    peak_amp=float(xv); peak_snr=float(yv)
        except (np.linalg.LinAlgError,ValueError,TypeError): pass
    return peak_snr, peak_amp
