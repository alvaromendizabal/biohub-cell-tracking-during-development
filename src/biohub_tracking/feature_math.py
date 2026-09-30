"""Pure image/geometry helpers. No data download, fitting, or ground-truth access.

Coordinates are Z,Y,X in micrometers. Intensity moments and threshold components
are patch descriptors, not cell segmentations. Missing evidence is NaN, not zero.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy import ndimage

EPS = 1e-10


def ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and abs(b) > EPS else float('nan')


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return ratio(float(np.dot(a.ravel(), b.ravel())), float(np.linalg.norm(a)*np.linalg.norm(b)))


def ncc(a, b) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape or a.size < 2:
        return float('nan')
    return cosine(a-a.mean(), b-b.mean())


def logratio(a: float, b: float) -> float:
    return float(np.log(a/b)) if np.isfinite(a) and np.isfinite(b) and a > EPS and b > EPS else float('nan')


def normalize(image):
    image = np.asarray(image)
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError('Image must be finite Z,Y,X data.')
    lo, hi = np.quantile(image, [.01, .998])
    normal = np.zeros(image.shape, np.float32) if hi <= lo else np.clip((image.astype(np.float32)-lo)/(hi-lo), 0, 1)
    return normal.astype(np.float32), {'low':float(lo), 'high':float(hi), 'flat': bool(hi<=lo)}


@dataclass
class Patch:
    raw: np.ndarray
    normal: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    center: tuple
    slices: tuple
    background: float
    noise: float
    spacing: np.ndarray


def patch(raw, normal, center, spacing, radius_um=6.) -> Patch:
    raw, normal = np.asarray(raw), np.asarray(normal)
    spacing = np.asarray(spacing, float)
    center = np.asarray(center)
    if raw.shape != normal.shape or raw.ndim != 3 or spacing.shape != (3,) or np.any(spacing<=0) or not np.isfinite(spacing).all():
        raise ValueError('Invalid image shape or physical spacing.')
    if center.shape != (3,) or not np.isfinite(center).all() or np.any(center != np.rint(center)):
        raise ValueError('Candidate must have integral Z,Y,X coordinates.')
    center = center.astype(int); radii = np.ceil(radius_um/spacing).astype(int)
    if np.any(center-radii<0) or np.any(center+radii>=np.array(raw.shape)):
        raise ValueError('Candidate patch crosses the retained image boundary.')
    sl=tuple(slice(int(c-r),int(c+r+1)) for c,r in zip(center,radii))
    xyz=np.stack(np.meshgrid(*[np.arange(-r,r+1)*s for r,s in zip(radii,spacing)],indexing='ij'),axis=-1)
    rad=np.linalg.norm(xyz,axis=-1)
    pixels=np.asarray(raw[sl],float); shell=pixels[(rad>=4.5)&(rad<=6.)]
    bg=float(np.median(shell)); noise=float(1.4826*np.median(abs(shell-bg)))
    return Patch(pixels,np.asarray(normal[sl],float),xyz,rad,tuple(radii),sl,bg,noise,spacing.copy())


def moments(p: Patch, radius_um=5.):
    mask=p.radius<=radius_um; points=p.xyz[mask]; w=np.maximum(p.raw[mask]-p.background,0.)
    mass=float(w.sum())
    if mass<=EPS:
        return mass,np.full(3,np.nan),np.full((3,3),np.nan),np.full(3,np.nan),np.full(3,np.nan)
    centroid=(points*w[:,None]).sum(axis=0)/mass; delta=points-centroid
    covariance=(delta.T*w)@delta/mass
    vals,vecs=np.linalg.eigh(covariance); vals=np.maximum(vals,0.)
    return mass,centroid,covariance,vals,vecs[:,-1]


def quantized_glcm(pixels, shift, levels=8):
    """Symmetric co-occurrence statistics; quantization fixed to normalized [0,1]."""
    a=np.asarray(pixels,float)
    shift=tuple(int(v) for v in shift)
    if len(shift)!=3 or all(v==0 for v in shift) or any(v<0 for v in shift):
        raise ValueError('Positive nonzero three-axis offset required.')
    if any(s>=n for s,n in zip(shift,a.shape)):
        return [float('nan')]*4
    first=tuple(slice(0,n-s) for n,s in zip(a.shape,shift))
    second=tuple(slice(s,n) for n,s in zip(a.shape,shift))
    bins=np.minimum((np.clip(a,0,1)*levels).astype(int),levels-1)
    x=bins[first].ravel(); y=bins[second].ravel()
    counts=np.bincount(x*levels+y,minlength=levels*levels).reshape(levels,levels).astype(float)
    counts+=counts.T.copy(); prob=counts/counts.sum()
    delta=np.arange(levels)[:,None]-np.arange(levels)[None,:]
    nz=prob[prob>0]
    return [float((prob*delta**2).sum()),float((prob/(1+delta**2)).sum()),float((prob**2).sum()),float(-(nz*np.log(nz)).sum())]


def histogram_similarity(a,b):
    ha=np.histogram(a,bins=16,range=(0.,1.))[0].astype(float)
    hb=np.histogram(b,bins=16,range=(0.,1.))[0].astype(float)
    if ha.sum()==0 or hb.sum()==0: return float('nan'),float('nan')
    ha/=ha.sum();hb/=hb.sum();m=(ha+hb)/2
    def kl(p):
        nz=p>0
        return float(np.sum(p[nz]*np.log(p[nz]/m[nz])))
    return .5*(kl(ha)+kl(hb)),float(np.minimum(ha,hb).sum())


def derivatives(normal, spacing, sigma_um):
    """Return physical, scale-normalized first/second derivatives, Z,Y,X."""
    spacing=np.asarray(spacing,float); sigma=float(sigma_um)/spacing
    gradient=[]; hessian={}
    for a in range(3):
        order=[0,0,0];order[a]=1
        gradient.append(ndimage.gaussian_filter(normal,sigma,order=order,mode='reflect')*(sigma_um/spacing[a]))
        for b in range(a,3):
            order=[0,0,0];order[a]+=1;order[b]+=1
            hessian[(a,b)]=ndimage.gaussian_filter(normal,sigma,order=order,mode='reflect')*(sigma_um**2/(spacing[a]*spacing[b]))
    return gradient,hessian


def sample_scale_features(normal, spacing, coords, scales=(1.5,2.5,3.5,5.)):
    values=[{} for _ in coords]
    for scale in scales:
        gradients,h=derivatives(normal,spacing,scale)
        for i,c in enumerate(coords):
            idx=tuple(int(x) for x in c)
            matrix=np.array([[h[(min(a,b),max(a,b))][idx] for b in range(3)] for a in range(3)])
            eig=np.linalg.eigvalsh(matrix); tag='s'+str(float(scale)).replace('.','p')
            for name,value in [('log',-np.trace(matrix)),('gradient',math.sqrt(sum(float(g[idx])**2 for g in gradients))),('hessian_min',eig[0]),('hessian_max',eig[-1])]:
                values[i][tag+'_'+name]=float(value)
    return values


def points_anisotropy(points):
    points=np.asarray(points,float)
    if len(points)<2: return float('nan')
    c=points-points.mean(axis=0); vals=np.maximum(np.linalg.eigvalsh(c.T@c/len(c)),0.)
    return ratio(vals[-1]-vals[0], vals.sum())


def finite_or_nan(values):
    """Do not silently substitute zeros for undefined image/geometry descriptors."""
    return {k:float(v) if np.isfinite(v) else float('nan') for k,v in values.items()}
