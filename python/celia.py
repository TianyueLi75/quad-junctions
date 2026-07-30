"""
Local surface as ONE continuous B-spline patch: sphere collar -> C^inf foot
blend -> cilium shaft (inward). Blend is interior to the patch, so smoothness
is preserved (no split seam). Requires: numpy, gmsh
"""
import numpy as np, gmsh

R, L, r0, tau, beta = 1.0, 0.25, 0.0125, 0.25, 2.0
s_b, s_max          = 0.15, 0.1
collar_ang          = 0.01          # sphere collar around the foot (rad)
N_U, N_TH           = 90, 48        # u runs collar->blend->shaft as ONE grid

def psi(x):
    x=np.asarray(x,float); o=np.zeros_like(x); m=x>1e-12
    o[m]=np.exp(-1/x[m]); return o
def Phi(u):
    u=np.asarray(u,float); a,b=psi(u),psi(1-u); d=a+b
    o=np.divide(a,d,out=np.zeros_like(a),where=d>0)
    o[u>=1]=1; o[u<=0]=0; return o
def r_of_s(s): return r0*(1+(beta-1)*(1-Phi(s/s_b)))*(1-tau*np.asarray(s,float))

p0=np.array([0.,0.,R]); n0=p0/R; axis=-n0
Nv,Bv=np.array([1.,0,0]),np.array([0.,1,0])
theta=np.linspace(0,2*np.pi,N_TH,endpoint=False)

# ---- ONE surface function over u in [-collar_extent, s_max] ----
# u<0 : pure sphere collar (points ON the sphere, no cilium)
# u in [0, s_b] : C^inf foot blend (flare lifts off the wall)
# u > s_b : cilium shaft
# The blend is driven by Phi so wall->sphere contact is infinite-order.
u_collar = np.linspace(-collar_ang, 0.0, N_U//3, endpoint=False)
u_cil    = np.linspace(0.0, s_max, N_U - N_U//3)
u_vals   = np.concatenate([u_collar, u_cil])

def surface_ring(u):
    ct, st = np.cos(theta), np.sin(theta)
    if u < 0.0:
        # pure sphere collar: latitude angle = foot angle + |u|
        # (radius on sphere grows as we move away from the axis)
        lat = _th_foot + (-u)
        z, rho = R*np.cos(lat), R*np.sin(lat)
        return np.stack([rho*ct, rho*st, np.full(N_TH, z)], axis=1)
    else:
        # cilium wall with C^inf foot term; as u->0+, Phi-blend seats it
        # ONTO the sphere to infinite order, matching the collar smoothly
        r = float(r_of_s(u))
        blend = Phi(1.0 - u/s_b) if u < s_b else 0.0   # 1 at foot, 0 past blend
        # seat height: interpolate C^inf from sphere surface to shaft axis
        C = p0 + L*u*axis + (R - np.sqrt(max(R**2 - r**2,0)))*blend*n0
        return C[None,:] + r*(ct[:,None]*Nv + st[:,None]*Bv)

# foot polar angle so collar and blend meet consistently
_r_foot = float(r_of_s(0.0))
_th_foot = np.arcsin(np.clip(_r_foot/R, -1, 1))

# grid = np.array([surface_ring(u) for u in u_vals])   # (N_U, N_TH, 3)
# base collar plane: through the foot point p0, normal = sphere normal n0.
# Cilium protrudes inward (-n0), so rings must stay on the -n0 side, short of it.
q      = p0.copy()          # foot point on the sphere = (0,0,R)
m      = n0.copy()          # collar-plane normal = outward sphere normal
margin = 0.01 * s_max * L         # stop this far short of the collar plane

def ring_clears_plane(u):
    ring = surface_ring(u)                 # (N_TH, 3)
    signed = (ring - q) @ m                # +ve = outward past collar, -ve = inward
    return np.all(signed < -margin)        # whole ring strictly inside, short of plane

u_keep = np.array([u for u in u_vals if ring_clears_plane(u)])
grid   = np.array([surface_ring(u) for u in u_keep])

# ---- fit ONE closed-in-theta B-spline surface ----
gmsh.initialize(); gmsh.model.add("local_patch"); occ=gmsh.model.occ
nu,nv,_=grid.shape
g=np.concatenate([grid, grid[:,:1,:]],axis=1); nv1=nv+1
pts=[occ.addPoint(*g[i,j]) for i in range(nu) for j in range(nv1)]
face=occ.addBSplineSurface(pts, nv1, degreeU=3, degreeV=3)
occ.synchronize()
gmsh.write("local_patch.brep")
print("wrote local_patch.brep — single patch, blend interior")
gmsh.finalize()
