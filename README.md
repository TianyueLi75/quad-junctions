Surface mesh with quadrilateral elements for Boundary Integral Equations solve to the Laplace and Stokes equations, coupled with CSBQ for slender connecting tubes to form large networks.

Current main geometries: (in include/quad_junctions/)
- Cilia: a collar mesh base on a plane with a cylinder standing normal to the plane. geometry is closed by a butterfly-mesh cap and smoothed by an arc to form watertight and well-behaved mesh. The collar can be mapped to non-flat surfaces to "mount" the cilia to the surface. (plane_cilia_hybrid_geom.hpp, stud_sphere_hybrid_geom.hpp)
- Y-bifurcation junction in equal angles:  junction base, quad- or csbq-arms, butterfly caps. (ybifurc_hybrid_geom.hpp)

These building blocks are connected to form more complex examples: src/cilia_carpet-bie.cpp for 2-periodic flow past top and bottom plates with cilia, and src/bifurc-vessels-flow-bie.cpp for large bifurcating channel (vessel).

The Dirichlet BIE solve happens in include/quad_junctions/hybrid_bie_tests.hpp, the periodic BIO are directly in cilia_carpet-bie.cpp.

Makefile defaults to no MPI or PVFMM, so make MPI=1 for MPI, and PVFMM=1 for pvfmm (and MPI).

Current problem with PVFMM: 
The 20-junction vessel geometry is currently incompatible with PVFMM. Running

```bash
export QJ_VESSELS_TAPER=0.80 PVFMM_DIR=$PWD/pvfmm-precomp   
for ord in 8 12; do                       # order nref fourier digits geom ncopy layout pitch gscale Ns
  /bin-pv/probe-fmm-vs-direct $ord 1 24 8 0 8 0 8 1 3
done
```

will show the difference between EvalDirect and EvalPVFMM functions on the same fmm tree being magnitudes different when using order 12 (>1e6 points) and the other paramters given or hard-coded, whereas in order 8 mesh (~6e5 points) they match. I've done the following checks:
- The single junction with caps has 240 patches, at order 16 it has 6e4 points, but works well with pvfmm and its evaluation tests match the non-fmm results
- The whole geometry in order 12 (20 junctions+2caps ~ 7e5, csbq ~ 3e5) using non-fmm evaluation satisfies the constant density DL test and the on-surface Green's Identity test to 1e-7, saturated by quadrature tolerance. 
- The cilia geometry runs well and self converges in a periodic BVP, and it totals a similar number of discretization points and higher.