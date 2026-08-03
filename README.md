Surface mesh with quadrilateral elements for Boundary Integral Equations solve to the Laplace and Stokes equations, coupled with CSBQ for slender connecting tubes to form large networks.

Current main geometries: (in include/quad_junctions/)
- Cilia: a collar mesh base on a plane with a cylinder standing normal to the plane. geometry is closed by a butterfly-mesh cap and smoothed by an arc to form watertight and well-behaved mesh. The collar can be mapped to non-flat surfaces to "mount" the cilia to the surface. (plane_cilia_hybrid_geom.hpp, stud_sphere_hybrid_geom.hpp)
- Y-bifurcation junction in equal angles:  junction base, quad- or csbq-arms, butterfly caps. (ybifurc_hybrid_geom.hpp)

These building blocks are connected to form more complex examples: src/cilia_carpet-bie.cpp for 2-periodic flow past top and bottom plates with cilia, and src/bifurc-vessels-flow-bie.cpp for large bifurcating channel (vessel).

The Dirichlet BIE solve happens in include/quad_junctions/hybrid_bie_tests.hpp, the periodic BIO are directly in cilia_carpet-bie.cpp.

Makefile defaults to no MPI or PVFMM, so make MPI=1 for MPI, and PVFMM=1 for pvfmm (and MPI).
