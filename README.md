Surface mesh with quadrilateral elements for Boundary Integral Equations solve to the Laplace and Stokes equations, coupled with CSBQ for slender connecting tubes to form large networks.

Current main geometries: (in include/quad_junctions/)
- Cilia: a collar mesh base on a plane with a cylinder standing normal to the plane. geometry is closed by a butterfly-mesh cap and smoothed by an arc to form watertight and well-behaved mesh. The collar can be mapped to non-flat surfaces (via a Mount functor) to "mount" the cilia to the surface. (plane_cilia_hybrid_geom.hpp; shared collar/cap/mount primitives in collar_mount_geom.hpp)
- Y-bifurcation junction in equal angles:  junction base, quad- or csbq-arms, butterfly caps. (ybifurc_hybrid_geom.hpp)

These building blocks are connected to form more complex examples: src/cilia_carpet-bie.cpp for 2-periodic flow past top and bottom plates with cilia, and src/bifurc-vessels-flow-bie.cpp for large bifurcating channel (vessel).

## Generating a generalized N-arm bifurcation

To build and verify a bifurcation mesh with **arbitrary branch angles, an arbitrary (including even) number of branches, and non-coplanar 3D arms** (`gaps:`/`dirs:` specs and presets like `y120`, `cross4`, `tetra4`), **read the playbook [bifurcation_meshing.md](bifurcation_meshing.md)**. It is the step-by-step command guide (for a human or an agent) covering the three drivers — `bin/bifurc-general-{geom,bie,load-bie}` — the runtime spec/resolution knobs, the watertightness + DL/Green identity acceptance tests, the Hybrid-vs-Duffy junction scheme (`SCTL_SELF_SCHEME`), and exporting the validated geometry as a reusable bundle. Start there before touching `src/bifurc-general-*.cpp` or `include/quad_junctions/gen_junction_geom.hpp`.

The Dirichlet BIE solve happens in include/quad_junctions/hybrid_bie_tests.hpp, the periodic BIO are directly in cilia_carpet-bie.cpp.

Makefile defaults to no MPI or PVFMM, so make MPI=1 for MPI, and PVFMM=1 for pvfmm (and MPI).
