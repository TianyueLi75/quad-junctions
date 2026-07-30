# Build the S_init benchmark drivers against the locally-built fmm3dbie static library.
# The fmm3dbie static lib already absorbs the FMM3D objects (ar -x), so we only add BLAS/LAPACK
# (MKL, matching the project toolchain), OpenMP, and the gfortran runtime.
#
#   module load gcc/13.3.0 (or: . ../../sctl_source)   # provides gfortran + MKLROOT
#   make
#
# Run single-threaded (paper-comparable Table 1c) with:  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./s_init_sweep

FC = gfortran
FFLAGS = -fPIC -O3 -funroll-loops -march=native -fopenmp -std=legacy

# fmm3dbie static lib (prefer the in-tree lib-static build; falls back to the install prefix)
FMMBIE_LIB := $(firstword $(wildcard \
    ../../extern/fmm3dbie/lib-static/libfmm3dbie.a \
    ../../extern/fmm3dbie-install/lib/libfmm3dbie.a \
    ../../extern/fmm3dbie-install/libfmm3dbie.a))

MKL = -L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
LIBS = $(FMMBIE_LIB) $(MKL) -lgomp -lgfortran

all: dump_rvnodes s_init_sweep stell_sinit bie_total_setup
	@echo "linked against: $(FMMBIE_LIB)"

# Unified Laplace/Stokes setup-time driver (kernel chosen at runtime); see scripts/setup_speed_sweep.sh
bie_total_setup: bie_total_setup.f90
	$(FC) $(FFLAGS) -o $@ $< $(LIBS)

dump_rvnodes: dump_rvnodes.f90
	$(FC) $(FFLAGS) -o $@ $< $(LIBS)

s_init_sweep: s_init_sweep.f90
	$(FC) $(FFLAGS) -o $@ $< $(LIBS)

stell_sinit: stell_sinit.f90
	$(FC) $(FFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f dump_rvnodes s_init_sweep stell_sinit bie_total_setup *.o *.mod
