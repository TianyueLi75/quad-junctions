# quad-junctions: quad-element junction geometries hybridized with CSBQ slender bodies.
# Structured on CSBQ's example Makefile, with the numerical-library flags the
# QuadElemList BIE code relies on (MKL BLAS/LAPACK, FFTW, quad precision).
#
# SCTL comes from PVFMM's own SCTL submodule (upstream, currently 7201e9a) -- there is
# exactly ONE sctl in the program, and it is the one libpvfmm.a was compiled against.
# The SCTL *fork* is no longer a dependency: its only local content was
# include/sctl/experimental/, which is now vendored into this repo at
# include/sctl/experimental/ (see the INCLUDES note below). CSBQ is header-only and is
# built against that same SCTL -- do NOT init CSBQ's nested SCTL submodule.

# PVFMM (extern/pvfmm -> a dmalhotra/pvfmm `develop` checkout, built in-tree with
# autotools, so the static lib lands in lib/.libs). Unlike SCTL/CSBQ this one is
# NOT header-only. Its SCTL submodule (extern/pvfmm/SCTL) is a plain checkout of
# upstream SCTL and is THE sctl for this project, hence SCTL_INCLUDE_DIR below.
# Note this makes extern/pvfmm a hard dependency of every build, PVFMM=1 or not,
# because that is where the sctl headers live.
# NB: called PVFMM_ROOT, not PVFMM_DIR -- pvfmm reads $PVFMM_DIR at *runtime* to
# locate its precomputed-data files, and an environment variable of that name
# would otherwise silently become this make variable.
PVFMM_ROOT    ?= ./extern/pvfmm
PVFMM_INC_DIR ?= $(PVFMM_ROOT)/include
PVFMM_LIB_DIR ?= $(PVFMM_ROOT)/lib/.libs

SCTL_INCLUDE_DIR ?= $(PVFMM_ROOT)/SCTL/include
CSBQ_INCLUDE_DIR ?= ./extern/CSBQ/include
SCTL_DATA_PATH   ?= ./data

CXX = g++ # requires g++-9 or newer
CXXFLAGS = -std=c++17 -fopenmp -Wall -Wfloat-conversion

# Optional debug/release
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -O0 -fsanitize=address,leak,undefined,pointer-compare,pointer-subtract,float-divide-by-zero,float-cast-overflow -fno-sanitize-recover=all -fstack-protector
	CXXFLAGS += -DSCTL_MEMDEBUG
else
	CXXFLAGS += -O3 -march=native -DNDEBUG
endif

OS = $(shell uname -s)
ifeq "$(OS)" "Darwin"
	CXXFLAGS += -g -rdynamic -Wl,-no_pie
else
	CXXFLAGS += -gdwarf-4 -g -rdynamic
	CXXFLAGS += -ldl
	CXXFLAGS += -mno-avx512fp16 # pre-2.38 binutils can't decode AVX-512-FP16 from -march=native
endif

# Opt-in PVFMM build. `make PVFMM=1 ...` defines SCTL_HAVE_PVFMM so sctl::ParticleFMM
# -- the far field of every BoundaryIntegralOp -- dispatches to PVFMM's PtFMM instead
# of direct summation. Flags are applied after the MPI block below, since PVFMM is
# MPI-only and thus implies MPI=1.
PVFMM ?= 0

# Opt-in distributed-memory build. `make MPI=1 ...` compiles with the MPI wrapper
# compiler and defines SCTL_HAVE_MPI so sctl::Comm::World() is a real MPI communicator
# (QuadElemList / CSBQ SlenderElemList partition their elements across ranks). Default
# build stays serial (g++). As with DEBUG, do NOT pass CXXFLAGS+= on the command line --
# it overrides (not appends to) the flags assigned here.
MPI ?= 0
# PVFMM has no serial build, so PVFMM=1 forces MPI on -- `override` so it wins even
# against an explicit command-line MPI=0. This is now load-bearing rather than belt-and-
# braces: older SCTL's common.hpp used to #define SCTL_HAVE_MPI itself whenever
# SCTL_HAVE_PVFMM was set, but upstream 7201e9a deleted that whole PVFMM block (along
# with the #include of the configure-generated pvfmm_config.h, which pvfmm's develop
# branch no longer produces), so nothing defines SCTL_HAVE_MPI for us any more.
ifeq ($(PVFMM), 1)
	override MPI := 1
endif
ifeq ($(MPI), 1)
	CXX = mpicxx
	CXXFLAGS += -DSCTL_HAVE_MPI
endif

ifeq ($(PVFMM), 1)
	CXXFLAGS += -DSCTL_HAVE_PVFMM
	PVFMM_INCLUDES = -I$(PVFMM_INC_DIR)
	# Static lib (only pvfmm-wrapper.o; the FMM itself is header-only templates).
	# Goes in LDLIBS, which the link rule places before CXXFLAGS' -lmkl*/-lfftw*.
	LDLIBS += $(PVFMM_LIB_DIR)/libpvfmm.a
endif

CXXFLAGS += -DSCTL_GLOBAL_MEM_BUFF=0
CXXFLAGS += -DSCTL_PROFILE=25 -DSCTL_VERBOSE
CXXFLAGS += -DSCTL_SIG_HANDLER
CXXFLAGS += -DSCTL_QUAD_T=__float128

# Quadrature-table path (CSBQ precomputed tables, symlinked at ./data)
CXXFLAGS += -DSCTL_DATA_PATH=$(SCTL_DATA_PATH)

# CSBQ needs one macro (SCTL_QUOTEME) that the SCTL we build against has dropped. Rather
# than patch CSBQ or SCTL, force-include a shim into every TU so the macro is in scope
# before <csbq.hpp> however a driver orders its includes. See the header for details.
# Kept out of CXXFLAGS on purpose: CXXFLAGS is also passed to the link rule, where a
# -include of a C++ header would be pointless work.
COMPAT_INCLUDE = -include $(INCDIR)/quad_junctions/csbq_sctl_compat.hpp

# MKL BLAS/LAPACK (non-Intel compiler), matching the SCTL fork's Makefile
CXXFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK

# FFTW
CXXFLAGS += -lfftw3_omp -DSCTL_FFTW_THREADS
CXXFLAGS += -lfftw3 -DSCTL_HAVE_FFTW
CXXFLAGS += -lfftw3f -DSCTL_HAVE_FFTWF
CXXFLAGS += -lfftw3l -DSCTL_HAVE_FFTWL

RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./include

# ORDER IS LOAD-BEARING: -I$(INCDIR) must come before -I$(SCTL_INCLUDE_DIR). Upstream SCTL
# ships its own (much older, ~690-line) include/sctl/experimental/quad_element.{hpp,cpp},
# and this project's vendored include/sctl/experimental/ (~3000 lines, the developed
# version from the fork) has to shadow it. If the order is ever flipped, the build picks
# up upstream's ancestor version and fails loudly on missing QuadElemList members --
# noisy, but easy to misdiagnose, hence this comment.
INCLUDES = -I$(INCDIR) -I$(CSBQ_INCLUDE_DIR) -I$(SCTL_INCLUDE_DIR) $(PVFMM_INCLUDES)

TARGET_BIN = \
	$(BINDIR)/ybifurc-hybrid-bie \
	$(BINDIR)/ybifurc-multi-bie \
	$(BINDIR)/ybifurc-tree-bie \
	$(BINDIR)/ybifurc-channel-bie \
	$(BINDIR)/ybifurc-vessels-bie \
	$(BINDIR)/ybifurc-vessels-flow-bie \
	$(BINDIR)/ybifurc-flow-bie \
	$(BINDIR)/ybifurc-genmesh \
	$(BINDIR)/ybifurc-export-fmm3dbie \
	$(BINDIR)/ybifurc-bie-selfsetup \
	$(BINDIR)/stud_sphere-geom \
	$(BINDIR)/stud_sphere-bie \
	$(BINDIR)/stud_sphere-hybrid-bie \
	$(BINDIR)/cilia_carpet-bie \
	$(BINDIR)/periodic-sphere-bie \
	$(BINDIR)/probe-fmm-vs-direct

all: $(TARGET_BIN)

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX) $^ $(LDLIBS) -o $@ $(CXXFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(COMPAT_INCLUDE) -c $^ -o $@

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*

.PHONY: all clean
