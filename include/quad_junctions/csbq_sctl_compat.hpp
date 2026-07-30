#ifndef QUAD_JUNCTIONS_CSBQ_SCTL_COMPAT_HPP_
#define QUAD_JUNCTIONS_CSBQ_SCTL_COMPAT_HPP_

/**
 * CSBQ-only shims for the SCTL version we build against.
 *
 * This project takes SCTL from PVFMM's submodule (upstream `7201e9a`), which is newer
 * than the SCTL that CSBQ pins (`2f1082e`). One symbol CSBQ still relies on was removed
 * from SCTL in between, so we supply it here rather than patching either dependency:
 *
 *   SCTL_QUOTEME -- upstream deleted it from `sctl/common.hpp`, but CSBQ's
 *   `slender_element.cpp` uses `SCTL_QUOTEME(SCTL_DATA_PATH)` in four places (lines 747,
 *   768, 794, 1580) to stringize the quadrature-table path handed in by -D.
 *
 * The Makefile force-includes this file into every translation unit (`-include`), so the
 * macros are in scope before <csbq.hpp> is reached no matter what order a driver includes
 * its headers in. Everything is #ifndef-guarded, so this stays inert if a future SCTL
 * reinstates the macro or a future CSBQ stops needing it -- at which point this file, and
 * the `-include` line in the Makefile, can simply be deleted.
 */

#ifndef SCTL_QUOTEME
#define SCTL_QUOTEME(x) SCTL_QUOTEME_1(x)
#define SCTL_QUOTEME_1(x) #x
#endif

#endif // QUAD_JUNCTIONS_CSBQ_SCTL_COMPAT_HPP_
