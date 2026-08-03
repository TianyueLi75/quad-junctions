#!/bin/bash
# Run a ParaView batch rendering script headlessly under a virtual X server.
module purge >/dev/null 2>&1
module load modules/2.3-20240529 paraview/5.10.1 >/dev/null 2>&1
exec xvfb-run -a -s "-screen 0 1600x1200x24" pvbatch "$@"
