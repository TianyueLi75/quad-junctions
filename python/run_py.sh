#!/bin/bash
# Run a python script with the numpy-capable python module.
module purge >/dev/null 2>&1
module load python/3.11.11 >/dev/null 2>&1
exec python3 "$@"
