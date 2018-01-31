#!/bin/bash

for f in $(squeue -u $USER | tail -n +2 | tr -s ' ' | cut -f2 -d' '); do echo $f; cat slurm-${f}.out; done
