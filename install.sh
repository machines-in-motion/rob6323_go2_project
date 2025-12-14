#!/usr/bin/env bash
mkdir logs
sbatch --job-name='rob6323_haizhou' --mail-user='hz3862@nyu.edu' install.slurm
cd /scratch/$USER
git clone https://github.com/isaac-sim/IsaacLab.git
