#!/bin/bash

#SBATCH --job-name=single_gp_rac    # job name (shows up in the queue)
#SBATCH --account=vuw03986
#SBATCH --time=24:00:00         # Walltime (HH:MM:SS)  7d+4hr
#SBATCH --mem=5G                     # Memory in MB
#SBATCH --cpus-per-task=50          # Will request 16 logical CPUs per task.
#SBATCH --mail-type=BEIGN,END,FAIL
#SBATCH --mail-user=2267670958@qq.com

module load Python/3.9.5-gimkl-2020a

python3 solve_multiprocessing.py -s $1