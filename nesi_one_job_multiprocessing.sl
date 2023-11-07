#!/bin/bash

#SBATCH --job-name=single_gp_rac50    # job name (shows up in the queue)
#SBATCH --account=vuw03986
#SBATCH --time=24:00:00         # Walltime (HH:MM:SS)  7d+4hr
#SBATCH --mem=1500MB                     # Memory in MB
#SBATCH --cpus-per-task=70          # Will request 16 logical CPUs per task.
#SBATCH --output=outputs/%x_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=2267670958@qq.com

module load Python/3.9.5-gimkl-2020a

python3 solve_multiprocessing.py -r $1 -s $2