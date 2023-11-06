import os

run = 30

for i in range(run):
    os.system(f"sbatch nesi_one_job.sl {i}")
