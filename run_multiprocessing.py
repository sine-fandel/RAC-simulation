import os

run = 30

for i in range(run):
    os.system(f"sbatch nesi_one_job_multiprocessing.sl {i} {i + i * 199}")
