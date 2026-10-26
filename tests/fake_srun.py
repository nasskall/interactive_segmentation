"""Stands in for deploy/slurm_train.sh in tests: runs the worker as a
'Slurm job' 4242."""
import os
import subprocess
import sys

print('fake srun:', os.environ.get('ISEG_JOB_NAME'), '|', os.environ.get('ISEG_JOB_COMMENT'),
      flush=True)
sys.exit(subprocess.call(sys.argv[1:], env={**os.environ, 'SLURM_JOB_ID': '4242'}))
