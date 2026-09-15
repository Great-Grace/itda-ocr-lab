import os, runpy
os.environ['BATCHED_LIMIT']='16'
runpy.run_path('/content/colab_batched_cpu_baseline_exec.py', run_name='__main__')
