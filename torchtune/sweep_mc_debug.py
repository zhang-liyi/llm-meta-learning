import subprocess
import time
import json
import datetime

tm = str(datetime.datetime.now())
TMSTR = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]

def write_run(i, extra=''):
    with open('temp.sh', 'w') as f:
        f.write(f"#!/bin/bash\n"
                "#SBATCH --job-name={0}\n"
                "#SBATCH --time=0:59:59\n"
                "#SBATCH --gres=gpu:1\n"
                # "#SBATCH --partition=pli\n"
                # "#SBATCH --account=bayesllm\n"
                "#SBATCH --constraint=gpu40\n"
                "#SBATCH --nodes=1\n"
                "#SBATCH --ntasks=1\n"
                "#SBATCH --cpus-per-task=1\n"
                "#SBATCH --mem-per-cpu=50G\n"
                "#SBATCH --mail-type=begin\n"
                "#SBATCH --mail-type=end\n"
                "#SBATCH --mail-user=zhangliyi97@gmail.com\n".format(i))

        cmd = "PYTHONPATH=${pwd}:PYTHONPATH tune run custom_lora.py --config config_files/llama3_8b_seed"
        cmd += f"{i}.yaml "
        cat = " >../results/" + TMSTR + "_" + i + ".out"
        f.write(cmd+extra+cat+'\n')

    subprocess.call('chmod +x temp.sh', shell=True)
    time.sleep(0.1)
    subprocess.call('sbatch temp.sh', shell=True)

for i in [1]:
    # save = '_'.join( [x.strip().split(' ')[-1] for x in ar.split('--') if len(x.strip()) > 0] )
    print(i)
    write_run(str(i))
