Example way to run:

1. With Command-Line, navigate to `torchtune/` folder
2. Type the following to use the llama3-8b-seed1 config

`PYTHONPATH=${pwd}:PYTHONPATH tune run custom_lora.py --config config_files/llama3_8b_seed1.yaml`

Alternatively, can use `python sweep_mc.py` to do slurm batch submission on university cluster.
