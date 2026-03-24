The code is modified from the torchtune open-source library for LLM fine-tuning and post-training.

Requirements before running the code:
* Clone or download this repo 
* Download either Llama3-8B or Qwen2-7B
* Inside any config file you will be using (those .yaml files inside the config_files/ folder), search for the MODEL_PATH variable, and set them to your local LLM folder path.

Example way to run the Bayesian Meta-Learning method:

1. With Command-Line, navigate to `torchtune-bayes/` folder
2. Type the following to use the llama3-8b-seed1 config (Llama3-8B) (the qwen .yaml file corresponds to running the same script with Qwen2-7B)

`PYTHONPATH=${pwd}:PYTHONPATH tune run custom_lora.py --config config_files/llama3_8b_seed1.yaml`

Example way to run the Reptile Meta-Learning method:

1. With Command-Line, navigate to `torchtune-reptile/` folder
2. Type the following to use the llama3-8b-seed1 config (Llama3-8B) (the qwen .yaml file corresponds to running the same script with Qwen2-7B)

`PYTHONPATH=${pwd}:PYTHONPATH tune run custom_lora_reptile.py --config config_files/llama3_8b_seed1.yaml`



