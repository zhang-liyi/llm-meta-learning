#
# In datasets/custom_clbench.py
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

train_filelist = [
    'domain-finance',
    'domain-healthcare',
    'domain-humanities',
    'domain-legal_advisory',
    'domain-lifestyle',
    'domain-management',
    'domain-science',
    'empirical-experimental_data',
    'empirical-observational_data',
    'empirical-simulation_environment',
    'procedure-instructional_procedures',
    'procedure-operational_procedures',
    'procedure-workflow_orchestration',
    'rule-game_mechanics',
    'rule-legal_regulatory',
    'rule-mathematical_formalism',
    'rule-programming_syntax',
    'rule-technical_standards',
]
test_filelist = [
    'domain-finance',
    'domain-healthcare',
    'domain-humanities',
    'domain-legal_advisory',
    'domain-lifestyle',
    'domain-management',
    'domain-science',
    'empirical-experimental_data',
    'empirical-observational_data',
    'empirical-simulation_environment',
    'procedure-instructional_procedures',
    'procedure-operational_procedures',
    'procedure-workflow_orchestration',
    'rule-game_mechanics',
    'rule-legal_regulatory',
    'rule-mathematical_formalism',
    'rule-programming_syntax',
    'rule-technical_standards',
]

def mc(tokenizer: ModelTokenizer, packed: bool = True):
    """
    CL-Bench dataset for context learning evaluation.
    """

    ds_list_train, ds_list_dev, ds_list_test = [], [], []

    for f in train_filelist:
        ds_train = SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            filter_fn=None,
            split="train",
            data_files=f'../data/{f}_inputoutput_train.json'
        )
        ds_list_train.append(ds_train)

    for f in test_filelist:
        ds_dev = SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            filter_fn=None,
            split="train",
            data_files=f'../data/{f}_inputoutput_dev.json'
        )
        ds_list_dev.append(ds_dev)
        ds_test = SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            filter_fn=None,
            split="train",
            data_files=f'../data/{f}_inputoutput_test.json'
        )
        ds_list_test.append(ds_test)

    ds = [ds_list_train, ds_list_dev, ds_list_test]

    if packed:
        return
    else:
        return ds
