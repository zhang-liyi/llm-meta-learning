#
# Custom dataset loader for domain-separated API-Bank (agentic tool use)
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

# Domain-based task lists for meta-learning
# These are populated after running classify_apibank_domains.py

train_filelist = [
    'healthcare_medical',
    'finance_banking',
    'shopping_ecommerce',
    'travel_booking',
    'productivity_calendar',
    'communication_messaging',
    'entertainment_media',
    'fitness_wellness',
    'food_dining',
    'smart_home_iot',
    'information_search',
    'utilities_tools',
]

test_filelist = [
    'healthcare_medical',
    'finance_banking',
    'shopping_ecommerce',
    'travel_booking',
    'productivity_calendar',
    'communication_messaging',
    'entertainment_media',
    'fitness_wellness',
    'smart_home_iot',
    'information_search',
    'utilities_tools',
]


def mc(tokenizer: ModelTokenizer, packed: bool = True, data_dir: str = '../data/apibank_domains'):
    """
    Custom dataset for API-Bank agentic tool use task.
    Domain-separated for meta-learning.

    Args:
        tokenizer: The model tokenizer
        packed: Whether to use packed dataset (not implemented)
        data_dir: Directory containing domain-separated JSON files
    """

    ds_list_train, ds_list_dev, ds_list_test = [], [], []

    for f in train_filelist:
        try:
            ds_train = SFTDataset(
                model_transform=tokenizer,
                source="json",
                message_transform=InputOutputToMessages(
                    column_map={"input": "input", "output": "output"},
                ),
                filter_fn=None,
                split="train",
                data_files=f'{data_dir}/{f}_train.json'
            )
            ds_list_train.append(ds_train)
        except Exception as e:
            print(f"Warning: Could not load train file for {f}: {e}")

    for f in test_filelist:
        # For meta-learning, we use test domains as dev/test
        try:
            ds_dev = SFTDataset(
                model_transform=tokenizer,
                source="json",
                message_transform=InputOutputToMessages(
                    column_map={"input": "input", "output": "output"},
                ),
                filter_fn=None,
                split="train",
                data_files=f'{data_dir}/{f}_test.json'
            )
            ds_list_dev.append(ds_dev)

            # Use same file for test (can be split later if needed)
            ds_test = SFTDataset(
                model_transform=tokenizer,
                source="json",
                message_transform=InputOutputToMessages(
                    column_map={"input": "input", "output": "output"},
                ),
                filter_fn=None,
                split="train",
                data_files=f'{data_dir}/{f}_test.json'
            )
            ds_list_test.append(ds_test)
        except Exception as e:
            print(f"Warning: Could not load test file for {f}: {e}")

    ds = [ds_list_train, ds_list_dev, ds_list_test]

    if packed:
        return
    else:
        return ds
