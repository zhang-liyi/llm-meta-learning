#
# In datasets/custom_dataset.py
#
from typing import Any, Dict

from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset
from torchtune.data import InputOutputToMessages
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.datasets._sft import SFTTransform
from torchtune.modules.tokenizers import ModelTokenizer

_SHUFFLE_SEED = 42


class _PreloadedSFTDataset(Dataset):
    """SFTDataset variant that accepts a pre-loaded (shuffled/selected) HF dataset."""

    def __init__(self, hf_data, message_transform, model_transform):
        self._data = hf_data
        self._prepare_sample = SFTTransform(
            message_transform=message_transform,
            model_transform=model_transform,
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._prepare_sample(self._data[index])

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune
def mc(tokenizer: ModelTokenizer, packed: bool = True):
    """
    Custom dataset for the counting task. Instruct and code response pairs.
    Uses ICL (in-context learning) versions for dev split.
    """

    ds_list_train, ds_list_dev_wino, ds_list_dev_cls45, ds_list_test_wino, ds_list_test_cls45 = [], [], [], [], []
    filelist = ['superglue-rte', 'tweet_eval-sentiment', 'glue-rte', 'superglue-wsc', 'glue-mrpc', 'tweet_eval-stance_hillary', 'tweet_eval-offensive', 'hatexplain', 'glue-cola', 'sick', 'paws', 'ethos-sexual_orientation', 'glue-qqp', 'tweet_eval-emotion', 'sms_spam', 'health_fact', 'glue-mnli', 'imdb', 'ethos-disability', 'glue-wnli', 'scitail', 'glue-sst2', 'tweet_eval-stance_abortion', 'tweet_eval-stance_climate', 'glue-qnli', 'ethos-directed_vs_generalized', 'ade_corpus_v2-classification', 'hate_speech_offensive', 'superglue-wic', 'google_wellformed_query', 'tweet_eval-irony', 'ethos-gender', 'rotten_tomatoes', 'kilt_fever']
    chembench_tasks = ['analytical_chemistry', 'chemical_preference', 'general_chemistry', 'inorganic_chemistry', 'materials_science', 'organic_chemistry', 'physical_chemistry', 'technical_chemistry', 'toxicity_and_safety']

    for f in filelist:
        ds_train = SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            filter_fn=None,
            split="train",
            data_files=f'../data/{f}_ins_inputoutput_icl_train.json'
        )
        ds_list_train.append(ds_train)
    ds_list_dev_wino.append(SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            filter_fn=None,
            split="train",
            data_files=f'../data/winogrande/winogrande_m_inputoutput_train.json'
        ))
    ds_list_test_wino.append(SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            filter_fn=None,
            split="train",
            data_files=f'../data/winogrande/winogrande_m_inputoutput_val.json'
        ))
    ds_list_dev_wino.append(SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            filter_fn=None,
            split="train",
            data_files=f'../data/openbookqa_inputoutput_dev.json'
        ))
    ds_list_test_wino.append(SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            filter_fn=None,
            split="train",
            data_files=f'../data/openbookqa_inputoutput_test.json'
        ))
    msg_transform = InputOutputToMessages(column_map={"input": "input", "output": "output"})
    for f in chembench_tasks:
        raw = hf_load_dataset(
            "json",
            data_files=f'../data/chembench/chembench_{f}_inputoutput_icl_dev.json',
            split="train",
        ).shuffle(seed=_SHUFFLE_SEED)
        half = len(raw) // 2
        ds_list_dev_cls45.append(_PreloadedSFTDataset(raw.select(range(half)), msg_transform, tokenizer))
        ds_list_test_cls45.append(_PreloadedSFTDataset(raw.select(range(half, len(raw))), msg_transform, tokenizer))
    ds = [ds_list_train, ds_list_dev_wino, ds_list_dev_cls45, ds_list_test_wino, ds_list_test_cls45]

    if packed:
        return
    else:
        return ds
