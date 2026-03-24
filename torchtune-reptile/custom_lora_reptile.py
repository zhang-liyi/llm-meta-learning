# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from warnings import warn

import torch
import torchtune.modules.common_utils as common_utils
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    get_adapter_params,
    get_adapter_state_dict,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchmetrics.classification import MulticlassCalibrationError

from transformers import AutoTokenizer

from tqdm import tqdm
from copy import deepcopy
import numpy as np

import logging



class LoRAFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    LoRA finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe is optimized
    for single GPU training. Training on CPU is not supported.

    Features:
        - Activation Checkpointing. This can be controlled using the ``enable_activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Activation Offloading. This can be controlled using the ``enable_activation_offloading``
            flag. Activation offloading is a technique similar to activations checkpointing that helps
            reduce the memory footprint to prevent OOMs on CUDA and enable bigger batches. Where activations
            checkpointing drops the activation in the forward to recompute it later in the backward,
            activations offloading will drop the activation in the forward to the CPU and bring it
            back during the backward pass. As always, there is a tradeoff--these savings in memory can
            come at the cost of training performance and CPU resources. To recover some runtime cost,
            we've added an option to enable offloading on a different stream to permit overlapping with
            the computation. This option is currently only available on PyTorch 2.5 or later and will
            be enabled by default if an acceptable torch version is found. Activation offloading can be
            used in conjunction with activation checkpointing.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * gradient accumulation steps.

            For example: with batch_size=1 and gradient_accumulation_steps=32 we get a total batch size of 32.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
            library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
            8-bit AdamW and Paged AdamW.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Currently we checkpoint both the adapter weights (trainable params only) and the
            complete merged weights (adapter weights added back to the base model). For more details
            please take a look at our LoRA tutorial
            (https://pytorch.org/torchtune/main/tutorials/lora_finetune.html).

            Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training. Resuming
            training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/tutorials/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator

    """

    def __init__(self, cfg: DictConfig) -> None:
        self.filename = FILENAME
        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        if self._log_peak_memory_stats and self._device.type == "cpu":
            log.info(
                "log_peak_memory_stats was set to True, however, training uses cpu. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.K = K
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            should_load_recipe_state=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not corrctly loaded
            # no need to check here
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self.cfg = cfg
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        self._compile = cfg.compile
        if cfg.device == "npu" and cfg.compile:
            raise ValueError(
                "NPU does not support model compilation. Please set `compile: False` in the config."
            )
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        # hack to toggle to the low cpu ram version of the reparametrize_as_dtype
        # hook based on the config.
        common_utils._use_low_cpu_ram = cfg.get("low_cpu_ram", False)

        # set up model
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            compile_model=cfg.compile,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            model=self._model,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if self._compile:
            self._loss_fn = training.compile_loss(self._loss_fn)

        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        log.info("Loss is initialized.")

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = 0
        for dataloader in self._dataloaderlist:
            self._steps_per_epoch += len(dataloader) // self._gradient_accumulation_steps
        
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
            self.global_step = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        # self._lr_scheduler = self._setup_lr_scheduler(
        #     cfg_lr_scheduler=cfg.lr_scheduler,
        #     num_training_steps=self.total_epochs * self._steps_per_epoch,
        #     last_epoch=self.global_step - 1,
        # )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
        """

        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        log.info(f" Profiler config after instantiation: {profiler_cfg}")

        self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
        if profiler_cfg["enabled"]:
            self.profiler_wait_steps = profiler_cfg["wait_steps"]
            self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
            self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        compile_model: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)
        self.adapter_params = get_adapter_params(model)
        self._is_dora = any(["magnitude" in k for k in self.adapter_params.keys()])
        set_trainable_params(model, self.adapter_params)

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        base_missing, base_unexpected = model.load_state_dict(
            base_model_state_dict, strict=False
        )
        # This is for any adapters that need to be initialized after base weights
        # have been loaded (e.g. DoRA).
        if self._is_dora:
            for m in model.modules():
                if hasattr(m, "initialize_dora_magnitude"):
                    m.initialize_dora_magnitude()
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(
                lora_weights_state_dict, strict=False
            )
        else:
            lora_missing, lora_unexpected = None, None

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # Validate model adapter params were loaded in with the expected dtype
        # TODO (rohan-varma): Further validation to ensure the appropriate base params
        # are NF4 vs bf16 based on the quantization config.
        training.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self._dtype
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        log.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)
        return model

    def _setup_optimizer(
        self, 
        cfg_optimizer: DictConfig, 
        model,
        opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler
    
    def _setup_data_helper(self, ds, shuffle, batch_size, collate_fn, packed):

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else padded_collate_packed
            ),
        )

        return sampler, dataloader

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = getattr(ds, "packed", False)
        else:
            # ds, ds2 = config.instantiate(cfg_dataset, self._tokenizer) # a possible way for multiple datasets
            # ds_wino, ds_wino_v, ds_arc, ds_arc_v, ds_obqa, ds_obqa_v
            if DATASET in ['cls45', 'obqa', 'winogrande', 'legalbench', 'chembench']:
                ds_list_train, ds_list_dev_wino, ds_list_dev_cls45, ds_list_test_wino, ds_list_test_cls45 = config.instantiate(cfg_dataset, self._tokenizer)
            else:
                ds_list_train, ds_list_dev, ds_list_test = config.instantiate(cfg_dataset, self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        self._samplerlist = []
        self._dataloaderlist = []
        for ds in ds_list_train:
            sampler, dataloader = self._setup_data_helper(ds, shuffle, batch_size, collate_fn, packed)
            self._samplerlist.append(sampler)
            self._dataloaderlist.append(dataloader)

        if DATASET == 'winogrande':
            sampler, dataloader = self._setup_data_helper(ds_list_dev_wino[0], False, batch_size, collate_fn, packed)
            self._sampler_v = [sampler]
            self._dataloader_v = [dataloader]
            sampler, dataloader = self._setup_data_helper(ds_list_test_wino[0], False, batch_size, collate_fn, packed)
            self._sampler_ood = [sampler]
            self._dataloader_ood = [dataloader]

        elif DATASET == 'obqa':
            sampler, dataloader = self._setup_data_helper(ds_list_dev_wino[1], False, batch_size, collate_fn, packed)
            self._sampler_v = [sampler]
            self._dataloader_v = [dataloader]
            sampler, dataloader = self._setup_data_helper(ds_list_test_wino[1], False, batch_size, collate_fn, packed)
            self._sampler_ood = [sampler]
            self._dataloader_ood = [dataloader]

        elif DATASET in ['cls45', 'legalbench', 'chembench']:
            self._sampler_v = []
            self._dataloader_v = []
            self._sampler_ood = []
            self._dataloader_ood = []
            for ds in ds_list_test_cls45:
                sampler, dataloader = self._setup_data_helper(ds, False, 1, collate_fn, packed)
                self._sampler_ood.append(sampler)
                self._dataloader_ood.append(dataloader)
            for ds in ds_list_dev_cls45:
                sampler, dataloader = self._setup_data_helper(ds, False, 1, collate_fn, packed)
                self._sampler_v.append(sampler)
                self._dataloader_v.append(dataloader)
            log.info(f'DEBUG {len(self._dataloaderlist)} {len(self._dataloader_v)} {len(self._dataloader_ood)}')
        
        elif DATASET in ['psych101', 'nli', 'para', 'cls23', 'qa', 'random', 'mixedspecific', 'legalbench-full']:
            self._sampler_v = []
            self._dataloader_v = []
            self._sampler_ood = []
            self._dataloader_ood = []
            for ds in ds_list_test:
                sampler, dataloader = self._setup_data_helper(ds, False, batch_size, collate_fn, packed)
                self._sampler_ood.append(sampler)
                self._dataloader_ood.append(dataloader)
            for ds in ds_list_dev:
                sampler, dataloader = self._setup_data_helper(ds, False, batch_size, collate_fn, packed)
                self._sampler_v.append(sampler)
                self._dataloader_v.append(dataloader)

        log.info("Dataset and Sampler are initialized.")

        return

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete
        - If the `self._save_adapter_weights_only` option is True, the checkpointer will save only the adapter weights

        To correctly resume from training, the adapter weights and recipe state must be provided along with the base model weights.
        """
        ckpt_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs
        # if training is in-progress, checkpoint the optimizer state as well
        if intermediate_checkpoint:
            ckpt_dict.update(
                {
                    training.OPT_KEY: self._optimizer.state_dict(),
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self.epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )

        adapter_state_dict = get_adapter_state_dict(self._model.state_dict())
        ckpt_dict.update({training.ADAPTER_KEY: adapter_state_dict})

        if not self._save_adapter_weights_only:
            # Construct the full state dict with LoRA weights merged into base LLM weights

            # Move to CPU to avoid a copy on GPU
            state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}

            merged_state_dict = get_merged_lora_ckpt(
                state_dict,
                rank=self._lora_rank,
                alpha=self._lora_alpha,
            )

            ckpt_dict.update({training.MODEL_KEY: merged_state_dict})

        adapter_config = {
            "r": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "target_modules": get_lora_module_names(
                self._lora_attn_modules,
                self._apply_lora_to_mlp,
                self._apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }
        ckpt_dict.update({training.ADAPTER_CONFIG: adapter_config})

        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=intermediate_checkpoint,
            adapter_only=self._save_adapter_weights_only,
        )

    def _loss_step(self, batch: Dict[str, torch.Tensor], model):
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")
        # run model
        with self.activations_handling_ctx:
            logits = model(**batch)

        # Shift labels to compute loss
        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
        # But this way we dont need to slice the logits. We just add an ignore index to labels.
        labels = torch.hstack(
            (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
        )
        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))

        loss = self._loss_fn(logits, labels)
        metrics = self.get_metrics(logits, labels)

        # free logits otherwise it peaks backward memory
        del logits

        return loss, metrics

    def get_metrics(self, logits, labels):
        # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
        ece_metric = MulticlassCalibrationError(num_classes=26, n_bins=10, norm='l1')
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self._loss_fn.num_output_chunks, dim=1)
        ]
        # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
        logits = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]
        correct = 0
        all = 0
        ece = 0
        for i in range(len(logits)):
            #log.info(f'LOGITS, {len(logits)}, {logits[i].shape}')
            tokens = torch.argmax(logits[i], 1)
            indices = torch.tensor(torch.tensor(labels[i]==32).int())
            for alph in range(1, 26):
                indices += torch.tensor(torch.tensor(labels[i]==32+alph).int())
            if torch.sum(indices) == 0:
                continue
            else:
                indices = indices.nonzero()
                for idx in indices:
                    if tokens[idx] == labels[i][idx]:
                        correct += 1
                    ece_label = torch.tensor([labels[i][idx]-32])
                    ece_logits = torch.unsqueeze(torch.squeeze(logits[i][idx, 32:58]), 0).detach().cpu()
                    ece += ece_metric(ece_logits, ece_label)
                    all += 1
        
        acc = correct/all
        ece = ece.numpy() / all

        del logits

        # log.info(f'ACCURACY: {acc}')

        return acc, ece
    

    def set_prune_pctg(self, prune_pctg, model):

        for layer in model.layers:

            attn = layer.attn

            attn.q_proj.prune_pctg = prune_pctg
            attn.v_proj.prune_pctg = prune_pctg
            attn.output_proj.prune_pctg = prune_pctg
            
            mlp = layer.mlp

            mlp.w1.prune_pctg = prune_pctg
            mlp.w2.prune_pctg = prune_pctg
            mlp.w3.prune_pctg = prune_pctg

        return


    def train(self) -> None:
        """
        The core training loop.
        """

        if self._compile:
            log.info(
                "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
            )

        running_loss = 0
        num_tokens = 0
        saved_losses_inner = []

        self.curr_acc = []
        self.curr_acc_v = []
        self.curr_acc_ood = []
        self.accs = []
        self.accs_v = []
        self.accs_ood = []

        self.curr_acc_ood_prune = []
        self.accs_ood_prune = []

        self.curr_ece = []
        self.curr_ece_v = []
        self.curr_ece_ood = []
        self.eces = []
        self.eces_v = []
        self.eces_ood = []

        self.curr_ece_ood_prune = []
        self.eces_ood_prune = []

        self.curr_ll = []
        self.curr_ll_v = []
        self.curr_ll_ood = []
        self.lls = []
        self.lls_v = []
        self.lls_ood = []
        best_acc = 0

        if DATASET in ['winogrande', 'obqa']:
            val_num = 500
            test_num = 500
        elif DATASET in ['cls45', 'cls23', 'qa', 'random', 'legalbench', 'chembench', 'legalbench-full']:
            val_num = 50
            test_num = 50
        elif DATASET in ['nli', 'mixedspecific']:
            val_num = 100
            test_num = 100
        elif DATASET in ['para']:
            val_num = 200
            test_num = 200
        elif DATASET == 'psych101':
            val_num = 100
            test_num = 100


        iters = []
        for i, ds in enumerate(self._dataloaderlist):
            iters.append(iter(ds))

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(0, self.total_epochs):

            frac_done = curr_epoch / self.total_epochs
            cur_meta_step_size = (1 - frac_done) * meta_lr
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True

            # Loop through datasets
            for i in range(len(self._dataloaderlist)):
                # Update the sampler to ensure data is correctly shuffled across epochs
                # in case shuffle is True
                self._samplerlist[i].set_epoch(curr_epoch)

                # model_copy does not change after inner loops
                self._model = self._model.to(torch.device('cpu'))
                model_copy = deepcopy(self._model).to(torch.device('cpu'))
                self._model = self._model.to(torch.device('cuda'))
                current_weights = model_copy.state_dict()

                ##### Inner loop
                # Take 5 batches from one dataset
                for inner_loop in range(self.K):
                    ok_batch = False
                    count = 0
                    while not ok_batch and count < 100:
                        try:
                            batch = next(iters[i])
                        except StopIteration:
                            iters[i] = iter(self._dataloaderlist[i])
                            batch = next(iters[i])
                        if batch['labels'].shape[-1] <= LENGTH_CUTOFF:
                            ok_batch = True
                        count += 1

                    if (
                        self.max_steps_per_epoch is not None
                        and (idx // self._gradient_accumulation_steps)
                        == self.max_steps_per_epoch
                    ):
                        break

                    # Start tracking CUDA memory for active steps for just the first epoch
                    if (
                        curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history()

                    utils.batch_to_device(batch, self._device)

                    # Calculate the number of unmasked tokens in the current batch
                    # and increment the total number of tokens seen in the step
                    current_num_tokens = (
                        batch["labels"] != self._loss_fn.ignore_index
                    ).sum()
                    num_tokens += current_num_tokens

                    # Loss is normalized by default so we multiply by the number of tokens
                    # This way we can normalize by the total number of tokens if we're accumulating gradients
                    current_loss, metrics = self._loss_step(batch, self._model)
                    current_loss = current_loss * current_num_tokens
                    current_loss.backward()
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    saved_losses_inner.append(current_loss.detach().cpu().numpy())
                    self.curr_acc.append(metrics[0])
                    self.curr_ece.append(metrics[1])
                    if np.sum(np.isnan(saved_losses_inner)) == 0:
                        np.savetxt(self.filename +'_innerloss.csv', np.array(saved_losses_inner))

                    # for theta in self._model.parameters():

                    #     # Get Parameter Gradient
                    #     if theta.grad is not None:
                    #         grad = theta.grad.data
                    #         # Update Model Parameter
                    #         theta.data -= inner_lr * grad
                        

                    # Reset running stats for the next step
                    num_tokens = 0
                    ##### END OF INNER LOOPS
                torch.cuda.empty_cache()  
                peak_memory = torch.cuda.max_memory_allocated()
                log.info(f"PEAK MEMORY INNER: Peak GPU memory allocated so far: {peak_memory / (1024**2):.2f} MB")

                ##### OUTER LOOP BEGINS HERE
                # Stop tracking CUDA memory now that active steps are complete
                if (
                    curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history(enabled=None)

                # Get Current Candidate Weights updated by the inner loop
                candidate_weights = self._model.state_dict()

                # Transfer Candidate Weights to Model State Checkpoint
                state_dict = {candidate: (current_weights[candidate].to(torch.device('cuda')) + cur_meta_step_size * 
                                        (candidate_weights[candidate] - current_weights[candidate].to(torch.device('cuda')))) 
                                            for candidate in candidate_weights}
                peak_memory = torch.cuda.max_memory_allocated()
                log.info(f"PEAK MEMORY OUTER: Peak GPU memory allocated so far: {peak_memory / (1024**2):.2f} MB")
                
                self._model.load_state_dict(state_dict)
                del state_dict
                del model_copy

            ##### EVAL #####
            report_every = 3 #if DATASET != 'nli' else 5
            if curr_epoch % report_every == 0:
                with torch.no_grad():
                    self._model.eval()

                    # Validation set of seen dataset
                    # Trivial step
                    log.info(len(self._dataloader_v))
                    for idx, batch in enumerate(self._dataloader_v[0]):
                        if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                            continue
                        utils.batch_to_device(batch, self._device)
                        #self.set_det_mode(True)
                        labels_copy = batch["labels"]
                        labels = batch.pop("labels")
                        # run model
                        with self.activations_handling_ctx:
                            logits = self._model(**batch)

                        batch["labels"] = labels_copy

                        del logits
                        del labels
                        break
                    torch.cuda.empty_cache()

                # Unseen data from unseen dataset
                for loader_idx, loader in enumerate(self._dataloader_v):
                    log.info(len(loader))
                    if len(loader) <= 10:
                        continue
                    self._model.to(torch.device('cpu'))
                    model_eval = deepcopy(self._model)
                    model_eval.to(torch.device('cuda'))
                    model_eval.train()
                    self.meta_optimizer = self._setup_optimizer(
                        cfg_optimizer=self.cfg.meta_optimizer,
                        model=model_eval,
                        opt_state_dict=(None
                        ),
                    )
                    self.meta_optimizer.zero_grad(set_to_none=True)

                    reject_batch_tracker = 0
                    last_seen_idx = -1
                    for idx, batch in enumerate(loader):
                        if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                            reject_batch_tracker += 1
                            continue

                        if idx - reject_batch_tracker >= metaeval_steps:
                            break

                        last_seen_idx = idx
                        utils.batch_to_device(batch, self._device)

                        loss, metrics = self._loss_step(batch, model_eval)
                        loss = loss * current_num_tokens
                        loss.backward()
                        self.meta_optimizer.step()
                        self.meta_optimizer.zero_grad(set_to_none=True)

                    log.info(f'CHECK VAL {len(loader)}, {reject_batch_tracker}')

                    with torch.no_grad():
                        model_eval.eval()
                        reject_tracker_val = 0
                        for idx, batch in enumerate(loader):
                            if idx == val_num + (last_seen_idx + 1) + reject_tracker_val:
                                break
                            if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                reject_tracker_val += 1
                                continue
                            if idx <= last_seen_idx:
                                continue
                            utils.batch_to_device(batch, self._device)
                            #self.set_det_mode(True)
                            labels_copy = batch["labels"]
                            labels = batch.pop("labels")
                            # run model
                            with self.activations_handling_ctx:
                                logits = model_eval(**batch)

                            labels = torch.hstack(
                                (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                            )
                            if not isinstance(logits, list):
                                labels = labels.reshape(-1)
                                logits = logits.reshape(-1, logits.size(-1))

                            acc, ece = self.get_metrics(logits, labels)
                            self.curr_acc_v.append(acc)
                            self.curr_ece_v.append(ece)
                            batch["labels"] = labels_copy

                            del logits

                        if self._dataloader_ood:
                            reject_tracker_ood = 0
                            for idx, batch in enumerate(self._dataloader_ood[loader_idx]):
                                if idx - reject_tracker_ood >= test_num:
                                    break
                                if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                    reject_tracker_ood += 1
                                    continue
                                utils.batch_to_device(batch, self._device)
                                #self.set_det_mode(True)
                                labels_copy = batch["labels"]
                                labels = batch.pop("labels")
                                # run model
                                with self.activations_handling_ctx:
                                    logits = model_eval(**batch)

                                labels = torch.hstack(
                                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                                )
                                if not isinstance(logits, list):
                                    labels = labels.reshape(-1)
                                    logits = logits.reshape(-1, logits.size(-1))

                                acc, ece = self.get_metrics(logits, labels)
                                self.curr_acc_ood.append(acc)
                                self.curr_ece_ood.append(ece)
                                batch["labels"] = labels_copy

                                del logits

                            log.info(f'CHECK OOD {len(self._dataloader_ood[loader_idx])}, {reject_tracker_ood}')
                        
                        # Trivial step, if the model never got to a forward pass, you can't delete it and erase memory cleanly
                        tmp_idx = 1 if DATASET == 'psych101' else 0
                        for idx, batch in enumerate(self._dataloader_v[tmp_idx]):
                            if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                continue
                            utils.batch_to_device(batch, self._device)
                            #self.set_det_mode(True)
                            labels_copy = batch["labels"]
                            labels = batch.pop("labels")
                            # run model
                            with self.activations_handling_ctx:
                                logits = model_eval(**batch)

                            batch["labels"] = labels_copy

                            del logits
                            del labels
                            break
                        torch.cuda.empty_cache()

                    torch.cuda.empty_cache()
                    del model_eval
                    log.info(f'MEMORY: end of one validation dataset, memory allocated {torch.cuda.memory_allocated()}')
                self._model.to(torch.device(self._device))
                self._model.train()
                # self._optimizer = self._setup_optimizer(
                #     cfg_optimizer=self.cfg.optimizer,
                #     opt_state_dict=(None
                #     ),
                # )

                if RUN_PRUNING and np.mean(self.curr_acc_v) > best_acc:
                    best_acc = np.mean(self.curr_acc_v)

                    with torch.no_grad():
                        self._model.eval()

                        # Validation set of seen dataset
                        # Trivial step
                        log.info(len(self._dataloader_v))
                        for idx, batch in enumerate(self._dataloader_v[0]):
                            if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                continue
                            utils.batch_to_device(batch, self._device)
                            #self.set_det_mode(True)
                            labels_copy = batch["labels"]
                            labels = batch.pop("labels")
                            # run model
                            with self.activations_handling_ctx:
                                logits = self._model(**batch)

                            batch["labels"] = labels_copy

                            del logits
                            del labels
                            break
                        torch.cuda.empty_cache()

                    # Unseen data from unseen dataset
                    for loader_idx, loader in enumerate(self._dataloader_v):
                        if len(loader) <= 10:
                            continue

                        self.curr_acc_ood_prune.append([])
                        self.curr_ece_ood_prune.append([])
                        log.info(len(loader))
                        
                        self._model.to(torch.device('cpu'))
                        model_eval = deepcopy(self._model)
                        model_eval.to(torch.device('cuda'))
                        model_eval.train()
                        self.meta_optimizer = self._setup_optimizer(
                            cfg_optimizer=self.cfg.meta_optimizer,
                            model=model_eval,
                            opt_state_dict=(None
                            ),
                        )
                        self.meta_optimizer.zero_grad(set_to_none=True)

                        reject_batch_tracker_prune = 0
                        for idx, batch in enumerate(loader):
                            if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                reject_batch_tracker_prune += 1
                                continue

                            if idx - reject_batch_tracker_prune >= metaeval_steps:
                                break

                            utils.batch_to_device(batch, self._device)

                            loss, metrics = self._loss_step(batch, model_eval)
                            loss = loss * current_num_tokens
                            loss.backward()
                            self.meta_optimizer.step()
                            self.meta_optimizer.zero_grad(set_to_none=True)

                        model_eval.eval()

                        torch.cuda.empty_cache() 

                        with torch.no_grad():
                            # run model_pruning
                            if RUN_PRUNING:
                                if curr_epoch >= 0:

                                    for prune_pctg in [0.01, 0.1, 0.2, 0.3]:

                                        self.curr_acc_ood_prune[-1].append([])
                                        self.curr_ece_ood_prune[-1].append([])

                                        self.set_prune_pctg(prune_pctg, model_eval)

                                        for loader_idx in range(len(self._dataloader_ood)):
                                            reject_tracker_ood = 0
                                            for idx, batch in enumerate(self._dataloader_ood[loader_idx]):
                                                if idx - reject_tracker_ood >= test_num:
                                                    break
                                                if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                                    reject_tracker_ood += 1
                                                    continue
                                                utils.batch_to_device(batch, self._device)
                                                #self.set_det_mode(True)
                                                labels_copy = batch["labels"]
                                                labels = batch.pop("labels")
                                                # run model
                                                logits = model_eval(**batch)

                                                labels = torch.hstack(
                                                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                                                )
                                                if not isinstance(logits, list):
                                                    labels = labels.reshape(-1)
                                                    logits = logits.reshape(-1, logits.size(-1))

                                                acc, ece = self.get_metrics(logits, labels)
                                                self.curr_acc_ood_prune[-1][-1].append(acc)
                                                self.curr_ece_ood_prune[-1][-1].append(ece)

                                                batch["labels"] = labels_copy

                                                del logits
                                                del labels

                                    self.set_prune_pctg(0, model_eval)

                            tmp_idx = 1 if DATASET == 'psych101' else 0
                            for idx, batch in enumerate(self._dataloader_v[tmp_idx]):
                                if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                    continue
                                utils.batch_to_device(batch, self._device)
                                #self.set_det_mode(True)
                                labels_copy = batch["labels"]
                                labels = batch.pop("labels")
                                # run model
                                with self.activations_handling_ctx:
                                    logits = model_eval(**batch)

                                batch["labels"] = labels_copy

                                del logits
                                del labels
                                break
                            torch.cuda.empty_cache()
                            torch.cuda.empty_cache() 
                            del model_eval
                            log.info(f'MEMORY: end of one validation dataset, memory allocated {torch.cuda.memory_allocated()}')

                else:

                    for loader_idx, loader in enumerate(self._dataloader_v):

                        if len(loader) <= 10:
                            continue

                        self.curr_acc_ood_prune.append([])
                        self.curr_ece_ood_prune.append([])

                        if RUN_PRUNING and self._dataloader_ood:
                            if curr_epoch >= 0:

                                for prune_pctg in [0.01, 0.1, 0.2, 0.3]:

                                    self.curr_acc_ood_prune[-1].append([])
                                    self.curr_ece_ood_prune[-1].append([])

                                    for idx, batch in enumerate(self._dataloader_ood[loader_idx]):

                                        acc = -1.
                                        ece = -1.

                                        self.curr_acc_ood_prune[-1][-1].append(acc)
                                        self.curr_ece_ood_prune[-1][-1].append(ece)


                self._model.to(torch.device(self._device))
                self._model.train()
                torch.cuda.empty_cache() 

                self.epochs_run += 1
                log.info('')
                log.info(f'----------- END OF EPOCH {curr_epoch} -----------')
                log.info('')
                log.info(f'TRAIN ACCURACY: {np.mean(self.curr_acc)}')
                log.info(f'VALIDATION ACCURACY: {np.mean(self.curr_acc_v)}')
                log.info(f'VALIDATION ACCURACY OOD: {np.mean(self.curr_acc_ood)}')
                log.info(f'VALIDATION ECE: {np.mean(self.curr_ece_v)}')
                log.info(f'VALIDATION ECE OOD: {np.mean(self.curr_ece_ood)}')
                log.info('')
                log.info(f'----------- END OF EPOCH {curr_epoch} -----------')
                log.info('')

                self.accs.append(np.mean(self.curr_acc))
                self.accs_v.append(np.mean(self.curr_acc_v))
                self.accs_ood.append(np.mean(self.curr_acc_ood))
                np.savetxt(self.filename +'_accs_train.csv', np.array(self.accs))
                np.savetxt(self.filename +'_accs_val.csv', np.array(self.accs_v))
                np.savetxt(self.filename +'_accs_ood.csv', np.array(self.accs_ood))
                self.curr_acc = []
                self.curr_acc_v = []
                self.curr_acc_ood = []

                self.eces.append(np.mean(self.curr_ece))
                self.eces_v.append(np.mean(self.curr_ece_v))
                self.eces_ood.append(np.mean(self.curr_ece_ood))
                np.savetxt(self.filename +'_eces_train.csv', np.array(self.eces))
                np.savetxt(self.filename +'_eces_val.csv', np.array(self.eces_v))
                np.savetxt(self.filename +'_eces_ood.csv', np.array(self.eces_ood))
                self.curr_ece = []
                self.curr_ece_v = []
                self.curr_ece_ood = []

                if RUN_PRUNING:
                    curr_acc_prune = [[] for i in range(4)]
                    for i in range(len(self.curr_acc_ood_prune)):
                        for prune_num in range(4):
                            curr_acc_prune[prune_num].extend(self.curr_acc_ood_prune[i][prune_num])
                    for i in range(4):
                        curr_acc_prune[i] = np.mean(curr_acc_prune[i])
                    curr_ece_prune = [[] for i in range(4)]
                    for i in range(len(self.curr_ece_ood_prune)):
                        for prune_num in range(4):
                            curr_ece_prune[prune_num].extend(self.curr_ece_ood_prune[i][prune_num])
                    for i in range(4):
                        curr_ece_prune[i] = np.mean(curr_ece_prune[i])
                    self.accs_ood_prune.append(curr_acc_prune)
                    self.eces_ood_prune.append(curr_ece_prune)
                    np.savetxt(self.filename + '_accs_prune.csv', np.array(self.accs_ood_prune))
                    np.savetxt(self.filename + '_eces_prune.csv', np.array(self.eces_ood_prune))

                self.curr_acc_ood_prune = []
                self.curr_ece_ood_prune = []






    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="LoRAFinetuneRecipeSingleDevice", cfg=cfg)

    global DATASET
    global DATA_PROPERTY
    global meta_lr
    global metaeval_steps
    global K
    global RUN_PRUNING
    global weight_decay
    global LENGTH_CUTOFF
    global tokenizer

    DATASET = str(cfg.dataset._component_).split('_')[1]
    if 'llama' in str(cfg.model._component_):
        MODEL = 'llama'
        model_path = str(cfg.model_path)
    elif 'qwen' in str(cfg.model._component_):
        MODEL = 'qwen'
        model_path = str(cfg.model_path)

    DATA_PROPERTY = '' # '' or '_limited' or '_icl'
    if 'fullmetaicl' in str(cfg.dataset._component_):
        DATA_PROPERTY = '_fullmetaicl'
        DATASET += '-full'
    elif 'metaicl' in str(cfg.dataset._component_):
        DATA_PROPERTY = '_metaicl'
    elif 'icl' in str(cfg.dataset._component_) and ('legalbench' in str(cfg.dataset._component_) or 'chembench' in str(cfg.dataset._component_)):
        DATA_PROPERTY = '_metaicl'
    elif 'icl' in str(cfg.dataset._component_):
        DATA_PROPERTY = '_icl'
    meta_lr = 0.5
    metaeval_steps = cfg.metaeval_steps
    K = 5
    RUN_PRUNING = False
    weight_decay = ''
    LENGTH_CUTOFF = 1500
    filename = f'{MODEL}-{DATASET}{DATA_PROPERTY}-reptile-5e-5-1e-5-meta_lr{meta_lr}-sgd-stepeval{metaeval_steps}_{RUN_PRUNING}{weight_decay}'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    global log
    global FILENAME

    FILENAME = filename + f'-seed{cfg.seed}'

    log = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{FILENAME}.log', encoding='utf-8', level=logging.DEBUG)
    log.debug('This message should go to the log file')
    log.info('So should this')
    log.warning('And this, too')


    recipe = LoRAFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
