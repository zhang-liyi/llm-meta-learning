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
import numpy as np
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
from transformers import AutoTokenizer
from torchmetrics.classification import MulticlassCalibrationError

from tqdm import tqdm
import copy
import pickle
import gc

import logging


def log_normal_pdf(sample, mean, logvar, dim=0):
    log2pi = torch.log(torch.tensor(2. * np.pi, device='cuda', dtype=sample.dtype))
    return torch.sum(
        -.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi),
        dim=dim)

def log_gamma_pdf(sample, a, b, dim=0):
    # a = a.to(torch.device('cuda'))
    # b = b.to(torch.device('cuda'))
    log_gamma_func = torch.lgamma(a)
    return torch.sum(
        a*b - log_gamma_func + (a-1) * torch.log(sample) - b*sample,
    dim=dim)

def mask_labels_psych101(labels):

    labels = torch.squeeze(labels)

    if len(labels.shape) == 1:
        labels = torch.unsqueeze(labels, 0)
    
    new_labels = []
    for labels_i in labels:
        new_labels.append([])
        for i, wordid in enumerate(labels_i):

            if i == 0 and wordid != 1134:
                new_labels[-1].append(-100)
            else:
                if labels_i[i-1] == 1134:
                    new_labels[-1].append(labels_i[i])
                else:
                    new_labels[-1].append(-100)

    return torch.tensor(new_labels)




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
        # ----- New ones added for Bayesian meta-learning -----+
        self.gamma_a = 1.
        self.gamma_b = 0.01
        self.K = K # number of innerloop updates
        self.filename = FILENAME
        # -----------------------------------------------------+
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

        self.inner_optimizer = self._setup_inner_optimizer(
            cfg.inner_optimizer,
            self._model,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        self.outer_optimizer = self._setup_outer_optimizer(
            cfg.outer_optimizer,
            self._model,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        self.cfg = cfg
        self.checkpoint_dict = checkpoint_dict

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
        self._steps_per_epoch = 200
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
    
    def _setup_inner_optimizer(
        self, 
        cfg_optimizer: DictConfig, 
        model,
        opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        
        inner_parameters = []
        for layer in model.layers:

            attn = layer.attn

            inner_parameters += attn.q_proj.lora_a_mui.parameters()
            inner_parameters += attn.q_proj.lora_b_mui.parameters()
            inner_parameters += attn.q_proj.lora_a_logvari.parameters()
            inner_parameters += attn.q_proj.lora_b_logvari.parameters()
            inner_parameters += attn.v_proj.lora_a_mui.parameters()
            inner_parameters += attn.v_proj.lora_b_mui.parameters()
            inner_parameters += attn.v_proj.lora_a_logvari.parameters()
            inner_parameters += attn.v_proj.lora_b_logvari.parameters()
            inner_parameters += attn.output_proj.lora_a_mui.parameters()
            inner_parameters += attn.output_proj.lora_b_mui.parameters()
            inner_parameters += attn.output_proj.lora_a_logvari.parameters()
            inner_parameters += attn.output_proj.lora_b_logvari.parameters()

            mlp = layer.mlp

            inner_parameters += mlp.w1.lora_a_mui.parameters()
            inner_parameters += mlp.w1.lora_b_mui.parameters()
            inner_parameters += mlp.w1.lora_a_logvari.parameters()
            inner_parameters += mlp.w1.lora_b_logvari.parameters()
            inner_parameters += mlp.w2.lora_a_mui.parameters()
            inner_parameters += mlp.w2.lora_b_mui.parameters()
            inner_parameters += mlp.w2.lora_a_logvari.parameters()
            inner_parameters += mlp.w2.lora_b_logvari.parameters()
            inner_parameters += mlp.w3.lora_a_mui.parameters()
            inner_parameters += mlp.w3.lora_b_mui.parameters()
            inner_parameters += mlp.w3.lora_a_logvari.parameters()
            inner_parameters += mlp.w3.lora_b_logvari.parameters()

        optimizer_inner = config.instantiate(cfg_optimizer, inner_parameters)

        log.info("Optimizer and loss are initialized.")
        return optimizer_inner

    def _setup_outer_optimizer(
        self, 
        cfg_optimizer: DictConfig, 
        model, 
        opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        
        outer_parameters = []
        for layer in model.layers:

            attn = layer.attn

            outer_parameters += attn.q_proj.lora_a_mutheta.parameters()
            outer_parameters += attn.q_proj.lora_b_mutheta.parameters()
            outer_parameters += attn.q_proj.lora_a_logvartheta.parameters()
            outer_parameters += attn.q_proj.lora_b_logvartheta.parameters()
            outer_parameters += attn.v_proj.lora_a_mutheta.parameters()
            outer_parameters += attn.v_proj.lora_b_mutheta.parameters()
            outer_parameters += attn.v_proj.lora_a_logvartheta.parameters()
            outer_parameters += attn.v_proj.lora_b_logvartheta.parameters()
            outer_parameters += attn.output_proj.lora_a_mutheta.parameters()
            outer_parameters += attn.output_proj.lora_b_mutheta.parameters()
            outer_parameters += attn.output_proj.lora_a_logvartheta.parameters()
            outer_parameters += attn.output_proj.lora_b_logvartheta.parameters()
            
            mlp = layer.mlp

            outer_parameters += mlp.w1.lora_a_mutheta.parameters()
            outer_parameters += mlp.w1.lora_b_mutheta.parameters()
            outer_parameters += mlp.w1.lora_a_logvartheta.parameters()
            outer_parameters += mlp.w1.lora_b_logvartheta.parameters()
            outer_parameters += mlp.w2.lora_a_mutheta.parameters()
            outer_parameters += mlp.w2.lora_b_mutheta.parameters()
            outer_parameters += mlp.w2.lora_a_logvartheta.parameters()
            outer_parameters += mlp.w2.lora_b_logvartheta.parameters()
            outer_parameters += mlp.w3.lora_a_mutheta.parameters()
            outer_parameters += mlp.w3.lora_b_mutheta.parameters()
            outer_parameters += mlp.w3.lora_a_logvartheta.parameters()
            outer_parameters += mlp.w3.lora_b_logvartheta.parameters()

        optimizer_outer = config.instantiate(cfg_optimizer, outer_parameters)
        # if opt_state_dict:
        #     optimizer.load_state_dict(opt_state_dict)

        log.info("Optimizer and loss are initialized.")
        return optimizer_outer

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
            if DATASET in ['cls45', 'obqa', 'winogrande']:
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

        elif DATASET == 'cls45':
            self._sampler_v = []
            self._dataloader_v = []
            self._sampler_ood = []
            self._dataloader_ood = []
            for ds in ds_list_test_cls45:
                sampler, dataloader = self._setup_data_helper(ds, False, batch_size, collate_fn, packed)
                self._sampler_ood.append(sampler)
                self._dataloader_ood.append(dataloader)
            for ds in ds_list_dev_cls45:
                sampler, dataloader = self._setup_data_helper(ds, False, batch_size, collate_fn, packed)
                self._sampler_v.append(sampler)
                self._dataloader_v.append(dataloader)
            log.info(f'DEBUG {len(self._dataloaderlist)} {len(self._dataloader_v)} {len(self._dataloader_ood)}')
        
        elif DATASET in ['psych101', 'nli', 'para', 'cls23', 'qa', 'random', 'mixedspecific']:
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
                    training.OPT_KEY: self.inner_optimizer.state_dict(),
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

    '''
    Bayesian meta-learning loss_steps:
    '''

    def reinitialize_phi(self, epoch):

        # loop over all mu_i, logvar_i parameters and re-initialize them to mu_theta, logvar_theta

        for layer in self._model.layers:

            attn = layer.attn

            attn.q_proj.lora_a_mui.weight.data = attn.q_proj.lora_a_mutheta.weight.data.clone().requires_grad_(True)
            attn.q_proj.lora_b_mui.weight.data = attn.q_proj.lora_b_mutheta.weight.data.clone().requires_grad_(True)
            attn.q_proj.lora_a_logvari.weight.data = attn.q_proj.lora_a_logvartheta.weight.data.clone().requires_grad_(True)
            attn.q_proj.lora_b_logvari.weight.data = attn.q_proj.lora_b_logvartheta.weight.data.clone().requires_grad_(True)
            attn.v_proj.lora_a_mui.weight.data = attn.v_proj.lora_a_mutheta.weight.data.clone().requires_grad_(True)
            attn.v_proj.lora_b_mui.weight.data = attn.v_proj.lora_b_mutheta.weight.data.clone().requires_grad_(True)
            attn.v_proj.lora_a_logvari.weight.data = attn.v_proj.lora_a_logvartheta.weight.data.clone().requires_grad_(True)
            attn.v_proj.lora_b_logvari.weight.data = attn.v_proj.lora_b_logvartheta.weight.data.clone().requires_grad_(True)
            attn.output_proj.lora_a_mui.weight.data = attn.output_proj.lora_a_mutheta.weight.data.clone().requires_grad_(True)
            attn.output_proj.lora_b_mui.weight.data = attn.output_proj.lora_b_mutheta.weight.data.clone().requires_grad_(True)
            attn.output_proj.lora_a_logvari.weight.data = attn.output_proj.lora_a_logvartheta.weight.data.clone().requires_grad_(True)
            attn.output_proj.lora_b_logvari.weight.data = attn.output_proj.lora_b_logvartheta.weight.data.clone().requires_grad_(True)
            
            mlp = layer.mlp

            mlp.w1.lora_a_mui.weight.data = mlp.w1.lora_a_mutheta.weight.data.clone().requires_grad_(True)
            mlp.w1.lora_b_mui.weight.data = mlp.w1.lora_b_mutheta.weight.data.clone().requires_grad_(True)
            mlp.w1.lora_a_logvari.weight.data = mlp.w1.lora_a_logvartheta.weight.data.clone().requires_grad_(True)
            mlp.w1.lora_b_logvari.weight.data = mlp.w1.lora_b_logvartheta.weight.data.clone().requires_grad_(True)
            mlp.w2.lora_a_mui.weight.data = mlp.w2.lora_a_mutheta.weight.data.clone().requires_grad_(True)
            mlp.w2.lora_b_mui.weight.data = mlp.w2.lora_b_mutheta.weight.data.clone().requires_grad_(True)
            mlp.w2.lora_a_logvari.weight.data = mlp.w2.lora_a_logvartheta.weight.data.clone().requires_grad_(True)
            mlp.w2.lora_b_logvari.weight.data = mlp.w2.lora_b_logvartheta.weight.data.clone().requires_grad_(True)
            mlp.w3.lora_a_mui.weight.data = mlp.w3.lora_a_mutheta.weight.data.clone().requires_grad_(True)
            mlp.w3.lora_b_mui.weight.data = mlp.w3.lora_b_mutheta.weight.data.clone().requires_grad_(True)
            mlp.w3.lora_a_logvari.weight.data = mlp.w3.lora_a_logvartheta.weight.data.clone().requires_grad_(True)
            mlp.w3.lora_b_logvari.weight.data = mlp.w3.lora_b_logvartheta.weight.data.clone().requires_grad_(True)

        return
    
    def save_parameters(self, loader_idx):

        # loop over all mu_i, logvar_i parameters and re-initialize them to mu_theta, logvar_theta

        for l, layer in enumerate(self._model.layers):

            weights = {}
            attn = layer.attn
            weights['q_proj'] = []
            weights['q_proj'].append(attn.q_proj.logvar_i.detach().float().cpu().numpy())
            weights['q_proj'].append(attn.q_proj.logvar_theta.detach().float().cpu().numpy())
            weights['v_proj'] = []
            weights['v_proj'].append(attn.v_proj.logvar_i.detach().float().cpu().numpy())
            weights['v_proj'].append(attn.v_proj.logvar_theta.detach().float().cpu().numpy())
            weights['out_proj'] = []
            weights['out_proj'].append(attn.output_proj.logvar_i.detach().float().cpu().numpy())
            weights['out_proj'].append(attn.output_proj.logvar_theta.detach().float().cpu().numpy())
            
            mlp = layer.mlp

            weights['w1'] = []
            weights['w1'].append(mlp.w1.logvar_i.detach().float().cpu().numpy())
            weights['w1'].append(mlp.w1.logvar_theta.detach().float().cpu().numpy())
            weights['w2'] = []
            weights['w2'].append(mlp.w2.logvar_i.detach().float().cpu().numpy())
            weights['w2'].append(mlp.w2.logvar_theta.detach().float().cpu().numpy())
            weights['w3'] = []
            weights['w3'].append(mlp.w3.logvar_i.detach().float().cpu().numpy())
            weights['w3'].append(mlp.w3.logvar_theta.detach().float().cpu().numpy())

            with open(f'./VAR-EMBEDS-LAYER{l}-TASK{loader_idx}-{FILENAME}.pickle', 'wb') as handle:
                pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return
    
    # def reset_internals(self):

    #     # loop over all mu_i, logvar_i parameters and re-initialize them to mu_theta, logvar_theta

    #     for layer in self._model.layers:

    #         attn = layer.attn

    #         attn.q_proj.phi_i = []
    #         attn.v_proj.phi_i = []
    #         attn.output_proj.phi_i = []
            
    #         mlp = layer.mlp

    #         mlp.w1.phi_i = []
    #         mlp.w2.phi_i = []
    #         mlp.w3.phi_i = []

    #     return
    
    def set_det_mode(self, det_mode, model):

        for layer in model.layers:

            attn = layer.attn

            attn.q_proj.det_mode = det_mode
            attn.v_proj.det_mode = det_mode
            attn.output_proj.det_mode = det_mode
            
            mlp = layer.mlp

            mlp.w1.det_mode = det_mode
            mlp.w2.det_mode = det_mode
            mlp.w3.det_mode = det_mode

        return
    
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
    
    def set_explicit_forward_mode(self, mode, model):

        for layer in model.layers:

            attn = layer.attn

            attn.q_proj.explicit_forward = mode
            attn.v_proj.explicit_forward = mode
            attn.output_proj.explicit_forward = mode
            
            mlp = layer.mlp

            mlp.w1.explicit_forward = mode
            mlp.w2.explicit_forward = mode
            mlp.w3.explicit_forward = mode

        return
    
    def get_prior_kl(self):
        # get kl (q(theta) || p(theta)), which is -log p(theta) after simplification
        kl = 0
        normal_logvar = np.log(1)

        for layer in self._model.layers:
            attn = layer.attn
            log_normal = log_normal_pdf(attn.q_proj.mu_theta.reshape(-1), 
                                        torch.tensor(0., device='cuda'), 
                                        torch.tensor(normal_logvar, device='cuda'))
            log_gamma = log_gamma_pdf(1/torch.exp(attn.q_proj.logvar_theta.reshape(-1)), 
                                      torch.tensor(self.gamma_a, device='cuda'), 
                                      torch.tensor(self.gamma_b, device='cuda'))
            kl -= log_normal + log_gamma
            log_normal = log_normal_pdf(attn.v_proj.mu_theta.reshape(-1), 
                                        torch.tensor(0., device='cuda'), 
                                        torch.tensor(normal_logvar, device='cuda'))
            log_gamma = log_gamma_pdf(1/torch.exp(attn.v_proj.logvar_theta.reshape(-1)), 
                                      torch.tensor(self.gamma_a, device='cuda'), 
                                      torch.tensor(self.gamma_b, device='cuda'))
            kl -= log_normal + log_gamma
            log_normal = log_normal_pdf(attn.output_proj.mu_theta.reshape(-1), 
                                        torch.tensor(0., device='cuda'), 
                                        torch.tensor(normal_logvar, device='cuda'))
            log_gamma = log_gamma_pdf(1/torch.exp(attn.output_proj.logvar_theta.reshape(-1)),
                                      torch.tensor(self.gamma_a, device='cuda'), 
                                      torch.tensor(self.gamma_b, device='cuda'))
            kl -= log_normal + log_gamma

            mlp = layer.mlp
            log_normal = log_normal_pdf(mlp.w1.mu_theta.reshape(-1), 
                                        torch.tensor(0., device='cuda'), 
                                        torch.tensor(normal_logvar, device='cuda'))
            log_gamma = log_gamma_pdf(1/torch.exp(mlp.w1.logvar_theta.reshape(-1)),
                                      torch.tensor(self.gamma_a, device='cuda'), 
                                      torch.tensor(self.gamma_b, device='cuda'))
            kl -= log_normal + log_gamma
            log_normal = log_normal_pdf(mlp.w2.mu_theta.reshape(-1), 
                                        torch.tensor(0., device='cuda'), 
                                        torch.tensor(normal_logvar, device='cuda'))
            log_gamma = log_gamma_pdf(1/torch.exp(mlp.w2.logvar_theta.reshape(-1)), 
                                      torch.tensor(self.gamma_a, device='cuda'), 
                                      torch.tensor(self.gamma_b, device='cuda'))
            kl -= log_normal + log_gamma
            log_normal = log_normal_pdf(mlp.w3.mu_theta.reshape(-1), 
                                        torch.tensor(0., device='cuda'), 
                                        torch.tensor(normal_logvar, device='cuda'))
            log_gamma = log_gamma_pdf(1/torch.exp(mlp.w3.logvar_theta.reshape(-1)), 
                                      torch.tensor(self.gamma_a, device='cuda'), 
                                      torch.tensor(self.gamma_b, device='cuda'))
            kl -= log_normal + log_gamma

        return kl.to(torch.float32)
    
    def get_log_q(self):

        log_q = 0

        for layer in self._model.layers:
            attn = layer.attn
            log_q_phi = log_normal_pdf(attn.q_proj.phi_i.reshape(-1), 
                                       attn.q_proj.mu_i.reshape(-1), 
                                       attn.q_proj.logvar_i.reshape(-1))
            log_q += log_q_phi
            log_q_phi = log_normal_pdf(attn.v_proj.phi_i.reshape(-1), 
                                       attn.v_proj.mu_i.reshape(-1), 
                                       attn.v_proj.logvar_i.reshape(-1))
            log_q += log_q_phi
            log_q_phi = log_normal_pdf(attn.output_proj.phi_i.reshape(-1), 
                                       attn.output_proj.mu_i.reshape(-1), 
                                       attn.output_proj.logvar_i.reshape(-1))
            log_q += log_q_phi
            
            mlp = layer.mlp
            log_q_phi = log_normal_pdf(mlp.w1.phi_i.reshape(-1), 
                                       mlp.w1.mu_i.reshape(-1), 
                                       mlp.w1.logvar_i.reshape(-1))
            log_q += log_q_phi
            log_q_phi = log_normal_pdf(mlp.w2.phi_i.reshape(-1), 
                                       mlp.w2.mu_i.reshape(-1), 
                                       mlp.w2.logvar_i.reshape(-1))
            log_q += log_q_phi
            log_q_phi = log_normal_pdf(mlp.w3.phi_i.reshape(-1), 
                                       mlp.w3.mu_i.reshape(-1), 
                                       mlp.w3.logvar_i.reshape(-1))
            log_q += log_q_phi

        return log_q.to(torch.float32)
    
    def get_log_p(self):

        log_p = 0

        for layer in self._model.layers:
            attn = layer.attn
            log_p_phi = log_normal_pdf(attn.q_proj.phi_i.reshape(-1), 
                                       attn.q_proj.mu_theta.reshape(-1), 
                                       attn.q_proj.logvar_theta.reshape(-1))
            log_p += log_p_phi
            log_p_phi = log_normal_pdf(attn.v_proj.phi_i.reshape(-1), 
                                       attn.v_proj.mu_theta.reshape(-1), 
                                       attn.v_proj.logvar_theta.reshape(-1))
            log_p += log_p_phi
            log_p_phi = log_normal_pdf(attn.output_proj.phi_i.reshape(-1), 
                                       attn.output_proj.mu_theta.reshape(-1), 
                                       attn.output_proj.logvar_theta.reshape(-1))
            log_p += log_p_phi

            mlp = layer.mlp
            log_p_phi = log_normal_pdf(mlp.w1.phi_i.reshape(-1), 
                                       mlp.w1.mu_theta.reshape(-1), 
                                       mlp.w1.logvar_theta.reshape(-1))
            log_p += log_p_phi
            log_p_phi = log_normal_pdf(mlp.w2.phi_i.reshape(-1), 
                                       mlp.w2.mu_theta.reshape(-1), 
                                       mlp.w2.logvar_theta.reshape(-1))
            log_p += log_p_phi
            log_p_phi = log_normal_pdf(mlp.w3.phi_i.reshape(-1), 
                                       mlp.w3.mu_theta.reshape(-1), 
                                       mlp.w3.logvar_theta.reshape(-1))
            log_p += log_p_phi
            
        return log_p.to(torch.float32)
    
    def generate(self, batch):

        labels_copy = batch["labels"]
        labels = batch.pop("labels")

        # run model
        logits = self._model(**batch)
        self.show_outputs(logits, generate_mode=True)

        batch["labels"] = labels_copy

        return
    
    def show_outputs(self, logits, generate_mode=False):
        str_concat = ''
        for i in range(len(logits)):
            #log.info(f'LOGITS, {len(logits)}, {logits[i].shape}')
            tokens = torch.argmax(logits[i], 2)
            response_string = tokenizer.decode(tokens[0], skip_special_tokens=False)
            str_concat += response_string
        #log.info(tokens.shape)
        if not generate_mode:
            log.info(f'STRING: {str_concat}')
        else:
            log.info(f'STRING GENERATE MODE: {str_concat}')
        # log.info(f'2nd to LAST TOKEN {tokens[0]}')
        # log.info(f'2nd to LAST TOKEN {tokenizer.decode(tokens[0,[-2]])}')
        # log.info(f'LAST TOKEN {tokenizer.decode(tokens[0,[-1]])}')

    def save_outputs(self, idx, logits, labels, batch=None, generate_mode=False):
        str_concat = ''
        for i in range(len(logits)):
            #log.info(f'LOGITS, {len(logits)}, {logits[i].shape}')
            tokens = torch.argmax(logits[i], 2)
            response_string = tokenizer.decode(tokens[0], skip_special_tokens=False)
            str_concat += response_string
        #log.info(tokens.shape)
        np.savetxt(f'str_output_batch{idx}_bayesian.csv', np.array([str_concat]), fmt='%s')
        with open(f'logits_labels_batch{idx}_bayesian.pickle', 'wb') as file:
            pickle.dump([logits, labels, batch], file)
        # log.info(f'2nd to LAST TOKEN {tokens[0]}')
        # log.info(f'2nd to LAST TOKEN {tokenizer.decode(tokens[0,[-2]])}')
        # log.info(f'LAST TOKEN {tokenizer.decode(tokens[0,[-1]])}')

    def get_metrics(self, logits, labels):
        # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self._loss_fn.num_output_chunks, dim=1)
        ]
        # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
        logits = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]
        correct = 0
        ece = 0
        all = 0

        if DATASET != 'psych101':
            ece_metric = MulticlassCalibrationError(num_classes=4, n_bins=10, norm='l1')
            for i in range(len(logits)):
                #log.info(f'LOGITS, {len(logits)}, {logits[i].shape}')
                tokens = torch.argmax(logits[i], 1)
                indices = torch.tensor(torch.tensor(labels[i]==32).int() + torch.tensor(labels[i]==33).int()) \
                    + torch.tensor(torch.tensor(labels[i]==34).int() + torch.tensor(labels[i]==35).int())# 32 for A, 33 for B
                if torch.sum(indices) == 0:
                    continue
                else:
                    indices = indices.nonzero()
                    for idx in indices:
                        if tokens[idx] == labels[i][idx]:
                            correct += 1
                        ece_label = torch.tensor([labels[i][idx]-32])
                        ece_logits = torch.unsqueeze(torch.squeeze(logits[i][idx, 32:36]), 0).detach().cpu()
                        ece += ece_metric(ece_logits, ece_label)
                        all += 1
            
            del logits
            acc = correct/all
            ece = ece.numpy() / all

        else:
            ece_metric = MulticlassCalibrationError(num_classes=26, n_bins=10, norm='l1')
            for i in range(len(logits)):
                #log.info(f'LOGITS, {len(logits)}, {logits[i].shape}')
                tokens = torch.argmax(logits[i], 1)
                indices = torch.tensor(torch.tensor(labels[i] != -100).int())
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
            
            del logits
            if all != 0:
                acc = correct/all
                ece = ece.numpy() / all
            else:
                log.info(f'ALL=0 DATAPOINT: {labels}')
                acc = 0
                ece = 0

        # log.info(f'ACCURACY: {acc}')

        return acc, ece
    
    def loss_step_abridged(self, batch: Dict[str, torch.Tensor], model, step, epoch) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels_copy = batch["labels"]
        labels = batch.pop("labels")
        # run model
        logits = model(**batch)

        # self.show_outputs(logits)
        # Shift labels to compute loss
        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
        # But this way we dont need to slice the logits. We just add an ignore index to labels.
        if DATASET == 'psych101':
            labels = mask_labels_psych101(labels).to(torch.device('cuda'))
        labels = torch.hstack(
            (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
        )
        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))

        loss = self._loss_fn(logits, labels) # neg log likelihood
        ll = -loss.detach().cpu().numpy()
        acc, ece = self.get_metrics(logits, labels)

        # free logits otherwise it peaks backward memory
        del logits

        log_q = self.get_log_q() # computing log_q is dependent on self._model(**batch) just being run, and corresponds to the same **batch
        log_p = self.get_log_p() # computing log_p is dependent on self._model(**batch) just being run, and corresponds to the same **batch

        info = [loss, log_q, log_p]
        metrics = [acc, ece, ll]

        batch["labels"] = labels_copy

        return loss, info, metrics
    
    def loss_step_weighted(self, batch: Dict[str, torch.Tensor], model, step, epoch) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels_copy = batch["labels"]
        labels = batch.pop("labels")
        # run model
        logits = model(**batch)

        # self.show_outputs(logits)
        # Shift labels to compute loss
        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
        # But this way we dont need to slice the logits. We just add an ignore index to labels.
        if DATASET == 'psych101':
            labels = mask_labels_psych101(labels).to(torch.device('cuda'))
            #log.info(f'DEBUG LABELS {labels}')
        labels = torch.hstack(
            (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
        )
        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))

        loss = self._loss_fn(logits, labels)
        acc, ece = self.get_metrics(logits, labels)


        # free logits otherwise it peaks backward memory
        del logits

        log_q = self.get_log_q() # computing log_q is dependent on self._model(**batch) just being run, and corresponds to the same **batch
        log_p = self.get_log_p() # computing log_p is dependent on self._model(**batch) just being run, and corresponds to the same **batch

        #w = min(1 / 2 / (10**10) * (step + epoch * 2000), 1.)
        w = min(1 / (10**(-weight)), 1.)

        ll = -loss.detach().cpu().numpy()

        loss = loss + w*(log_q - log_p)
        #loss = log_q - log_p
        info = [loss, log_q, log_p]
        metrics = [acc, ece, ll]

        batch["labels"] = labels_copy

        return loss, info, metrics

    # Original _loss_step:
    # def _loss_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    #     # Shape [b, s], needed for the loss not the model
    #     labels = batch.pop("labels")
    #     # run model
    #     with self.activations_handling_ctx:
    #         logits = self._model(**batch)

    #     # Shift labels to compute loss
    #     # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
    #     # But this way we dont need to slice the logits. We just add an ignore index to labels.
    #     labels = torch.hstack(
    #         (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
    #     )
    #     if not isinstance(logits, list):
    #         labels = labels.reshape(-1)
    #         logits = logits.reshape(-1, logits.size(-1))

    #     loss = self._loss_fn(logits, labels)

    #     # free logits otherwise it peaks backward memory
    #     del logits

    #     return loss

    def train(self) -> None:
        """
        The core training loop.
        """

        if self._compile:
            log.info(
                "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
            )

        log.info(f'IGNORE INDEX: {self._loss_fn.ignore_index}')

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        #num_tokens = 0
        saved_losses_inner = [] # recons loss, q loss, p loss
        saved_losses_outer = [] # recons loss, q loss, p loss, prior

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
        elif DATASET in ['cls45', 'cls23', 'qa', 'random']:
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

        self.set_det_mode(False, self._model)
        self.set_explicit_forward_mode(True, self._model)

        iters = []
        for i, ds in enumerate(self._dataloaderlist):
            iters.append(iter(ds))

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            self._model.train()

            pbar = tqdm(total=self._steps_per_epoch)
            log.info(len(self._dataloaderlist))

            batches = []
            for iter_idx, ds in enumerate(self._dataloaderlist):
                # Update the sampler to ensure data is correctly shuffled across epochs
                # in case shuffle is True
                self._samplerlist[iter_idx].set_epoch(curr_epoch)
                if not VANILLA:
                    self.reinitialize_phi(curr_epoch)

                # Loop through 'tasks', or 'datasets'; one batch from each dataset
                for inner_k in range(self.K):

                    ok_batch = False
                    count = 0
                    while not ok_batch and count < 100:
                        try:
                            batch = next(iters[iter_idx])
                        except StopIteration:
                            iters[iter_idx] = iter(self._dataloaderlist[iter_idx])
                            batch = next(iters[iter_idx])
                        if batch['labels'].shape[-1] <= LENGTH_CUTOFF:
                            ok_batch = True
                        count += 1

                    utils.batch_to_device(batch, self._device)

                    # Loss is normalized by default so we multiply by the number of tokens
                    # This way we can normalize by the total number of tokens if we're accumulating gradients

                    ##### Inner Loop
                    # Calculate the number of unmasked tokens in the current batch
                    # and increment the total number of tokens seen in the step
                    # current_num_tokens = (
                    #     batch["labels"] != self._loss_fn.ignore_index
                    # ).sum()
                    # num_tokens += current_num_tokens
                    
                    # self.set_det_mode(True)
                    # self.generate(batch)
                    # self.set_det_mode(False)
                    self.inner_optimizer.zero_grad(set_to_none=True) 
                    self.outer_optimizer.zero_grad(set_to_none=True)
                    if not VANILLA:
                        loss, info, metrics = self.loss_step_weighted(batch, self._model, inner_k, curr_epoch)
                    else:
                        loss, info, metrics = self.loss_step_abridged(batch, self._model, inner_k, curr_epoch)
                    current_loss = loss
                    acc = metrics[0]
                    ece = metrics[1]
                    ll = metrics[2]
                    if inner_k == self.K - 1:
                        self.curr_acc.append(acc)
                        self.curr_ece.append(ece)
                        self.curr_ll.append(ll)
                    #log.info(f'1DEBUG PARAMS: {self._model.layers[-1].attn.q_proj.phi_i}, {len(self._model.layers[-1].attn.q_proj.phi_i)}')

                    if not EPOCH_0_NO_TRAIN or curr_epoch != 0:
                        self.set_explicit_forward_mode(False, self._model)
                        current_loss.backward()
                        self.set_explicit_forward_mode(True, self._model)
                        #log.info(f'2DEBUG PARAMS: {self._model.layers[-1].attn.q_proj.phi_i}, {len(self._model.layers[-1].attn.q_proj.phi_i)}')
                        if np.sum(np.isnan(saved_losses_inner)) == 0:
                            np.savetxt(self.filename +'_innerloss.csv', np.array(saved_losses_inner))
                        # Step with optimizer
                        # if (idx + 1) % 1 == 0:
                        #     training.scale_grads(self._model, 1 / num_tokens)
                        #     if self._clip_grad_norm is not None:
                        #         grad_norm = torch.nn.utils.clip_grad_norm_(
                        #             self._model.parameters(),
                        #             max_norm=float(self._clip_grad_norm),
                        #         )
                            #log.info(f'3DEBUG PARAMS: {self._model.layers[-1].attn.q_proj.phi_i}, {len(self._model.layers[-1].attn.q_proj.phi_i)}')
                        self.inner_optimizer.step() 
                            #log.info(f'4DEBUG PARAMS: {self._model.layers[-1].attn.q_proj.phi_i}, {len(self._model.layers[-1].attn.q_proj.phi_i)}')
                        info = [x.detach().cpu().float().numpy() for x in info]
                        saved_losses_inner.append(info)
                    # log.info(' ')
                    # log.info(' ')
                    # log.info(' ')  
                    # log.info(f'Epoch {curr_epoch}, Step {idx}, Inner loop {inner_k} | Loss {current_loss.detach().cpu().numpy()}')
                    # log.info(f'Breakdown: {info}')
                    # log.info(f'Params: {torch.mean(self._model.layers[-1].attn.q_proj.logvar_i)}')
                    # log.info(' ')
                    # log.info(' ')
                    # log.info(' ')
                    # self.reset_internals()
                    torch.cuda.empty_cache()
                        

                    # self.set_det_mode(True)
                    # self.generate(batch)
                    # self.set_det_mode(False)

                ##### Outer loop
                if not VANILLA:
                    self.inner_optimizer.zero_grad(set_to_none=True)
                    self.outer_optimizer.zero_grad(set_to_none=True)
                    # current_num_tokens = (
                    #     batch["labels"] != self._loss_fn.ignore_index
                    # ).sum()
                    # num_tokens += current_num_tokens
                    loss, info, metrics = self.loss_step_weighted(batch, self._model, iter_idx, curr_epoch) 
                    prior_loss = 1/len(self._dataloaderlist) * self.get_prior_kl() * 10**(weight_prior)
                    outer_loss = loss + prior_loss
                    info.append(prior_loss)
                    #log.info(f'DEBUG PARAMS: {self._model.layers[-1].attn.q_proj.phi_i}, {len(self._model.layers[-1].attn.q_proj.phi_i)}')

                    if not EPOCH_0_NO_TRAIN or curr_epoch != 0:
                        self.set_explicit_forward_mode(False, self._model)
                        outer_loss.backward()
                        self.set_explicit_forward_mode(True, self._model)
                        if np.sum(np.isnan(saved_losses_outer)) == 0:
                            np.savetxt(self.filename +'_outerloss.csv', np.array(saved_losses_outer))
                        # Step with optimizer
                        # if (idx + 1) % 1 == 0:
                        #     training.scale_grads(self._model, 1 / num_tokens)
                        #     if self._clip_grad_norm is not None:
                        #         grad_norm = torch.nn.utils.clip_grad_norm_(
                        #             self._model.parameters(),
                        #             max_norm=float(self._clip_grad_norm),
                        #         )
                        self.outer_optimizer.step()
                        info = [x.detach().cpu().float().numpy() for x in info]
                        saved_losses_outer.append(info)
                        peak_memory = torch.cuda.max_memory_allocated()
                        log.info(f"PEAK MEMORY: Peak GPU memory allocated so far: {peak_memory / (1024**2):.2f} MB")

                    log.info(' ')
                    log.info(' ')
                    log.info(' ')  
                    log.info(f'Epoch {curr_epoch}, Step {iter_idx}, Outer loop | Loss {outer_loss.detach().cpu().numpy()}')
                    log.info(f'Breakdown: {info}')
                    
                    # self.reset_internals()
                    # log.info(f'Params: {self._model.layers[-1].attn.q_proj.logvar_i}')
                    # log.info(f'Params: {self._model.layers[-1].attn.v_proj.logvar_i}')
                    # log.info(f'Params: {self._model.layers[-10].attn.q_proj.logvar_i}')
                    # log.info(f'Params: {self._model.layers[-10].attn.v_proj.logvar_i}')
                    # log.info(f'Params: {self._model.layers[-1].attn.q_proj.lora_a_logvartheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-1].attn.v_proj.lora_a_logvartheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-10].attn.q_proj.lora_a_logvartheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-10].attn.v_proj.lora_a_logvartheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-1].attn.q_proj.lora_a_mutheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-1].attn.v_proj.lora_a_mutheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-10].attn.q_proj.lora_a_mutheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-10].attn.v_proj.lora_a_mutheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-1].attn.q_proj.lora_b_logvartheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-1].attn.v_proj.lora_b_logvartheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-1].attn.q_proj.lora_b_mutheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-1].attn.v_proj.lora_b_mutheta.weight.data}')
                    # log.info(f'Params: {self._model.layers[-1].attn.q_proj.lora_a_logvartheta.weight.data.shape}')
                    # log.info(f'Params: {self._model.layers[-1].attn.v_proj.lora_a_logvartheta.weight.data.shape}')
                    # log.info(f'Params: {self._model.layers[-10].attn.q_proj.lora_b_logvartheta.weight.data.shape}')
                    # log.info(f'Params: {self._model.layers[-10].attn.v_proj.lora_b_logvartheta.weight.data.shape}')
                    log.info(f'Params: {torch.mean(self._model.layers[-1].attn.q_proj.lora_b_mui.weight.data)} | {torch.mean(self._model.layers[-1].attn.q_proj.lora_b_mutheta.weight.data)}')
                    log.info(' ')
                    log.info(' ')
                    log.info(' ')
                    utils.batch_to_device(batch, 'cpu')
                    torch.cuda.empty_cache()

                    #self._lr_scheduler.step()
            report_every = 3 #if DATASET != 'nli' else 5
            if curr_epoch % report_every == 0:
                # Unseen data from unseen dataset
                if not VANILLA:
                    self.reinitialize_phi(curr_epoch)
                with torch.no_grad():
                    self._model.eval()
                    self.set_det_mode(True, self._model)

                    # Trivial step, just pass data in model forward mode once so we can later deepcopy it
                    log.info(len(self._dataloader_v))
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
                            logits = self._model(**batch)

                        batch["labels"] = labels_copy

                        del logits
                        del labels
                        break
                    torch.cuda.empty_cache()

                for loader_idx, loader in enumerate(self._dataloader_v):

                    reject_batch_tracker = 0
                    reject_tracker_val = 0
                    
                    log.info(len(loader))
                    if len(loader) <= 10:
                        continue
                    else:
                        self.curr_acc_ood.append([])
                        self.curr_ece_ood.append([])
                        self._model.to(torch.device('cpu'))
                        model_eval = copy.deepcopy(self._model)
                        model_eval.train()
                        model_eval.to(torch.device('cuda'))
                        self.set_det_mode(False, model_eval)
                        self.set_explicit_forward_mode(True, model_eval)
                        self.inner_optimizer_eval = self._setup_inner_optimizer(
                            self.cfg.inner_optimizer,
                            model_eval,
                            opt_state_dict=(
                                self.checkpoint_dict[training.OPT_KEY]
                                if self._resume_from_checkpoint
                                else None
                            ),
                        )
                        log.info(f'MEMORY: After creating new model in unseen validation, memory allocated {torch.cuda.memory_allocated()}')

                        last_seen_idx = len(loader) - 1

                        # Training for a few steps on this unseen dataset
                        for idx, batch in enumerate(loader):

                            if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                reject_batch_tracker += 1
                                continue

                            if idx - reject_batch_tracker >= metaeval_steps:
                                break

                            last_seen_idx = idx
                            
                            utils.batch_to_device(batch, self._device)
                            self.inner_optimizer_eval.zero_grad(set_to_none=True) 
                            # current_num_tokens = (
                            #     batch["labels"] != self._loss_fn.ignore_index
                            # ).sum()
                            # num_tokens += current_num_tokens

                            if not VANILLA:
                                loss, info, metrics = self.loss_step_weighted(batch, model_eval, idx, curr_epoch)
                            else:
                                loss, info, metrics = self.loss_step_abridged(batch, model_eval, idx, curr_epoch)

                            if not EPOCH_0_NO_TRAIN or curr_epoch != 0:
                                self.set_explicit_forward_mode(False, model_eval)
                                loss.backward()
                                self.set_explicit_forward_mode(True, model_eval)

                                # Step with optimizer
                                # training.scale_grads(model_eval, 1 / num_tokens)
                                self.inner_optimizer_eval.step()
                            del loss 
                            del info
                            del metrics

                        log.info(f'CHECK VAL {len(loader)}, {reject_batch_tracker}')
                        
                        self.inner_optimizer_eval.zero_grad(set_to_none=True) 

                        with torch.no_grad():
                            self.set_det_mode(True, model_eval)
                            model_eval.eval()
                            for idx, batch in enumerate(loader):
                                if idx == val_num + last_seen_idx + reject_tracker_val:
                                    break
                                if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                    reject_tracker_val += 1
                                    continue
                                if idx > last_seen_idx:
                                    utils.batch_to_device(batch, self._device)
                                    #self.set_det_mode(True)
                                    labels_copy = batch["labels"]
                                    labels = batch.pop("labels")
                                    # run model
                                    logits = model_eval(**batch)

                                    if DATASET == 'psych101':
                                        labels = mask_labels_psych101(labels).to(torch.device('cuda'))
                                    labels = torch.hstack(
                                        (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                                    )
                                    if not isinstance(logits, list):
                                        labels = labels.reshape(-1)
                                        logits = logits.reshape(-1, logits.size(-1))
                                    loss = self._loss_fn(logits, labels) # neg log likelihood
                                    ll = -loss.detach().cpu().numpy()

                                    acc, ece = self.get_metrics(logits, labels)
                                    self.curr_acc_v.append(acc)
                                    self.curr_ece_v.append(ece)
                                    self.curr_ll_v.append(ll)

                                    batch["labels"] = labels_copy

                                    del logits
                                    del labels

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

                                if DATASET == 'psych101':
                                    labels = mask_labels_psych101(labels).to(torch.device('cuda'))
                                labels = torch.hstack(
                                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                                )
                                if not isinstance(logits, list):
                                    labels = labels.reshape(-1)
                                    logits = logits.reshape(-1, logits.size(-1))
                                loss = self._loss_fn(logits, labels) # neg log likelihood
                                ll = -loss.detach().cpu().numpy()

                                acc, ece = self.get_metrics(logits, labels)
                                self.curr_acc_ood[-1].append(acc)
                                self.curr_ece_ood[-1].append(ece)
                                self.curr_ll_ood.append(ll)

                                batch["labels"] = labels_copy

                                del logits
                                del labels

                            log.info(f'CHECK OOD {len(self._dataloader_ood[loader_idx])}, {reject_tracker_ood}')

                            # Trivial step, if the model never got to a forward pass, you can't delete it and erase memory cleanly
                            if reject_tracker_ood == len(self._dataloader_ood[loader_idx]):
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

                        #model_eval = model_eval.to(torch.device('cpu'))
                        log.info(f'MEMORY: After validation before model delete, memory allocated {torch.cuda.memory_allocated()}')
                        del model_eval
                        log.info(f'MEMORY: After validation before optimizer delete, memory allocated {torch.cuda.memory_allocated()}')
                        del self.inner_optimizer_eval
                        gc.collect()
                        
                        self.set_det_mode(False, self._model)
                        log.info(f'MEMORY: The end of one dataset\'s validation, memory allocated {torch.cuda.memory_allocated()}')

                        torch.cuda.empty_cache()

                log.info(f'MEMORY: After validation after everything delete, memory allocated {torch.cuda.memory_allocated()}')
                self._model.to(torch.device(self._device))
                self._model.train()
                log.info(f'MEMORY: After validation after creating back the model, memory allocated {torch.cuda.memory_allocated()}')
                ##### END OF A TRAIN AND VAL EPOCH
                # Update the number of steps when the weights are updated
                self.global_step += 1

                # loss_to_log = outer_loss.item() / num_tokens

                # # Log per-step metrics
                # if self.global_step % self._log_every_n_steps == 0:
                #     time_per_step = time.perf_counter() - t0
                #     log_dict = {
                #         "loss": loss_to_log,
                #         "inner_lr": self.inner_optimizer.param_groups[0]["lr"],
                #         "outer_lr": self.outer_optimizer.param_groups[0]["lr"],
                #         "tokens_per_second_per_gpu": num_tokens / time_per_step,
                #     }
                #     if (
                #         self._device.type != "cpu"
                #         and self._log_peak_memory_stats
                #     ):
                #         log_dict.update(
                #             training.get_memory_stats(device=self._device)
                #         )
                #     if self._clip_grad_norm is not None:
                #         log_dict.update({"grad_norm": grad_norm})
                #     # self._metric_logger.log_dict(
                #     #     log_dict,
                #     #     step=self.global_step,
                #     # )

                t0 = time.perf_counter()
                num_tokens = 0

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
                    log.info('GPU MEMORY: ')
                    log.info(torch.cuda.memory_summary(device=None, abbreviated=False))

                torch.cuda.empty_cache() 
                acc_temp = []
                for i in range(len(self.curr_acc_ood)):
                    acc_temp.append(np.mean(self.curr_acc_ood[i]))

                ece_temp = []
                for i in range(len(self.curr_ece_ood)):
                    ece_temp.append(np.mean(self.curr_ece_ood[i]))

                self.curr_acc_ood = np.concatenate(self.curr_acc_ood, 0)
                self.curr_ece_ood = np.concatenate(self.curr_ece_ood, 0)

                
                
                if np.mean(self.curr_acc_v) > best_acc:
                    best_acc = np.mean(self.curr_acc_v)

                    with torch.no_grad():
                        self._model.eval()
                        self.set_det_mode(True, self._model)

                        # Trivial step, just pass data in model forward mode once so we can later deepcopy it
                        log.info(len(self._dataloader_v))
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
                                logits = self._model(**batch)

                            batch["labels"] = labels_copy

                            del logits
                            del labels
                            break
                        torch.cuda.empty_cache()

                    for loader_idx, loader in enumerate(self._dataloader_v):
                        
                        log.info(len(loader))
                        
                        if len(loader) <= 10:
                            continue
                        else:
                            reject_batch_tracker = 0
                            reject_tracker_val = 0
                            self.curr_acc_ood_prune.append([])
                            self.curr_ece_ood_prune.append([])

                            self._model.to(torch.device('cpu'))
                            model_eval = copy.deepcopy(self._model)
                            model_eval.train()
                            model_eval.to(torch.device('cuda'))
                            self.set_det_mode(False, model_eval)
                            self.set_explicit_forward_mode(True, model_eval)
                            self.inner_optimizer_eval = self._setup_inner_optimizer(
                                self.cfg.inner_optimizer,
                                model_eval,
                                opt_state_dict=(
                                    self.checkpoint_dict[training.OPT_KEY]
                                    if self._resume_from_checkpoint
                                    else None
                                ),
                            )
                            log.info(f'MEMORY: After creating new model in unseen validation, memory allocated {torch.cuda.memory_allocated()}')

                            last_seen_idx = len(loader) - 1

                            # Training for a few steps on this unseen dataset
                            for idx, batch in enumerate(loader):

                                if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                    reject_batch_tracker += 1
                                    continue

                                if idx - reject_batch_tracker >= metaeval_steps:
                                    break

                                last_seen_idx = idx
                                
                                utils.batch_to_device(batch, self._device)
                                self.inner_optimizer_eval.zero_grad(set_to_none=True) 
                                # current_num_tokens = (
                                #     batch["labels"] != self._loss_fn.ignore_index
                                # ).sum()
                                # num_tokens += current_num_tokens

                                if not VANILLA:
                                    loss, info, metrics = self.loss_step_weighted(batch, model_eval, idx, curr_epoch)
                                else:
                                    loss, info, metrics = self.loss_step_abridged(batch, model_eval, idx, curr_epoch)

                                if not EPOCH_0_NO_TRAIN or curr_epoch != 0:
                                    self.set_explicit_forward_mode(False, model_eval)
                                    loss.backward()
                                    self.set_explicit_forward_mode(True, model_eval)

                                    # Step with optimizer
                                    # training.scale_grads(model_eval, 1 / num_tokens)
                                    self.inner_optimizer_eval.step()
                                del loss 
                                del info
                                del metrics

                            log.info(f'CHECK VAL {len(loader)}, {reject_batch_tracker}')
                            
                            self.inner_optimizer_eval.zero_grad(set_to_none=True) 

                            with torch.no_grad():
                                self.set_det_mode(True, model_eval)
                                model_eval.eval()

                                if SAVE_WEIGHTS:
                                    for idx, batch in enumerate(self._dataloader_ood[loader_idx]):
                                        if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                                            continue
                                        utils.batch_to_device(batch, self._device)
                                        #self.set_det_mode(True)
                                        labels_copy = batch["labels"]
                                        labels = batch.pop("labels")
                                        # run model
                                        logits = model_eval(**batch)
                                        #log.info('PRUNE {prune_pctg} \n {logits} \n')
                                        self.save_parameters(loader_idx)
                                        del logits
                                        del labels
                                        break

                                if RUN_PRUNING:
                                    if curr_epoch >= 0:

                                        for prune_pctg in [0.01, 0.1, 0.2, 0.3]:

                                            self.curr_acc_ood_prune[-1].append([])
                                            self.curr_ece_ood_prune[-1].append([])

                                            self.set_prune_pctg(prune_pctg, model_eval)

                                            reject_tracker_ood = 0
                                            for idx, batch in enumerate(self._dataloader_ood[loader_idx]):
                                                if idx >= test_num:
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
                                                #log.info('PRUNE {prune_pctg} \n {logits} \n')

                                                if DATASET == 'psych101':
                                                    labels = mask_labels_psych101(labels).to(torch.device('cuda'))
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

                            #model_eval = model_eval.to(torch.device('cpu'))
                            log.info(f'MEMORY: After validation before model delete, memory allocated {torch.cuda.memory_allocated()}')
                            del model_eval
                            log.info(f'MEMORY: After validation before optimizer delete, memory allocated {torch.cuda.memory_allocated()}')
                            del self.inner_optimizer_eval
                            gc.collect()
                            
                            self.set_det_mode(False, self._model)
                            log.info(f'MEMORY: The end of one dataset\'s validation, memory allocated {torch.cuda.memory_allocated()}')

                            torch.cuda.empty_cache()

                    log.info(f'MEMORY: After validation after everything delete, memory allocated {torch.cuda.memory_allocated()}')
                    self._model.to(torch.device(self._device))
                    self._model.train()
                    log.info(f'MEMORY: After validation after creating back the model, memory allocated {torch.cuda.memory_allocated()}')

                else:
                    for loader_idx, loader in enumerate(self._dataloader_v):
                        if len(loader) <= 10:
                            continue

                        self.curr_acc_ood_prune.append([])
                        self.curr_ece_ood_prune.append([])

                        if RUN_PRUNING:
                            if curr_epoch >= 0:

                                for prune_pctg in [0.01, 0.1, 0.2, 0.3]:

                                    self.curr_acc_ood_prune[-1].append([])
                                    self.curr_ece_ood_prune[-1].append([])

                                    for idx, batch in enumerate(self._dataloader_ood[loader_idx]):

                                        acc = -1.
                                        ece = -1.

                                        self.curr_acc_ood_prune[-1][-1].append(acc)
                                        self.curr_ece_ood_prune[-1][-1].append(ece)

                # log.info('')
                # log.info(f'----------- END OF EPOCH {curr_epoch} -----------')
                # log.info('')
                # log.info(f'TRAIN ACCURACY: {np.mean(self.curr_acc)}')
                # log.info(f'VALIDATION ACCURACY: {np.mean(self.curr_acc_v)}')
                # log.info(f'VALIDATION ACCURACY OOD: {np.mean(self.curr_acc_ood)}')
                # log.info(f'VALIDATION ECE: {np.mean(self.curr_ece_v)}')
                # log.info(f'VALIDATION ECE OOD: {np.mean(self.curr_ece_ood)}')
                # log.info('')
                # log.info(f'----------- END OF EPOCH {curr_epoch} -----------')
                # log.info('')
                log.info('')
                log.info(f'----------- END OF EPOCH {curr_epoch} -----------')
                log.info('')
                log.info(f'TRAIN ACCURACY: {np.mean(self.curr_acc)}')
                log.info(f'VALIDATION ACCURACY: {np.mean(self.curr_acc_v)}')
                log.info(f'VALIDATION ACCURACY OOD: {acc_temp}')
                log.info(f'VALIDATION ECE: {np.mean(self.curr_ece_v)}')
                log.info(f'VALIDATION ECE OOD: {ece_temp}')
                log.info(f'VALIDATION LL: {np.mean(self.curr_ll_v)}')
                log.info(f'VALIDATION LL OOD: {np.mean(self.curr_ll_ood)}')
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

                self.lls.append(np.mean(self.curr_ll))
                self.lls_v.append(np.mean(self.curr_ll_v))
                self.lls_ood.append(np.mean(self.curr_ll_ood))
                np.savetxt(self.filename +'_lls_train.csv', np.array(self.lls))
                np.savetxt(self.filename +'_lls_val.csv', np.array(self.lls_v))
                np.savetxt(self.filename +'_lls_ood.csv', np.array(self.lls_ood))
                self.curr_ll = []
                self.curr_ll_v = []
                self.curr_ll_ood = []

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

            self.epochs_run += 1


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

    global VANILLA
    global DATASET
    global LIMITED_DATA
    global RUN_PRUNING
    global SAVE_WEIGHTS
    global MODEL
    global K
    global metaeval_steps
    global weight
    global weight_prior
    global LENGTH_CUTOFF
    global tokenizer
    global EPOCH_0_NO_TRAIN

    VANILLA = False
    DATASET = str(cfg.dataset._component_).split('_')[1]
    if 'llama' in str(cfg.model._component_):
        MODEL = 'llama'
        model_path = "/scratch/gpfs/GRIFFITHS/lz3156/resources/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"
    elif 'qwen' in str(cfg.model._component_):
        MODEL = 'qwen'
        model_path = "/scratch/gpfs/GRIFFITHS/lz3156/resources/qwen/Qwen2-7B-Instruct"

    LIMITED_DATA = '' # '_limited', otherwise empty string ''
    RUN_PRUNING = False
    SAVE_WEIGHTS = False
    K = 5
    metaeval_steps = 5
    weight = -8
    weight_prior = weight
    LENGTH_CUTOFF = 1500

    if not VANILLA:
        filename = f'{MODEL}-{DATASET}{LIMITED_DATA}-bayesian-uniweight{weight}-defaultprior-lr1t10neg5-5t10neg5-1t10neg5-reinit-stepeval{metaeval_steps}_{RUN_PRUNING}' # no '.log'
    else:
        filename = f'{MODEL}-{DATASET}{LIMITED_DATA}-vanilla-lr1t10neg5-1t10neg5-stepeval{metaeval_steps}_{RUN_PRUNING}'

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    EPOCH_0_NO_TRAIN = False

    global log
    global FILENAME
    FILENAME = filename + f'-seed{cfg.seed}'
    log = logging.getLogger(__name__)
    logging.basicConfig(filename=FILENAME + '.log', encoding='utf-8', level=logging.DEBUG)
    log.debug('This message should go to the log file')
    log.info('So should this')
    log.warning('And this, too')
    torch.set_printoptions(precision=10)

    log.info(f'{cfg.dataset._component_}')
    log.info(f'{cfg.model._component_}')


    recipe = LoRAFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    print('Starting..')
    recipe_main()
