# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os, glob
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq import metrics, search
from fairseq.data import Dictionary, encoders
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING, II
import numpy as np
from argparse import Namespace

DBG=True if len(sys.argv) == 1 else False


if DBG:
    from vsp_llm_dataset_icl import VSP_LLM_dataset
else:
    from .vsp_llm_dataset_icl import VSP_LLM_dataset

logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# #task config
@dataclass
class VSP_LLM_TrainingConfig(FairseqDataclass):
    train_iter_time: int = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    icl_data: bool = field(
        default=MISSING, metadata={"help": "path to data directory"}
    )
    save_path: str = field(
        default=MISSING, metadata={"help": "path to data directory"}
    )
    
    image_adaptive_mask: bool = field(
        default=False,
        metadata={
            "help": "if true, adapt image_adaptive_mask in trainingset"
        },
    )
    pretrained_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, load from pretrained model",
        },
    )

    data: str = field(
        default=MISSING, metadata={"help": "path to data directory"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    visual_decoder_path: str = field(
        default=MISSING, metadata={"help": "path to llama checkpoint"}
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to keep in training"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to keep in training"},
    )
    max_trim_sample_size: Optional[int] = field(
        default=II("task.max_sample_size"),
        metadata={"help": "max sample size to trim to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys "
            "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    pdb: Optional[bool] = field(
        default=False,
        metadata={"help": "pdb"},
    )
    stack_order_audio: int = field(
        default=1,
        metadata={"help": "concatenate n consecutive audio frames for one step"},
    )
    skip_verify: Optional[bool] = field(
        default=False,
        metadata={"help": "skip verifying label-audio alignment"},
    )
    image_aug: bool = field(default=False, metadata={'help': 'image data augmentation'})
    image_crop_size: int = field(
        default=88, metadata={"help": "image ROI size"})
    image_mean: float = field(
        default=0.421, metadata={"help": "image mean"})
    image_std: float = field(
        default=0.165, metadata={"help": "image std"})
    modalities: Optional[List[str]] = field(default_factory=lambda: ["audio", "video"], metadata={'help': 'modalities to load'})
    is_s2s: bool=field(default=False, metadata={'help': 'seq2seq fine-tuning only'})
    tokenizer_bpe_name: Optional[str] = field(default=None, metadata={'help': 'tokenizer model name'})
    tokenizer_bpe_model: Optional[str] = field(default=None, metadata={'help': 'tokenizer model path'})
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'manifest of noise wav files (one wav file path per line)'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: Optional[str] = field(default='0', metadata={'help': 'noise SNR in audio'})
    noise_num: int = field(default=1, metadata={'help': 'number of noise wav files to mix'})
    fine_tuning: bool = field(default=False, metadata={"help": "set to true if fine-tuning AV-Hubert"})


@register_task("vsp_llm_training", dataclass=VSP_LLM_TrainingConfig)
class VSP_LLM_TrainingTask(FairseqTask):

    cfg: VSP_LLM_TrainingConfig

    def __init__(
        self,
        cfg: VSP_LLM_TrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"AVHubertPretrainingTask Config {cfg}")

        self.fine_tuning = cfg.fine_tuning    
        self.blank_symbol = "<s>"
    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return None
    
    @property
    def dictionaries(self) -> List[Dictionary]:
        return None

    @classmethod
    def setup_task(
        cls, cfg: VSP_LLM_TrainingConfig, **kwargs
    ) -> "Avhubert_Llama_Cluster_Trans_PretrainingTask":
        if cfg.pdb:
            import pdb
            pdb.set_trace()
        return cls(cfg)



    def load_dataset(self, split: str, **kwargs) -> None:
        logger.info(f"Using tokenizer")
        label_path = self.cfg.label_dir + '/' + split + '.csv'
        image_aug = self.cfg.image_aug if split == 'train' or split == 'trainval' else False
        self.datasets[split] = VSP_LLM_dataset(
            sample_rate=self.cfg.sample_rate,
            llm_ckpt_path=self.cfg.visual_decoder_path,
            label_path=label_path,
            label_rates=self.cfg.label_rate,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_trim_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=True,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            stack_order_audio=self.cfg.stack_order_audio,
            skip_verify=self.cfg.skip_verify,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            image_aug=image_aug,
            modalities=self.cfg.modalities,
            is_s2s=self.cfg.is_s2s,
            cfg=self.cfg
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self, indices: np.array, *args, **kwargs
    ) -> np.array:
        return indices
