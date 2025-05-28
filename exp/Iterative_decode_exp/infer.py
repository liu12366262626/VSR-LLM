import multiprocessing

from fairseq import tasks
from transformers import AutoTokenizer
import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace
import pdb
import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    GenerationConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf, MISSING
import sacrebleu
import logging



from hydra._internal.utils import (
    get_args,
)  # pylint: disable=import-outside-toplevel
cfg_dir = get_args().config_dir
cfg_name = get_args().config_name
print(f'using \ncfg_dir:{cfg_dir}    cfg_name:{cfg_name}')


@dataclass
class OverrideConfig(FairseqDataclass):
    iter_time: int = field(default=1, metadata={'help': 'iter time'})
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: ["video"], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})
    eval_bleu: bool = field(default=False, metadata={'help': 'evaluate bleu score'})
    visual_decoder_path: str = field(default=MISSING, metadata={'help': 'path to llama checkpoint'})
    repetition_penalty: float = field(default=0)
    max_length: float = field(default=0)
    min_length: float = field(default=0)

@dataclass
class SaveConfig(FairseqDataclass):
    infer_result: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the inference results"}
    )

@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )
    save: SaveConfig = SaveConfig()  # 添加这个字段



def calculate_cer_details(reference, hypothesis):
    import Levenshtein
    # 移除空格，因为CER是基于字符计算的
    reference = reference.replace(" ", "")
    hypothesis = hypothesis.replace(" ", "")
    
    # 计算编辑距离
    distance = Levenshtein.distance(reference, hypothesis)
    
    # 计算总的字符数
    total_tokens = len(reference)
    
    # 计算CER
    cer = distance / total_tokens
    
    return cer, distance, total_tokens

def task_on_gpu(gpu_id, cfg, fns, return_dic, task_id, model_override_cfg):

    import torch
    import os
    from tqdm import tqdm
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")
    print(f"Running on {torch.cuda.get_device_name(device)}")



    utils.import_user_module(cfg.common)
    tokenizer = AutoTokenizer.from_pretrained(cfg.override.visual_decoder_path)

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path],model_override_cfg,strict=False)
    models = [model.eval() for model in models]
    saved_cfg.task.modalities = cfg.override.modalities
    model = models[0]
    model.encoder.cuda()
    model.avfeat_to_llm.cuda()
    model.cross_attention.cuda()
    model.proj_encoder.cuda()
    model.half()

    
    result = {}
    for i in range(1, cfg.override.iter_time + 1):
        result[f"iter_{i}"] = []
        result[f"iter_{i}_error"] = 0
        result[f"iter_{i}_total"] = 0



    for sample in tqdm(fns, desc=f"TASK_ID:{task_id}   Processing samples"):
        sample = utils.move_to_cuda(sample)
        
        sample['data']['llm_input']['video'] = sample['data']['llm_input']['video'].to(torch.half)
        sample['data']['encoder_input']['source']['video'] = sample['data']['encoder_input']['source']['video'].to(torch.half)


        iter_result = model.iter_generate(target_list='无', 
                                iter_time = cfg.override.iter_time,
                                num_beams=cfg.generation.beam, 
                                length_penalty=cfg.generation.lenpen,
                                repetition_penalty = cfg.override.repetition_penalty,
                                max_length = int(cfg.override.max_length),
                                min_length = int(cfg.override.min_length),
                                **sample)

        for i in range(len(iter_result)):
            best_hypo , instrcuct_text = iter_result[i]
            hypo_str = best_hypo
            llm_data = sample['data']['llm_input']
            temp_dic = {}
            temp_dic['utt_id'] = sample['data']['fid']

            target = llm_data['label'].masked_fill(
                llm_data['label'] == -100, 0
            )
            ref_sent = tokenizer.decode(target.int().cpu().tolist()[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)


            input_instruction = instrcuct_text
            temp_dic['instruction'] = input_instruction
            hypo, ref = hypo_str.strip(), ref_sent.strip()

            cer, error, token =  calculate_cer_details(ref, hypo)
            cer = cer * 100
            error = editdistance.eval(hypo, ref)
            token = len(ref)


            result[f'iter_{i + 1}_error'] = result[f'iter_{i + 1}_error'] + error
            result[f'iter_{i + 1}_total'] = result[f'iter_{i + 1}_total'] + token
            print(f"\nTASK_ID:{task_id}\nINST:{input_instruction}\nREF:{ref_sent}\nHYP:{hypo_str}\nWER:{cer}%\n")
            temp_dic['ref'] = ref_sent
            temp_dic['hypo'] = hypo_str
            temp_dic['cer'] = cer
            result[f'iter_{i + 1}'].append(temp_dic)

    return_dic[task_id] = (result)
        


def load_data(cfg, model_override_cfg):
    import random
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    utils.import_user_module(cfg.common)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path],model_override_cfg,strict=False)
    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)

    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir

    task.cfg.visual_decoder_path = cfg.override.visual_decoder_path
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=cfg.task)

    # Load dataset (possibly sharded)
    cfg.dataset.batch_size = 1
    cfg.dataset.max_tokens = 1000

    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        # num_shards=cfg.distributed_training.distributed_world_size,
        # shard_id=cfg.distributed_training.distributed_rank,
        num_shards=1,
        shard_id=0,        
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=True)


    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    fns = []
    count = 0
    for sample in progress:
        # if count == 4:
        #     break
        count += 1
        fns.append(sample)
    # random.shuffle(fns)
    logging.info(f'load test data {len(fns)}')

    return fns

@hydra.main(config_path=cfg_dir, config_name=cfg_name)
def main(cfg):
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    multiprocessing.set_start_method('spawn', force=True)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    # print("⚠️ Debugger waiting for attachment at port 5679")
    # import debugpy
    # # 设置debugpy监听特定端口
    # debugpy.listen(('0.0.0.0', 5679))  # 使用0.0.0.0允许从任何IP连接
    # debugpy.wait_for_client()  # 代码将在这里暂停，直到调试器连接


    # model_override_cfg = {'model':{'visual_decoder_path':cfg.override.visual_decoder_path,
    #                                'pretrained_model':'/home/liuzehua/task/VSR/VSP-LLM/result/Single-90h/model_v6/pretrained_model/origin_model/model-v6/model/checkpoint_best.pt'},
    #                         'task':{'visual_decoder_path':cfg.override.visual_decoder_path,
    #                                'pretrained_model':'/home/liuzehua/task/VSR/VSP-LLM/result/Single-90h/model_v6/pretrained_model/origin_model/model-v6/model/checkpoint_best.pt'},
                                   
    #                       }
    model_override_cfg = {'model':{'visual_decoder_path':cfg.override.visual_decoder_path,
                                   'visual_encoder_path':'/home/liuzehua/task/VSR/VSP-LLM/checkpoints/visual_encoder/conformer/model_avg_cncvs_cnvsrc-single.pth',
                                   },
                            'task':{'icl_data': False}}
    
    fns = load_data(cfg, model_override_cfg)


    # 您所拥有的GPU列表

    available_gpus = [0,1,2,3,4,5]
    gpu_process_num = [0,0,1,1,1,1]
    split_num = sum(gpu_process_num)

            
    # 使用Python的多进程
    processes = []
    last = 0
    count = 0

    task_path_list = fns 


    for i in range(len(available_gpus)):
        for j in range(gpu_process_num[i]):

            avg = len(task_path_list) // split_num
            remain = len(task_path_list) % split_num
            size = avg + 1 if count < remain else avg
            temp_input = task_path_list[last:last + size]
            last += size
            gpu_id = available_gpus[i]
            count = count + 1
            p = multiprocessing.Process(target=task_on_gpu, args=(gpu_id, cfg, temp_input, return_dict, count, model_override_cfg))
            p.start()
            processes.append(p)
            
    # 等待所有进程完成
    for p in processes:
        p.join()


    for i in range(1, cfg.override.iter_time + 1):
        # 收集所有结果
        sum_infos = []
        sum_error = 0
        sum_total = 0
        for key in return_dict:
            result = return_dict[key]
            infos = result[f'iter_{i}']
            sum_error = result[f'iter_{i}_error'] + sum_error
            sum_total = result[f'iter_{i}_total'] + sum_total
            sum_infos.extend(infos)

        print('show')
        dic_cer = {'total_cer': str(float(100 * sum_error / sum_total)) + '%' ,
                'total_error': sum_error,
                'total_token': sum_total}
        sum_infos.insert(0, dic_cer)

        print(f'total cer: {sum_error*100 / sum_total}%')

        with open(cfg.save.infer_result + f'iter{i}.json', 'w', encoding='utf-8') as f:
            json.dump(sum_infos, f, ensure_ascii=False, indent=4)

def cli_main() -> None:
    from hydra._internal.utils import (
        get_args,
    )  # pylint: disable=import-outside-toplevel

    cfg_name = get_args().config_name or "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    main()  # pylint: disable=no-value-for-parameter

    return 



if __name__ == '__main__': 
    cli_main()
