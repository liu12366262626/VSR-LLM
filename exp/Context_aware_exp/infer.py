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
import utils_vsp_llm as custom_utils


from hydra._internal.utils import (
    get_args,
)  # pylint: disable=import-outside-toplevel
cfg_dir = get_args().config_dir
cfg_name = get_args().config_name
print(f'using \ncfg_dir:{cfg_dir}    cfg_name:{cfg_name}')


@dataclass
class OverrideConfig(FairseqDataclass):
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

def process_mp4(mp4_path):

    feats = custom_utils.load_video(mp4_path)
    transform = custom_utils.Compose([
    custom_utils.Normalize( 0.0,255.0 ),
    custom_utils.CenterCrop((88, 88)),
    custom_utils.Normalize(0.421, 0.165) ])
    feats = transform(feats)
    feats = np.expand_dims(feats, axis=-1)
    feats = torch.from_numpy(feats.astype(np.float32))
    feats = feats.unsqueeze(0)
    feats = feats.permute((0, 4, 1, 2, 3)).contiguous()
    feats = feats.half()
    return feats

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
    model.half()


    total_error = 0
    total_token = 0

    
    result = []
    for sample in tqdm(fns, desc=f"TASK_ID:{task_id}   Processing samples"):
        sample['video'] = process_mp4(sample['path'])
        sample = utils.move_to_cuda(sample)
        
    # try:
        best_hypo, input_instruction = model.generate(
                                num_beams=cfg.generation.beam, 
                                length_penalty=cfg.generation.lenpen,
                                repetition_penalty = cfg.override.repetition_penalty,
                                max_length = int(cfg.override.max_length),
                                min_length = int(cfg.override.min_length),
                                **sample)
        torch.cuda.empty_cache()
        best_hypo = tokenizer.batch_decode(
                best_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        temp_dic = {}
        temp_dic['utt_id'] = sample['path']


        ref_sent = sample['label']
        hypo_str = best_hypo[0]
        # input_instruction = tokenizer.decode(sample['net_input']['source']['text'][0].int().cpu().tolist()[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # output_instruction = tokenizer.decode(sample['net_input']['source']['text'][1].int().cpu().tolist()[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # result_dict['instruction'].append((input_instruction, output_instruction))
        # input_instruction = tokenizer.decode(sample['net_input']['source']['text'][i].int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)

        temp_dic['instruction'] = input_instruction
        hypo, ref = hypo_str.strip(), ref_sent.strip()

        cer, error, token =  calculate_cer_details(ref, hypo)
        cer = cer * 100
        error = editdistance.eval(hypo, ref)
        token = len(ref)

        total_error = total_error + error
        total_token = total_token + token
        print(f"\nTASK_ID:{task_id}\nINST:{input_instruction}\nREF:{ref_sent}\nHYP:{hypo_str}\nCER:{cer}%\n")
        temp_dic['ref'] = ref_sent
        temp_dic['context_before'] = sample['context']['before_text'] if model_override_cfg['task']['icl_data'] else 'None'
        temp_dic['hypo'] = hypo_str
        temp_dic['context_after'] = sample['context']['after_text'] if model_override_cfg['task']['icl_data'] else 'None'
        temp_dic['cer'] = cer
        result.append(temp_dic)
    print(f'TASK_ID:{task_id}       CER: {100 * total_error/total_token}%')
    return_dic[task_id] = (total_error, total_token, result)
        
def load_data(label_path, icl_data, context_max_length):
    file = os.path.basename(label_path)
    sample = []
    import csv
    with open(label_path, newline='', encoding='utf-8') as csvfile:
        
        reader = csv.reader(csvfile)
        for row in reader:
            if 'test' not in file:
                video_path = row[0]
                audio_path = row[1]
                video_frame = int(row[2])
                audio_frame = int(row[3])
                label = row[4]
            else:
                video_path = row[0]
                video_frame = int(row[1])
                label = row[2]


            if icl_data :
                if 'test' not in file:
                    before_text = row[5]
                    after_text = row[6]
                else:
                    before_text = row[3]
                    after_text = row[4]  
                context = {}
                context['before_text'] = before_text[context_max_length:] if len(before_text) > context_max_length else before_text
                context['after_text'] = after_text[context_max_length] if len(after_text) > context_max_length else after_text
                
            else:
                context = None

            
            if os.path.exists(video_path):
                sample.append({
                    'path': video_path,
                    'frame_size': video_frame,
                    'label': label,
                    'context': context
                })
    print(f'load total test data: {len(sample)}')
    
    return sample



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


    # model_override_cfg = {'model':{'visual_decoder_path':cfg.override.visual_decoder_path},
    #                     #   'task':{'context_max_length': 100000000000}
    #                       }
    model_override_cfg = {'model':{'visual_decoder_path':cfg.override.visual_decoder_path,
                                   'visual_encoder_path':'/home/liuzehua/task/VSR/VSP-LLM/checkpoints/visual_encoder/conformer/model_avg_cncvs_cnvsrc-single.pth',
                                   },
                            'task':{'icl_data': True,
                                    'context_max_length': 100000000000}}
    
    fns = load_data(os.path.join(cfg.override.data, cfg.dataset.gen_subset) + '.csv', icl_data= model_override_cfg['task']['icl_data'],
                    context_max_length=model_override_cfg['task']['context_max_length'])


    # 您所拥有的GPU列表

    available_gpus = [0,1,2,3,4,5]
    gpu_process_num = [1,1,1,0,0,0]
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


    # 收集所有结果
    sum_infos = []
    sum_error = 0
    sum_total = 0
    for key in return_dict:
        error, total, infos = return_dict[key]
        sum_infos.extend(infos)
        sum_error = sum_error + error
        sum_total = sum_total + total
    print('show')
    dic_cer = {'total_cer': str(float(100 * sum_error / sum_total)) + '%' ,
               'total_error': sum_error,
               'total_token': sum_total}
    sum_infos.insert(0, dic_cer)

    print(f'total cer: {sum_error*100 / sum_total}%')

    with open(cfg.save.infer_result, 'w', encoding='utf-8') as f:
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
