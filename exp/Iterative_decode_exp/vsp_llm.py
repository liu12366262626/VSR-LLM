# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Any
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from einops import repeat

from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from omegaconf import II, MISSING
from transformers import AutoTokenizer
from attention import RelPositionCrossMultiHeadedAttention
import torch.nn.functional as F
from joblib import load
import random
logger = logging.getLogger(__name__)


MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)



#model config
@dataclass
class VSPLLMConfig(FairseqDataclass):
    # save config
    train_iter_time: int = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    icl_data: bool = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    save_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    apply_cluster: bool = field(
        default=True, metadata={"help": "if True ,use cluster to apply downsample"}
    )
    pretrained_model: Optional[str] = None
    visual_encoder_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    visual_decoder_path: str = field(
        default=MISSING, metadata={"help": "path to llama model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
                    "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
                    "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
                    "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    masking_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None
    encoder_embed_dim: int = field(
        default=1024, metadata={"help": "encoder embedding dimension"}
    )
    decoder_embed_dim: int = field(
        default=4096, metadata={"help": "decoder embedding dimension"}
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )

class HubertEncoderWrapper(FairseqEncoder):
    def __init__(self, w2v_model):
        super().__init__(None)
        self.w2v_model = w2v_model

    def forward_(self, source, padding_mask, **kwargs):
        src ={}
        src['video'] = source
        src['audio'] = None
        w2v_args = {
            "source": src,
            "padding_mask": padding_mask,
        }

        x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask
        }


    def forward(self, source, padding_mask, **kwargs):
            w2v_args = {
                "source": source,
                "padding_mask": padding_mask,
            }
            

            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)

            return x
            # return {
            #     "encoder_out": x,  # T x B x C
            #     "encoder_padding_mask": padding_mask,  # B x T
            #     "padding_mask": padding_mask
            # }
 

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class Encoders:
    def __init__(self, encoder_path, cfg):
        self.cfg = cfg
        self.visual_encoder_path = encoder_path
        self.dic_encoder = {
            'conformer': self.load_conformer,
            'avhubert': self.load_avhubert
        }

    def get_encoder(self):
        # 遍历字典的 key 和 value
        for key, value in self.dic_encoder.items():
            if key in self.visual_encoder_path:
                return value()

    # conformer train on cncvs + single
    def load_conformer(self):
        import sys
        sys.path.append('../../checkpoints')
        from visual_encoder.conformer.conformer import VSR_frontend
        """Build a new model instance."""
        import yaml
        # 打开并读取 YAML 文件
        with open('../../checkpoints/visual_encoder/conformer/conformer.yaml', 'r', encoding='utf-8') as file:
            vsr_frontend_cfg = yaml.safe_load(file)
        checkpoint_path = self.visual_encoder_path
        ckpt = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
        temp = []
        modified_dict = {}
        for key, value in ckpt.items():
            if 'encoder' not in key:
                continue
            key = key[8:]
            modified_dict[key] = value
            temp.append((key,value))

        encoder = VSR_frontend(vsr_frontend_cfg['visual_backbone'])

        result = encoder.load_state_dict(modified_dict, strict= True)

        logging.info(f'load visual encoder from {self.visual_encoder_path}')

        return encoder
    #avhubert train on English
    def load_avhubert(self):
        cfg = self.cfg
        """Build a new model instance."""

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.visual_encoder_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )
        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()

        return encoder


class Decoders:
    def __init__(self, decoder_path):
        self.visual_decoder_path = decoder_path
        self.dic_decoder = {
            'Atom-7b': self.load_llm,
            'Llama-2-7b-hf': self.load_llm,
            'Qwen2.5-7B': self.load_llm,
            'Qwen2.5-14B' : self.load_llm,
        }

    def get_decoder(self):
        # 遍历字典的 key 和 value
        for key, value in self.dic_decoder.items():
            if key in self.visual_decoder_path:
                return value(self.visual_decoder_path)

    # conformer train on cncvs + single
    def load_llm(self, decoder_path):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        decoder_4bit = AutoModelForCausalLM.from_pretrained(decoder_path, quantization_config=bnb_config)            

        config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "v_proj", "k_proj"], 
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM" 
        )

        decoder_4bit = get_peft_model(decoder_4bit, config)
        decoder_4bit.print_trainable_parameters()

        logging.info(f'load visual decoder from {decoder_path}')

        return decoder_4bit





@register_model("vsp_llm", dataclass=VSPLLMConfig)
class avhubert_llm_seq2seq_cluster_count(BaseFairseqModel):
    def __init__(self, encoder, decoder, cfg, avfeat_to_llm, cross_attention, proj_encoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder
        self.avfeat_to_llm = avfeat_to_llm
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.visual_decoder_path)
        self.cross_attention = cross_attention
        self.proj_encoder = proj_encoder



    @classmethod
    def build_model(cls, cfg, task):

        if task.cfg.pretrained_model is None:
            encoder = Encoders(cfg.visual_encoder_path, cfg).get_encoder()
            decoder_4bit = Decoders(cfg.visual_decoder_path).get_decoder()
            embedding_dim = decoder_4bit.base_model.model.model.embed_tokens.embedding_dim
            avfeat_to_llm = nn.Linear(768, embedding_dim)
            cross_attention = RelPositionCrossMultiHeadedAttention(n_head=8, n_feat=embedding_dim, dropout_rate=0.1)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
            proj_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        
        else:
                model_override_cfg = {'model':{'visual_decoder_path':cfg.visual_decoder_path, 
                                               'visual_encoder_path':cfg.visual_encoder_path,
                                               'icl_data': False}}
                model, _, _ = checkpoint_utils.load_model_ensemble_and_task([task.cfg.pretrained_model],model_override_cfg,strict=False)
                model = model[0]
                decoder_4bit = model.decoder
                encoder = model.encoder
                avfeat_to_llm = model.avfeat_to_llm
                cross_attention = model.cross_attention
                proj_encoder = model.proj_encoder
        


            
        return avhubert_llm_seq2seq_cluster_count(encoder, decoder_4bit, cfg, avfeat_to_llm, cross_attention, proj_encoder)


    # def forward(self, **kwargs):
    #     ft = self.freeze_finetune_updates <= self.num_updates
    #     context, input_data = kwargs['context'], kwargs['data']


    #     encoder_input, llm_input = input_data['encoder_input'], input_data['llm_input']

    #     with torch.no_grad() if not ft else contextlib.ExitStack():
    #         #这里的output是过了前端＋AV-Hubert之后的feature
    #         output = self.encoder(encoder_input['source']['video'])
    #         # output = self.encoder(**encoder_input)
    
    #     # video embedding
    #     video_embedding = self.avfeat_to_llm(output)
    #     token_vs = self.llm_tokenizer('<|vision_start|>', return_tensors="pt").input_ids[0].unsqueeze(0)
    #     token_ve = self.llm_tokenizer('<|vision_end|>', return_tensors="pt").input_ids[0].unsqueeze(0)
    #     token_vs_emb = self.decoder.model.model.embed_tokens(token_vs)
    #     token_ve_emb = self.decoder.model.model.embed_tokens(token_ve)
    #     video_embedding = torch.cat((token_vs_emb, video_embedding, token_ve_emb), dim=1)

    #     # label embedding
    #     labels = llm_input['label'].clone()
    #     labels_embedding = self.decoder.model.model.embed_tokens(labels)
    #     llm_labels = labels.clone()
    #     llm_labels[llm_labels == 0] = -100
    #     result_llm = {}
    #     decode_iter_target = '无'


    #     for i in range(self.cfg.train_iter_time):


    #         instrcuct_text = f"根据这段信息:{decode_iter_target}\n把视觉信息识别为中文,输入:<|endoftext|>"


    #         B, T, D = video_embedding.size()


    #         input_instruction = self.llm_tokenizer(instrcuct_text, return_tensors="pt").input_ids[0]
    #         input_instruction = input_instruction.unsqueeze(0)
    #         input_instruction_embedding = self.decoder.model.model.embed_tokens(input_instruction)



    #         llm_input_embedding = torch.cat((input_instruction_embedding, video_embedding, labels_embedding), dim=1)

            
    #         _, input_instruction_embedding_t, _ = input_instruction_embedding.size()

    #         target_ids = torch.full((B, T + input_instruction_embedding_t),-100).long().to(labels.device)
    #         llm_labels = torch.cat((target_ids, llm_labels), dim=1)
    #         result_llm[f'iter_{i+1}'] = self.decoder(inputs_embeds=llm_input_embedding, labels=llm_labels, return_dict=True)
            
            
    #         target = llm_input['prev_output_tokens']
    #         b,t = target.size()
    #         mask = llm_input['label_attn_mask'] == 1
    #         decode_iter_token = result_llm[f'iter_{i+1}'].logits[:,-t:].argmax(2).masked_select(mask)
    #         decode_iter_target = self.llm_tokenizer.decode(decode_iter_token)





    #     return result_llm
    
    def forward(self, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates
        context, input_data = kwargs['context'], kwargs['data']


        encoder_input, llm_input = input_data['encoder_input'], input_data['llm_input']

        with torch.no_grad() if not ft else contextlib.ExitStack():
            #这里的output是过了前端＋AV-Hubert之后的feature
            output = self.encoder(encoder_input['source']['video'])
            # output = self.encoder(**encoder_input)
    
        # video embedding
        video_embedding = self.avfeat_to_llm(output)
        token_vs = self.llm_tokenizer('<|vision_start|>', return_tensors="pt").input_ids[0].unsqueeze(0)
        token_ve = self.llm_tokenizer('<|vision_end|>', return_tensors="pt").input_ids[0].unsqueeze(0)
        token_vs_emb = self.decoder.model.model.embed_tokens(token_vs)
        token_ve_emb = self.decoder.model.model.embed_tokens(token_ve)
        video_embedding = torch.cat((token_vs_emb, video_embedding, token_ve_emb), dim=1)



        instrcuct_text = f"根据这段信息:{'无'}\n把视觉信息识别为中文,输入:<|endoftext|>"


        B, T, D = video_embedding.size()


        input_instruction = self.llm_tokenizer(instrcuct_text, return_tensors="pt").input_ids[0]
        input_instruction = input_instruction.unsqueeze(0)
        input_instruction_embedding = self.decoder.model.model.embed_tokens(input_instruction)


        # label embedding
        labels = llm_input['label'].clone()
        labels_embedding = self.decoder.model.model.embed_tokens(labels)


        llm_input_embedding = torch.cat((input_instruction_embedding, video_embedding, labels_embedding), dim=1)
        llm_labels = labels.clone()
        llm_labels[llm_labels == 0] = -100
        
        _, input_instruction_embedding_t, _ = input_instruction_embedding.size()

        target_ids = torch.full((B, T + input_instruction_embedding_t),-100).long().to(labels.device)
        llm_labels = torch.cat((target_ids, llm_labels), dim=1)
        llm_out1 = self.decoder(inputs_embeds=llm_input_embedding, labels=llm_labels, return_dict=True)
        
        
        target = llm_input['prev_output_tokens']
        b,t = target.size()
        mask = llm_input['label_attn_mask'] == 1
        decode_iter1_token = llm_out1.logits[:,-t:].argmax(2).masked_select(mask)
        decode_iter1_target = self.llm_tokenizer.decode(decode_iter1_token)

        instrcuct_text = f"根据这段信息:{decode_iter1_target}\n把视觉信息识别为中文,输入:<|endoftext|>"


        input_instruction = self.llm_tokenizer(instrcuct_text, return_tensors="pt").input_ids[0]
        input_instruction = input_instruction.unsqueeze(0)
        input_instruction_embedding = self.decoder.model.model.embed_tokens(input_instruction)



        llm_input_embedding = torch.cat((input_instruction_embedding, video_embedding, labels_embedding), dim=1)
        llm_labels = labels.clone()

        
        llm_labels[llm_labels == 0] = -100
        
        _, input_instruction_embedding_t, _ = input_instruction_embedding.size()

        target_ids = torch.full((B, T + input_instruction_embedding_t),-100).long().to(labels.device)
        llm_labels = torch.cat((target_ids, llm_labels), dim=1)
        llm_out2 = self.decoder(inputs_embeds=llm_input_embedding, labels=llm_labels, return_dict=True)
        

        decode_iter2_token = llm_out2.logits[:,-t:].argmax(2).masked_select(mask)
        decode_iter2_target = self.llm_tokenizer.decode(decode_iter2_token)

        instrcuct_text = f"根据这段信息:{decode_iter2_target}\n把视觉信息识别为中文,输入:<|endoftext|>"


        input_instruction = self.llm_tokenizer(instrcuct_text, return_tensors="pt").input_ids[0]
        input_instruction = input_instruction.unsqueeze(0)
        input_instruction_embedding = self.decoder.model.model.embed_tokens(input_instruction)



        llm_input_embedding = torch.cat((input_instruction_embedding, video_embedding, labels_embedding), dim=1)
        llm_labels = labels.clone()

        
        llm_labels[llm_labels == 0] = -100
        
        _, input_instruction_embedding_t, _ = input_instruction_embedding.size()

        target_ids = torch.full((B, T + input_instruction_embedding_t),-100).long().to(labels.device)
        llm_labels = torch.cat((target_ids, llm_labels), dim=1)
        llm_out3 = self.decoder(inputs_embeds=llm_input_embedding, labels=llm_labels, return_dict=True)
        

        return llm_out1.loss, llm_out1.logits, llm_out2.loss, llm_out2.logits, llm_out3.loss, llm_out3.logits
    



    @torch.no_grad()
    def generate(self,
                num_beams=20,
                max_length=200,
                min_length=5,
                top_p=0.9,
                repetition_penalty=5.0,
                length_penalty=0.0,
                **kwargs,
                ):

        
        context, input_data = kwargs['context'], kwargs['data']
        target_token = kwargs['target_list']
        device = input_data['llm_input']['video'].device
        encoder_input, llm_input = input_data['encoder_input'], input_data['llm_input']


        #这里的output是过了前端＋AV-Hubert之后的feature
        output = self.encoder(encoder_input['source']['video'])
        # output = self.encoder(**encoder_input)

        # video embedding
        video_embedding = self.avfeat_to_llm(output)
        token_vs = self.llm_tokenizer('<|vision_start|>', return_tensors="pt").input_ids[0].unsqueeze(0)
        token_ve = self.llm_tokenizer('<|vision_end|>', return_tensors="pt").input_ids[0].unsqueeze(0)
        token_vs_emb = self.decoder.model.model.embed_tokens(token_vs)
        token_ve_emb = self.decoder.model.model.embed_tokens(token_ve)
        video_embedding = torch.cat((token_vs_emb, video_embedding, token_ve_emb), dim=1)



        # instruction embedding


        instrcuct_text = f"根据这段信息:{target_token}\n把视觉信息识别为中文,输入:<|endoftext|>"


        print(f'instruct: {instrcuct_text}')
        input_instruction = self.llm_tokenizer(instrcuct_text, return_tensors="pt").input_ids[0]
        input_instruction = input_instruction.unsqueeze(0)
        input_instruction_embedding = self.decoder.model.model.embed_tokens(input_instruction)

        logging.info(instrcuct_text)
        #logging.info('Recognize this speech in Chinese. Input: video_embedding Output: ')
        llm_input = torch.cat((input_instruction_embedding, video_embedding), dim=1)


        self.decoder.config.use_cache = True
        outputs = self.decoder.generate(inputs_embeds=llm_input,
                        top_p=top_p,
                        num_beams=num_beams,
                        max_new_tokens=max_length,
                        min_length=min_length,
                        repetition_penalty=repetition_penalty,
                        do_sample=False,
                        length_penalty=length_penalty,
                        )
        
        return outputs, instrcuct_text


    @torch.no_grad()
    def iter_generate(self,
                num_beams=20,
                max_length=200,
                min_length=5,
                top_p=0.9,
                repetition_penalty=5.0,
                length_penalty=0.0,
                  **kwargs,
                ):
        result = []
        iter_time = kwargs['iter_time']
        target_list = kwargs['target_list']
        sample = {
            'context': kwargs['context'],
            'data': kwargs['data']
        }
        for i in range(iter_time):
            target_list, instrcuct_text = self.generate(target_list=target_list, 
                                    num_beams=num_beams, 
                                    length_penalty=length_penalty,
                                    repetition_penalty = repetition_penalty,
                                    max_length = int(max_length),
                                    min_length = int(min_length),
                                    **sample)
            target_list = self.llm_tokenizer.batch_decode(
                target_list, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            target_list = str(target_list[0])
            print(i)
            result.append([target_list, instrcuct_text])

        
        return result
        




    def get_ctc_target(self, sample):
        return sample["target"], sample["target_lengths"]

    def get_ctc_output(self, encoder_out, sample):
        en_out = encoder_out["encoder_out"]
        logits = self.ctc_proj(en_out)  # T x B x C
        out = utils.log_softmax(logits.float(), dim=-1)
        padding_mask = encoder_out["encoder_padding_mask"]
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def state_dict(self):
        old_state = super().state_dict()
        state = {k:v for k,v in old_state.items() if 'lora' in k or 'avfeat_to_llm' in k
                  or 'encoder' in k or 'cross_attention' in k or 'visual_uints' in k
                  or 'proj_encoder' in k}
        return state














