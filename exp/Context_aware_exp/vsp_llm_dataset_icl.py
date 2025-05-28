# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union
import random
import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from python_speech_features import logfbank
from scipy.io import wavfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import utils_vsp_llm as custom_utils

logger = logging.getLogger(__name__)


def load_data(label_path, max_keep, min_keep):
    paths = []
    inds = []
    tot = 0
    sizes = []
    labels = []

    over_tot = 0 
    below_tot = 0

    import csv
    with open(label_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            video_path = row[0]
            audio_path = row[1]
            video_frame = int(row[2])
            audio_frame = int(row[3])
            label = row[4]

            if os.path.exists(video_path) and os.path.exists(audio_path):
                if video_frame > max_keep:
                    over_tot = over_tot + 1
                    continue
                if video_frame < min_keep:
                    below_tot = below_tot + 1
                    continue
                inds.append(tot)
                tot = tot + 1
                paths.append((video_path, audio_path))
                sizes.append(video_frame)
                labels.append(label)
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {tot}, skipped {below_tot} short and {over_tot} long "
            f"longest-video-loaded={max(sizes)} frames, shortest-video-loaded={min(sizes)} frames"
        )
    )
    
    return paths, inds, tot, sizes, labels

def load_data_icl(label_path, max_keep, min_keep):
    paths = []
    inds = []
    tot = 0
    sizes = []
    labels = []
    before_texts = []
    after_texts = []
    over_tot = 0 
    below_tot = 0

    import csv
    with open(label_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            video_path = row[0]
            audio_path = row[1]
            video_frame = int(row[2])
            audio_frame = int(row[3])
            label = row[4]
            before_text = row[5]
            after_text = row[6]

            if os.path.exists(video_path) and os.path.exists(audio_path):
                if video_frame > max_keep:
                    over_tot = over_tot + 1
                    continue
                if video_frame < min_keep:
                    below_tot = below_tot + 1
                    continue
                inds.append(tot)
                tot = tot + 1
                paths.append((video_path, audio_path))
                sizes.append(video_frame)
                labels.append(label)
                before_texts.append(before_text)
                after_texts.append(after_text)
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {tot}, skipped {below_tot} short and {over_tot} long "
            f"longest-video-loaded={max(sizes)} frames, shortest-video-loaded={min(sizes)} frames"
        )
    )
    
    return paths, inds, tot, sizes, labels, before_texts, after_texts

def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, 88, 88]
        cloned = np.copy(x)
        length = cloned.shape[0]
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = np.random.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            if t_end > length:
                t_end = length
            cloned[t_start:t_end, :, :] = 0
        return cloned


class VSP_LLM_dataset(FairseqDataset):
    def __init__(
            self,
            sample_rate: float,
            llm_ckpt_path: str,
            label_path: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            stack_order_audio: int=1,
            skip_verify: bool=False,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            is_s2s=False,
            cfg=None
    ):
        self.cfg = cfg
        self.dataset_name = os.path.basename(label_path)[:-4]
        
        self.label_rates = [-1]
        self.modalities = set(modalities)
        
        self.data = []
        if  not cfg.icl_data:
            paths, inds, tot, self.sizes, labels = load_data(label_path, max_keep_sample_size, min_keep_sample_size)


            for i in range(len(paths)):
                item_dic = {}
                item_dic['context'] = None
                item_dic['video_path'] = paths[i][0]
                item_dic['audio_path'] = paths[i][1]
                item_dic['video_frame'] = self.sizes[i]
                item_dic['label'] = labels[i]
                self.data.append(item_dic)
        else:
        #in-context learning
            paths, inds, tot, self.sizes, labels, before_texts, after_texts = load_data_icl(label_path, max_keep_sample_size, min_keep_sample_size)


            for i in range(len(paths)):
                item_dic = {}
                item_dic['context'] = {'before_text': before_texts[i], 'after_text': after_texts[i]}
                item_dic['video_path'] = paths[i][0]
                item_dic['audio_path'] = paths[i][1]
                item_dic['video_frame'] = self.sizes[i]
                item_dic['label'] = labels[i]
                self.data.append(item_dic)

        
            

        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_ckpt_path)
        self.num_labels = len(labels)
        self.single_target = single_target
        self.store_labels = store_labels
        self.is_s2s = is_s2s

        assert self.single_target == (self.label_rates[0] == -1), f"single target should be equivalent to sequence label (label_rate==-1)"

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        if image_aug:
            if self.cfg.image_adaptive_mask == True: 
                self.transform = custom_utils.Compose([
                    custom_utils.Normalize( 0.0,255.0 ),
                    custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                    custom_utils.HorizontalFlip(0.5),
                    AdaptiveTimeMask(10, 25),
                    custom_utils.Normalize(image_mean, image_std) ])
            else:
                self.transform = custom_utils.Compose([
                    custom_utils.Normalize( 0.0,255.0 ),
                    custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                    custom_utils.HorizontalFlip(0.5),
                    custom_utils.Normalize(image_mean, image_std) ])

        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        logger.info(f"image transform: {self.transform}")

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")

    def load_feature(self, mix_name):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        video_fn, audio_fn = mix_name
        if 'video' in self.modalities:
            video_feats = self.load_video(video_fn) # [T, H, W, 1]
        else:
            video_feats = None
        if 'audio' in self.modalities:
            audio_fn = audio_fn.split(':')[0]
            audio_npy = audio_fn.replace('.wav','.npy')
            if os.path.exists(audio_npy) == False:
                sample_rate, wav_data = wavfile.read(audio_fn)
                assert sample_rate == 16_000 and len(wav_data.shape) == 1
                audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
                audio_feats = stacker(audio_feats, self.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
                np.save(audio_npy[:-4], audio_feats)
            else:
                audio_feats = np.load(audio_npy)
                
        else:
            audio_feats = None
        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
            if diff < 0:
                audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
            elif diff > 0:
                audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def load_video(self, video_path):
        feats = custom_utils.load_video(video_path)
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def __getitem__(self, index):

        data = self.data[index]

        context = None
        if self.cfg.icl_data:
            context = data['context']
            if context['before_text'] == '':
                context['before_text'] = '无'
            if context['after_text'] == '':
                context['after_text'] = '无'
            # 截取文本，超出长度才截取，否则保留原文本

            context['before_text'] = context['before_text'][-self.cfg.context_max_length:] if len(context['before_text']) > self.cfg.context_max_length else context['before_text']
            context['after_text'] = context['after_text'][:self.cfg.context_max_length] if len(context['after_text']) > self.cfg.context_max_length else context['after_text']
            

        path = (data['video_path'], data['audio_path'])
        video_feats, audio_feats = self.load_feature(path)
        audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None


        labels = [self.llm_tokenizer( data['label'], return_tensors="pt").input_ids[0]]
        
        fid = data['video_path']



        return {"context": context ,"id": index, 'fid': fid, "video_source": video_feats, 'audio_source': audio_feats, "label_list": labels}



    def __len__(self):
        return len(self.data)

    def crop_to_max_size(self, video, target_size):
        size = len(video)
        diff = size - target_size
        if diff <= 0:
            return video, 0
        # longer utterances
        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return video[start:end], start

    def collater(self, samples):
        try:
            input_data = samples[0]

            batch = {'context': input_data['context']}
            audio_source = input_data['audio_source']
            video_source = input_data['video_source']
            label_source = input_data['label_list']
            fid_source = input_data['fid']
            if audio_source is None:
                collated_audios = None
            if video_source is not None:
                max_video_size = len(video_source)
                collated_videos, padding_mask = self.collater_video([video_source], max_video_size)
            encoder_input = {"source": {"audio": collated_audios, "video": collated_videos}, "padding_mask": padding_mask}


            #  origin label + eos
            label = torch.cat((label_source[0], torch.tensor([self.llm_tokenizer.eos_token_id])))
            prev_label = torch.cat((label[1:], torch.tensor([self.llm_tokenizer.eos_token_id])))
            label_len = label.size()[0]
            label = label.unsqueeze(0)
            prev_label = prev_label.unsqueeze(0)
            target_attn_mask = label != 0

            llm_input = { "video": collated_videos}


            llm_input["label_lengths"] = torch.tensor(label_len)
            llm_input["ntokens"] = label_len

            llm_input['label'], llm_input['prev_output_tokens'] = label, prev_label
            llm_input['label_attn_mask'] = target_attn_mask

            batch['data'] = {'id': input_data['id'], 'fid': input_data['fid'], 'encoder_input': encoder_input, 'llm_input': llm_input}

            return batch
        
        except:
            print('------------error---------------')
            print(samples)
            print('------------error---------------')


    def collater_video(self, videos, video_frame):
        video_shape = list(videos[0].shape[1:]) #[H, W, C]
        collated_videos = videos[0].new_zeros([len(videos), video_frame] + video_shape) #[B, T, H, W, C]
        padding_mask = (
            torch.BoolTensor(len(videos), video_frame).fill_(False) 
        )
        video_starts = [0 for _ in videos]
        for i, video in enumerate(videos):
            diff = len(video) - video_frame
            if diff == 0:
                collated_videos[i] = video
            elif diff < 0:
                assert self.pad_audio
                collated_videos[i] = torch.cat(
                    [video, video.new_full([-diff]+video_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_videos[i], video_starts[i] = self.crop_to_max_size(
                    video, video_frame, video_starts[i]
                )
        if len(videos[0].shape) == 2:
            collated_videos = collated_videos.transpose(1, 2) # [B, T, F] -> [B, F, T]
        else:
            collated_videos = collated_videos.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_videos, padding_mask


    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad
    ):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate # num label per sample
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s: s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens


    def collater_seq_label_llm(self, targets):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        pad, eos = 0, self.llm_tokenizer.eos_token_id
        targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
       
        new_targets = []
        for tar in targets:
            new_targets.append(tar[1:])

        prev_output_tokens = data_utils.collate_tokens(new_targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=False)
        
        
        prev_list = []
        for prev_tokens in prev_output_tokens:
            padding_start_idx = torch.sum(prev_tokens == 0) * -1
            if padding_start_idx == 0:
                prev_list.append(torch.cat((prev_tokens, torch.tensor([2]).long())))
            else:
                prev_tokens[padding_start_idx] = 2
                prev_list.append(torch.cat((prev_tokens, torch.tensor([0]).long())))
        
        prev_output_tokens = torch.stack(prev_list, dim=0)
        return (targets_, prev_output_tokens), lengths, ntokens


    def collater_label(self, targets_by_label):
        targets_list, lengths_list, ntokens_list = [], [], []
        for targets in targets_by_label:
            targets, lengths, ntokens = self.collater_seq_label_llm(targets)
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list


    def collate_tokens(self,
        values,
        pad_idx,
        eos_idxs,
        left_pad=False,
        move_eos_to_beginning=False,
        pad_to_length=None,
        pad_to_multiple=1,
        pad_to_bsz=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

        batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
        res = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst, eos_idx):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                if eos_idx is None:
                    # if no eos_idx is specified, then use the last token in src
                    dst[0] = src[-1]
                else:
                    dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)], eos_idxs[i])
        return res



    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]


