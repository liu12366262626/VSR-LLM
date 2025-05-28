import sys
sys.path.append('/home/liuzehua/task/VSR/akvsr_plus')
import torch
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.nets_utils import (
    get_subsample,
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
import torch
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict

# from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from vsr2asr.model4.Phase3_vsr2asr_v2.attention import (
    RelPositionMultiHeadedAttention,  # noqa: H301
    CrossMultiHeadedAttention,
    RelPositionCrossMultiHeadedAttention
)
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.transformer.embedding import (
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.repeat import repeat
import torch.nn as nn
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
import logging
def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)







# including VSR frontend + 4 layers Transformer + 4 layers hybird AVEncoder + 2 layers cross-attention
class VSR_frontend(torch.nn.Module):
    def __init__(self,args,
                normalize_before = True, 
                concat_after=False,) -> None:
        super(VSR_frontend, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)  
        # init params
        relu_type = getattr(args, "relu_type", "swish")
        attention_dim = attention_dim=args['adim']
        positional_dropout_rate=args['dropout_rate']
        linear_units=args['eunits']
        dropout_rate=args['dropout_rate']
        attention_heads=args['aheads']
        attention_dropout_rate=args['transformer_attn_dropout_rate']
        zero_triu=getattr(args, "zero_triu", False)
        cnn_module_kernel=args['cnn_module_kernel']
        num_blocks = args['elayers']
        use_cnn_module=args['use_cnn_module']
        macaron_style=args['macaron_style']


        # init vsr frontend
        pos_enc_class = RelPositionalEncoding
        self.frontend = Conv3dResNet(relu_type=relu_type)

        self.embed = torch.nn.Sequential(
            torch.nn.Linear(512, attention_dim),
            pos_enc_class(attention_dim, positional_dropout_rate),  # drop positional information randomly 
        )
        
        # init vsr transformer part
        self.normalize_before = normalize_before
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate) 


        encoder_attn_layer = RelPositionMultiHeadedAttention
        encoder_attn_layer_args = (
            attention_heads,
            attention_dim,
            attention_dropout_rate,
            zero_triu,
        )

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)

        #encoder is equal to multihead transformers
        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                macaron_style,
            ),
        )

        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)





    def forward(self, xs):

        xs = xs.transpose(1, 2)
        B, T, C, H, W = xs.size()
        masks = torch.ones((B, 1, T), dtype=torch.bool ).to(xs.device)
        xs = self.frontend(xs)

        xs = self.embed(xs)#a linear layer ＋ positional code = 2 output

        xs, masks = self.encoders(xs, None)

        xs = xs[0]

        xs = self.after_norm(xs)
        

        return xs