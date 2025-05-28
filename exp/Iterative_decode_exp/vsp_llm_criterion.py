# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import editdistance

@register_criterion("decoder_only_language_modeling_loss", dataclass=FairseqDataclass)
class decoder_only_language_modeling_loss(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        loss1, loss1_lprob, loss2, loss2_lprob, loss3, loss3_lprob = model(**sample)


        data = sample['data']['llm_input']

        sample_size = (
            data["label"].size()[0]
        )

        logging_output = {
            "loss1": loss1.data,
            "loss2": loss2.data,
            "loss3": loss3.data,
            "ntokens": data["ntokens"],
            "nsentences": data["label"].size(0),
            "sample_size": sample_size,
        }


        n_correct1, total = self.compute_accuracy(loss1_lprob, data)
        logging_output["n_correct1"] = utils.item(n_correct1.data)

        n_correct2, _ = self.compute_accuracy(loss2_lprob, data)
        logging_output["n_correct2"] = utils.item(n_correct2.data)

        n_correct3, _ = self.compute_accuracy(loss3_lprob, data)
        logging_output["n_correct3"] = utils.item(n_correct3.data)

        logging_output["total"] = utils.item(total.data)


        loss = loss1 + loss2 + loss3
        return loss, sample_size, logging_output

    
    def compute_accuracy(self, lprobs, data):
        target = data['prev_output_tokens']
        
        b,t = target.size()
        mask = data['label_attn_mask'] == 1
        n_correct = torch.sum(lprobs[:,-t:].argmax(2).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)

        return n_correct, total

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum1 = sum(log.get("loss1", 0) for log in logging_outputs)
        loss_sum2 = sum(log.get("loss2", 0) for log in logging_outputs)
        loss_sum3 = sum(log.get("loss3", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss1", loss_sum1 / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss2", loss_sum2 / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss3", loss_sum3 / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss", loss_sum3 / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct1 = utils.item(
                sum(log.get("n_correct1", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct1", n_correct1)
            metrics.log_derived(
                "accuracy1",
                lambda meters: round(
                    meters["n_correct1"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
            n_correct2 = utils.item(
                sum(log.get("n_correct2", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct2", n_correct2)
            metrics.log_derived(
                "accuracy2",
                lambda meters: round(
                    meters["n_correct2"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
            n_correct2 = utils.item(
                sum(log.get("n_correct3", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct3", n_correct2)
            metrics.log_derived(
                "accuracy3",
                lambda meters: round(
                    meters["n_correct3"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct3"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False









# # Copyright (c) Facebook, Inc. and its affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# import math
# import re
# from dataclasses import dataclass, field
# from typing import List, Optional

# import torch
# import torch.nn.functional as F
# from fairseq import metrics, utils
# from fairseq.criterions import FairseqCriterion, register_criterion
# from fairseq.dataclass import FairseqDataclass
# import editdistance

# @register_criterion("decoder_only_language_modeling_loss", dataclass=FairseqDataclass)
# class decoder_only_language_modeling_loss(FairseqCriterion):
#     def __init__(self, task):
#         super().__init__(task)


#     def forward(self, model, sample):
#         """Compute the loss for the given sample.
#         Returns a tuple with three elements:
#         1) the loss
#         2) the sample size, which is used as the denominator for the gradient
#         3) logging outputs to display while training
#         """
        
#         result = model(**sample)

#         data = sample['data']['llm_input']

#         sample_size = (
#             data["label"].size()[0]
#         )

#         logging_output = {
#             "ntokens": data["ntokens"],
#             "nsentences": data["label"].size(0),
#             "sample_size": sample_size,
#             "iter_time": len(result)
#         }



#         for i in range(len(result)):
#             temp_loss = result[f'iter_{i+1}'].loss
#             loss_lprob = result[f'iter_{i+1}'].logits
#             logging_output[f"iter_{i + 1}_loss"] = temp_loss

#             n_correct, total = self.compute_accuracy(loss_lprob, data)
#             logging_output[f"n_correct_{i+1}"] = utils.item(n_correct.data)
#             logging_output["total"] = utils.item(total.data)

#         # for i in range(len(result)):
#         loss = result[f'iter_1'].loss + result[f'iter_2'].loss + result[f'iter_3'].loss + result[f'iter_4'].loss



#         return loss, sample_size, logging_output

    
#     def compute_accuracy(self, lprobs, data):
#         target = data['prev_output_tokens']
        
#         b,t = target.size()
#         mask = data['label_attn_mask'] == 1
#         n_correct = torch.sum(lprobs[:,-t:].argmax(2).masked_select(mask).eq(target.masked_select(mask)))
#         total = torch.sum(mask)

#         return n_correct, total

#     @staticmethod
#     def reduce_metrics(logging_outputs) -> None:
#         """Aggregate logging outputs from data parallel training."""
#         iter_time = logging_outputs[0]['iter_time']
#         sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
#         metrics.log_derived(
#             "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
#         )
#         total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
#         metrics.log_scalar("total", total)
#         for i in range(iter_time):

#             loss_sum = sum(log.get(f"iter_{i + 1}_loss", 0) for log in logging_outputs)
#             metrics.log_scalar(
#                 f"loss_iter_{i+1}", loss_sum / sample_size / math.log(2), sample_size, round=3
#             )
#             n_correct = utils.item(
#                 sum(log.get(f"n_correct_{i+1}", 0) for log in logging_outputs)
#             )
#             metrics.log_scalar(f"n_correct_{i + 1}", n_correct)
#             metrics.log_derived(
#                 f"accuracy_iter_{i + 1}",
#                 lambda meters: round(
#                     meters[f"n_correct_{i+1}"].sum * 100.0 / meters["total"].sum, 3
#                 )
#                 if meters["total"].sum > 0
#                 else float("nan"),
#             )


#             metrics.log_scalar(
#                 "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
#             )
#             metrics.log_derived(
#                 "accuracy",
#                 lambda meters: round(
#                     meters[f"n_correct_{i+1}"].sum * 100.0 / meters["total"].sum, 3
#                 )
#                 if meters["total"].sum > 0
#                 else float("nan"),
#             )
            









#     @staticmethod
#     def logging_outputs_can_be_summed() -> bool:
#         """
#         Whether the logging outputs returned by `forward` can be summed
#         across workers prior to calling `reduce_metrics`. Setting this
#         to True will improves distributed training speed.
#         """
#         return False

