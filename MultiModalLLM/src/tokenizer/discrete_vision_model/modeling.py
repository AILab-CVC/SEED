from .quantizer import VectorQuantizer2
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from transformers import Blip2VisionModel
from torch.nn import functional as F

# from diffusers import StableDiffusionPipeline
from diffusers import StableUnCLIPImg2ImgPipeline
from transformers import CLIPVisionModelWithProjection
from .modeling_clip import CLIPEncoder
from transformers import CLIPVisionConfig, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional

import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class SimpleConfig:

    def __init__(self, in_dim, out_dim, num_token):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tok = num_token


class DiscreteVisionModel(nn.Module):

    def __init__(self,
                 visual_encoder,
                 quant_encoder,
                 quant_decoder,
                 rec_loss,
                 loss_scale_rec=1.0,
                 loss_scale_quant=1.0,
                 loss_scale_contrastive=1.0,
                 tie_projection=True,
                 n_embed=8192,
                 embed_dim=32):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.quant_encoder = quant_encoder
        self.quant_decoder = quant_decoder

        self.rec_loss = rec_loss
        self.loss_scale_rec = loss_scale_rec
        self.loss_scale_quant = loss_scale_quant
        self.loss_scale_contrastive = loss_scale_contrastive
        self.tie_projection = tie_projection

        self.quantizer = VectorQuantizer2(n_embed, embed_dim, beta=0.25, remap=None, sane_index_shape=False, legacy=False)

        self.encode_task_layer = nn.Linear(self.quant_encoder.config.hidden_size, embed_dim)
        self.decode_task_layer = nn.Linear(embed_dim, self.quant_decoder.config.hidden_size)

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

        if self.tie_projection:
            self.post_layernorm = visual_encoder.vision_model.post_layernorm
            self.visual_projection = visual_encoder.visual_projection
        else:
            raise NotImplementedError

        self.frozen_model()

    def frozen_model(self, ):
        self.visual_encoder.eval()
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def forward(self, pixel_values, **kwargs):
        with torch.no_grad():
            vision_output = self.visual_encoder(pixel_values, return_dict=True)

        iti_target = vision_output.image_embeds
        rec_target = vision_output.last_hidden_state

        hidden_state = self.quant_encoder(rec_target, return_dict=True).last_hidden_state
        hidden_state = self.encode_task_layer(hidden_state)

        quant, loss_embed, embed_ind = self.quantize(hidden_state)

        hidden_state = self.decode_task_layer(quant)
        predict_state = self.quant_decoder(hidden_state, return_dict=True).last_hidden_state

        pooled_output = self.post_layernorm(predict_state[:, 0, :])
        predict_embed = self.visual_projection(pooled_output)

        loss_rec = self.rec_loss(predict_state, rec_target)

        loss_iti = self.contrastive_loss(predict_embed, iti_target)

        print('loss_rec: ', loss_rec.item(), 'loss_embed: ', loss_embed.item(), 'loss_iti: ', loss_iti.item())

        loss = self.loss_scale_rec * loss_rec + self.loss_scale_quant * loss_embed + self.loss_scale_contrastive * loss_iti
        # loss = self.loss_scale_rec * loss_rec + self.loss_scale_quant * loss_embed

        return loss

    def contrastive_loss(self, embed_x, embed_y):

        all_embed_x = concat_all_gather(embed_x)
        all_embed_y = concat_all_gather(embed_y)

        logits_per_x = self.logit_scale * all_embed_x @ all_embed_y.t()
        logits_per_y = logits_per_x.t()

        targets = torch.arange(len(logits_per_x)).long().cuda(logits_per_x.device)

        # loss_contrastive = (F.cross_entropy(logits_per_x, targets, label_smoothing=0.1) +
        #                     F.cross_entropy(logits_per_y, targets, label_smoothing=0.1)) / 2
        loss_contrastive = (F.cross_entropy(logits_per_x, targets) + F.cross_entropy(logits_per_y, targets)) / 2

        return loss_contrastive


class CLIPEmbedEmbeddings(nn.Module):

    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size
        self.embed_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels

        self.num_patches = (self.embed_size // self.patch_size)
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        self.patch_embedding = nn.Conv1d(in_channels=self.num_channels,
                                         out_channels=self.embed_dim,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size,
                                         bias=False)

    def forward(self, clip_embed):
        # clip_embed with shape [batch_size, self.num_channels, embed_dim]
        patch_embeds = self.patch_embedding(clip_embed)
        patch_embeds = patch_embeds.permute(0, 2, 1)

        embeddings = patch_embeds + self.position_embedding(self.position_ids)
        return embeddings


class CLIPEmbedEncoder(nn.Module):

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = CLIPEmbedEmbeddings(config)
        self.encoder = CLIPEncoder(config)

    def forward(
        self,
        embed,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        hidden_states = self.embeddings(embed)  # b x n x d

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPEmbedDecoder(nn.Module):

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.encoder = CLIPEncoder(config)
        dim_projection = config.image_size // config.patch_size * config.hidden_size
        self.projection = nn.Linear(dim_projection, config.image_size)

    def forward(
        self,
        embed,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        encoder_outputs = self.encoder(
            inputs_embeds=embed,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.projection(last_hidden_state.flatten(1))

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class DiscreteVisionModelFroClipEmbed(nn.Module):

    def __init__(
        self,
        visual_encoder,
        quant_encoder,
        quant_decoder,
        quantizer,
        rec_loss,
        loss_scale_rec=1.0,
        loss_scale_quant=1.0,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.quant_encoder = quant_encoder
        self.quant_decoder = quant_decoder

        self.rec_loss = rec_loss
        self.loss_scale_rec = loss_scale_rec
        self.loss_scale_quant = loss_scale_quant

        # self.quantizer = VectorQuantizer2(n_embed, embed_dim, beta=0.25, remap=None, sane_index_shape=False, legacy=False)
        self.quantizer = quantizer

        # self.encode_task_layer = nn.Linear(self.quant_encoder.config.hidden_size, self.quantizer.embed_dim)
        # self.decode_task_layer = nn.Linear(self.quantizer.embed_dim, self.quant_decoder.config.hidden_size)
        self.encode_task_layer = nn.Linear(self.quant_encoder.out_dim, self.quantizer.embed_dim)
        self.decode_task_layer = nn.Linear(self.quantizer.embed_dim, self.quant_decoder.in_dim)

        self.frozen_model()

    def frozen_model(self, ):
        self.visual_encoder.eval()
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        with torch.no_grad():
            vision_output = self.visual_encoder(pixel_values, return_dict=True)

        clip_embed = vision_output.image_embeds.unsqueeze(1)

        hidden_state = self.quant_encoder(clip_embed, return_dict=True).last_hidden_state
        hidden_state = self.encode_task_layer(hidden_state)

        quant, loss_embed, embed_ind = self.quantizer(hidden_state)

        hidden_state = self.decode_task_layer(quant)
        predict_embed = self.quant_decoder(hidden_state, return_dict=True).last_hidden_state

        loss_rec = self.rec_loss(predict_embed, clip_embed.squeeze(1))
        # print('loss_embed: ', loss_embed, 'loss_rec: ', loss_rec)

        loss = self.loss_scale_quant * loss_embed + self.loss_scale_rec * loss_rec

        return {
            'loss': loss,
            'loss_quant': loss_embed,
            'loss_rec': loss_rec,
            'predict_embed': predict_embed,
        }


class SimpleEmbedEncoder(nn.Module):

    def __init__(self, in_dim, out_dim, num_token, act_type='no'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_token = num_token

        if act_type == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

        self.linear = nn.Linear(in_dim, out_dim * num_token)

    def forward(
        self,
        embed,
        return_dict: Optional[bool] = None,
    ):
        batch_size = embed.shape[0]
        output = self.linear(embed).view(batch_size, self.num_token, self.out_dim)
        output = self.act(output)
        return BaseModelOutput(
            last_hidden_state=output,
            hidden_states=None,
            attentions=None,
        )


class SimpleEmbedDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, num_token, act_type='no'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_token = num_token

        if act_type == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()
        self.linear = nn.Linear(in_dim * num_token, out_dim)

    def forward(
        self,
        embed,
        return_dict: Optional[bool] = None,
    ):
        batch_size = embed.shape[0]

        embed = self.act(embed)
        output = self.linear(embed.flatten(1))
        return BaseModelOutput(
            last_hidden_state=output.view(batch_size, self.out_dim),
            hidden_states=None,
            attentions=None,
        )
