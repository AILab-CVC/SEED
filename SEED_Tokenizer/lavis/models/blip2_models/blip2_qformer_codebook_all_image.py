"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from functools import partial

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from lavis.models.quantize_semantic import VectorQuantizer2 as VectorQuantizer
#from stable_diffusion.ldm.modules.encoders.modules import FrozenCLIPEmbedder
from lavis.models.vit import Block
from transformers import CLIPVisionModelWithProjection


@registry.register_model("blip2_codebook_all_image")
class Blip2QformerCodebookAllImage(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
            self.ln_vision.weight.requires_grad = False
            self.ln_vision.bias.requires_grad = False

        self.recon_s = True
        self.distill_image = True
        self.blocks_for_image = True
        self.distill_text = False
        self.blocks_for_text = True
        self.discrete = True
        self.use_qformer_image = True
        codebook_embed_dim = 32
        n_embed = 8192
        
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
        self.query_tokens.requires_grad = False

        if self.discrete:
            self.quantize = VectorQuantizer(n_embed, codebook_embed_dim, beta=0.25,
                                            remap=None, sane_index_shape=False)

            self.encode_task_layer = nn.Sequential(
                nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.Qformer.config.hidden_size, codebook_embed_dim) # for quantize
            )

            self.decode_task_layer = nn.Sequential(
                nn.Linear(codebook_embed_dim, codebook_embed_dim),
                nn.Tanh(),
                nn.Linear(codebook_embed_dim, self.Qformer.config.hidden_size) # for quantize
            )

        if self.recon_s:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
            self.depth = 4
            self.blocks = nn.ModuleList([
                Block(
                    dim=self.Qformer.config.hidden_size, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                    drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                for i in range(self.depth)])


        if self.distill_image:
            clip_path = 'pretrained/CLIP-ViT-H-14-laion2B-s32B-b79K'
            self.clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_path, torch_dtype=torch.float16).to("cuda")
            for name, param in self.clip_model.named_parameters():
                param.requires_grad = False
                
            image_features_dim = 1024
            num_reverse_token = 1

            if self.blocks_for_image:
                self.pos_embed_image = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
                self.blocks_image = nn.ModuleList([
                    Block(
                        dim=self.Qformer.config.hidden_size, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                        drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                    for i in range(self.depth)])

            if self.use_qformer_image:
                
                self.Reverse_Qformer, self.reverse_tokens = self.init_Qformer(
                    num_reverse_token, self.Qformer.config.hidden_size
                )
                self.Reverse_Qformer.cls = None
                self.Reverse_Qformer.bert.embeddings.word_embeddings = None
                self.Reverse_Qformer.bert.embeddings.position_embeddings = None
                for layer in self.Reverse_Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None
                self.distill_image_proj = nn.Linear(self.Qformer.config.hidden_size, image_features_dim)

            else:
                self.image_down = nn.Sequential(
                   nn.Linear(self.Qformer.config.hidden_size, 256, bias=False),
                   nn.ReLU(),
                   nn.Linear(256, 128, bias=False),
                   nn.ReLU(),
                   nn.Linear(128, 32, bias=False),
                )
                self.distill_image_proj = nn.Linear(num_query_token * 32, image_features_dim)

        if self.distill_text:

            text_features_dim = 768
            self.text_encoder = FrozenCLIPEmbedder()
            
            if self.blocks_for_text:
                self.pos_embed_text = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
                self.blocks_text = nn.ModuleList([
                    Block(
                        dim=self.Qformer.config.hidden_size, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                        drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                    for i in range(self.depth)])

            num_reverse_token_text = 77

            self.Reverse_Qformer_text, self.reverse_tokens_text = self.init_Qformer(
                num_reverse_token_text, self.Qformer.config.hidden_size
            )

            self.Reverse_Qformer_text.cls = None
            self.Reverse_Qformer_text.bert.embeddings.word_embeddings = None
            self.Reverse_Qformer_text.bert.embeddings.position_embeddings = None
            for layer in self.Reverse_Qformer_text.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

            self.distill_text_proj = nn.Linear(self.Qformer.config.hidden_size, text_features_dim)

            
        self.iter = 0
        
        self.mse = torch.nn.MSELoss() 

        self.max_txt_len = max_txt_len

    def calculate_rec_loss(self, rec, target):  
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()
        return rec_loss

    def forward(self, samples):
        image = samples["image"]
        text = samples["caption"]

        with torch.no_grad():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                use_cache=True,
                return_dict=True,
            )

        loss_embed = 0.0

        query_output_down = self.encode_task_layer(query_output.last_hidden_state)
        quant, loss_embed, embed_ind = self.quantize(query_output_down)
        embed_ind = embed_ind.reshape(quant.shape[0], -1)

        query_output_up = self.decode_task_layer(quant)

        loss_recon_s = 0.0
        if self.recon_s:
            pos_embed = self.pos_embed.repeat(query_output_up.shape[0], 1, 1)
            query_output_up_pos = query_output_up + pos_embed
            for blk in self.blocks:
                query_output_up_pos = blk(query_output_up_pos)
            recon_s = query_output_up_pos
            loss_recon_s = self.calculate_rec_loss(recon_s, query_output.last_hidden_state)

        loss_distil_image = 0.0
        if self.distill_image:

            if self.blocks_for_image:
                pos_embed_image = self.pos_embed_image.repeat(query_output_up.shape[0], 1, 1)
                query_output_up_pos_image = query_output_up + pos_embed_image
                for blk in self.blocks_image:
                    query_output_up_pos_image = blk(query_output_up_pos_image)
                
                if self.use_qformer_image:
                    query_atts = torch.ones(query_output_up.size()[:-1], dtype=torch.long).to(image.device)
                    reverse_tokens = self.reverse_tokens.expand(image_embeds.shape[0], -1, -1)
                    reverse_output = self.Reverse_Qformer.bert(
                        query_embeds=reverse_tokens,
                        encoder_hidden_states=query_output_up_pos_image,
                        encoder_attention_mask=query_atts,
                        return_dict=True,
                    )
                    reverse_output = reverse_output.last_hidden_state
                    reverse_output_proj = self.distill_image_proj(reverse_output).squeeze(1)
                else:
                    reverse_output = self.image_down(query_output_up_pos_image)
                    reverse_output = reverse_output.reshape(reverse_output.shape[0], -1)
                    reverse_output_proj = self.distill_image_proj(reverse_output)

            else:
                query_atts = torch.ones(query_output_up.size()[:-1], dtype=torch.long).to(image.device)
                reverse_tokens = self.reverse_tokens.expand(image_embeds.shape[0], -1, -1)
                reverse_output = self.Reverse_Qformer.bert(
                    query_embeds=reverse_tokens,
                    encoder_hidden_states=query_output_up,
                    encoder_attention_mask=query_atts,
                    return_dict=True,
                )
                reverse_output = reverse_output.last_hidden_state
                reverse_output_proj = self.distill_image_proj(reverse_output).squeeze(1)
            
            with torch.no_grad():
                image_features = self.clip_model(image).image_embeds

            loss_distil_image = self.mse(reverse_output_proj, image_features)
            
        loss_distil_text = 0.0
        if self.distill_text:

            query_atts_text = torch.ones(query_output_up.size()[:-1], dtype=torch.long).to(image.device)

            reverse_tokens_text = self.reverse_tokens_text.expand(image_embeds.shape[0], -1, -1)

            if self.blocks_for_text:
                pos_embed_text = self.pos_embed_text.repeat(query_output_up.shape[0], 1, 1)
                query_output_up_pos_text = query_output_up + pos_embed_text
                for blk in self.blocks_text:
                    query_output_up_pos_text = blk(query_output_up_pos_text)

                reverse_output_text = self.Reverse_Qformer_text.bert(
                    query_embeds=reverse_tokens_text,
                    encoder_hidden_states=query_output_up_pos_text,
                    encoder_attention_mask=query_atts_text,
                    return_dict=True,
                )
            else:
                reverse_output_text = self.Reverse_Qformer_text.bert(
                    query_embeds=reverse_tokens_text,
                    encoder_hidden_states=query_output_up,
                    encoder_attention_mask=query_atts_text,
                    return_dict=True,
                )
            reverse_output_text = reverse_output_text.last_hidden_state
            
            with torch.no_grad():
                text_features, attn_mask = self.text_encoder(text)

            reverse_output_text_proj =  self.distill_text_proj(reverse_output_text)

            loss_distil_text = self.mse(reverse_output_text_proj, text_features)

        return BlipOutput(
            loss=loss_embed*5 + loss_distil_image*0.5 + loss_distil_text*0.5 + loss_recon_s*2,
            loss_distil_image=loss_distil_image*0.5,
            loss_distil_text=loss_distil_text*0.5,            
            loss_recon_s=loss_recon_s*2,
            loss_embed=loss_embed*5
        )

    def clip_text_features(self, text):
        with torch.no_grad():
            text_features, attn_mask = self.text_encoder(text)
        return text_features
    

    def get_discrete_tokens(self, image):
        with torch.no_grad():
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_down = self.encode_task_layer(query_output.last_hidden_state)
            quant, loss_embed, embed_ind = self.quantize(query_output_down)
            embed_ind = embed_ind.reshape(quant.shape[0], -1)

        return embed_ind

     def get_discrete_features_for_decode(self, image):
        with torch.no_grad():
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_down = self.encode_task_layer(query_output.last_hidden_state)
            quant, loss_embed, embed_ind = self.quantize(query_output_down)
            embed_ind = embed_ind.reshape(quant.shape[0], -1)
           
            query_output_up = self.decode_task_layer(quant)

        pos_embed_image = self.pos_embed_image.repeat(query_output_up.shape[0], 1, 1)
        query_output_up_pos_image = query_output_up + pos_embed_image
        for blk in self.blocks_image:
            query_output_up_pos_image = blk(query_output_up_pos_image)
        query_output_up = query_output_up_pos_image

        if self.use_qformer_image:
            query_atts = torch.ones(query_output_up.size()[:-1], dtype=torch.long).to(query_output_up.device)
            reverse_tokens = self.reverse_tokens.expand(query_output_up.shape[0], -1, -1)
            reverse_output = self.Reverse_Qformer.bert(
                query_embeds=reverse_tokens,
                encoder_hidden_states=query_output_up,
                encoder_attention_mask=query_atts,
                return_dict=True,
            )
            reverse_output = reverse_output.last_hidden_state
            reverse_output_proj = self.distill_image_proj(reverse_output).squeeze(1)
        else:
            reverse_output = self.image_down(query_output_up)
            reverse_output = reverse_output.reshape(reverse_output.shape[0], -1)
            reverse_output_proj = self.distill_image_proj(reverse_output)
         
        return reverse_output_proj
      
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        query_output_down = self.encode_task_layer(query_output.last_hidden_state)
        quant, loss_embed, embed_ind = self.quantize(query_output_down)
        embed_ind = embed_ind.reshape(quant.shape[0], -1)

        query_output_up = self.decode_task_layer(quant)

        pos_embed = self.pos_embed.repeat(query_output_up.shape[0], 1, 1)
        query_output_up_pos = query_output_up + pos_embed
        for blk in self.blocks:
            query_output_up_pos = blk(query_output_up_pos)
        query_output_up = query_output_up_pos

        return query_output_up, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
