"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from functools import partial

from models.eva_vit import create_eva_vit_g
from models.Qformer_casual import BertConfig, BertLMHeadModel

from models.quantize_semantic import VectorQuantizer2 as VectorQuantizer
from models.vit import Block

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
    
def init_Qformer(num_query_token, vision_width, cross_attention_freq=2):
    """init Causal Q-Former and Reverse Q-Former"""
    
    encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    encoder_config.encoder_width = vision_width
    # for Reverse Q-Former
    if num_query_token != 32:
        encoder_config.num_hidden_layers = 4

    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    Qformer = BertLMHeadModel.from_pretrained(
        "bert-base-uncased", config=encoder_config
    )
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens


class SEEDVisualTokenizer(nn.Module):

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        num_query_token=32,
        max_txt_len=32,
    ):
        super().__init__()

        self.visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, vit_precision
            )
        self.ln_vision = LayerNorm(self.visual_encoder.num_features)
        
        self.visual_encoder = self.visual_encoder.eval()

        codebook_embed_dim = 32
        n_embed = 8192

        self.Qformer, self.query_tokens = init_Qformer(num_query_token, self.visual_encoder.num_features)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.quantize = VectorQuantizer(n_embed, codebook_embed_dim, beta=0.25, remap=None, sane_index_shape=False)

        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.Qformer.config.hidden_size, codebook_embed_dim)  # for quantize
        )

        self.decode_task_layer = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, self.Qformer.config.hidden_size)  # for quantize
        )
        self.quantize = self.quantize.eval()
        self.quantize.training = False
            

        text_features_dim = 768
        num_reverse_token = 77
        self.depth = 4

        self.pos_embed_text = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
        self.blocks_text = nn.ModuleList([
            Block(dim=self.Qformer.config.hidden_size,
                    num_heads=12,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)) for i in range(self.depth)
        ])

        self.Reverse_Qformer, self.reverse_tokens = init_Qformer(num_reverse_token, self.Qformer.config.hidden_size)
        self.Reverse_Qformer.cls = None
        self.Reverse_Qformer.bert.embeddings.word_embeddings = None
        self.Reverse_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Reverse_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.distill_text_proj = nn.Linear(self.Qformer.config.hidden_size, text_features_dim)

        self.max_txt_len = max_txt_len

        qformer_path = "pretrained/causal_qformer.pth"
        codebook_path = "pretrained/seed_tokenizer.pth"
        
        checkpoint = torch.load(qformer_path, map_location="cpu")
        state_dict = checkpoint["model"]
        msg = self.load_state_dict(state_dict, strict=False)

        codebook_dict = torch.load(codebook_path, map_location="cpu")["model"]
        msg = self.load_state_dict(codebook_dict, strict=False)

    def predict(self, image):
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            # Causal embeddings
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            query_output_down = self.encode_task_layer(query_output.last_hidden_state)
            # Causal tokens
            quant, _, embed_ind = self.quantize(query_output_down)
            embed_ind = embed_ind.reshape(quant.shape[0], -1)

            query_output_up = self.decode_task_layer(quant)

            pos_embed_text = self.pos_embed_text.repeat(query_output_up.shape[0], 1, 1)
            query_output_up_pos_text = query_output_up + pos_embed_text
            for blk in self.blocks_text:
                query_output_up_pos_text = blk(query_output_up_pos_text)
            query_output_up = query_output_up_pos_text

            query_atts = torch.ones(query_output_up.size()[:-1], dtype=torch.long).to(image.device)

            reverse_tokens = self.reverse_tokens.expand(image_embeds.shape[0], -1, -1)
            reverse_output = self.Reverse_Qformer.bert(
                query_embeds=reverse_tokens,
                encoder_hidden_states=query_output_up,
                encoder_attention_mask=query_atts,
                return_dict=True,
            )
            reverse_output = reverse_output.last_hidden_state
            # Generation embeddings as the input of the stable diffusion UNet
            reverse_output_proj = self.distill_text_proj(reverse_output)

            return embed_ind, reverse_output_proj
