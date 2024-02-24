import torch
import peft
import os
import transformers
from typing import Optional
from dataclasses import dataclass, field
import hydra
from omegaconf import OmegaConf
import pyrootutils

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)


@dataclass
class Arguments:
    model_cfg: Optional[str] = field(default=None, metadata={"help": "config path of model used to initialize model"})
    tokenizer_cfg: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer"})
    base_model: Optional[str] = field(default=None, metadata={"help": "ckpt path of LLM model"})
    lora_model: Optional[str] = field(default=None, metadata={"help": "ckpt path of lora model"})
    save_path: Optional[str] = field(default=None, metadata={"help": "save path of unloaded model"})


parser = transformers.HfArgumentParser(Arguments)
args, = parser.parse_args_into_dataclasses()


def main():
    assert args.lora_model is not None
    model_cfg = OmegaConf.load(args.model_cfg)
    if args.base_model is not None:
        model_cfg.model.pretrained_model_name_or_path = args.base_model

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    tokenizer_cfg = OmegaConf.load(args.tokenizer_cfg)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    peft_model = hydra.utils.instantiate(model_cfg, model_id=args.lora_model, tokenizer=tokenizer)
    print('Load model done!')
    merged_model = peft_model.merge_and_unload()
    print('Prepare to save.')
    merged_model.save_pretrained(args.save_path)
    print('Done!')


if __name__ == '__main__':
    main()


# python3 src/tools/merge_lora_weights.py --model_cfg configs/model/vicuna_7b_lora_pretrained.yaml --tokenizer_cfg configs/tokenizer/seed_llama_tokenizer.yaml --base_model pretrained/vicuna-7b-v1.1 --lora_model log/seed_vicuna-7b_lora_pretrain/checkpoint-10000 --save_path log/seed_vicuna-7b_lora_pretrain/checkpoint-merged-10000
