from peft import (
    LoraConfig,
    PeftModel,
    LoraModel,
    PeftModelForCausalLM,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.utils import _set_trainable, PromptLearningConfig
from peft.utils import PeftConfig

import torch
from transformers import LlamaForCausalLM
from omegaconf import DictConfig
import hydra


class PeftModelWithNullKey(PeftModelForCausalLM):
    def __init__(self, model, peft_config: PeftConfig):
        torch.nn.Module.__init__(self)
        self.peft_config = peft_config
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None
        if getattr(self.peft_config, 'target_modules', None) is None:
            self.base_model = model
        elif isinstance(self.peft_config, PromptLearningConfig):
            self._setup_prompt_encoder()
        else:
            self.base_model = LoraModel(peft_config, model)
        if getattr(self.peft_config, "modules_to_save", None) is not None:
            self.modules_to_save = self.peft_config.modules_to_save
            _set_trainable(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation


class PeftModelWithNullKeyV2(PeftModelForCausalLM):
    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        torch.nn.Module.__init__(self)

        self.base_model = model
        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        self.modules_to_save = None
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        if not isinstance(peft_config, PromptLearningConfig):
            self.peft_config[adapter_name] = peft_config
            if getattr(self.peft_config, 'target_modules', None) is not None:
                self.base_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type](self.base_model, self.peft_config,
                                                                                    adapter_name)
            self.set_additional_trainable_modules(peft_config, adapter_name)
        else:
            self.add_adapter(adapter_name, peft_config)

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation


def get_peft_model_with_trainable_embedding(model, peft_config):

    print('peft config: ', peft_config)
    # print(type(peft_config.target_modules))
    peft_model = get_peft_model(model=model, peft_config=peft_config)

    for name, param in peft_model.get_input_embeddings().named_parameters():
        param.requires_grad = True

    for name, param in peft_model.get_output_embeddings().named_parameters():
        param.requires_grad = True

    peft_model.print_trainable_parameters()

    return peft_model


def get_peft_model_with_resize_embedding(model, peft_config=None, model_id=None, tokenizer=None, torch_dtype='bf16'):
    if torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if isinstance(model, DictConfig):
        model = hydra.utils.instantiate(model, torch_dtype=torch_dtype)

    assert (peft_config is None) + (model_id is None) == 1

    # print(type(peft_config.target_modules))
    if tokenizer is not None:
        print(f'Length of tokenizer and resize embedding: {len(tokenizer)}')
        model.resize_token_embeddings(len(tokenizer))

    if peft_config is not None:
        print('peft config: ', peft_config)
        peft_model = get_peft_model(model=model, peft_config=peft_config)
        peft_model.print_trainable_parameters()

        param_count = 0
        if peft_model.modules_to_save is not None:
            for name, param in peft_model.named_parameters():
                if any(module_name in name for module_name in peft_model.modules_to_save):
                    param_count += param.numel()
                    print(name, param.numel())

    else:
        peft_model = PeftModel.from_pretrained(model=model, model_id=model_id)

    return peft_model


def get_learnable_image_embedding_model(model, peft_config=None, model_id=None, tokenizer=None):
    assert (peft_config is None) + (model_id is None) == 1

    # print(type(peft_config.target_modules))
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    if tokenizer is not None:
        print(f'Length of tokenizer and resize embedding: {len(tokenizer)}')
        model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = model.get_input_embeddings().weight.shape[0]

    mask = torch.ones(new_vocab_size, dtype=torch.bool)
    mask[old_vocab_size:new_vocab_size] = False

    def embed_grad_hook(grad):
        grad[mask, :] = 0.
        return grad

    # def head_grad_hook(grad):
    #     grad[:, mask] = 0.
    #     return grad

    if peft_config is not None:
        for n, p in model.named_parameters():
            p.requires_grad = False
        print('peft config: ', peft_config)
        peft_model = PeftModelWithNullKeyV2(model=model, peft_config=peft_config)
        peft_model.print_trainable_parameters()

        print(f'old_vocab_size: {old_vocab_size}, new_vocab_size: {new_vocab_size}')
        # print(
        #     f'input embed size: {model.get_input_embeddings().weight.shape}, output embed size: {model.get_output_embeddings().weight.shape}'
        # )
        # model.get_input_embeddings().weight.register_hook(embed_grad_hook)
        # model.get_output_embeddings().weight.register_hook(embed_grad_hook)

        for name, module in model.get_input_embeddings().named_children():
            if hasattr(module, 'weight'):
                print(f'Set grad hook to {name}.weight')
                module.weight.register_hook(embed_grad_hook)

        for name, module in model.get_output_embeddings().named_children():
            if hasattr(module, 'weight'):
                print(f'Set grad hook to {name}.weight')
                module.weight.register_hook(embed_grad_hook)

        param_count = 0
        if peft_model.modules_to_save is not None:
            for name, param in peft_model.named_parameters():
                if any(module_name in name for module_name in peft_model.modules_to_save):
                    param_count += param.numel()
                    print(name, param.numel())
    else:
        peft_model = PeftModelWithNullKeyV2.from_pretrained(model=model, model_id=model_id)

    return peft_model


def get_frozen_embedding_model(model, tokenizer=None):
    for name, param in model.get_input_embeddings().named_parameters():
        print(f'{name} will be frozen: {param.numel()}')
        param.requires_grad = False
    return model
