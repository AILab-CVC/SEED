import transformers
from typing import Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import hydra
import pyrootutils
import logging
# from typing import List
# import deepspeed
# from transformers import LlamaForCausalLM
# deepspeed.ops.op_builder.CPUAdamBuilder().load()

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.train.trainer import CustomTrainer, compute_metrics

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)


@dataclass
class ConfigPathArguments:
    model: Optional[str] = field(default=None, metadata={"help": "config path of model used to initialize LM model"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    train_data: Optional[str] = field(default=None, metadata={"help": "config path of train dataset"})
    eval_data: Optional[str] = field(default=None, metadata={"help": "config path of eval dataset"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(
        default=None, metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    overwrite_output_dir: bool = field(default=False, metadata={"help": "Overwrite the content of the output directory"})
    optim: str = field(default="adamw_hf")
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    min_lr_ratio: float = field(
        default=0.1, metadata={"help": "The min lr ratio reqpect to the learning rate, only used to cosine lr scheduler"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1, metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."})

    lr_scheduler_type: str = field(default='cosine', metadata={"help": "The scheduler type to use."})
    # report_to: Optional[List[str]] = field(default=['tensorboard'],
    #                                        metadata={"help": "The list of integrations to report the results and logs to."})
    save_steps: int = field(default=1000, metadata={"help": "The interval between saving the model checkpoint."})
    bf16: bool = field(default=False,
                       metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"})
    fp16: bool = field(default=False,
                       metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"})
    dataloader_num_workers: int = field(default=8, metadata={"help": "The number of workers to use for data loading."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for training."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    run_name: str = field(default=None, metadata={"help": "The name of the run."})

    torch_compile: bool = field(default=False, metadata={"help": "Whether to use torch.jit.trace to compile the model."})
    coco_caption_root: str = field(
        default=None, metadata={"help": "root path of coco karpathy which is used to comput caption metrics during training."})


# from transformers import LlamaForCausalLM


class Trainer(CustomTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_labels = inputs.get('labels', None)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_labels)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ConfigPathArguments, TrainingArguments))
    cfg_path, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    train_data_cfg = OmegaConf.load(cfg_path.train_data)
    model_cfg = OmegaConf.load(cfg_path.model)
    tokenizer_cfg = OmegaConf.load(cfg_path.tokenizer)

    use_peft = 'peft' in model_cfg._target_ or 'lora' in model_cfg._target_
    print('Use peft or not: ', use_peft)

    print('Init tokenizer')
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)
    tokenizer.pad_token = tokenizer.unk_token
    print('Init train data')
    train_data = hydra.utils.instantiate(train_data_cfg, tokenizer=tokenizer)
    print('Init model')
    if use_peft:
        model = hydra.utils.instantiate(model_cfg, tokenizer=tokenizer)
    else:
        model = hydra.utils.instantiate(model_cfg)
        print(f'Length of tokenizer and resize embedding: {len(tokenizer)}')
        model.resize_token_embeddings(len(tokenizer))

    if cfg_path.eval_data is not None:
        eval_data_cfg = OmegaConf.load(cfg_path.eval_data)
        eval_data = hydra.utils.instantiate(eval_data_cfg, tokenizer=tokenizer)
    else:
        eval_data = None

    print('Init done.')
    model.config.use_cache = False

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=data_module['train_dataset'],
    #     eval_dataset=data_module['eval_dataset'],
    #     tokenizer=tokenizer,
    #     data_collator=data_module['collate_fn'],
    # )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # trainer.train(resume_from_checkpoint=True)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # trainer.save_state()
    # trainer.evaluate()


if __name__ == '__main__':
    train()
