import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from transformers.utils import is_datasets_available, logging
from transformers.trainer_utils import seed_worker, has_length
from transformers.deepspeed import deepspeed_init

import datasets
from typing import Dict, List
import time
from tqdm import tqdm

from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, DistributedReadingService, SequentialReadingService

import torch.distributed as dist
import json

import os
# from pycocoevalcap.eval import COCOEvalCap
# from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url
import pyrootutils

logger = logging.get_logger(__name__)
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.train.optimization import get_scheduler


@torch.no_grad()
def gather_together(data):
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


class CustomTrainer(transformers.Trainer):
    """Override the Trainer to avoid IterableDatasetShard"""
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            # return DataLoader(
            #     train_dataset,
            #     batch_size=self._train_batch_size,
            #     collate_fn=data_collator,
            #     num_workers=self.args.dataloader_num_workers,
            #     pin_memory=self.args.dataloader_pin_memory,
            # )
            print('Using Dataloader2')
            mp_rs = MultiProcessingReadingService(num_workers=self.args.dataloader_num_workers)
            dist_rs = DistributedReadingService()
            rs = SequentialReadingService(dist_rs, mp_rs)

            return DataLoader2(train_dataset, reading_service=rs)

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            # return DataLoader(
            #     eval_dataset,
            #     batch_size=self.args.eval_batch_size,
            #     collate_fn=data_collator,
            #     num_workers=self.args.dataloader_num_workers,
            #     pin_memory=self.args.dataloader_pin_memory,
            # )
            print('Using Dataloader2')
            mp_rs = MultiProcessingReadingService(num_workers=self.args.dataloader_num_workers)
            dist_rs = DistributedReadingService()
            rs = SequentialReadingService(dist_rs, mp_rs)

            return DataLoader2(eval_dataset, reading_service=rs)

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            # return DataLoader(
            #     test_dataset,
            #     batch_size=self.args.eval_batch_size,
            #     collate_fn=data_collator,
            #     num_workers=self.args.dataloader_num_workers,
            #     pin_memory=self.args.dataloader_pin_memory,
            # )
            print('Using Dataloader2')
            mp_rs = MultiProcessingReadingService(num_workers=self.args.dataloader_num_workers)
            dist_rs = DistributedReadingService()
            rs = SequentialReadingService(dist_rs, mp_rs)

            return DataLoader2(test_dataset, reading_service=rs)

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        print('comput_metrics: ', self.compute_metrics)
        eval_results = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        eval_results_sync = gather_together(eval_results)

        eval_results = []
        for item in eval_results_sync:
            eval_results.extend(item)

        metrics = self.compute_metrics(eval_results=eval_results, coco_gt_root=self.args.coco_caption_root, split='test')

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> List:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None, inference=True)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        generation_config = transformers.GenerationConfig(
            temperature=0.7,
            num_beams=5,
        )

        eval_results = []
        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader)):
            input_ids = inputs['input_ids'].to(device=self.args.device)

            generate_ids = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                max_new_tokens=120,
            )
            generate_ids = generate_ids[0][input_ids.shape[1]:]
            generate_text = self.tokenizer.decode(generate_ids, skip_special_tokens=True)
            eval_results.append({'image_id': inputs['image_id'][0].item(), 'caption': generate_text})

        return eval_results

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        print('optimizer', type(self.optimizer), self.optimizer)
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(self.args.lr_scheduler_type,
                                              optimizer=self.optimizer if optimizer is None else optimizer,
                                              num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                                              num_training_steps=num_training_steps,
                                              min_lr_ratio=self.args.min_lr_ratio)
        return self.lr_scheduler


def compute_metrics(eval_results, coco_gt_root, split='test'):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(eval_results)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval.eval


if __name__ == '__main__':
    predict_path = 'test_epochbest.json'
    coco_gt_root = 'data/coco_caption'

    with open(predict_path, 'r') as f:
        eval_results = json.load(f)

    print(eval_results[0:5])

    compute_metrics(eval_results=eval_results, coco_gt_root=coco_gt_root)
