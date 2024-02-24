"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset
from lavis.datasets.datasets.laion_dataset import LaionDataset
from lavis.datasets.datasets.laion_dataset_400M import LaionDataset400M
from lavis.datasets.datasets.laion_dataset_coco import LaionDatasetCoCo
from lavis.datasets.datasets.mmc4_dataset import MMC4Dataset
from lavis.datasets.datasets.cc3m_dataset_web import CC3MDatasetWeb
from lavis.datasets.datasets.cc12m_dataset_web import CC12MDatasetWeb


@registry.register_builder("coco_llava")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_llava.yaml"
    }
    
@registry.register_builder("coco_pair")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_pair.yaml"
    }
    

@registry.register_builder("coco_brush")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_brush.yaml"
    }
    
@registry.register_builder("conceptual_caption_3m")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m.yaml"
    }


@registry.register_builder("conceptual_caption_12m")
class ConceptualCaption12MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_12m.yaml"
    }


@registry.register_builder("conceptual_caption_3m_web")
class CC3MDatasetWebBuilder(BaseDatasetBuilder):
    train_dataset_cls = CC3MDatasetWeb

    DATASET_CONFIG_DICT = {"default": "configs/datasets/conceptual_caption/defaults_3m_web.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
    
@registry.register_builder("conceptual_caption_12m_web")
class CC12MDatasetWebBuilder(BaseDatasetBuilder):
    train_dataset_cls = CC12MDatasetWeb

    DATASET_CONFIG_DICT = {"default": "configs/datasets/conceptual_caption/defaults_12m_web.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
     
    
@registry.register_builder("sbu_caption")
class SBUCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sbu_caption/defaults.yaml"}


@registry.register_builder("vg_caption")
class VGCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption.yaml"}


@registry.register_builder("laion2B_multi")
class Laion2BMultiBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults_2B_multi.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets

@registry.register_builder("laion400M_multi")
class Laion400MMultiBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset400M

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults_400M_multi.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets

@registry.register_builder("laion_coco_600M")
class LaionCoCo600MBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDatasetCoCo

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults_coco_600M.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
    
   
@registry.register_builder("MMC4")
class MMC4Builder(BaseDatasetBuilder):
    train_dataset_cls = MMC4Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults_mmc4.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
     
    
@registry.register_builder("datacomp_small")
class DatacompSmallMultiBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset400M

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults_datacomp_small.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets