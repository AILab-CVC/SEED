"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random
from random import choice
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import random

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class ImageTextPairDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        captions = ann["caption"]
        if isinstance(captions, list):
            caption = choice(captions)
        else:
            caption = captions
        
        caption = self.text_processor(caption)
        image_path = ann["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        
        return {"image": image, "caption": caption}
