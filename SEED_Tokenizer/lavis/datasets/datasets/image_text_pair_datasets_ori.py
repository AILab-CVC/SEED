"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


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

    # ori
    # def __getitem__(self, index):

    #     # TODO this assumes image input, not general enough
    #     ann = self.annotation[index]

    #     # try:
    #     #     image_path = ann["image"]
    #     #     #image_path = os.path.join(self.vis_root, ann["image"])
    #     #     image = Image.open(image_path).convert("RGB")
    #     # except:
    #     #     ann = self.annotation[0]
    #     #     image_path = ann["image"]
    #     #     image = Image.open(image_path).convert("RGB")

    #     image_path = ann["image"]
    #     image = Image.open(image_path).convert("RGB")

    #     image = self.vis_processor(image)
    #     caption = self.text_processor(ann["caption"])

    #     #print(image_path, caption)

    #     num_words = len(caption.split(' '))
    #     split_indx = random.randint(0, int(num_words/2)-1)
    #     text_input = ''
    #     for i in range(split_indx+1):
    #         text_input = text_input + caption.split(' ')[i] + ' '
    #     text_input = text_input.strip()
    #     text_output = caption.replace(text_input, '').strip()
    #     #print(1, caption, 2, text_input, 3, text_output)

    #     #return {"image": image, "text_input": caption}
    #     return {"image": image, "text_input": text_input, "text_output": text_output}

    def __getitem__(self, index):
        #index = 204684

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        # try:
        #     image_path = ann["image"]
        #     #image_path = os.path.join(self.vis_root, ann["image"])
        #     image = Image.open(image_path).convert("RGB")
        # except:
        #     ann = self.annotation[0]
        #     image_path = ann["image"]
        #     image = Image.open(image_path).convert("RGB")

        image_path = ann["image"]
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])
        caption = 'Generate an image using the caption: ' + caption
        #print(index, image_path, caption)

        return {"image": image, "text_input": caption, "text_output": caption}