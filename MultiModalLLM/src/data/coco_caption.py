from torch.utils.data import Dataset
# from pycocotools.coco import COCO
import json
from typing import Any
from collections import defaultdict
from PIL import Image
import os
from torchvision import transforms


class CocoCapEvalDataset(Dataset):
    def __init__(self, ann_path, image_root, transform=None) -> None:
        super().__init__()
        with open(ann_path, 'r') as f:
            self.ann_data = json.load(f)

        self.image_root = image_root
        self.transform = transform
        self.createIndex()

        if transform is None:
            self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])

    def createIndex(self):

        self.img_ids = []
        self.id_to_file_name = {}
        self.img_to_anns = defaultdict(list)

        for img in self.ann_data['images']:
            self.img_ids.append(img['id'])
            self.id_to_file_name[img['id']] = img['file_name']

        for ann in self.ann_data['annotations']:
            self.img_to_anns[ann['image_id']].append(ann['caption'])

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index) -> Any:

        img_id = self.img_ids[index]
        img_name = self.id_to_file_name[img_id]
        caption = self.img_to_anns[img_id]

        img_path = os.path.join(self.image_root, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return {'image': image, 'image_id': img_id, 'caption': caption}


class CocoCapAnn30K(Dataset):
    def __init__(self, ann_path) -> None:
        super().__init__()
        with open(ann_path, 'r') as f:
            self.ann_data = json.load(f)

    def __len__(self) -> int:
        return len(self.ann_data)

    def __getitem__(self, index) -> Any:
        sample = self.ann_data[index]
        return {"image_id": sample['image_id'], "id": sample["id"], "caption": sample["caption"]}
