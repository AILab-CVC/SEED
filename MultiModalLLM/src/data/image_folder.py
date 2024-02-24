'''
写一个 Pytorch Dataset，使用 glob 在文件夹 data_root 中找到所有文件，data_root 可能存在子文件夹，使用PIL.Image 打开，并在 getitem 中返回 image_name image_tensor， image_name 为该文件在 data_root 中的相对路径
'''

import torch.utils.data as data
from PIL import Image
import os
from torchvision import transforms
# from torchvision.transforms.functional import pil_to_tensor
import torch


class ImageFolder(data.Dataset):

    def __init__(self, data_root, image_size=256, transform=None) -> None:
        super().__init__()
        self.data_root = data_root
        self.file_paths = []

        for dirpath, dirnames, filenames in os.walk(self.data_root):
            # 遍历该文件夹以及子文件夹中的所有文件
            for filename in filenames:
                # 将每个文件的相对路径添加到列表中
                relative_path = os.path.relpath(os.path.join(dirpath, filename), self.data_root)
                self.file_paths.append(relative_path)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image_path = self.file_paths[index]

        try:
            # print(self.data_root, image_path)
            image_pil = Image.open(os.path.join(self.data_root, image_path)).convert('RGB')
            # image_tensor = pil_to_tensor(image_pil)
            # print(image_tensor.shape)
            image_tensor = self.transform(image_pil)
            # image_tensor = image_tensor.float() / 255.0
        except Exception as e:
            print('Some error: ', e)
            return None

        return {'image_name': image_path, 'image_tensor': image_tensor}


def _filter_out_invalid_data(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
