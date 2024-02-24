from torch.utils.data import Dataset
import pandas as pd
import lmdb
import os
import pickle
import torch
import numpy as np

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'


class CC3mLmdbDataset(Dataset):

    def __init__(self, meta_path, lmdb_dir, tokenizer, max_num_tokens=256, reverse_ratio=0.5) -> None:
        super().__init__()
        self.meta_path = meta_path
        self.lmdb_dir = lmdb_dir
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens

        df = pd.read_csv(meta_path, sep='\t', header=None)
        self.captions = df[0].tolist()
        self.img_names = df[1].tolist()

        self.image_lmdb = lmdb.open(lmdb_dir, readonly=True, max_readers=1024, create=False, lock=False)
        self.image_txn = self.image_lmdb.begin(buffers=True)
        self.reverse_ratio = reverse_ratio

    def __del__(self):
        self.image_lmdb.close()

    def __len__(self):
        return len(self.captions)

    def get_item(self, index):
        img_name = self.img_names[index]
        caption = self.captions[index]
        img_ids = self.image_txn.get(img_name.encode('utf-8'))
        if img_ids is None:
            return None, None, None, None

        img_ids = pickle.loads(img_ids)

        img_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in img_ids]) + EOI_TOKEN

        if np.random.uniform(0, 1) < self.reverse_ratio:
            unified_tokens = self.tokenizer.bos_token + img_tokens + caption + self.tokenizer.eos_token
        else:
            unified_tokens = self.tokenizer.bos_token + caption + img_tokens + self.tokenizer.eos_token

        tokenized = self.tokenizer(unified_tokens,
                                   max_length=self.max_num_tokens,
                                   add_special_tokens=False,
                                   truncation=True,
                                   padding='max_length',
                                   return_tensors='pt')

        input_ids = tokenized['input_ids'][0]
        attention_mask = tokenized['attention_mask'][0]

        # bos_idx = torch.where(input_ids == self.tokenizer.bos_token_id)[0][0]

        labels = torch.clone(input_ids)
        # labels[:bos_idx + 1] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels, caption

    def __getitem__(self, index) -> dict:
        input_ids, attention_mask, labels, caption = self.get_item(index)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'caption': caption}


def _filter_out_invalid_data(batch):
    batch = list(filter(lambda x: x['input_ids'] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def _filter_out_invalid_data_with_pad(batch):
    old_size = len(batch)
    batch = list(filter(lambda x: x['input_ids'] is not None, batch))
    new_size = len(batch)
    if old_size != new_size:
        batch += batch[:old_size - new_size]
    return torch.utils.data.dataloader.default_collate(batch)


def get_cc3m_dataset(meta_dir, lmdb_dir, tokenizer, max_num_tokens=512):
    train_set_path = os.path.join(meta_dir, 'cc3m_training_success_full.tsv')
    valid_set_path = os.path.join(meta_dir, 'cc3m_validation_success_full.tsv')
    assert os.path.exists(train_set_path)
    assert os.path.exists(valid_set_path)
    assert os.path.exists(lmdb_dir)

    train_dataset = CC3mLmdbDataset(meta_path=train_set_path,
                                    lmdb_dir=lmdb_dir,
                                    tokenizer=tokenizer,
                                    max_num_tokens=max_num_tokens)
    valid_dataset = CC3mLmdbDataset(meta_path=valid_set_path,
                                    lmdb_dir=lmdb_dir,
                                    tokenizer=tokenizer,
                                    max_num_tokens=max_num_tokens)

    return dict(train_dataset=train_dataset, eval_dataset=valid_dataset, collate_fn=_filter_out_invalid_data_with_pad)
