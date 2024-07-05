#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions that load the three datasets
for training and evaluating multitask BERT.

Feel free to edit code in this file if you wish to modify the way in which the data
examples are preprocessed.
'''

import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import BertTokenizer


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        # TODO: Make this available offline
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # no-internet connection problem

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sents': sents,
            'sent_ids': sent_ids
        }

        return batched_data


# Unlike SentenceClassificationDataset, we do not load labels in SentenceClassificationTestDataset.
class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sents': sents,
            'sent_ids': sent_ids
        }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])
        if self.isRegression:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels, sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
            'token_ids_1': token_ids,
            'token_type_ids_1': token_type_ids,
            'attention_mask_1': attention_mask,
            'token_ids_2': token_ids2,
            'token_type_ids_2': token_type_ids2,
            'attention_mask_2': attention_mask2,
            'labels': labels,
            'sent_ids': sent_ids
        }

        return batched_data


# Unlike SentencePairDataset, we do not load labels in SentencePairTestDataset.
class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
            'token_ids_1': token_ids,
            'token_type_ids_1': token_type_ids,
            'attention_mask_1': attention_mask,
            'token_ids_2': token_ids2,
            'token_type_ids_2': token_type_ids2,
            'attention_mask_2': attention_mask2,
            'sent_ids': sent_ids
        }

        return batched_data


def load_multitask_data(sentiment_filename, paraphrase_filename, similarity_filename, split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent, sent_id))
    else:
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label, sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])), sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        , sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        np.float32(record['similarity']), sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data


def create_train_dataloaders(args):
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train,
                                                                                      args.sts_train, split='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size, num_workers=8,
                                      drop_last=True,
                                      persistent_workers=True, prefetch_factor=4, collate_fn=sst_train_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=False, batch_size=args.batch_size, num_workers=8,
                                       drop_last=True,
                                       persistent_workers=True, prefetch_factor=4,
                                       collate_fn=para_train_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=False, batch_size=args.batch_size, num_workers=8,
                                      drop_last=True,
                                      persistent_workers=True, prefetch_factor=4, collate_fn=sts_train_data.collate_fn)

    return sst_train_dataloader, para_train_dataloader, sts_train_dataloader


def create_dev_dataloaders(args):
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = \
        load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    return sst_dev_dataloader, para_dev_dataloader, sts_dev_data, sts_dev_dataloader, num_labels


def create_test_dataloaders(args):
    sst_test_data, num_labels, para_test_data, sts_test_data = \
        load_multitask_data(args.sst_test, args.para_test, args.sts_test, split='test')

    sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
    sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sst_test_data.collate_fn)

    para_test_data = SentencePairTestDataset(para_test_data, args)
    para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_test_data.collate_fn)

    sts_test_data = SentencePairTestDataset(sts_test_data, args)
    sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sts_test_data.collate_fn)

    return sst_test_dataloader, para_test_dataloader, sts_test_dataloader
