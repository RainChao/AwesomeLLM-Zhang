from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import os
import requests
import tiktoken
import torch
from random import Random


# Using TikToken (Same as GPT3) to tokenize the source text
local_tokenizer = None


def get_local_tokenizer():
    """Get global tokenizer for this project"""
    global local_tokenizer
    if local_tokenizer is None:
        local_tokenizer = tiktoken.get_encoding("cl100k_base")
    return local_tokenizer


def get_text_data():
    """Load training data"""
    if not os.path.exists('data/sales_textbook.txt'):
        url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
        with open('data/sales_textbook.txt', 'w') as f:
            f.write(requests.get(url, timeout=60).text)

    with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    tokenized_text = get_local_tokenizer().encode(text)
    # the maximum value of the tokenized numbers
    max_token_value = max(tokenized_text) + 1
    # put tokenized text into tensor
    tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)

    # Split train and validation
    split_idx = int(len(tokenized_text) * 0.9)
    train_data = tokenized_text[:split_idx]
    val_data = tokenized_text[split_idx:]
    return train_data, val_data, max_token_value


class SequenceBatch(Dataset):
    """"Generate user defined dataset for DataParallel Training Mode"""
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __getitem__(self, index):
        return (self.data[index:index + self.context_length],
                self.data[index+1:index + self.context_length + 1])

    def __len__(self):
        return len(self.data)-self.context_length


def get_dataloader(batch_size, context_length):
    """"Generate train and valuate data for DataParallel Training Mode"""
    torch.manual_seed(123)
    train_data, val_data, max_token_value = get_text_data()
    train_dataset = SequenceBatch(train_data, context_length)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_dataset = SequenceBatch(val_data, context_length)
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    return (train_loader, eval_loader, max_token_value)


def text_to_ids(input_text):
    """Converts a string to a list of ids"""
    return get_local_tokenizer().encode(input_text)


def idx_to_text(input_ids):
    """Converts a list of ids to a string"""
    return get_local_tokenizer().decode(input_ids)


class Partitoner(Dataset):
    """ds = Partitoner(data, indexes, context_length)"""

    def __init__(self, data, indexes, context_length):
        self.data = data
        self.indexes = indexes
        self.context_length = context_length

    def __getitem__(self, index):
        offset = self.indexes[index]
        return (self.data[offset:offset + self.context_length],
                self.data[offset+1:offset + self.context_length + 1])

    def __len__(self):
        return len(self.indexes)


class DataPartitioner(object):
    """dst = DataPartitioner(data, partition_sizes, context_length, seed=1234)"""

    def __init__(self, data, partition_sizes, context_length, seed=1234):
        self.data = data
        self.partition_sizes = partition_sizes
        self.context_length = context_length
        indexes = [index for index in range(len(data) - self.context_length)]
        rng = Random(seed)
        rng.seed(seed)
        rng.shuffle(indexes)
        self.partition_index = []
        for size in partition_sizes:
            index = int(size*(len(data) - self.context_length))
            self.partition_index.append(indexes[:index])
            indexes = indexes[index:]

    def use(self, partiton: int):
        """default: use all data"""
        return Partitoner(self.data, self.partition_index[partiton], self.context_length)


def partition_dataset(batch_size, context_length):
    """get partitioned data for ddp mode"""
    train_data, eval_dataset, max_token_value = get_text_data()
    world_size = dist.get_world_size()
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    part_traindata = DataPartitioner(
        train_data, partition_sizes, context_length)
    part_train_dataset = part_traindata.use(partiton=dist.get_rank())

    mini_batch_size = batch_size // dist.get_world_size()
    train_loader = DataLoader(
        dataset=part_train_dataset,
        batch_size=mini_batch_size,
        shuffle=False,
        drop_last=True,
    )
    
    eval_dataset = SequenceBatch(eval_dataset, context_length)
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=mini_batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, eval_loader, max_token_value
