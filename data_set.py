from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import requests
import tiktoken
import torch


# Load training data
if not os.path.exists('data/sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Using TikToken (Same as GPT3) to tokenize the source text
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
# the maximum value of the tokenized numbers
max_token_value = max(tokenized_text) + 1
# put tokenized text into tensor
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)

# Split train and validation
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]


class SequenceBatch(Dataset):
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __getitem__(self, index):
        return (self.data[index:index + self.context_length],
                self.data[index+1:index + self.context_length + 1])

    def __len__(self):
        return len(self.data)-self.context_length


def get_dataloader(batch_size, context_length):
    torch.manual_seed(123)
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
    return encoding.encode(input_text)


def idx_to_text(input_ids):
    return encoding.decode(input_ids)
