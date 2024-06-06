import os
import pickle

import torch
from model import GPTConfig, GPT
from data_set import *

# Hyperparameters
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

data_dir = ''
out_dir = 'out'
batch_size = 32  # How many batches per training step
init_from = 'scratch'
eval_interval = 50  # Number of iterations to average for evaluation
learning_rate = 1e-3  # 0.001

n_layer = 3
n_head = 12
n_embd = 120
block_size = 128
bias = True
vocab_size = None
dropout = 0.0
# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

train_loader, eval_loader, max_token_value = get_dataloader(
    batch_size, block_size)


if init_from == 'scratch':
    print('Initializing model from scratch')
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    pass
model = model.to(device)
# initialize a GradScaler. If enabled=False scaler is a no-op
# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

@torch.no_grad()
def evaluate_loss(data_set):
    losses = []
    model.eval()
    for x_batch, y_batch in data_set:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        _, loss = model(x_batch, y_batch)
        losses.append(loss.item())
    model.train()
    return torch.mean(torch.tensor(losses))


def train_loop():
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    step = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        _, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if step % eval_interval == 0:
            eval_loss = evaluate_loss(eval_loader)
            print('Step:', step, 'Evaluate Loss:', round(
                eval_loss.item(), 3), 'Train Loss:', round(loss.item(), 3))


train_loop()


print(f"saving checkpoint to {out_dir}")
# Save the model state dictionary
torch.save(model.state_dict(), os.path.join(out_dir, 'ckpt.pt'))
