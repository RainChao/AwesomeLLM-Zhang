import os
import pickle
import numpy as np
import torch
from model import GPTConfig, GPT
from data_set import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# cluster related hyperparameters
is_main_rank = True
backend = 'nccl'  # nccl or gloo
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # this process will do logging, checkpointing etc.
    is_main_rank = (ddp_rank == 0)
    seed_offset = ddp_rank
torch.manual_seed(1337 + seed_offset)

# train related hyperparameters
dataset = 'openwebtext'
out_dir = 'checkpoints/'
batch_size = 32  # How many batches per training step
init_from = 'scratch'
eval_interval = 80  # Number of iterations to average for evaluation
eval_iters = 20  # Number of iterations
max_iters = 600000
learning_rate = 1e-3  # 0.001

# model related hyperparameters
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024  # Length of the context, in tokens
bias = True
vocab_size = None
dropout = 0.0
# attempt to derive vocab_size from the dataset
meta_vocab_size = None
meta_path = os.path.join('data', dataset, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

# state related variables
best_val_loss = 1e9
iter_num = 0

if init_from == 'scratch':
    print('Initializing model from scratch')
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
model = model.to(device)
# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # unwrap DDP container if needed

# initialize a GradScaler. If enabled=False scaler is a no-op
# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# Use AdamW optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data_dir = os.path.join('data', dataset)
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'),
                         dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'),
                         dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate_loss():
    losses = []
    model.eval()
    for _ in range(eval_iters):
        x_batch, y_batch = get_batch(split='val')
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        _, loss = model(x_batch, y_batch)
        losses.append(loss.item())
    model.train()
    return torch.mean(torch.tensor(losses))


x_batch, y_batch = get_batch(split='train')
while True:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    _, loss = model(x_batch, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    iter_num += 1
    if iter_num == max_iters:
        break
    if iter_num % eval_interval == 0 and is_main_rank:
        eval_loss = evaluate_loss()
        print('Step:', iter_num, 'Evaluate Loss:', round(
              eval_loss.item(), 3), 'Train Loss:', round(loss.item(), 3))
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            print(f"saving checkpoint to {out_dir}")
            checkpoint = {"model": raw_model.state_dict(),
                          "model_args": model_args,
                          "optimizer": optimizer.state_dict(),
                          "iter_num": iter_num,
                          "best_val_loss": best_val_loss}
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    x_batch, y_batch = get_batch(split='train')

if ddp:
    destroy_process_group()
