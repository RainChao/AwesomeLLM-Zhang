import torch
from data_set import *
import tiktoken
from model import GPTConfig, GPT
from data_set import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


device = 'cpu'
model_args = {}
checkpoint = torch.load('checkpoints/ckpt.pt', map_location='cpu')
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


start = "What is the answer to life, the universe, and everything?"
#start = "Who are you?"
enc = tiktoken.get_encoding("gpt2")
start_ids = enc.encode(start, allowed_special={"<|endoftext|>"})
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
# run generation
max_new_tokens = 100
with torch.no_grad():
    y = model.generate(x, max_new_tokens, temperature=0.8, top_k=100)
    print(enc.decode(y[0].tolist()))
    print('---------------')
