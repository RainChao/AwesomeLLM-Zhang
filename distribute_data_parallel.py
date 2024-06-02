import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from model import TransformerLanguageModel
from data_set import *


# Hyperparameters
epoch = 1
batch_size = 128  # How many batches per training step
# Length of the token chunk each batch
context_length = 64
learning_rate = 1e-3  # 0.001
eval_iters = 20  # Number of iterations to average for evaluation
# Use GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)


def average_gradients(model):
    """Average gradients across processes."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        if isinstance(param, torch.Tensor):
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size


@torch.no_grad()
def evaluate_loss(model, data_set, device):
    """Evaluate loss on a data set."""
    losses = []
    model.eval()
    for x_batch, y_batch in data_set:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        _, loss = model(x_batch, y_batch)
        losses.append(loss.item())
    model.train()
    return torch.mean(torch.tensor(losses))


def train_loop(local_rank):
    """Main train loop for one data parallel process."""
    device = torch.device(local_rank)
    train_loader, eval_loader, max_token_value = partition_dataset(
        batch_size, context_length)
    model = TransformerLanguageModel(context_length=context_length,
                                     max_token_value=max_token_value, device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(epoch):
        step = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            _, loss = model(x_batch, y_batch)
            loss.backward()
            average_gradients(model)
            optimizer.step()
            step += 1
            if (0 == local_rank) and (step % eval_iters == 0):
                eval_loss = evaluate_loss(model, eval_loader, device)
                print('Step:', step, 'Evaluate Loss:', round(
                    eval_loss.item(), 3), 'Train Loss:', round(loss.item(), 3))
    if 0 == local_rank:
        # Save the model state dictionary
        torch.save(model.state_dict(), 'model-ckpt.pt')


def init_processes(local_rank, world_size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=local_rank, world_size=world_size)
    fn(local_rank)


if __name__ == "__main__":
    WORLD_SIZE = 2
    processes = []
    for rank in range(WORLD_SIZE):
        p = Process(target=init_processes, args=(rank, WORLD_SIZE, train_loop))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
