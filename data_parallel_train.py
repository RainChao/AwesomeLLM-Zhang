import torch
import torch.nn as nn
from model import TransformerLanguageModel
from data_set import *


# 总结
# 1. 数据并行模式下，在每个model的forward运行过程，需要保证算子的所有tensor都位于同一个device上。
# 2. distribute_model.forward返回的tensor是位于cuda:0上，而且会将每个modle副本返回的tensor在batch维度进行cat:
#    例子1: model0返回scalar0, model1返回scalar1, distribute_model返回[scalar0, scalar1]
#    例子2: model0返回tensor0 (shape=[2,3]), model1返回tensor1(shape=[2,3]), distribute_model返回shape=[4,3]

# Hyperparameters
epoch = 1
batch_size = 128  # How many batches per training step
# Length of the token chunk each batch
context_length = 64
learning_rate = 1e-3  # 0.001
eval_iters = 20  # Number of iterations to average for evaluation
# Use GPU if it's available.
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

train_loader, eval_loader, max_token_value = get_dataloader(
    batch_size, context_length)
model = TransformerLanguageModel(context_length=context_length,
                                 max_token_value=max_token_value, device=device)
model = model.to(device)
model = nn.DataParallel(model)  # 就在这里wrap一下，模型就会使用所有的GPU


@torch.no_grad()
def evaluate_loss(data_set):
    losses = []
    model.eval()
    for x_batch, y_batch in data_set:
        x_batch.to(device)
        y_batch.to(device)
        _, loss = model(x_batch, y_batch)
        losses.append(loss.mean().item())
    model.train()
    return torch.mean(torch.tensor(losses))


def train_loop():
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    for _ in range(epoch):
        step = 0
        for x_batch, y_batch in train_loader:
            x_batch.to(device)
            y_batch.to(device)
            _, loss = model(x_batch, y_batch)
            loss = loss.mean()  # Mean loss across gpus
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % eval_iters == 0:
                eval_loss = evaluate_loss(eval_loader)
                print('Step:', step, 'Evaluate Loss:', round(
                    eval_loss.item(), 3), 'Train Loss:', round(loss.item(), 3))
        print('Epoch={} end, Start next Epoch={}'.format(epoch, epoch+1))


train_loop()
# Save the model state dictionary
torch.save(model.state_dict(), 'model-ckpt.pt')
