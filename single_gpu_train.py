import torch
from model import TransformerLanguageModel
from data_set import *

# Hyperparameters
epoch = 1
batch_size = 4  # How many batches per training step
context_length = 64  # Length of the token chunk each batch
learning_rate = 1e-3  # 0.001
eval_iters = 20  # Number of iterations to average for evaluation
# Use GPU if it's available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

train_loader, eval_loader, max_token_value = get_dataloader(
    batch_size, context_length)
model = TransformerLanguageModel(context_length=context_length,
                                 max_token_value=max_token_value, device=device)
model = model.to(device)


@torch.no_grad()
def evaluate_loss(data_set):
    losses = []
    model.eval()
    for x_batch, y_batch in data_set:
        x_batch.to(device)
        y_batch.to(device)
        _, loss = model(x_batch, y_batch)
        losses.append(loss.item())
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

# Generate
model.eval()
start = 'The salesperson'
start_ids = text_to_ids(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(idx_to_text(y[0].tolist()))
print('---------------')
