import torch
from model import TransformerLanguageModel
from data_set import *


context_length = 64  # Length of the token chunk each batch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
_, _, max_token_value = get_dataloader(1, context_length)
predict_model = TransformerLanguageModel(context_length=context_length,
                                         max_token_value=max_token_value, device=device)
loaded_state_dict = torch.load('model-ckpt.pt')
new_loaded_state_dict = {}
for param_key in loaded_state_dict:
    new_param_key = param_key[len('module')+1:]
    new_loaded_state_dict[new_param_key] = loaded_state_dict[param_key]
predict_model.load_state_dict(new_loaded_state_dict)
predict_model = predict_model.to(device)
predict_model.eval()
start = 'The salesperson'
start_ids = text_to_ids(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = predict_model.generate(x, max_new_tokens=100)
print('---------------')
print(idx_to_text(y[0].tolist()))
print('---------------')
