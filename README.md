## Transformer from scratch

This is a **Transformer** based **Large Language Model (LLM)** which reproduces GPT-2 (124M) on OpenWebText dataset.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), I wrote this demo to show how to train a LLM from scratch using PyTorch. 
The code is very simple and easy to understand. It's a good start point for beginners to learn how to train a LLM.


## Get Started & Reproducing GPT-2

(1). Install dependencies

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

(2). Prepare datasets

Here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes.

(3). Training

Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

``` 
Step: 0 Training Loss: 11.68 Validation Loss: 11.681
Step: 20 Training Loss: 10.322 Validation Loss: 10.287
Step: 40 Training Loss: 8.689 Validation Loss: 8.783
Step: 60 Training Loss: 7.198 Validation Loss: 7.617
Step: 80 Training Loss: 6.795 Validation Loss: 7.353
Step: 100 Training Loss: 6.598 Validation Loss: 6.789
...
```
 
The training loss will decrease as the training goes on. After 5000 iterations, the training will stop and the losses are down to around `2.807`. The model will be saved under name `model-ckpt.pt`.

Feel free to change some of the hyperparameters on the top of the `model.py` file, and see how it affects the training process.


## Other contents in this repo

- inference.py: inference with trained model(model-ckpt.pt)
- distribute_data_parallel.py and data_parallel.py: Show different distributed parallel training methods using pytorch


## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) Andrej Karpathy's famous video tutorial on how to build a GPT model from scratch.
- [Transformers from Scratch](https://blog.matdmiller.com/posts/2023-06-10_transformers/notebook.html) A clear and easy implementation of Andrej's video contents by Mat Miller.
- [Attention is all you need](https://arxiv.org/abs/1706.03762) The original paper of Transformer architecture.
