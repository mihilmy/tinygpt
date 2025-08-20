import torch
import torch.nn as nn
from torch.nn import functional as F

# Get vocabulary from our training set

with open("./shakespeare.txt", "r") as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
ch_to_idx = {ch: idx for idx, ch in enumerate(vocab)}


def encode(s: str):
    return [ch_to_idx[ch] for ch in s]


def decode(tokens: list[int]):
    return "".join([vocab[idx] for idx in tokens])


# Setup a machine learning tensor (come on we're not fucking around here)

data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data = data[:split]
test_data = data[split:]

# Setup batching

torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batch(split: str):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    context = torch.stack([data[i : i + block_size] for i in ix])
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return context, targets


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: int, targets: torch.Tensor):
        logits = self.token_embedding_table(idx)
        # Reshape such that it can adhere to how pytorch accepts inputs
        B, T, C = logits.shape
        print(B, T, C)
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        # Calculate the loss based on the prediction of tokens
        loss = F.cross_entropy(logits, targets)

        return logits, loss


batch, targets = get_batch("train")
model = BigramLanguageModel(vocab_size)
logits, loss = model(batch, targets)
print(logits.shape, loss)
