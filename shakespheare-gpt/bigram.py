import math
import torch
from torch import nn
from torch.nn import functional as F

# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length of predictions?
max_iters = 5000
eval_interval = 100
eval_iters = 200
learning_rate = 3e-4
n_embd = 384
n_heads = 6
n_layer = 6
dropout = 0.2
# ====== End Of Hyperparameters ======

torch.manual_seed(1337)

text = ""
with open("./dataset/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create a set of all the unique characters from the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}  # This is really sexy
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]  # encoder: takes a string and return a list of integers
decode = lambda l: "".join([itos[ch] for ch in l])  # decoder: takes a list of int and return text

# Split dataset into train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be for training
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    random_offset = torch.randint(len(data) - block_size, (batch_size,))
    contexts = torch.stack([data[i:i + block_size] for i in random_offset])
    targets = torch.stack([data[i + 1:i + block_size + 1] for i in random_offset])
    contexts, targets = contexts.to(device), targets.to(device)
    return contexts, targets


# torch.no_grad tells pytorch to not prepare for backpropagation since we don't train anything here
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # Set the mode to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Set it back to training mode
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Produce a query, key, value by passing it through a linear layer
        q = self.query(x)  # (B, T, hs)
        k = self.key(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # do the scaled attention formula refer to attention is all you need paper
        wei = q @ k.transpose(-2, -1) * (1 / math.sqrt(C))  # (B, T, hs) @ (B, hs, T) = (B, T, T) * (1 / sqrt(C)
        # mask the weight so it can't communicate to the future since this is a decoder block
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # normalize it to every column sums up to 1
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # multiply the weight to the value
        out = wei @ v  # (B, T, T) @ (B, T, hs) = (B, T, hs)

        # return the result
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attention_heads = MultiHeadAttention(n_head,
                                                  head_size)  # since we have a lot of heads now split the work evenly
        self.ffn = FeedForward(n_embd)
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Communication
        x = x + self.attention_heads(self.ln_1(x))  # apply one head of self attention. (B, T, C)
        # Calculation
        x = x + self.ffn(self.ln_2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) Tensor of integers
        token_embedding = self.token_embedding_table(idx)  # (B, T, C)
        position_embedding = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_embedding + position_embedding  # (B, T, C)
        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:,
                       -block_size:]  # will always show last -block_size item so -1 will always be the next predicted
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# xb, yb = get_batch("train")
model = BigramLanguageModel()
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# Pytorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training Loops
for iter in range(max_iters):
    # Sample a batch of data
    xb, yb = get_batch("train")
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))

torch.save(model.state_dict(), "./quotesgptv1.pth")