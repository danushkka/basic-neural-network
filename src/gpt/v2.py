import torch
from torch import nn
from torch.nn import functional as F

batch_size = 16  # the number of sequences to be processed in parallel
block_size = 128  # the number of characters to be processed as context
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 192  # dimensions of the "meaning vector"
n_head = 6
n_layer = 4
n_dropout = 0.1

torch.manual_seed(228_1488)

with open("./src/gpt/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data

    # generates 1d tensor containing `batch_size` random numbers
    # from 0 to `len(data) - block_size` - 1
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # take `block_size` number of `x` starting at random index in the sequence
    x = torch.stack([data[i : i + block_size] for i in ix])

    # take `block_size` number of `y` starting at random (index + 1) in the sequence
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(n_dropout)

    def forward(self, x):
        _B, T, C = x.shape
        k = self.key(x)  # B, T, C
        q = self.query(x)  # B, T, C

        # compute attention scores ("affinites")
        # scale them by 1/sqrt(C) so the softmax won't give extreme 0.0 or 1.0
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) => (B, T, T)

        # get a lower-triangular matrix which hides the "future" chars
        # by setting their scores to -inf, so softmax turns them into 0.0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # softmax the weights
        wei = torch.softmax(wei, dim=-1)  # (B, T, T)

        # apply the dropout
        wei = self.dropout(wei)

        # perform the weighted aggregation
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(n_dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # multiply by 4 to create extra space for new specific features
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # preserve only positive values
            nn.Linear(4 * n_embd, n_embd),  # squash back to 32d working size
            nn.Dropout(n_dropout),  # apply the dropout
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForward(n_embd)

        # normalization accross each row
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # gives each symbol in vocab its 32d vector
        self.token_emb_table = nn.Embedding(vocab_size, n_embd)

        # gives each char in the context its 32d vector with its current position
        self.position_emb_table = nn.Embedding(block_size, n_embd)

        # create blocks of MHA and FF
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)],
            nn.LayerNorm(n_embd),
        )

        # hidden layer that translates 32d "meaning vector" into vocab logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # embed tokens into 32d meaning vector
        tok_emb = self.token_emb_table(idx)  # (B, T, C)

        # embed 32d position vector
        pos_emb = self.position_emb_table(torch.arange(T, device=device))  # (T, C)

        # add position vector to the meaning vector for the specific context
        x = tok_emb + pos_emb  # (B, T, C)

        # # apply one layer of self-attention (e.g. Communication layer)
        # x = self.sa_heads(x)  # (B, T, C)

        # # run `x` through feed forward layer (e.g. Computation layer)
        # x = self.ffwd(x)  # (B, T, C)

        # apply blocks of MHA and FF
        x = self.blocks(x)

        # get logits of the specified context
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # flatten the matrixes so the loss function can read it
            # e.g. (32, 8, 65) => (256, 65)
            logits = logits.view(B * T, C)

            targets = targets.view(B * T)  # e.g. (32, 8) => (256,)

            # calculate the loss (cross_entropy() needs 2d tensor)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            # when we call `param = self(x)` PyTorch calls `self.forward(x)`
            # therefore in this line it returns logits, loss for `idx_cond`
            logits, _loss = self(idx_cond)

            # take all batches, the last timestamp (i.e. the 8th char in each batch)
            # and all vocab logits for this char
            logits = logits[:, -1, :]  # (B, C)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
