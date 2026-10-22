from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    batch_size: int = 16  # the number of sequences to be processed in parallel
    block_size: int = 128  # the number of characters to be processed as context
    max_iters: int = 10_000
    eval_interval: int = 1000
    learning_rate: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters: int = 200
    n_embd: int = 384  # dimensions of the "meaning vector"
    n_head: int = 6
    n_layer: int = 6
    n_dropout: float = 0.1


config = GPTConfig()

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
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    # take `block_size` number of `x` starting at random index in the sequence
    x = torch.stack([data[i : i + config.block_size] for i in ix])

    # take `block_size` number of `y` starting at random (index + 1) in the sequence
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])

    x, y = x.to(config.device), y.to(config.device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            _logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.dropout = nn.Dropout(config.n_dropout)

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
        self.proj = nn.Linear(
            config.n_embd, config.n_embd
        )  # retrieves logits from the Heads
        self.dropout = nn.Dropout(config.n_dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # multiply by 4 to create extra space for new specific features
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),  # preserve only positive values
            nn.Linear(
                4 * config.n_embd, config.n_embd
            ),  # squash back to `n_emb` working size
            nn.Dropout(config.n_dropout),  # apply the dropout
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = config.n_embd // config.n_head  # 384 // 16 = 64
        self.sa = MultiHeadAttention(num_heads=config.n_head, head_size=head_size)
        self.ffwd = FeedForward(config.n_embd)

        # normalization accross each row
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class BigramLanguageModel(nn.Module):
    # initialize weights within a small distribution to minimize the initial loss
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def __init__(self):
        super().__init__()

        # gives each symbol in vocab its `n_emb`-dimensional vector
        self.token_emb_table = nn.Embedding(vocab_size, config.n_embd)

        # gives each char in the context its `n_emb`-dimensional vector with its current position
        self.position_emb_table = nn.Embedding(config.block_size, config.n_embd)

        # create blocks of MHA and FF
        self.blocks = nn.Sequential(
            *[
                Block(config.n_embd, n_head=config.n_head)
                for _ in range(config.n_layer)
            ],
            nn.LayerNorm(config.n_embd),
        )

        # hidden layer that translates `n_emb`-dimensional "meaning vector" into vocab logits
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        # apply the custom initialization to all sub-modules
        self.apply(self._init_weights)

        # iterate through every parameter tensor
        # pn - parameter name (e.g. "blocks.2.ffwd.net.0.weight")
        # p - parameter itself
        for pn, p in self.named_parameters():
            # find a tensor that ends with self.proj
            if pn.endswith("c_proj.weight"):
                # apply normalization so that params won't explode after proj (Linear layer)
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5
                )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # embed tokens into `n_emb`-dimensional meaning vector
        tok_emb = self.token_emb_table(idx)  # (B, T, C)

        # embed `n_emb`-dimensional position vector
        pos_emb = self.position_emb_table(
            torch.arange(T, device=config.device)
        )  # (T, C)

        # add position vector to the meaning vector for the embedded context
        x = tok_emb + pos_emb  # (B, T, C)

        # apply blocks of MHA and FF
        x = self.blocks(x)

        # get logits of the embedded context
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # flatten the matrixes so the loss function can read it
            # e.g. (384, 6, 65) => (2304, 65)
            logits = logits.view(B * T, C)

            targets = targets.view(B * T)  # e.g. (384, 6) => (2304,)

            # calculate the loss (cross_entropy() needs 2d tensor)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size :]

            # when we call `param = self(x)` PyTorch calls `self.forward(x)`
            # therefore this line returns logits, loss for `idx_cond`
            logits, _loss = self(idx_cond)

            # take all batches, the last timestamp (i.e. the 8th char in each batch)
            # and all vocab logits for this char
            logits = logits[:, -1, :]  # (B, C)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel()
m = model.to(config.device)

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

for iter in range(config.max_iters + 1):
    if iter % config.eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
