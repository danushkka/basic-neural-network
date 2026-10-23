# Character-Level GPT (from scratch)

A fully functional, character-level Transformer (GPT) built from scratch in PyTorch. 

This project is the culmination of Andrej Karpathy's "Neural Networks: Zero to Hero" course. It starts with a scalar autograd engine (`micrograd`), progresses through Multi-Layer Perceptrons (MLPs), and ends with a complete, optimized Transformer architecture capable of generating coherent Shakespearean text.

## 📂 Project Structure

```text
karpathy-zero-to-hero/
├── src/
│   ├── micrograd/        # Scalar autograd engine (built from scratch)
│   ├── makemore/         # MLP, BatchNorm, and Bigram models
│   └── gpt/              # The final Transformer model
|       ├── output_shakespeare.txt
|       ├── shakespeare.txt
│       └── gpt.py        # Full GPT implementation
├── scripts/              # Training and evaluation entry points
├── checkpoints/          # Saved model weights
└── README.md
```

## 🏗️ Architecture Scheme

The model is a decoder-only Transformer. Here is the complete data flow from raw text to generated output:

```text
[ RAW TEXT ] "First Citizen:"
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 1. TOKENIZATION & BATCHING                             │
│  Text ──► Integers ──► Batches of (16, 128)            │
│  (x = inputs, y = targets shifted by +1)               │
└────────────────────────────────────────────────────────┘
      │
      ▼ x shape: (Batch=16, Time=128) -> Integer IDs
┌────────────────────────────────────────────────────────┐
│ 2. EMBEDDING LAYER                                     │
│  Token Embedding + Position Embedding                  │
│  (16, 128) ──► (16, 128, 384)                          │
└────────────────────────────────────────────────────────┘
      │
      ▼ x shape: (Batch, Time, Channels) = (16, 128, 384)
┌────────────────────────────────────────────────────────┐
│ 3. THE TRANSFORMER BLOCKS (Repeated 6 times)           │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  A. LayerNorm                                    │  │
│  │  B. Multi-Head Attention (6 heads)               │  │
│  │     - Tokens talk to each other (causal mask)    │  │
│  │  C. Residual: x = x + attention(x)               │  │
│  ├──────────────────────────────────────────────────┤  │
│  │  D. LayerNorm                                    │  │
│  │  E. Feed-Forward (384 -> 1536 -> 384)            │  │
│  │     - Tokens "think" individually (ReLU)         │  │
│  │  F. Residual: x = x + ffwd(x)                    │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
      │
      ▼ x shape: (16, 128, 384)
┌────────────────────────────────────────────────────────┐
│ 4. FINAL LAYER NORM & OUTPUT HEAD                      │
│  LayerNorm(x) ──► Linear(384 -> 65)                    │
└────────────────────────────────────────────────────────┘
      │
      ▼
[ LOGITS ] shape: (16, 128, 65)
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 5. LOSS & GENERATION                                   │
│  Training: Cross-Entropy Loss ──► Backprop ──► AdamW   │
│  Generation: Sample from last token logits             │
└────────────────────────────────────────────────────────┘
```

### 🧠 The Four Pillars of the Transformer

| Component | The Job | The Analogy |
|-----------|---------|-------------|
| **Embeddings** | Translate IDs to meaning + position | Giving actors a script and a stage position |
| **Attention** | Tokens share context with each other | Actors talking to each other |
| **Feed-Forward** | Tokens process context individually | Actors going backstage to think |
| **Residuals** | Preserve original signal (gradient highway) | Stage manager ensuring the script isn't lost |
| **LayerNorm** | Keep activations stable (mean 0, std 1) | Sound engineer adjusting the volume |

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA support if you have an NVIDIA GPU)
- Matplotlib (for visualizations)

### Installation
```bash
# Clone the repo
git clone https://github.com/danushkka/karpathy-zero-to-hero.git
cd karpathy-zero-to-hero

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch matplotlib
```

### Run the GPT Model
```bash
python -m src.gpt.v2
```

## ⚙️ Hyperparameters (Current Config)

These settings are optimized for a **4GB VRAM GPU** (like a GTX 1650):

```python
batch_size = 16  # Reduced from 64 to fit VRAM
block_size = 128  # Reduced from 256 to fit VRAM
max_iters = 10000  # Early stopping around 8k-10k to prevent overfitting
n_embd = 384  # Embedding dimension (Karpathy's original)
n_head = 6  # 384 // 6 = 64 head size
n_layer = 6  # Number of Transformer blocks
dropout = 0.2  # Regularization
```

## 📈 Results

After 10,000 training steps on the Tiny Shakespeare dataset:

- **Train Loss:** 1.09
- **Validation Loss:** 1.50
- **Model Size:** ~10.7M parameters

### Generated Sample

> FRIAR LAURENCE:
> O, must hear your wrongs that he big or hate,
> And safe well paying.
>
> GLOUCESTER:
> I am most royal, but whath the state,
> Take power with burning the wancers: urged and
> Whom that twain and advancied strew the
> common willancer took at our tender...

## 🛠️ Built With

- **PyTorch** - The deep learning framework.
- **NumPy** - Numerical computing.
- **Graphviz / Matplotlib** - Visualization of computational graphs and loss curves.

## 🙏 Acknowledgments

- **Andrej Karpathy** for the incredible "Zero to Hero" course.
- The "Attention is All You Need" paper (Vaswani et al., 2017).
- **DeepSeek** for serving as an AI pair-programmer, debugging assistant, and architecture tutor throughout this learning journey.