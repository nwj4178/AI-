import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import pandas as pd
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")
train_iter = WikiText2(split='train')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def data_process(raw_text_iter):
    return torch.cat([torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter])

def batchify(data, batch_size):
    nbatch = data.size(0) // batch_size
    return data[:nbatch * batch_size].view(batch_size, -1).t().contiguous()

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target

class BaseModel(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, input_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        raise NotImplementedError

    def predict(self, x):
        emb = self.embed(x)
        out = self.forward(emb)
        return self.output(out)

class RNNModel(BaseModel):
    def __init__(self, vocab_size, input_dim, hidden_dim):
        super().__init__(vocab_size, input_dim, hidden_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out

class GRUModel(BaseModel):
    def __init__(self, vocab_size, input_dim, hidden_dim):
        super().__init__(vocab_size, input_dim, hidden_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out

class LSTMModel(BaseModel):
    def __init__(self, vocab_size, input_dim, hidden_dim):
        super().__init__(vocab_size, input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out

class TransformerModel(BaseModel):
    def __init__(self, vocab_size, input_dim, hidden_dim):
        super().__init__(vocab_size, input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True),
            num_layers=1
        )

    def forward(self, x):
        return self.encoder(x)

class MambaModel(BaseModel):
    def __init__(self, vocab_size, input_dim, hidden_dim):
        super().__init__(vocab_size, input_dim, hidden_dim)
        self.linear_u = nn.Linear(input_dim, hidden_dim)
        self.linear_g = nn.Linear(input_dim, hidden_dim)
        self.A = nn.Parameter(torch.eye(hidden_dim) * 0.6)
        self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.C = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.D = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        u = F.gelu(self.linear_u(x)) * torch.sigmoid(self.linear_g(x))
        s = torch.zeros(x.size(0), x.size(1), device=x.device)
        outputs = []
        for t in range(x.size(1)):
            s = torch.matmul(s, self.A.T) + torch.matmul(u[:, t], self.B.T)
            y = torch.matmul(s, self.C.T) + self.D
            outputs.append(y.unsqueeze(1))
        return torch.cat(outputs, dim=1)

class TitanModel(BaseModel):
    def __init__(self, vocab_size, input_dim, hidden_dim):
        super().__init__(vocab_size, input_dim, hidden_dim)
        self.mamba = MambaModel(vocab_size, input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.out_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.memory = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        mamba_out = self.mamba(x)
        attn_out, _ = self.attn(mamba_out, mamba_out, mamba_out)
        out = torch.cat([mamba_out, attn_out], dim=-1)
        return self.out_proj(out)

    def memory_update(self, loss):
        if self.memory.requires_grad:
            grad = torch.autograd.grad(loss, self.memory, retain_graph=True, allow_unused=True)[0]
            if grad is not None:
                with torch.no_grad():
                    self.memory[:] = 0.9 * self.memory - 0.01 * grad

def train(model, data, vocab_size, device='cpu', epochs=1):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, data.size(0) - 21, 20):
            x, y = get_batch(data, i, 20)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model.predict(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            if isinstance(model, TitanModel):
                model.memory_update(loss)
            total_loss += loss.item()
        print(f"[{model.__class__.__name__}] Epoch {epoch+1}, Loss: {total_loss:.4f}")

def evaluate_model(model, name, vocab_size, data, device='cpu'):
    model.to(device)
    model.eval()
    x, _ = get_batch(data, 0, 20)
    x = x.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    with torch.no_grad():
        model.predict(x)
    latency = (time.time() - start) * 1000
    mem = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
    return {
        "Model": name,
        "Params": sum(p.numel() for p in model.parameters()),
        "Latency (ms)": round(latency, 2),
        "Memory (MB)": round(mem, 2)
    }

device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim, hidden_dim, batch_size, seq_len = 64, 128, 4, 20
train_data = batchify(data_process(WikiText2(split='train')), batch_size)
vocab_size = len(vocab)

models = [
    RNNModel(vocab_size, input_dim, hidden_dim),
    GRUModel(vocab_size, input_dim, hidden_dim),
    LSTMModel(vocab_size, input_dim, hidden_dim),
    TransformerModel(vocab_size, input_dim, hidden_dim),
    MambaModel(vocab_size, input_dim, hidden_dim),
    TitanModel(vocab_size, input_dim, hidden_dim),
]

stats = []
for model in models:
    train(model, train_data, vocab_size, device, epochs=1)
    stats.append(evaluate_model(model, model.__class__.__name__, vocab_size, train_data, device))

df = pd.DataFrame(stats)
print("\n모델 비교 결과:\n", df)

df.set_index("Model")[["Latency (ms)", "Memory (MB)"]].plot(kind='bar', figsize=(10, 5), title="모델 성능 비교")
plt.grid(True)
plt.tight_layout()
plt.show()