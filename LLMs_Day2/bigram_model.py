# ================================
# BIGRAM LANGUAGE MODEL FROM SCRATCH (BEGINNER-FRIENDLY COMMENTS)
# ================================

# STEP 1: LOAD THE DATASET
import torch
import torch.nn as nn
import torch.nn.functional as F


file_path = r"C:\Users\GenITeam\Desktop\GenIteam_Solutions_Internship_Work\LLMs_Day2\Rumi_poetry.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print("Dataset length in characters:", len(text))
print("Sample:\n", text[:500])  # Display first 500 characters

# STEP 2: BUILD VOCABULARY
chars = sorted(list(set(text)))  # Get all unique characters
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Mapping from character to index (and vice versa)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # Converts string to list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # Converts list of integers to string

# Encode and decode test
print(encode("hii there"))
print(decode(encode("hii there")))

# STEP 3: CONVERT ENTIRE TEXT TO TENSOR
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)  # Shape and data type
print(data[:1000])  # Print the first 1000 encoded characters

# STEP 4: SPLIT DATA INTO TRAIN AND VALIDATION SETS
n = int(0.9*len(data))
train_data = data[:n]  # 90% for training
val_data = data[n:]    # 10% for validation

# Visualizing how input (x) and target (y) are created
block_size = 8  # Sequence length used for training
x = train_data[:block_size]            # Input sequence
y = train_data[1:block_size+1]         # Next characters (targets)
for t in range(block_size):
    context = x[:t+1]  # Increasing context length
    target = y[t]
    print(f"when input is {context} the target: {target}")

# STEP 5: FUNCTION TO FETCH TRAINING BATCHES
torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])  # Input sequences
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # Corresponding targets
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

# Visualizing each token prediction in batch
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

# STEP 6: BUILD THE BIGRAM LANGUAGE MODEL
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly maps to the logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are (B, T) tensors
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)  # Classification loss
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Get last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # Append prediction
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# STEP 7: SAMPLE GENERATED TEXT BEFORE TRAINING
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# STEP 8: TRAIN THE MODEL
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(100):  # Can increase steps for better results
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())  # Final loss after training

# STEP 9: SAMPLE FINAL TEXT AFTER TRAINING
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

# STEP 10: TOY MATRIX EXAMPLE - WEIGHTED AVERAGE
# This shows how attention-like operations use weighted sums

torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))  # Lower triangular matrix
# Normalize each row
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b  # Matrix multiply for aggregation
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)

# STEP 11: TOY EXAMPLE FOR AVERAGE OVER TIME
B,T,C = 4,8,2
x = torch.randn(B,T,C)
# Compute running mean using loops
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, 0)

# VERSION 2: Weighted average using matrix multiply
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x

# VERSION 3: Use Softmax to normalize
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x

# VERSION 4: FULL SELF-ATTENTION (ONE HEAD)
B,T,C = 4,8,32
x = torch.randn(B,T,C)
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)
wei =  q @ k.transpose(-2, -1)  # Similarity scores
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))  # Mask future tokens
wei = F.softmax(wei, dim=-1)
v = value(x)
out = wei @ v  # Final output of attention
out.shape

# STEP 12: INSPECT SELF-ATTENTION STABILITY
k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5  # Scale scores
k.var()
q.var()
wei.var()
torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)
torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])*8, dim=-1)  # Sharper

# STEP 13: IMPLEMENTING LAYER NORM (like BatchNorm)
class LayerNorm1d:
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    xmean = x.mean(1, keepdim=True)
    xvar = x.var(1, keepdim=True)
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

# Example usage:
torch.manual_seed(1337)
module = LayerNorm1d(100)
x = torch.randn(32, 100)  # 32 samples, 100 features each
x = module(x)
print(x.shape)
print("Mean:", x[:,0].mean(), "Std:", x[:,0].std())  # Across batch
print("Mean (single):", x[0,:].mean(), "Std (single):", x[0,:].std())  # Single input
