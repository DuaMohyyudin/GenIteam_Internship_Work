'''
Enhanceents Made:

1. Temperature Sampling (Better Text Generation)
What it does: Controls how "creative" or "predictable" the AI's writing is.

Simple analogy: Imagine picking words from a hat:

High temperature (1.5) = Reach in blindly (more surprising/creative)

Medium (1.0) = Glance quickly before picking (balanced)

Low (0.5) = Study each word carefully before picking (more predictable)

Why it's good: You can adjust how wild or tame the poems should be!

2. Learning Rate Scheduling (Better Training)
What it does: Automatically adjusts how fast the AI learns during training.

Simple analogy: Like learning to ride a bike:

Start with training wheels (small learning rate)

Then go faster (increase learning rate)

Finally slow down to perfect your technique (decrease learning rate)

Why it's good: Helps the AI learn better and faster without crashing!

3. Gradient Clipping (Training Stability)
What it does: Prevents the AI from making "too big" mistakes during learning.

Simple analogy: Like putting bumpers in a bowling lane - prevents the ball (learning process) from going completely wild.

Why it's good: Stops the AI from getting completely confused by one bad example.

4. Rhyme Helper (Poetry-Specific)
What it does: Helps find rhyming words.

How it works:

Looks at the last 3 letters of words

Groups words with similar endings (like "love" and "dove")

Suggests rhymes when needed

Why it's good: Makes the poems sound more musical and traditional!

5. GLU (Gated Linear Unit) - Smarter Thinking
What it does: Makes the AI's "brain cells" work better.

Simple analogy: Regular version is like a light switch (on/off). GLU is like a dimmer switch - can be partly on.

Why it's good: Helps the AI understand more complex poetry patterns.

6. Mixed Precision Training (Faster Training)
What it does: Uses less computer memory to train faster.

Simple analogy: Like packing a suitcase:

Regular: Fold everything perfectly (slow but precise)

Mixed: Some things folded, some rolled (faster but still organized)

Why it's good: Lets you train the AI quicker if you have a good graphics card.

How These Help Make Better Poems:
Better Learning: The training improvements help the AI understand poetry patterns faster and more reliably.

Better Writing: Temperature control lets you choose between wild, creative poems or more traditional-sounding ones.

Poetic Features: The rhyme helper makes the output sound more like real poetry.'''






import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

file_path = r"C:\Users\GenITeam\Desktop\GenIteam_Solutions_Internship_Work\LLMs_Day2\Rumi_poetry.txt"
# Read the text data
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print("Dataset length in characters:", len(text))
print("Sample:\n", text[:500])  # Print the first 500 characters to verify

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ Gated Linear Unit version """
    def __init__(self, n_embd):
        super().__init__()
        self.gate = nn.Linear(n_embd, 4 * n_embd)
        self.up = nn.Linear(n_embd, 4 * n_embd)
        self.down = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.down(self.dropout(F.silu(self.gate(x)) * self.up(x)))

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class RhymeHelper:
    def __init__(self, text):
        self.rhyme_dict = self.build_simple_rhyme_dict(text)
    
    def build_simple_rhyme_dict(self, text):
        words = set(text.split())
        rhyme_dict = {}
        for word in words:
            if len(word) >= 3:
                ending = word[-3:].lower()
                if ending not in rhyme_dict:
                    rhyme_dict[ending] = []
                rhyme_dict[ending].append(word)
        return rhyme_dict
    
    def get_rhyming_words(self, word):
        if len(word) >= 3:
            ending = word[-3:].lower()
            return self.rhyme_dict.get(ending, [])
        return []

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.rhyme_helper = RhymeHelper(text)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
                
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Create optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=learning_rate*10,
    total_steps=max_iters
)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
        logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

# Generate with different sampling methods
print("\nCreative generation (temperature=1.2):")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200, temperature=1.2)[0].tolist()))

print("\nConservative generation (temperature=0.7, top_k=40):")
print(decode(m.generate(context, max_new_tokens=200, temperature=0.7, top_k=40)[0].tolist()))

# Example of using rhyme helper
print("\nRhyming words for 'love':")
print(m.rhyme_helper.get_rhyming_words("love"))