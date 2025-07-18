import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict, Counter
import regex as re

# Hyperparameters (updated for better performance)
batch_size = 64
block_size = 128  # Increased to capture poetic lines
max_iters = 10000
eval_interval = 250
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128  # Increased model capacity
n_head = 8    # More attention heads
n_layer = 8   # More layers
dropout = 0.1  # Added dropout for regularization
# ------------

torch.manual_seed(1337)

# Read the Robert Frost poetry dataset
file_path = r"C:\Users\GenITeam\Downloads\robert_frost_poems.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print("Dataset length in characters:", len(text))
print("Sample:\n", text[:500])

# Enhanced BPE Tokenizer
class BPETokenizer:
    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def train(self, text, vocab_size, verbose=False):
        # Preprocess text - preserve poetic structure
        text = text.replace('\n', ' \n ')  # Treat newlines as separate tokens
        
        # Tokenize into words and punctuation first
        words = self.pattern.findall(text)
        if verbose:
            print(f"Found {len(words)} word-level tokens")
            
        # Then train on byte pairs within these words
        tokens = list(' '.join(words).encode('utf-8'))
        num_merges = vocab_size - 256
        
        for i in range(num_merges):
            stats = defaultdict(int)
            for pair in zip(tokens, tokens[1:]):
                stats[pair] += 1
                
            if not stats:
                break
                
            best_pair = max(stats, key=stats.get)
            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            if verbose and i % 50 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_id} ({stats[best_pair]} merges)")
                
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
        
        if verbose:
            print(f"Final vocab size: {256 + len(self.merges)}")
            print("Top 10 most common tokens:")
            token_counts = Counter(self.encode(text))
            for token, count in token_counts.most_common(10):
               print(f"{token}: {repr(self.vocab.get(token, b'<unk>').decode('latin1'))} ({count} occurrences)")

        
    def encode(self, text):
        # First apply word-level tokenization
        words = self.pattern.findall(text)
        full_tokens = []
        
        for word in words:
            # Convert each word to bytes
            tokens = list(word.encode('utf-8'))
            
            # Apply merges
            while len(tokens) >= 2:
                # Find the next merge that can be applied
                min_pos = None
                min_id = float('inf')
                
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i+1])
                    if pair in self.merges and self.merges[pair] < min_id:
                        min_id = self.merges[pair]
                        min_pos = i
                        
                if min_pos is None:
                    break  # No more merges possible
                    
                # Apply the merge
                pair = (tokens[min_pos], tokens[min_pos+1])
                new_id = self.merges[pair]
                tokens = tokens[:min_pos] + [new_id] + tokens[min_pos+2:]
                
            full_tokens.extend(tokens)
            
        return full_tokens
        
    def decode(self, tokens):
        byte_string = b''.join([self.vocab.get(idx, b'<?>') for idx in tokens])
        return byte_string.decode('utf-8', errors='replace')

# Initialize and train tokenizer with verbose output
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=512, verbose=True)

# Test the tokenizer
test_lines = [
    "The woods are lovely, dark and deep,",
    "But I have promises to keep,",
    "And miles to go before I sleep,",
    "And miles to go before I sleep."
]

print("\nTokenization tests:")
for line in test_lines:
    encoded = tokenizer.encode(line)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {line}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Match: {line == decoded}\n")

# Encode the entire dataset
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Train and validation splits
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
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
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
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
        self.rhyme_dict = self.build_rhyme_dict(text)
    
    def build_rhyme_dict(self, text):
        # More sophisticated rhyme detection
        lines = [line.strip().lower() for line in text.split('\n') if line.strip()]
        rhyme_dict = {}
        
        for line in lines:
            words = line.split()
            if not words:
                continue
                
            last_word = words[-1].rstrip('.,;!?')
            if len(last_word) >= 2:
                ending = last_word[-2:]  # Look at last 2 characters for rhyme
                if ending not in rhyme_dict:
                    rhyme_dict[ending] = []
                if last_word not in rhyme_dict[ending]:
                    rhyme_dict[ending].append(last_word)
                    
        return rhyme_dict
    
    def get_rhyming_words(self, word):
        word = word.lower().rstrip('.,;!?')
        if len(word) >= 2:
            ending = word[-2:]
            return self.rhyme_dict.get(ending, [])
        return []

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.rhyme_helper = RhymeHelper(text)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.dropout(x)
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

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, rhyme_word=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Rhyme guidance (experimental)
            if rhyme_word and _ > max_new_tokens // 2:
                rhyming_words = self.rhyme_helper.get_rhyming_words(rhyme_word)
                if rhyming_words:
                    rhyme_indices = [tokenizer.encode(word)[-1] for word in rhyming_words if tokenizer.encode(word)]
                    if rhyme_indices:
                        rhyme_logits = torch.zeros_like(logits) - float('Inf')
                        rhyme_logits[:, rhyme_indices] = logits[:, rhyme_indices]
                        logits = rhyme_logits  # Only allow rhyming words
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

vocab_size = len(tokenizer.vocab)
model = BigramLanguageModel(vocab_size)
m = model.to(device)
print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")

# Create optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=learning_rate*10,
    total_steps=max_iters,
    pct_start=0.1
)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

# Training loop with gradient monitoring
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Generate samples during training
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = m.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
        print("Sample:\n", tokenizer.decode(generated[0].tolist()))
        print("---")

    xb, yb = get_batch('train')

    with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
        logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    
    # Gradient clipping and monitoring
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if torch.isnan(grad_norm):
        print("NaN gradients detected!")
        break
    
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if iter % 100 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"iter {iter}: loss {loss.item():.4f}, lr {current_lr:.2e}, grad_norm {grad_norm:.2f}")

# Final generation examples
print("\n=== Final Generations ===")
temperatures = [0.5, 0.8, 1.0, 1.2]
for temp in temperatures:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = m.generate(context, max_new_tokens=150, temperature=temp, top_k=40)
    print(f"\nTemperature {temp}:")
    print(tokenizer.decode(generated[0].tolist()))

# Rhyme-aware generation example
print("\n=== Rhyme-Aware Generation ===")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = m.generate(context, max_new_tokens=50, temperature=0.8, top_k=40, rhyme_word="snow")
print("Poem ending with rhyme for 'snow':")
print(tokenizer.decode(generated[0].tolist()))

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'tokenizer': tokenizer,
}, 'frost_poetry_model.pth')