# !pip install torch tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- 0. Tokenizer Setup (Using HuggingFace Tokenizers) ---
# For demonstration, we train a quick BPE on dummy data.
# In production, load a pretrained tokenizer: Tokenizer.from_file("path.json")

def get_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    # Dummy training data
    files = ["The robot walked to the store.", "Transformers are cool.", "Hello world."]
    tokenizer.train_from_iterator(files, trainer)
    return tokenizer

# Global constants for demo
tokenizer = get_tokenizer()
VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_IDX = tokenizer.token_to_id("[PAD]")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, use_cache=False, past_kv=None):
        batch_size = q.size(0)
        
        # Linear projections
        # If using cache (decoding step > 0), q is usually just the current token [Batch, 1, Dim]
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # --- KV Caching Logic ---
        current_k, current_v = K, V
        if use_cache and past_kv is not None:
            # past_kv is typically (past_k, past_v)
            past_k, past_v = past_kv
            # Concatenate past keys/values with current ones along sequence dimension
            K = torch.cat([past_k, K], dim=-2)
            V = torch.cat([past_v, V], dim=-2)
        
        # Save new cache
        new_kv = (K, V) if use_cache else None
        # ------------------------

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Apply mask (e.g., look-ahead mask). 
            # Note: During generation with cache, mask shape needs careful handling.
            # Usually, if caching, we only attend to past, so mask might not be needed or is all 1s.
            if mask.size(-1) == scores.size(-1): # basic shape check
                scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        # Concatenate heads and put through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output), new_kv

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self Attention
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads) # For Seq2Seq
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask, use_cache=False, past_kv=None):
        # Unpack past_kv if using cache (it handles self_attn cache)
        # For simplicity in this generic layer, we assume past_kv is specific to self_attn
        
        # Self Attention (masked for causality)
        attn_out, new_kv = self.self_attn(x, x, x, tgt_mask, use_cache=use_cache, past_kv=past_kv)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross Attention (if memory/encoder_output is provided)
        if memory is not None:
            cross_out, _ = self.cross_attn(x, memory, memory, src_mask)
            x = self.norm2(x + self.dropout(cross_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = (self.norm3 if memory is not None else self.norm2)(x + self.dropout(ffn_out))
        
        return x, new_kv
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.embedding(src)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, src_mask, tgt_mask, use_cache=False, past_kvs=None):
        x = self.embedding(tgt)
        x = self.pos_encoder(x)
        
        new_kvs = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs else None
            x, new_kv = layer(x, memory, src_mask, tgt_mask, use_cache=use_cache, past_kv=past_kv)
            if use_cache: new_kvs.append(new_kv)
            
        return self.norm(x), new_kvs
    
# --- 1. Seq2Seq Transformer ---
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff)
        self.decoder = Decoder(vocab_size, d_model, num_layers, num_heads, d_ff)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Weight Tying
        self.output_projection.weight = self.decoder.embedding.weight

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(src, src_mask)
        out, _ = self.decoder(tgt, memory, src_mask, tgt_mask)
        return self.output_projection(out)

# --- 2. BERT (Encoder Only) ---
class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, d_ff=3072):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff)
        # MLM Head usually includes a transform + layernorm + decoder weight
        self.mlm_head = nn.Linear(d_model, vocab_size) 
        
        # Weight Tying
        self.mlm_head.weight = self.encoder.embedding.weight

    def forward(self, src, mask=None):
        # BERT uses simple padding masks, no causal masks
        features = self.encoder(src, mask)
        return self.mlm_head(features)

# --- 3. GPT (Decoder Only) ---
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, d_ff=3072):
        super().__init__()
        # GPT is essentially a Decoder without cross-attention
        self.decoder = Decoder(vocab_size, d_model, num_layers, num_heads, d_ff)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying
        self.lm_head.weight = self.decoder.embedding.weight

    def forward(self, idx, use_cache=False, past_kvs=None):
        # Create Causal Mask
        seq_len = idx.size(1)
        # If using cache, we only process the last token, so mask is trivial (or handled by attention logic)
        if use_cache and past_kvs is not None:
             tgt_mask = None # No masking needed for single token step interacting with past
        else:
            # Standard causal mask for training
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(idx.device)
            tgt_mask = ~tgt_mask # Convert to keep_mask -> 1s on diagonal and below
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq, seq]

        # Memory is None because GPT is decoder-only (no encoder output)
        out, new_kvs = self.decoder(idx, memory=None, src_mask=None, tgt_mask=tgt_mask, 
                                    use_cache=use_cache, past_kvs=past_kvs)
        logits = self.lm_head(out)
        return logits, new_kvs
    
# --- Training Steps for BERT and GPT ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train_bert_step(model, optimizer, batch_input):
    """
    BERT Training: Masked Language Modeling
    batch_input: [batch, seq_len]
    """
    model.train()
    optimizer.zero_grad()
    
    # 1. Create Masks labels (Simplistic random masking for demo)
    inputs = batch_input.clone()
    labels = batch_input.clone()
    
    # Probability matrix for masking
    prob_matrix = torch.full(inputs.shape, 0.15)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    
    labels[~masked_indices] = -100  # We only calculate loss on masked tokens
    inputs[masked_indices] = tokenizer.token_to_id("[MASK]")
    
    # 2. Forward
    # Padding mask (batch, 1, 1, seq_len)
    padding_mask = (inputs != PAD_IDX).unsqueeze(1).unsqueeze(2)
    logits = model(inputs.to(device), padding_mask.to(device))
    
    # 3. Loss
    loss = criterion(logits.view(-1, VOCAB_SIZE), labels.to(device).view(-1))
    
    loss.backward()
    optimizer.step()
    return loss.item()

def train_gpt_step(model, optimizer, batch_input):
    """
    GPT Training: Next Token Prediction (Causal LM)
    batch_input: [batch, seq_len]
    """
    model.train()
    optimizer.zero_grad()
    
    inputs = batch_input[:, :-1].to(device) # Input: t_0 ... t_N-1
    targets = batch_input[:, 1:].contiguous().to(device) # Target: t_1 ... t_N
    
    # Forward (Mask handling is inside GPTModel)
    logits, _ = model(inputs)
    
    # Loss
    loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
    
    loss.backward()
    optimizer.step()
    return loss.item()

def generate_text_kv_cache(model, start_prompt, max_tokens=20):
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor(tokenizer.encode(start_prompt).ids).unsqueeze(0).to(device)
    
    # To store generated tokens
    generated = input_ids
    
    # Initialize KV cache
    past_kvs = None
    
    print(f"Prompt: {start_prompt}")
    
    with torch.no_grad():
        for _ in range(max_tokens):
            if past_kvs is None:
                # First step: Process entire prompt
                logits, past_kvs = model(input_ids, use_cache=True, past_kvs=None)
                # Take the logits of the last token
                next_token_logits = logits[:, -1, :]
            else:
                # Subsequent steps: Process ONLY the last generated token
                # input_ids is just [Batch, 1]
                last_token = generated[:, -1:]
                logits, past_kvs = model(last_token, use_cache=True, past_kvs=past_kvs)
                next_token_logits = logits[:, -1, :] # actually shape is [Batch, 1, vocab] -> [Batch, vocab]
            
            # Greedy decoding
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append result
            generated = torch.cat([generated, next_token_id], dim=1)
            
            # Stop if EOS (optional, not implemented here as we use fixed length)
    
    decoded_output = tokenizer.decode(generated[0].tolist())
    print(f"Generated: {decoded_output}")
    return decoded_output

def save_checkpoint(model, optimizer, epoch, loss, filepath="checkpoint.pth"):
    """
    Saves the model and training state to a file.
    """
    print(f"Saving checkpoint to {filepath}...")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print("Saved.")

def load_checkpoint(model, optimizer, filepath, device="cpu"):
    """
    Loads model and optimizer state. 
    Returns the epoch and loss to allow resuming training.
    """
    print(f"Loading checkpoint from {filepath}...")
    
    # Map_location ensures we can load a GPU-trained model onto a CPU if needed
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from Epoch {epoch} with Loss {loss:.4f}.")
    return epoch, loss

# --- Usage Example ---

if __name__ == "__main__":
    # Initialize Models
    bert = BERTModel(vocab_size=VOCAB_SIZE, d_model=256, num_layers=4, num_heads=4).to(device)
    gpt = GPTModel(vocab_size=VOCAB_SIZE, d_model=256, num_layers=4, num_heads=4).to(device)
    
    optimizer_bert = torch.optim.Adam(bert.parameters(), lr=1e-4)
    optimizer_gpt = torch.optim.Adam(gpt.parameters(), lr=1e-4)

    # Dummy Data Batch
    dummy_input = torch.randint(0, VOCAB_SIZE, (2, 10)) # Batch 2, Seq 10

    # ... (Previous model and optimizer initialization) ...

    # 1. Check if a checkpoint exists to resume from
    checkpoint_path = "gpt_checkpoint.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        user_input = input(f"Found {checkpoint_path}. Resume training? (y/n): ")
        if user_input.lower() == 'y':
            # Load the model and optimizer states
            start_epoch, last_loss = load_checkpoint(gpt, optimizer_gpt, checkpoint_path, device=device)
            start_epoch += 1 # Resume from the next epoch

    # 2. Training Loop with Save Logic
    num_epochs = 5

    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"--- Epoch {epoch} ---")

            # (Your training step here)
            loss = train_gpt_step(gpt, optimizer_gpt, dummy_input) 
            print(f"Loss: {loss:.4f}")

            # Save checkpoint at the end of every epoch
            save_checkpoint(gpt, optimizer_gpt, epoch, loss, checkpoint_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted manually.")
        save_request = input("Do you want to save the current state? (y/n): ")
        if save_request.lower() == 'y':
            save_checkpoint(gpt, optimizer_gpt, epoch, loss, checkpoint_path)

    # Inference with KV Caching
    # Initialize a blank model structure (must match the architecture of the saved model)
    inference_model = GPTModel(vocab_size=VOCAB_SIZE, d_model=256, num_layers=4, num_heads=4).to(device)

    # Load weights only
    load_checkpoint(inference_model, optimizer=None, filepath="gpt_checkpoint.pth", device=device)

    # Set to evaluation mode (crucial for Dropout/LayerNorm)
    inference_model.eval()

    # Run generation
    generate_text_kv_cache(inference_model, "The robot walked")