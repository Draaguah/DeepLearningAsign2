import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_key):
        super().__init__()
        # Three separate linear layers for the queries, keys, and values
        self.w_q = nn.Linear(d_model, d_key)
        self.w_k = nn.Linear(d_model, d_key)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Compute the self attention weights
        a = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5) 
        a = F.softmax(a, dim=-1)

        # Apply the self attention weights
        z = a @ v
        return z
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_key, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_model, d_key) for _ in range(n_heads)])
        # Down projection back to model dimension
        # Alternatively, we could also split the input into n_heads and concatenate the output
        self.w_o = nn.Linear(n_heads * d_model, d_model)

    def forward(self, x):
        return self.w_o(torch.cat([h(x) for h in self.heads], dim=-1))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_key, n_heads, mlp_factor=4):
        super().__init__()
        # We need to init two layer norms because they have parameters
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, d_key, n_heads)
        self.ln2 = nn.LayerNorm(d_model)

        # a feedforward module with one internal hidden layer
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_factor * d_model),
            nn.SiLU(),  # Swish activation function, f(x) = x * sigmoid(x)
            nn.Linear(mlp_factor * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, n_embeds, n_classes, d_model=256, d_key=64, n_heads=4, mlp_factor=4, n_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(n_embeds, d_model)
        self.transformer_model = nn.Sequential(*[TransformerBlock(d_model, d_key, n_heads, mlp_factor) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_classes))

    def forward(self, x):
        return self.classifier(self.final_layer_norm(torch.mean(self.transformer_model(self.token_embedding(x)), 1)))
