
import torch
from torch import nn 
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        # Use PyTorch's built-in multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
    def forward(self, embeddings, attention_mask=None):
        # Expand query to batch size
        batch_size = embeddings.size(0)
        query = self.query.expand(batch_size, -1, -1)
        
        # Create key padding mask from attention mask if provided
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # Convert to boolean mask where True means ignore
        
        # Apply attention: query attends to embeddings
        context, _ = self.attention(
            query=query,                  # [batch_size, 1, hidden_size]
            key=embeddings,               # [batch_size, seq_len, hidden_size]
            value=embeddings,             # [batch_size, seq_len, hidden_size]
            key_padding_mask=key_padding_mask
        )
        
        # Squeeze out the singleton dimension
        return context.squeeze(1)         # [batch_size, hidden_size]
class DNAClassifierModelTrainer(nn.Module):
    def __init__(self, 
                model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModelForMaskedLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            config=self.config,
        )
        self.dna_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.config.embed_dim
        self.two_token_classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 2),
        )
        self.ref_pool = SelfAttentionPooling(self.hidden_size)
        self.alt_pool = SelfAttentionPooling(self.hidden_size)

    def forward(self, batch):
        ref_ids = batch["ref_ids"]
        alt_ids = batch["alt_ids"]
        ref_outputs = self.backbone(ref_ids, output_hidden_states=True)
        alt_outputs = self.backbone(alt_ids, output_hidden_states=True)
        ref_embeddings = ref_outputs.hidden_states[-1]
        alt_embeddings = alt_outputs.hidden_states[-1]
        ref_embeddings = self.ref_pool(ref_embeddings)
        alt_embeddings = self.alt_pool(alt_embeddings)
        
        combined_embeddings = torch.cat((ref_embeddings, alt_embeddings), dim=1)
        
        logits = self.two_token_classifier(combined_embeddings)
        return logits
