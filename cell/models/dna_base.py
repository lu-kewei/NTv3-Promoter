import torch
import torch.nn as nn


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


class DNAClassifierModel(nn.Module):
    """
    A simple classifier that uses a DNA model with a classification head.
    """

    def __init__(
        self,
        dna_model_name: str,
        cache_dir: str = None,
        max_length_dna: int = 4096,
        num_classes: int = 2,  # Binary classification by default
        dna_is_evo2: bool = False,
        dna_embedding_layer: str = None,
        train_just_classifier: bool = True
    ):
        """
        Initialize the DNAClassifierModel.

        Args:
            dna_model_name (str): Name of the DNA model to use
            cache_dir (str): Directory to cache models
            max_length_dna (int): Maximum sequence length
            num_classes (int): Number of output classes
            dna_is_evo2: Whether the DNA model is Evo2. Defaults to False
            dna_embedding_layer: Name of the layer to use for the Evo2 model. Defaults to None
            train_just_classifier: Whether to train just the classifier. Defaults to True
        """
        super().__init__()

        self.dna_model_name = dna_model_name
        self.cache_dir = cache_dir
        self.max_length_dna = max_length_dna
        self.num_classes = num_classes
        self.dna_is_evo2 = dna_is_evo2
        self.dna_embedding_layer = dna_embedding_layer
        self.train_just_classifier = train_just_classifier

        # Load the DNA model and tokenizer
        if not self.dna_is_evo2:
            # self.dna_model = AutoModelForMaskedLM.from_pretrained(
            #     dna_model_name, cache_dir=cache_dir, trust_remote_code=True
            # )
            
            from modelscope import AutoTokenizer, AutoModelForMaskedLM
            self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)

            # Import the tokenizer and the model
            self.dna_model = AutoModelForMaskedLM.from_pretrained(dna_model_name, trust_remote_code=True)
            self.dna_config = self.dna_model.config

        else:
            from evo2 import Evo2
            from cell.models.evo2_tokenizer import Evo2Tokenizer
            self.dna_model = Evo2(dna_model_name)
            self.dna_tokenizer = Evo2Tokenizer(self.dna_model.tokenizer)
            self.dna_config = self.dna_model.model.config
            self.dna_embedding_layer = self.dna_embedding_layer

        # Get hidden size from model config
        self.hidden_size = self.dna_config.hidden_size

        # Add the self-attention pooling module
        self.pooler = SelfAttentionPooling(self.hidden_size)

        # Create classification head that takes concatenated embeddings from both sequences
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_classes),
        )
        self.two_token_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_classes),
        )
        self.max_length_dna = max_length_dna

    def get_dna_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Get DNA embedding for a single DNA sequence using self-attention pooling.

        Args:
            input_ids: DNA tokenized sequence
            attention_mask: DNA tokenized sequence attention mask

        Returns:
            torch.Tensor: Tensor containing the self-attention pooled DNA embedding
        """
        # Add batch dimension if not present
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze()  # [1, seq_len]
        
        # Handle attention mask - create if not provided or add batch dimension
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze()  # [1, seq_len]
        
        # Get embeddings from DNA model
        with torch.set_grad_enabled(not self.train_just_classifier):  # Enable gradients for fine-tuning

            if self.dna_is_evo2 and self.dna_embedding_layer is not None:  # Evo2 model
                # Get embeddings from the specific layer in Evo2
                _, embeddings = self.dna_model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[self.dna_embedding_layer]
                )
                
                # Get embeddings for the specified layer
                hidden_states = embeddings[self.dna_embedding_layer]
            
            else:
                # Get embeddings from the last hidden state
                outputs = self.dna_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Get the last hidden state
                hidden_states = outputs.hidden_states[-1]
        
        # Apply self-attention pooling to get a weighted representation
        sequence_embedding = self.pooler(hidden_states, attention_mask)
        # change lukewei
        # return sequence_embedding.squeeze(0)
        return sequence_embedding


    def single_token_forward(self, batch):
        ids = batch["ids"]
        attention_mask = batch["attention_mask"]
        embeddings = self.get_dna_embedding(ids, attention_mask)
        logits = self.classifier(embeddings)
        return logits
    
    def two_token_forward(self, batch):
        ref_ids = batch["ref_ids"]
        alt_ids = batch["alt_ids"]
        ref_attention_mask = batch["ref_attention_mask"]
        alt_attention_mask = batch["alt_attention_mask"]
        ref_embeddings = self.get_dna_embedding(ref_ids, ref_attention_mask)
        alt_embeddings = self.get_dna_embedding(alt_ids, alt_attention_mask)
        combined_embeddings = torch.cat((ref_embeddings, alt_embeddings), dim=1)
        # chang lukewei end 
        # Pass through classifier
        logits = self.two_token_classifier(combined_embeddings)
        return logits
    
    def forward(self,batch,single_token_flag=True):
        if single_token_flag:
            logits = self.single_token_forward(batch)
        else:
            logits = self.two_token_forward(batch)
        return logits