
from torch import nn 
from cell.models.dna_base import DNAClassifierModel

class DNAClassifierModelTrainer(nn.Module):
    def __init__(self, 
                dna_model_name,
                cache_dir,
                max_length_dna,
                num_classes,
                dna_is_evo2,
                dna_embedding_layer,
                train_just_classifier,
                single_token_flag):
        super().__init__()
        self.dna_model = DNAClassifierModel(
            dna_model_name,
            cache_dir,
            max_length_dna,
            num_classes,
            dna_is_evo2,
            dna_embedding_layer,
            train_just_classifier,
        )
        self.dna_tokenizer = self.dna_model.dna_tokenizer
        # Set the training mode for the classifier and pooler
        self.dna_model.pooler.train()
        self.dna_model.classifier.train()

        # Freeze the DNA model parameters
        if dna_is_evo2:
            self.dna_model_params = self.dna_model.dna_model.model.parameters()
        else:
            self.dna_model_params = self.dna_model.dna_model.parameters()

        if train_just_classifier:
            for param in self.dna_model_params:
                param.requires_grad = False
        self.single_token_flag = single_token_flag

    def forward(self, batch):
        
        logits = self.dna_model(batch,single_token_flag=self.single_token_flag)
        return logits
