import torch 
from transformers import get_cosine_schedule_with_warmup

def configure_optimizers(model, total_steps, kwargs):
    scheduler = None
    if kwargs["type"] == "dna_class":
        classifier_params = [
            {
                "params": model.dna_model.classifier.parameters(),
                "lr": kwargs["lr"],
            },
            {
                "params": model.dna_model.pooler.parameters(),
                "lr": kwargs["lr"],
            }
        ]
        dna_model_params = [
            {
                "params": model.dna_model_params,
                "lr": kwargs["lr"] * 0.1,
            },
        ]
        optimizer = torch.optim.AdamW(
                classifier_params + dna_model_params,
                weight_decay=kwargs["weight_decay"],
            )

        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
    elif kwargs["type"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
            )
    return optimizer, scheduler
    