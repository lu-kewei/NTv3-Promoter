
from modelscope import AutoModelForMaskedLM, AutoModel
import torch 
from torch import nn 

def hard_reset_unet1d(unet):
    for m in unet.modules():
        # Conv1D
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Linear（如果 final_block 里有）
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Norm
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class SegmentNT(nn.Module):
    def __init__(self, nt_name,segmentnt_name,label_idx):
        super().__init__()
        nt = AutoModelForMaskedLM.from_pretrained(nt_name, trust_remote_code=True)
        segmentnt = AutoModel.from_pretrained(segmentnt_name, trust_remote_code=True)
        seg_esm = segmentnt.esm
        nt_esm = nt.esm

        nt_sd = nt_esm.state_dict()
        seg_sd = seg_esm.state_dict()

        new_sd = {}
        loaded, skipped = 0, 0

        for k, v in nt_sd.items():
            if "rescaling" in k:
                print("rescal: ",k)
            if k in seg_sd and v.shape == seg_sd[k].shape:
                new_sd[k] = v
                loaded += 1
            else:
                skipped += 1

        print("matched:", loaded, "skipped:", skipped)
        seg_esm.load_state_dict(new_sd, strict=False)

        # 新的fc
        old_fc = segmentnt.fc
        embed_dim = old_fc.in_features

        segmentnt.config.features = [segmentnt.config.features[i] for i in label_idx]
        segmentnt.num_features = len(segmentnt.config.features)
        # NEW_NUM_FEATURES = len(segmentnt.config.features)
        # 新输出维度
        NEW_OUT = 6 * 2 * segmentnt.num_features
        segmentnt.fc = nn.Linear(embed_dim, NEW_OUT)
        
        hard_reset_unet1d(segmentnt.unet)
        hard_reset_unet1d(segmentnt.fc)
        
        self.segmentnt = segmentnt
        
    def forward(self,tokens, attention_mask):
        with torch.no_grad():
            outputs = self.segmentnt.esm(tokens, attention_mask=attention_mask)
            
        sequence_output = outputs[0]
        # Remove CLS token
        sequence_output = sequence_output[:,1:,:]
        

        # Invert the channels and sequence length channel
        sequence_output = torch.transpose(sequence_output, 2,1)

        x = self.segmentnt.activation_fn(self.segmentnt.unet(sequence_output))

        # Invert the channels and sequence length channel
        x = torch.transpose(x, 2,1)

        logits = self.segmentnt.fc(x)

        # Final reshape to have logits per nucleotides, per feature
        logits = torch.reshape(logits, (x.shape[0], x.shape[1] * 6, self.segmentnt.num_features, 2))

        # Add logits to the ESM outputs
        outputs["logits"] = logits

        return outputs
        # outs = self.segmentnt(tokens, attention_mask=attention_mask)
        # return outs
