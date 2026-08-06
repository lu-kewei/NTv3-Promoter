


from torch.utils.data import Dataset
import numpy as np 
from cell.utils.utils import load_genome_fasta
import os.path as osp 
from numpy.lib.stride_tricks import sliding_window_view
from modelscope import AutoTokenizer

# def project_labels(offsets, base_labels):
#     token_labels = []
#     for seq_offsets, seq_base in zip(offsets, base_labels):
#         tl = []
#         for (s,e) in seq_offsets:
#             if s == e:    # CLS / SEP
#                 tl.append(-100)
#                 continue
#             votes = seq_base[s:e]
            
#             if len(votes)==0:
#                 tl.append(-100)
#             else:
#                 tl.append(int(np.mean(votes) >= 0.5))
#         token_labels.append(tl)
#     token_labels = np.array(token_labels)
#     return token_labels
def project_labels_multi(offsets, base_labels, threshold=0.5):
    batch_out = []

    for seq_offsets, seq_base in zip(offsets, base_labels):
        seq_base = np.asarray(seq_base)        # (K, L)
        seq_offsets = np.asarray(seq_offsets) 
        L,K = seq_base.shape

        starts = seq_offsets[:,0]
        ends   = seq_offsets[:,1]

        T = len(starts)
        token_out = np.full((T, K), -100, dtype=np.int64)

        for t in range(T):
            s, e = starts[t], ends[t]
            if s < e:
                span = seq_base[s:e,:]
                token_out[t] = (span.mean(axis=0) >= threshold)

        batch_out.append(token_out)
    return np.array(batch_out)

class TrainDataset(Dataset):
    def __init__(self, fasta_file,label_dir,token_num,label_idx,dnabert2_name,batch_size):
        self.genome = load_genome_fasta(fasta_file)
        self.keys = self.genome.keys()
        self.label_dir = label_dir
        self.key_num = len(self.keys)
        self._len = self.key_num*20
        self.token_num = token_num
        self.label_idx = label_idx
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(dnabert2_name, model_max_length=1000, padding_side="right", trust_remote_code=True)

    def __getitem__(self, idx):
        key = list(self.keys)[idx%self.key_num]
        sequence = self.genome[key]
        label = np.load(osp.join(self.label_dir,"LABELS_{}.npy".format(key)))
        
        idx = np.where(label[:,10] == 1)[0]
        sel = np.random.choice(idx, size=self.batch_size//2, replace=False)
        sel_shifted = np.maximum(0, sel - np.random.randint(0, self.token_num//2, size=sel.shape))
        
        random_begin = np.random.randint(0,len(sequence) - self.token_num,size = (self.batch_size//2))
        random_begin = np.concatenate([random_begin, sel_shifted])
        
        # all random 
        # random_begin = np.random.randint(0,len(sequence)-self.token_num,size=(self.batch_size))
        
        # window_seq = sliding_window_view(sequence, self.token_num)
        sequences = [sequence[begin:begin+self.token_num] for  begin in random_begin]
        # N_mask = [np.array(list(sequence)) != 'N' for  sequence in sequences]
        window_label = sliding_window_view(label, (self.token_num,label.shape[1]))  
        labels = window_label[random_begin] 
        batch = self.tokenizer.batch_encode_plus(
                sequences,
                return_tensors="pt",
                padding="max_length",
                max_length=self.token_num//3+1,
                truncation=True,
                return_offsets_mapping=True
            )
        tokens = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = labels[:,0,]
        
        labels = labels[:, :, self.label_idx]
        token_labels = project_labels_multi(batch["offset_mapping"],labels)
        N_mask = token_labels[:,:,0]>-50
        return  tokens, token_labels, attention_mask, N_mask
        
    def __len__(self):
        return self._len

class TestDataset(Dataset):
    def __init__(self, fasta_file,label_dir,token_num,label_idx):
        self.genome = load_genome_fasta(fasta_file,test_flag=True)
        self.keys = self.genome.keys()
        self.label_dir = label_dir
        self.key_num = len(self.keys)
        self._len = self.key_num
        self.token_num = token_num
        self.label_idx = label_idx
        

    def __getitem__(self, idx):
        key = list(self.keys)[idx%self.key_num]
        sequence = self.genome[key]
        labels = np.load(osp.join(self.label_dir,"LABELS_{}.npy".format(key)))
        labels = labels[:,self.label_idx]
        return  sequence, labels
        
    def __len__(self):
        return self._len
    

    
