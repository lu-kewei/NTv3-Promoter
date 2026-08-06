
from torch.utils.data import Dataset
import numpy as np 
from cell.utils.utils import load_genome_fasta
import os.path as osp 
from numpy.lib.stride_tricks import sliding_window_view
from vortex.model.tokenizer import CharLevelTokenizer

class TrainDataset(Dataset):
    def __init__(self, fasta_file,label_dir,token_num,batch_size,label_idx):
        self.genome = load_genome_fasta(fasta_file)
        self.keys = self.genome.keys()
        self.label_dir = label_dir
        self.key_num = len(self.keys)
        self._len = self.key_num*20
        self.batch_size = batch_size
        self.token_num = token_num
        self.label_idx = label_idx
        self.tokenizer = CharLevelTokenizer(512)

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
        N_mask = [np.array(list(sequence)) != 'N' for  sequence in sequences]
        window_label = sliding_window_view(label, (self.token_num,label.shape[1]))  
        labels = window_label[random_begin] 
        tokens = np.array(
            self.tokenizer.tokenize_batch(sequences),dtype=np.int32
        )
        # tokens = self.tokenizer.batch_encode(
        #         sequences,
        #         return_tensors="pt",
        #         padding="max_length",
        #         max_length=self.token_num//6+1,
        #         truncation=True
        #     )["input_ids"]
        # attention_mask = (tokens != self.tokenizer.pad_token_id)
        
        labels = labels[:,0,]
        
        labels = labels[:, :, self.label_idx]
        attention_mask = N_mask
        return  tokens, labels, attention_mask, N_mask
        
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
    

    
