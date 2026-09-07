import pandas as pd
from torch.utils.data import Dataset
import os
from transformers import BertTokenizer
import numpy as np
import torch

class CustomTextDataset(Dataset):
    '''
    __init__: 
        function is run once when instantiating the Dataset object. 
        We initialize the directory containing the text, the annotations file, and both transforms (covered in more detail in the next section).
    
    '''
    def __init__(self, train_file_path, device, transform=None, target_transform=None):

        self.data = pd.read_csv(train_file_path)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.device = device


    def bert_tokenizer(self, text, max_len = None):

       return self.tokenizer(
                        text,
                        padding=True,          # Pad sequences to the maximum length in the batch
                        truncation=True,       # Cut off text longer than BERT's max limit (512 tokens)
                        max_length=max_len,         # Set an explicit sequence length limit (optional)
                        return_tensors="pt"    # Crucial: Returns PyTorch tensors ('pt')
                    )

    def __len__(self):
        ''' datset length '''
        return len(self.data)

    def __getitem__(self, idx):
        '''model_a,model_b,prompt,response_a,response_b,'''

        device = self.device
        inputs = {
            'prompt':self.bert_tokenizer(self.data.iloc[idx]['prompt']).to(device),
            'model_a': self.data.iloc[idx]['model_a'],
            'model_b': self.data.iloc[idx]['model_b'],
            'response_a': self.bert_tokenizer(self.data.iloc[idx]['response_a']).to(device),
            'response_b': self.bert_tokenizer(self.data.iloc[idx]['response_b']).to(device)
        }

        result = self.data.iloc[idx]["winner_model_a winner_model_b winner_tie".split()].to_numpy()

        return inputs, result

if __name__ == '__main__':
 print('test')

 os.chdir('..')

 print(print(torch.__version__))

 #path = os.getcwd() + '/data/train.csv'

 #ds = CustomTextDataset(train_file_path=path)

 #print(ds.__getitem__(2))