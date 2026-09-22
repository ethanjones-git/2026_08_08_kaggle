import pandas as pd
from torch.utils.data import Dataset
import os
from transformers import BertTokenizer
import numpy as np
import torch

def dataset_transform(df:pd.DataFrame) -> pd.DataFrame:
    
    '''
    Feature creation / transformation
    '''
    # attach prompt to each response
    for i in ['response_a','response_b']:
        df[i] = "PROMPT: " + df['prompt'] + " RESPONSE: " + df[i]

    # give ties as indiviual winners
    for i in ['a','b']:
        df[f'winner_model_{i}'] = np.where(df['winner_tie'] == 1, 1, df[f'winner_model_{i}'])

    '''
    Dataset Transformation, wide to long    
    '''
    # wide to long transformation
    data = pd.melt(
        df, 
        id_vars=['id','prompt'], 
        value_vars=['response_a', 'response_b'],
        var_name='response_type', 
        value_name='response'
    ).reset_index(drop=True) # Reset index so it matches the new row count

    # adjust features
    for i in ['winner_model','model']:
        data[i] = pd.melt(
            df, 
            id_vars=['prompt'], 
            value_vars=[f'{i}_a',f'{i}_b'],
            var_name= f'{i}_type', 
            value_name= i
        ).reset_index(drop=True)[i]

    # subset and rename
    data = data['id response winner_model model'.split()].rename(columns = {'response':'text','winner_model':'results'})

    # add model features
    data = data.merge(pd.read_csv(data_path + '/_feature_model_categories.csv')['model model_family submodel version'.split()], how = 'left', on = 'model')

    return data

class CustomTextDataset(Dataset):
    '''
    General dataset notes:

        INPUT FORMAT:
        _________________________
        |   COLUMN NAME  | TYPE |
        _________________________
        |      id        | int  | 
        |    mode_a      | str  |
        |    model_b     | str  |
        |    prompt      | list | 
        |   response_a   | list |
        |   response_b   | list |
        | winner_model_a | bool |
        | winner_model_b | bool |
        |    winner_tie  | bool |
        _________________________

        OUTPUT FORMAT:
            text = str, prompt : response
            target = [int] (0,1)

        MODEL FORMAT:

            model_1 : response to result
            model_2 ; multiple responses result

        NOTES:
            - Some rows have multipe prompt values. How prevelant is
              this and are responses similarly structured?
                Ex: id = 30192, Prompts on women in managerial 
                positons & pizza
            
            - There are also instances where text is not UTF-8, if 
              using BERT should confirm this data is tokenized.
                EX: id = 441448, changing text to Russian
        
    __init__: 
        function is run once when instantiating the Dataset object. 
        We initialize the directory containing the text, the annotations file, and both transforms (covered in more detail in the next section).
    

    '''
    def __init__(self, data_path:str, transform=None, target_transform=None):

        # pull train data
        for i in os.listdir():
           if i == 'train.csv':
              df = pd.read_csv(data_path + '/' + i)

        # wide to long transformation
        self.data = dataset_transform(df)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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

        # We pass raw strings to the DataLoader so we can pad them evenly per batch
        row = self.data.iloc[idx]
        text_input = f"Prompt: {row['prompt']} [SEP] Response: {row['response_a']}" # Example combination
        
        # Parse targets (one-hot vector to class index)
        target_vector = row["winner_model_a winner_model_b winner_tie".split()].to_numpy().astype(np.float32)
        target_label = np.argmax(target_vector) # Convert [1, 0, 0] to index 0
        
        return text_input, target_label
    
if __name__ == '__main__':

    os.chdir('..')

    out = CustomTextDataset().__getitem__(3)

    print(out)
 #path = os.getcwd() + '/data/train.csv'

 #ds = CustomTextDataset(train_file_path=path)

 #print(ds.__getitem__(2))