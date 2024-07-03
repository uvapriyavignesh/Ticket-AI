import torch
import constant as co
import pandas as pd
import Tokenizer
def get_torch_data():
    torch.manual_seed(co.RANDOM_SEED)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch,DEVICE

def get_data_source():
    data_source=pd.read_csv(co.TRAIN_DATA_PATH)
    # mix data set
    data_source=data_source.sample(frac=1).reset_index(drop=True)
    data_source=data_source.sample(frac=1).reset_index(drop=True)
    data_source=data_source.sample(frac=1).reset_index(drop=True)
    return data_source

def get_tokenizer_object(data):
    # if isinstance(data,pd):
        sample_text=[i for i in data.get(co.TEXT_COLUMN_NAME).values]
        return Tokenizer.Tokenizer(sample_text,padding_size=1000)
    # else:
    #     raise Exception("args must be pandas object")
    
def get_out(data):
    li=[]
    for j in data.split('$'):
         li.append(int(j))
    return li
         

def tokenize_input_out_put_field(token,data):
    input_data_sources=data.get(co.TEXT_COLUMN_NAME).apply(lambda x: token.encode([x])[0].ids ).tolist()
    # out_data_sources=[[1 if j else 0 for j in i] for i in pd.get_dummies(data[co.LABEL_COLUMN_NAME]).values.tolist()]
    out_data_sources=[ get_out(i) for i in data[co.LABEL_COLUMN_NAME].values.tolist()  ]
    return input_data_sources,out_data_sources


def _dictDate(input,output,device,torch_object,sequence):
    input_date=torch_object.tensor(input,dtype=torch.int64,device=device)
    data={
        "input":input_date,
        "sequence": (input_date != 0).sum(dim=1)
    }
    if output is not None:
        data["output"]=torch_object.tensor(output,dtype=torch.float32,device=device)
    if sequence:
         data['isSeq']=True
    return data

def batch_spliter(input_data_sources,out_data_sources,sequence=False):
    batch_list=[]
    count=0
    _,device=get_torch_data()
    while count+ co.BATCH_SIZE<len(input_data_sources):
        batch_list.append(_dictDate(input_data_sources[count:count+ co.BATCH_SIZE],out_data_sources[count:count+ co.BATCH_SIZE] if out_data_sources is not None else None,device,_,sequence))
        count=count+ co.BATCH_SIZE
    batch_list.append(_dictDate(input_data_sources[count:],out_data_sources[count:] if out_data_sources is not None else None,device,_,sequence))
    return batch_list

