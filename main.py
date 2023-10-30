import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM,BioGptForCausalLM,LlamaForCausalLM
from utils import *
import os,fire,json,re
from pathlib import Path
from tqdm import tqdm
from trainer import trainer

def load(model, ckpt_dir: str):
    checkpoints = [str(filepath) for filepath in Path(ckpt_dir).glob("*.pth")]
    if not checkpoints:
        print(f"No *.pth files found in {ckpt_dir} directory.")
        return 0
    else:
        # load from checkpoint of the latest training epoch
        def epoch_sort_key(s):
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)][-2]
        checkpoints.sort(key=epoch_sort_key)
        print(f"Loading from {checkpoints[-1]}")
        model_params = torch.load(checkpoints[-1])
        model.biogpt.embed_tokens.load_state_dict(model_params)
        return epoch_sort_key(checkpoints[-1])+1

def main(ckpt_dir: str = './checkpoints', data_dir: str = './dataset/processed_data', lr: float = 1e-3, num_epochs: int = 20):
    # load dataset
    print('Loading data')
    function_tokens = load_obj(data_dir+'/function_tokens.list')
    train_document = load_obj(data_dir+'/train_document')
    train_task = load_obj(data_dir+'/train_task')
    train_dataset = train_document + train_task
    train_dataset = DataLoader(train_dataset,batch_size=2,shuffle=True)
    # load model
    print('Loading model')
    decoder_based_model = 'microsoft/biogpt'
    tokenizer = AutoTokenizer.from_pretrained(decoder_based_model)
    model = AutoModelForCausalLM.from_pretrained(decoder_based_model)

    # add tokens & load checkpoints
    print('Preparing training')
    num_added_tokens = tokenizer.add_tokens(function_tokens)
    model.resize_token_embeddings(len(tokenizer))
    starting_epoch = 0
    if ckpt_dir:
        starting_epoch = load(model,ckpt_dir)

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    model.biogpt.embed_tokens.weight.requires_grad = True
    # train the extra embeddings
    llm_trainer= trainer(model, tokenizer, train_dataset,lr,num_epochs,num_added_tokens,starting_epoch)
    llm_trainer.train()



if __name__ == "__main__":
    fire.Fire(main)
