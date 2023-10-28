import torch.nn.functional as F
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from utils import *
import os
import fire

def load_dataset(path):
    pass

def main():
    # load dataset

    # load model
    decoder_based_model = 'microsoft/biogpt'
    tokenizer = AutoTokenizer.from_pretrained(decoder_based_model)
    model = AutoModelForCausalLM.from_pretrained(decoder_based_model)

    # add tokens
    extra_tokens = ['askhim']
    num_added_tokens = tokenizer.add_tokens(extra_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # Only train the extra embeddings
    for param in model.biogpt.embed_tokens.weight[-num_added_tokens:]:
        param.requires_grad = True

    inputs = tokenizer('you can askhim', truncation=True, padding=True, return_tensors="pt")
    labels = torch.tensor(tokenizer.encode('you can askhim'))
    output = model(**inputs)
    # print(output.last_hidden_state.shape)
    logits = output.logits.view(-1, output.logits.shape[-1])
    loss = F.cross_entropy(logits, labels.view(-1), ignore_index=-100)
    # check p, r, f1 for each function
    pred = torch.argmax(output.logits, dim=-1)  # (bsz, seqlen)
    pred = pred.view(-1)
    labels = labels.view(-1)

if __name__ == "__main__":
    fire.Fire(main)
