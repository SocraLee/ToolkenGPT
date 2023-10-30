import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np

class trainer():
    def __init__(self, model, tokenizer, trainset, lr, num_epoch, num_added_tokens,starting_epoch):
        self.model = model
        #self.added_embedding = nn.Embedding(num_added_tokens,model.output_projection.weight.shape[0],padding_idx=1)# padding index align with chosen model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam([model.biogpt.embed_tokens.weight], lr=lr)
        self.num_epoch = num_epoch
        self.trainset = trainset
        self.num_added_tokens = num_added_tokens
        self.starting_epoch = starting_epoch
        self.frozen_embedding = model.biogpt.embed_tokens.weight[:-num_added_tokens].clone()

    def train_one_epoch(self):
        epoch_loss = []
        for text in tqdm(self.trainset[:10]):
            inputs = self.tokenizer(text, return_tensors="pt")
            labels = torch.tensor(self.tokenizer.encode(text))
            output = self.model(**inputs)
            # print(output.last_hidden_state.shape)
            logits = output.logits.view(-1, output.logits.shape[-1])
            loss = F.cross_entropy(logits, labels.view(-1))
            loss.backward()
            epoch_loss.append(loss.item())
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.model.biogpt.embed_tokens.weight.data[:-self.num_added_tokens]=self.frozen_embedding
        print(f'Epoch loss {np.mean(epoch_loss)}')
    def train(self):
        print(f'Start training from {self.starting_epoch}')
        for epoch in range(self.starting_epoch,self.num_epoch):
            self.train_one_epoch()
            # save the parameters of func_embed every epoch
            save_dir = f"checkpoints/"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.model.biogpt.embed_tokens.state_dict(), f"{save_dir}/epoch_{epoch}.pth")
