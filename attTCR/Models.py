import math

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
attTCR network architecture

inputs:
- input_dim: The embedding dim of each AA and position combination. Determined by the autoencoder you pretrained.
- input_len: The max len of a TCR in the dataset, the input len of the autoencoder. Determined by the autoencoder you pretrained.
- vgenes_dim: The embedding dim of the vgene of eac TCR. Determined by the autoencoder you pretrained.
- encoding_dim: The encoding dim of the encoder. The dimension each TCR is projected to. Determined by the autoencoder you pretrained.
- auto_encoder_state_dict: The weights dict of the autoencoder you pretrained.
- emb_dict: A dict that has as keys all the reactive TCR numbers, and as values their embedding (an input to the encoder).
- hidden_layer: Hidden layer size
'''


class attTCR(nn.Module):
    def __init__(self, input_dim, input_len, vgenes_dim, encoding_dim, auto_encoder_state_dict, emb_dict,
                 hidden_layer=30):
        super(attTCR, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.vgenes_dim = vgenes_dim
        self.hidden_layer = hidden_layer
        self.emb_dict = emb_dict
        self.encoding_dim = encoding_dim
        self.betas = nn.Linear(1, 1)
        self.betas.weight = torch.nn.Parameter(torch.tensor([1.0]).view(1, 1))
        self.betas.bias = torch.nn.Parameter(torch.tensor(0.0).view(1, 1))
        self.output_layer = nn.Linear(encoding_dim, 1)
        self.key_layer1 = nn.Linear(encoding_dim, hidden_layer)
        self.key_layer2 = nn.Linear(hidden_layer, hidden_layer)
        self.attention_vector = nn.Linear(hidden_layer, 1)
        self.bn1 = nn.BatchNorm1d(hidden_layer, affine=False)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim * (self.input_len + 1), 800),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(800, 1100),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(1100, self.encoding_dim)
        )
        self.vocab_size = self.input_len * 20 + 2 + self.vgenes_dim
        self.embedding = nn.Embedding(self.vocab_size, self.input_dim, padding_idx=-1)
        self.embedding.weight = torch.nn.Parameter(auto_encoder_state_dict["embedding.weight"])
        self.encoder[0].weight = torch.nn.Parameter(auto_encoder_state_dict["encoder.0.weight"])
        self.encoder[0].weight.requires_grad = True
        self.encoder[0].bias = torch.nn.Parameter(auto_encoder_state_dict["encoder.0.bias"])
        self.encoder[0].bias.requires_grad = True
        self.encoder[3].weight = torch.nn.Parameter(auto_encoder_state_dict["encoder.3.weight"])
        self.encoder[3].weight.requires_grad = True
        self.encoder[3].bias = torch.nn.Parameter(auto_encoder_state_dict["encoder.3.bias"])
        self.encoder[3].bias.requires_grad = True
        self.encoder[6].weight = torch.nn.Parameter(auto_encoder_state_dict["mu.weight"])
        self.encoder[6].weight.requires_grad = True
        self.encoder[6].bias = torch.nn.Parameter(auto_encoder_state_dict["mu.bias"])
        self.encoder[6].bias.requires_grad = True

    def embed_for_encoding(self, x):
        tensors = []
        for tcr_idx in x.view(-1).tolist():
            tensors.append(self.emb_dict[tcr_idx].long())
        return torch.stack(tensors)

    def forward(self, x):
        emb = self.embed_for_encoding(x)
        emb = self.embedding(emb.view(-1, self.input_len + 1))
        enc = self.encoder(emb.view(-1, self.input_dim * (self.input_len + 1)))
        keys = self.key_layer1(enc)
        keys = self.key_layer2(F.relu(keys))
        scores = self.attention_vector(keys)
        scores = torch.div(scores, math.sqrt(self.hidden_layer))
        scores = torch.transpose(scores, 0, 1)
        scores = scores.view(1, -1)
        scores = torch.sigmoid(scores)
        sum_scores = torch.sum(scores, dim=1).view(1, 1)
        return torch.sigmoid(self.betas(sum_scores))
