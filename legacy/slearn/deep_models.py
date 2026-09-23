# This is the example models for testing, user can design their own models following this template.

import torch
import math
import torch.nn as nn

NUM_HEAD = 8  # Number of attention heads, can be adjusted based on model requirements

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

# Causal Mask Generation
def generate_causal_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.d_model = 512
        self.input_proj = nn.Linear(input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=NUM_HEAD,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.output_proj = nn.Linear(self.d_model, output_size)
    
    def forward(self, src, tgt):
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.input_proj(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.output_proj(output[:, -1, :])



class GPTLikeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.d_model = 512
        self.input_proj = nn.Linear(input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=NUM_HEAD, dim_feedforward=hidden_size, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(self.d_model, output_size)
    
    def forward(self, x):
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        tgt_mask = generate_causal_mask(x.size(1)).to(x.device)
        memory = torch.zeros_like(x).to(x.device)
        output = self.decoder(x, memory, tgt_mask=tgt_mask)
        return self.output_proj(output[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hn = self.gru(x)
        return self.fc(hn[-1])