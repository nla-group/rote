"""Model zoo for symbolic next-token experiments.

All models consume one-hot windows shaped ``(batch, window, alphabet)`` and
return logits shaped ``(batch, alphabet)``.  The encoder-decoder Transformer
keeps its historical ``forward(src, tgt)`` signature for old experiment scripts.
"""

import math
import os

import torch
import torch.nn as nn


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

NUM_HEAD = 8


def _optional_import(package_name, import_name=None):
    try:
        module = __import__(package_name, fromlist=[import_name] if import_name else [])
    except ImportError as exc:
        raise ImportError(
            f"Model dependency '{package_name}' is not installed. "
            "Run exps/scripts/install_experiment_deps.sh or install the package manually."
        ) from exc
    return getattr(module, import_name) if import_name else module


def _num_heads(d_model):
    for heads in range(min(NUM_HEAD, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


def generate_causal_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, d_model=512):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=_num_heads(d_model),
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            batch_first=True,
        )
        self.output_proj = nn.Linear(d_model, output_size)

    def forward(self, src, tgt=None):
        src = self.pos_encoder(self.input_proj(src) * math.sqrt(self.d_model))
        if tgt is None:
            tgt = src
        else:
            tgt = self.pos_encoder(self.input_proj(tgt) * math.sqrt(self.d_model))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.output_proj(output[:, -1, :])


class BERTClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, d_model=512):
        super().__init__()
        BertModel = _optional_import("transformers", "BertModel")
        BertConfig = _optional_import("transformers", "BertConfig")
        self.input_proj = nn.Linear(input_size, d_model)
        config = BertConfig(
            hidden_size=d_model,
            intermediate_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=_num_heads(d_model),
        )
        self.bert = BertModel(config)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        attention_mask = torch.ones(x.size(0), x.size(1), device=x.device)
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        return self.fc(outputs.mean(dim=1))


class GPTLikeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, d_model=512):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=_num_heads(d_model),
            dim_feedforward=hidden_size,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.pos_encoder(self.input_proj(x) * math.sqrt(self.d_model))
        tgt_mask = generate_causal_mask(x.size(1)).to(x.device)
        memory = torch.zeros_like(x)
        output = self.decoder(x, memory, tgt_mask=tgt_mask)
        return self.output_proj(output[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, d_model=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, d_model=None):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hn = self.gru(x)
        return self.fc(hn[-1])


class MinGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, d_model=None):
        super().__init__()
        minGRU = _optional_import("minGRU_pytorch", "minGRU")
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([minGRU(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.norm(x[:, -1, :]))


class MinLSTMLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gates = nn.Linear(dim, 2 * dim)
        self.candidate = nn.Linear(dim, dim)

    def forward(self, x):
        h = x.new_zeros(x.size(0), x.size(2))
        outputs = []
        for xt in x.unbind(dim=1):
            f, i = torch.sigmoid(self.gates(xt)).chunk(2, dim=-1)
            z = f + i + 1e-6
            candidate = torch.tanh(self.candidate(xt))
            h = (f / z) * h + (i / z) * candidate
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class MinLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, d_model=None):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([MinLSTMLayer(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.norm(x[:, -1, :]))


class LinearAttentionTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, d_model=512, max_seq_len=512):
        super().__init__()
        LinearAttentionTransformer = _optional_import(
            "linear_attention_transformer", "LinearAttentionTransformer"
        )
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        self.encoder = LinearAttentionTransformer(
            dim=d_model,
            depth=num_layers,
            max_seq_len=max_seq_len,
            heads=_num_heads(d_model),
            causal=True,
            ff_dropout=0.0,
            attn_dropout=0.0,
            n_local_attn_heads=0,
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.pos_encoder(self.input_proj(x))
        x = self.encoder(x)
        return self.fc(x[:, -1, :])


class PerformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, d_model=512, max_seq_len=512):
        super().__init__()
        Performer = _optional_import("performer_pytorch", "Performer")
        heads = _num_heads(d_model)
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        self.encoder = Performer(
            dim=d_model,
            depth=num_layers,
            heads=heads,
            dim_head=max(16, d_model // heads),
            causal=True,
            ff_mult=max(1, hidden_size // max(1, d_model)),
            auto_check_redraw=False,
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.pos_encoder(self.input_proj(x))
        x = self.encoder(x)
        return self.fc(x[:, -1, :])


class RWKVBlock(nn.Module):
    """A compact RWKV-style time-mixing block for controlled small experiments."""

    def __init__(self, dim, hidden_size):
        super().__init__()
        self.time_mix_r = nn.Parameter(torch.rand(dim))
        self.time_mix_k = nn.Parameter(torch.rand(dim))
        self.time_mix_v = nn.Parameter(torch.rand(dim))
        self.time_decay = nn.Parameter(torch.zeros(dim))
        self.receptance = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def _mix(self, x, prev, mix):
        return x * mix + prev * (1.0 - mix)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        r = torch.sigmoid(self.receptance(self._mix(x, prev, self.time_mix_r)))
        k = torch.sigmoid(self.key(self._mix(x, prev, self.time_mix_k)))
        v = self.value(self._mix(x, prev, self.time_mix_v))
        decay = torch.sigmoid(self.time_decay).view(1, -1)
        state = x.new_zeros(x.size(0), x.size(2))
        outputs = []
        for kt, vt, rt in zip(k.unbind(dim=1), v.unbind(dim=1), r.unbind(dim=1)):
            state = decay * state + kt * vt
            outputs.append(self.output(rt * state))
        x = residual + torch.stack(outputs, dim=1)
        return x + self.ffn(x)


class RWKVModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, d_model=512):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([RWKVBlock(d_model, hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.fc(self.norm(x[:, -1, :]))


MODEL_REGISTRY = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "minGRU": MinGRUModel,
    "minLSTM": MinLSTMModel,
    "Transformer": TransformerModel,
    "BERT": BERTClassificationModel,
    "GPT": GPTLikeModel,
    "LinearAttention": LinearAttentionTransformerModel,
    "Performer": PerformerModel,
    "RWKV": RWKVModel,
}

TRANSFORMER_STYLE_MODELS = {"Transformer", "BERT", "GPT", "LinearAttention", "Performer", "RWKV"}


def get_model(model_name, input_size, hidden_size, output_size, num_layers=1, d_model=512, window_size=100):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[model_name]
    if model_name in TRANSFORMER_STYLE_MODELS:
        return cls(
            input_size,
            hidden_size,
            output_size,
            num_layers=num_layers,
            d_model=d_model,
            max_seq_len=window_size,
        ) if model_name in {"LinearAttention", "Performer"} else cls(
            input_size,
            hidden_size,
            output_size,
            num_layers=num_layers,
            d_model=d_model,
        )
    return cls(input_size, hidden_size, output_size, num_layers=num_layers)
