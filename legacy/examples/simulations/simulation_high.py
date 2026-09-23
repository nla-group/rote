import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from slearn import lzw_string_library
from slearn.dmetric import (normalized_damerau_levenshtein_distance, normalized_jaro_winkler_distance)
from transformers import T5EncoderModel, BertModel, T5Config, BertConfig
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import os
from torch.cuda.amp import autocast, GradScaler
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

symbols_list = [2, 3, 4, 5, 6]
complexities = list(range(13, 21))
min_length = 1100
window_size = 100
validation_length = 100
learning_rate = 0.01
max_epochs = 999
stopping_loss = 0.1
num_runs = 5
units = [25, 50, 100, 200, 250]
layers = [1, 2, 3]
batch_size = 128

os.makedirs("figures_high", exist_ok=True)
os.makedirs("results_partial_high", exist_ok=True)

def generate_strings(symbols, complexities):
    df_strings = lzw_string_library(symbols=symbols, complexity=complexities, random_state=0)
    return df_strings


def prepare_data(seed_string, window_size, validation_length):
    repeats = int(np.ceil(min_length / len(seed_string)))
    s = seed_string * repeats
    s = s[:min_length]
    v = s[-validation_length:]
    train_test = s[:-validation_length]
    
    X, y = [], []
    for i in range(len(train_test) - window_size):
        X.append(list(train_test[i:i + window_size]))
        y.append(train_test[i + window_size])
    
    symbols = list(set(seed_string))
    enc = OneHotEncoder(sparse_output=False, categories=[symbols] * window_size)
    X = np.array(X).reshape(len(X), window_size, 1)
    X_encoded = enc.fit_transform(X.reshape(len(X), -1)).reshape(len(X), window_size, -1)
    y_encoded = OneHotEncoder(sparse_output=False, categories=[symbols]).fit_transform(np.array(y).reshape(-1, 1))
    
    train_size = int(0.90 * len(X_encoded))
    valid_size = int(0.05 * len(X_encoded))
    X_train = X_encoded[:train_size]
    X_valid = X_encoded[train_size:train_size + valid_size]
    X_test = X_encoded[train_size + valid_size:]
    y_train = y_encoded[:train_size]
    y_valid = y_encoded[train_size:train_size + valid_size]
    y_test = y_encoded[train_size + valid_size:]
    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, symbols

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TransformerModel, self).__init__()
        nhead = 2 if input_size % 2 == 0 else 1
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out

class xLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(xLSTMModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=1,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=window_size,
            num_blocks=num_layers,
            embedding_dim=hidden_size,
            slstm_at=list(range(num_layers)),
        )
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.xlstm_stack(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out


class T5ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(T5ClassificationModel, self).__init__()
        config = T5Config(d_model=input_size, d_ff=hidden_size, num_layers=num_layers, num_heads=2 if input_size % 2 == 0 else 1)
        self.t5 = T5EncoderModel(config)
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        outputs = self.t5(inputs_embeds=x).last_hidden_state
        pooled = outputs.mean(dim=1)
        out = self.fc(pooled)
        return out


class BERTClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BERTClassificationModel, self).__init__()
        config = BertConfig(hidden_size=input_size, intermediate_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=2 if input_size % 2 == 0 else 1)
        self.bert = BertModel(config)
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        outputs = self.bert(inputs_embeds=x).last_hidden_state
        pooled = outputs.mean(dim=1)
        out = self.fc(pooled)
        return out


class GPTLikeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GPTLikeModel, self).__init__()
        nhead = 2 if input_size % 2 == 0 else 1
        decoder_layer = nn.TransformerDecoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        memory = torch.zeros_like(x).to(x.device)
        x = self.decoder(x, memory)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out


def train_and_evaluate(model, X_train, y_train, X_valid, y_valid, X_test, y_test, validation_string, enc, symbols, layer, unit):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate if not isinstance(model, (T5ClassificationModel, BERTClassificationModel, GPTLikeModel)) else 5e-5)
    scaler = GradScaler()
    
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    start_time = time.time()
    best_loss = float('inf')
    patience = 10
    epochs_used = 0
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                with autocast():
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
        val_loss /= len(valid_loader)
        
        epochs_used += 1
        if val_loss <= stopping_loss:
            break
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    training_time = time.time() - start_time
    memory_usage = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    model.eval()
    forecast = []
    current_seq = list(validation_string[:window_size])
    with torch.no_grad():
        for _ in range(validation_length):
            seq_encoded = enc.transform(np.array(current_seq).reshape(1, -1)).reshape(1, window_size, -1)
            seq_tensor = torch.tensor(seq_encoded, dtype=torch.float32).to(device)
            with autocast():
                pred = model(seq_tensor)
            next_symbol = symbols[torch.argmax(pred, dim=1).item()]
            forecast.append(next_symbol)
            current_seq = current_seq[1:] + [next_symbol]
    
    forecast_str = ''.join(forecast)
    dl_dist = normalized_damerau_levenshtein_distance(validation_string, forecast_str)
    jw_dist = normalized_jaro_winkler_distance(validation_string, forecast_str)
    
    return training_time, dl_dist, jw_dist, epochs_used, memory_usage


def generate_visualizations(results_df):
    for metric, title, ylabel, filename, log_scale in [
        ('train_time', 'Training Time by Model and Layers (Low Complexity)', 'Time (seconds)', 'time_violin.png', True),
        ('DL', 'Normalized Damerau-Levenshtein Distance by Model and Layers', 'DL Distance', 'dl_violin.png', False),
        ('JW', 'Normalized Jaro-Winkler Distance by Model and Layers', 'JW Distance', 'jw_violin.png', False),
        ('epochs', 'Epochs to Convergence by Model and Layers', 'Epochs', 'epochs_violin.png', False),
        ('memory', 'Memory Usage by Model and Layers (Low Complexity)', 'Memory (MB)', 'memory_violin.png', False)
    ]:
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='model', y=metric, hue='layers', data=results_df, split=True)
        plt.title(title)
        plt.ylabel(ylabel)
        if log_scale:
            plt.yscale('log')
        plt.legend(title='Layers')
        plt.tight_layout()
        plt.savefig(f'figures_high/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

    for metric in ['DL', 'JW']:
        plt.figure(figsize=(12, 8))
        pivot = results_df.pivot_table(values=metric, index='layers', columns='units', aggfunc='mean')
        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
        plt.title(f'Mean {metric} by Layers and Units (All Models)')
        plt.tight_layout()
        plt.savefig(f'figures_high/{metric.lower()}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    summary = results_df.groupby(['model', 'layers']).agg({
        'train_time': ['median', 'std'],
        'DL': ['median', 'std'],
        'JW': ['median', 'std'],
        'epochs': ['median', 'std'],
        'memory': ['median', 'std']
    }).round(3)
    print("\nSummary Table:")
    print(summary)
    summary.to_csv('figures_high/summary_table.csv')


def run_experiments():
    for symbols in symbols_list:
        print(f"Processing symbols: {symbols}")
        for complexity in complexities:
            if complexity < symbols:
                continue


            result_file = f"results_partial_high/symbols_{symbols}_complexity_{complexity}.csv"
            if os.path.exists(result_file):
                print(f"Skipping symbols: {symbols}, complexity: {complexity} (results already exist)")
                continue
            print(f"Processing complexity: {complexity}")
            results = []
            df_strings = generate_strings(symbols, [complexity])
            for _, row in df_strings.iterrows():
                seed_string = row['string']
                X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, unique_symbols = prepare_data(seed_string, window_size, validation_length)
                
                for layer in layers:
                    for unit in units:
                        for run in range(num_runs):
                            # LSTM
                            lstm_model = LSTMModel(len(unique_symbols), unit, len(unique_symbols), num_layers=layer)
                            lstm_time, lstm_dl, lstm_jw, lstm_epochs, lstm_memory = train_and_evaluate(
                                lstm_model, X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, unique_symbols, layer, unit)
                            
                            # Transformer
                            transformer_model = TransformerModel(len(unique_symbols), unit, len(unique_symbols), num_layers=layer)
                            trans_time, trans_dl, trans_jw, trans_epochs, trans_memory = train_and_evaluate(
                                transformer_model, X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, unique_symbols, layer, unit)
                            
                            # xLSTM
                            xlstm_model = xLSTMModel(len(unique_symbols), unit, len(unique_symbols), num_layers=layer)
                            xlstm_time, xlstm_dl, xlstm_jw, xlstm_epochs, xlstm_memory = train_and_evaluate(
                                xlstm_model, X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, unique_symbols, layer, unit)
                            
                            # T5
                            t5_model = T5ClassificationModel(len(unique_symbols), unit, len(unique_symbols), num_layers=layer)
                            t5_time, t5_dl, t5_jw, t5_epochs, t5_memory = train_and_evaluate(
                                t5_model, X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, unique_symbols, layer, unit)
                            
                            # BERT
                            bert_model = BERTClassificationModel(len(unique_symbols), unit, len(unique_symbols), num_layers=layer)
                            bert_time, bert_dl, bert_jw, bert_epochs, bert_memory = train_and_evaluate(
                                bert_model, X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, unique_symbols, layer, unit)
                            
                            # GPT-like
                            gpt_model = GPTLikeModel(len(unique_symbols), unit, len(unique_symbols), num_layers=layer)
                            gpt_time, gpt_dl, gpt_jw, gpt_epochs, gpt_memory = train_and_evaluate(
                                gpt_model, X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, unique_symbols, layer, unit)
                            
                            # Store results
                            for model, train_time, dl, jw, epochs, memory in [
                                ('LSTM', lstm_time, lstm_dl, lstm_jw, lstm_epochs, lstm_memory),
                                ('Transformer', trans_time, trans_dl, trans_jw, trans_epochs, trans_memory),
                                ('xLSTM', xlstm_time, xlstm_dl, xlstm_jw, xlstm_epochs, xlstm_memory),
                                ('T5', t5_time, t5_dl, t5_jw, t5_epochs, t5_memory),
                                ('BERT', bert_time, bert_dl, bert_jw, bert_epochs, bert_memory),
                                ('GPT-like', gpt_time, gpt_dl, gpt_jw, gpt_epochs, gpt_memory)
                            ]:
                                results.append({
                                    'symbols': symbols,
                                    'complexity': complexity,
                                    'layers': layer,
                                    'units': unit,
                                    'run': run,
                                    'model': model,
                                    'train_time': train_time,
                                    'DL': dl,
                                    'JW': jw,
                                    'epochs': epochs,
                                    'memory': memory
                                })

            partial_df = pd.DataFrame(results)
            partial_df.to_csv(result_file, index=False)
            print(f"Saved results for symbols: {symbols}, complexity: {complexity}")


def merge_results():
    all_results = []
    for symbols in symbols_list:
        for complexity in complexities:
            if complexity < symbols:
                continue
            result_file = f"results_partial_high/symbols_{symbols}_complexity_{complexity}.csv"
            if os.path.exists(result_file):
                partial_df = pd.read_csv(result_file)
                all_results.append(partial_df)
            else:
                print(f"Missing results for symbols: {symbols}, complexity: {complexity}")
    if not all_results:
        raise ValueError("No partial results found to merge.")
    return pd.concat(all_results, ignore_index=True)

if __name__ == "__main__":
    run_experiments()
    
    results_df = merge_results()
    generate_visualizations(results_df)