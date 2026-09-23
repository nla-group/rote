import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from slearn import lzw_string_seeds
from slearn.dmetric import (normalized_damerau_levenshtein_distance, normalized_jaro_winkler_distance)
from sklearn.preprocessing import OneHotEncoder
import time
import os
import shutil
import logging
import random
import warnings
import sys
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from models import get_model, TransformerModel, GPTLikeModel
from itertools import product 

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("experiment_log.log"),
    logging.StreamHandler(sys.stdout)
])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 3407
random.seed(42)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True
np.random.seed(2)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- EXPERIMENT CONFIGURATION ---
symbols_list = [2, 4, 6, 8]
complexities = [10, 30, 50, 70, 90]
window_size = 100
validation_length = 100
base_stopping_loss = 0.1
max_epochs = 1500
num_runs = 2 # I set it to 3 for averaged results; Increased runs for better statistical significance

# Scaling Law Constant (T = C * N)
SCALING_TOKENS_PER_MILLION_PARAMS = 20 

# ---  ENSURING MINIMUM DATA SAMPLES
# Required minimum sequence length to generate enough X, y pairs for training/validation/test split.
MIN_TRAINING_SAMPLES_BUFFER = 200
MIN_TOTAL_SEQUENCE_LENGTH = window_size + validation_length + MIN_TRAINING_SAMPLES_BUFFER 
# ----------------------------------------

# Tuned Parameters
units = [64, 128, 256] 
layers = [1, 2, 3]
batch_sizes = [128, 256] 


tuning_params = [ 
    {'optim': 'AdamW', 'lr': 1e-4, 'wd': 0.01}, 
] 


LR_STEP_SIZE = 100 
LR_GAMMA = 0.5    

# Parameters for Transformer/BERT/GPT
d_models = [256, 512] 
NUM_HEAD = 8 
max_strings_per_complexity = 1000

# Create output directories
os.makedirs("figures_low", exist_ok=True)
os.makedirs("results_partial_low", exist_ok=True)

# Clear PyTorch extensions cache
cache_dir = os.path.expanduser("~/.cache/torch_extensions")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    logging.info(f"Cleared cache directory: {cache_dir}")


# --- UTILITY FUNCTIONS ---
def generate_strings(symbols, complexities):
    all_strings = []
    target_num_strings = symbols * len(complexities)
    for complexity in complexities:
        try:
            df = lzw_string_seeds(symbols=symbols, complexity=[complexity], random_state=42)
            if df.empty or not all(isinstance(s, str) and len(s) > 0 for s in df['string']):
                logging.warning(f"No valid strings generated for symbols={symbols}, complexity={complexity}")
                continue
            num_to_sample = min(len(df), max_strings_per_complexity)
            sampled_df = df.sample(n=num_to_sample, random_state=0) if num_to_sample < len(df) else df
            all_strings.append(sampled_df)
        except Exception as e:
            logging.warning(f"Error generating strings for symbols={symbols}, complexity={complexity}: {str(e)}")
            continue
    
    if not all_strings:
        logging.error(f"No valid strings generated for symbols={symbols} across all complexities")
        raise ValueError("No valid strings generated")
    
    df_strings = pd.concat(all_strings, ignore_index=True)
    if len(df_strings) < target_num_strings:
        logging.warning(f"Only {len(df_strings)} strings generated for symbols={symbols}, expected {target_num_strings}")
    elif len(df_strings) > target_num_strings:
        df_strings = df_strings.sample(n=target_num_strings, random_state=0)
    
    logging.debug(f"Generated {len(df_strings)} strings for symbols={symbols}")
    return df_strings

def prepare_data(seed_string, window_size, validation_length, target_length):
    repeats = int(np.ceil(target_length / len(seed_string)))
    s = seed_string * repeats
    s = s[:target_length]
    v = s[-validation_length:]
    train_test = s[:-validation_length]
    
    X, y = [], []
    for i in range(len(train_test) - window_size):
        X.append(list(train_test[i:i + window_size]))
        y.append(train_test[i + window_size])
    
    symbols = sorted(list(set(seed_string)))
    enc = OneHotEncoder(sparse_output=False, categories=[symbols] * window_size)
    X_encoded = enc.fit_transform(X).reshape(len(X), window_size, -1)
    y_encoded = OneHotEncoder(sparse_output=False, categories=[symbols]).fit_transform(np.array(y).reshape(-1, 1))
    
    total_samples = len(X_encoded)
    train_size = int(0.8 * total_samples)
    valid_size = int(0.1 * total_samples)
    
    X_train = X_encoded[:train_size]
    X_valid = X_encoded[train_size:train_size + valid_size]
    X_test = X_encoded[train_size + valid_size: total_samples]
    
    y_train = y_encoded[:train_size]
    y_valid = y_encoded[train_size:train_size + valid_size]
    y_test = y_encoded[train_size + valid_size: total_samples]
    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Check if data splits are empty after tokenization/sampling
    if X_valid.shape[0] == 0 or y_valid.shape[0] == 0 or X_train.shape[0] == 0:
        logging.error("Empty validation data generated")
        raise ValueError("Validation data is empty")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, symbols

def bootstrap_ci(data, n_boot=1000, ci=95):
    """Calculates bootstrap confidence interval for the median."""
    if len(data) == 0: return np.nan, np.nan
    bootstraps = [np.median(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(bootstraps, (100 - ci) / 2)
    upper = np.percentile(bootstraps, 100 - (100 - ci) / 2)
    return lower, upper


# --- EVALUATION AND SCALING FUNCTION ---
def train_and_evaluate(model, X_train, y_train, X_valid, y_valid, X_test, y_test, validation_string, enc, symbols, params, model_size, batch_size):
    """Trains the model and evaluates performance and efficiency."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if params['optim'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    elif params['optim'] == 'AdamW':
        # This branch handles the AdamW optimization with weight decay (wd)
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    else:
        raise ValueError(f"Unknown optimizer: {params['optim']}")
    
    # This applies the StepLR scheduler to the chosen optimizer
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    if len(valid_loader) == 0:
        raise ValueError("Empty validation loader")
    if len(test_loader) == 0:
        raise ValueError("Empty test loader")
    
    start_time = time.time()
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    epochs_used = 0
    
    # Adaptive stopping loss threshold (Scaling Law concept)
    stopping_loss = base_stopping_loss * (1 + np.log10(max(1, model_size / 1e6)))
    
    try:
        for epoch in range(max_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Transformer requires a target sequence
                if isinstance(model, (TransformerModel)):
                    # Target is the input sequence shifted right by one, plus the actual target token
                    tgt = torch.cat([X_batch[:, 1:, :], y_batch.unsqueeze(1)], dim=1)
                    outputs = model(X_batch, tgt)
                else: 
                    outputs = model(X_batch)
                    
                loss = criterion(outputs, y_batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"Invalid training loss: {loss.item()}")
                    raise ValueError("Invalid training loss")
                
                loss.backward()
                optimizer.step()
            
            scheduler.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    if isinstance(model, (TransformerModel)):
                        tgt = torch.cat([X_batch[:, 1:, :], y_batch.unsqueeze(1)], dim=1)
                        outputs = model(X_batch, tgt)
                    else:
                        outputs = model(X_batch)
                        
                    batch_loss = criterion(outputs, y_batch).item()
                    if not np.isnan(batch_loss) and not np.isinf(batch_loss):
                        val_loss += batch_loss
                    else:
                        logging.error(f"Invalid validation batch loss: {batch_loss}")
                        raise ValueError("Invalid validation batch loss")
            val_loss /= len(valid_loader)
            
            if np.isnan(val_loss) or np.isinf(val_loss):
                logging.error(f"Invalid validation loss: {val_loss}")
                raise ValueError("Invalid validation loss")
            
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
    except ZeroDivisionError as e:
        logging.error(f"ZeroDivisionError in training: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if isinstance(model, (TransformerModel)):
                tgt = torch.cat([X_batch[:, 1:, :], y_batch.unsqueeze(1)], dim=1)
                outputs = model(X_batch, tgt)
            else:
                outputs = model(X_batch)
                
            batch_loss = criterion(outputs, y_batch).item()
            if not np.isnan(batch_loss) and not np.isinf(batch_loss):
                test_loss += batch_loss
            else:
                logging.error(f"Invalid test batch loss: {batch_loss}")
                raise ValueError("Invalid test batch loss")
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(y_batch, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = correct / total if total > 0 else 0
    
    training_time = time.time() - start_time
    memory_usage = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
    time_per_epoch = training_time / epochs_used if epochs_used > 0 else 0
    
    current_seq = list(validation_string[-window_size:])
    forecast = []
    
    with torch.no_grad():
        for _ in range(validation_length):
            seq_encoded = enc.transform(np.array(current_seq).reshape(1, -1)).reshape(1, window_size, -1)
            seq_tensor = torch.tensor(seq_encoded, dtype=torch.float32).to(device)
            
            # For Transformer-based models, the input sequence is used as both source and target in auto-regressive manner
            if isinstance(model, (TransformerModel, GPTLikeModel)):
                if isinstance(model, TransformerModel):
                    # For prediction, we use the current sequence as both src and tgt
                    outputs = model(seq_tensor, seq_tensor) 
                elif isinstance(model, GPTLikeModel):
                    outputs = model(seq_tensor)
            else:
                outputs = model(seq_tensor)

            # Predict the next token (last output of the sequence)
            pred = outputs[:, -1, :] if outputs.ndim == 3 else outputs 

            next_symbol = symbols[torch.argmax(pred, dim=-1).item()]
            forecast.append(next_symbol)
            current_seq = current_seq[1:] + [next_symbol]
    
    forecast_str = ''.join(forecast)
    dl_dist = normalized_damerau_levenshtein_distance(validation_string, forecast_str)
    jw_dist = normalized_jaro_winkler_distance(validation_string, forecast_str)
    
    return training_time, dl_dist, jw_dist, epochs_used, memory_usage, model_size, time_per_epoch, test_loss, test_accuracy


def power_law_fit(N, A, alpha, C):
    """Power law function for scaling law analysis: Loss = A/N^alpha + C"""
    return A * (N ** (-alpha)) + C

def plot_scaling_laws(results_df):
    """Analyzes and plots the relationship between model size and test loss."""
    logging.info("Starting Scaling Law Analysis...")
    
    # 1. Prepare data: Calculate median loss for each model size configuration
    scaling_df = results_df.groupby(['model', 'model_size']).agg({
        'test_loss': 'median',
        'test_accuracy': 'median'
    }).reset_index()
    
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='model_size', y='test_loss', hue='model', data=scaling_df, style='model', s=100)
    
    # 2. Fit power law for each model type
    model_types = scaling_df['model'].unique()
    for model_name in model_types:
        df_model = scaling_df[scaling_df['model'] == model_name].copy()
        
        # Ensure there are enough distinct model sizes to fit a curve
        if len(df_model['model_size'].unique()) < 3:
            logging.warning(f"Skipping power law fit for {model_name}: insufficient model sizes.")
            continue
            
        N = df_model['model_size'].values
        Loss = df_model['test_loss'].values
        
        # Attempt to fit the power law
        try:
            # Initial guesses for A, alpha, C
            p0 = [1.0, 0.5, Loss.min() * 0.9] 
            popt, pcov = curve_fit(power_law_fit, N, Loss, p0=p0, maxfev=5000)
            
            A, alpha, C = popt
            
            # Plot the fitted curve
            N_fit = np.logspace(np.log10(N.min()), np.log10(N.max()), 100)
            Loss_fit = power_law_fit(N_fit, *popt)
            plt.plot(N_fit, Loss_fit, label=f'{model_name} (α={alpha:.2f})', linestyle='--')
            
            logging.info(f"{model_name} Scaling Law: Loss = {A:.2f}/N^{alpha:.2f} + {C:.4f}")
            
        except RuntimeError:
            logging.warning(f"Could not fit power law for {model_name}.")

    plt.xscale('log')
    plt.title(r"Scaling Laws: Median Test Loss vs. Model Size (M) $[L \propto N^{-\alpha}]$")
    plt.xlabel('Model Size (Millions of Parameters, log scale)')
    plt.ylabel('Median Test Loss (CrossEntropy)')
    plt.legend(title='Model Type')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('figures_low/scaling_law_test_loss.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_sequence_analysis(results_df):
    """Plots performance metrics as a function of sequence length."""
    logging.info("Starting Sequence Length Analysis...")
    
    # Calculate median performance for each model and sequence length
    seq_df = results_df.groupby(['model', 'sequence_length']).agg({
        'test_accuracy': 'median',
        'DL': 'median'
    }).reset_index()
    
    # Plot 1: Test Accuracy vs. Sequence Length
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='sequence_length', y='test_accuracy', hue='model', data=seq_df, marker='o', linewidth=2.5, palette='tab10')
    plt.title('Median Test Accuracy vs. Input Sequence Length')
    plt.xlabel('Training Sequence Length')
    plt.ylabel('Median Test Accuracy')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig('figures_low/seq_len_accuracy.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: DL Distance vs. Sequence Length
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='sequence_length', y='DL', hue='model', data=seq_df, marker='o', linewidth=2.5, palette='tab10')
    plt.title('Median Damerau-Levenshtein Distance vs. Input Sequence Length')
    plt.xlabel('Training Sequence Length')
    plt.ylabel('Median DL Distance (Lower is Better)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig('figures_low/seq_len_dl_distance.jpg', dpi=300, bbox_inches='tight')
    plt.close()


def generate_visualizations(results_df):
    """Generates visualizations and summary statistics."""
    
    # --- 1. Model Size Normalization  
    results_df['acc_per_M'] = results_df['test_accuracy'] / results_df['model_size']
    results_df['time_per_M'] = results_df['train_time'] / results_df['model_size']
    
    # Plotting metrics (includes normalized metrics)
    metrics = [
        ('train_time', 'Training Time by Model and Hyperparameters', 'Time (seconds)', 'time_violin.jpg', True),
        ('time_per_M', 'Training Time per Million Parameters', 'Time (s) / M Params', 'time_per_M_violin.jpg', True),
        ('DL', 'Damerau-Levenshtein Distance by Model and Hyperparameters', 'DL Distance', 'dl_violin.jpg', False),
        ('test_accuracy', 'Test Accuracy by Model and Hyperparameters', 'Accuracy', 'test_accuracy_violin.jpg', False),
        ('acc_per_M', 'Test Accuracy per Million Parameters', 'Accuracy / M Params', 'acc_per_M_violin.jpg', False),
    ]

    sns.set_style("whitegrid")
    
    # Plotting for Model Comparison (Violin Plots)
    for metric, title, ylabel, filename, log_scale in metrics:
        if metric not in results_df.columns: continue
        plt.figure(figsize=(14, 7))
        sns.violinplot(x='model', y=metric, data=results_df, inner='quartile', linewidth=1.5, palette='viridis')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Model')
        if log_scale:
            plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'figures_low/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

    # Heatmap of performance by LR and WD
    for metric in ['test_accuracy', 'test_loss']:
        if metric in results_df.columns:
            # Note: The Hyper_Group creation will be simpler now as optimizer/lr/wd is fixed
            results_df['Hyper_Group'] = results_df['optimizer'] + '_' + results_df['lr'].astype(str) + '_wd' + results_df['wd'].astype(str) + '_bs' + results_df['batch_size'].astype(str)
            pivot = results_df.pivot_table(values=metric, index='Hyper_Group', columns='model', aggfunc='mean')
            
            plt.figure(figsize=(15, 8))
            sns.heatmap(pivot, annot=True, cmap='coolwarm', fmt='.4f')
            plt.title(f'Mean {metric} by Hyperparameter Group and Model')
            plt.tight_layout()
            plt.savefig(f'figures_low/{metric.lower()}_hyperparam_heatmap.jpg', dpi=300, bbox_inches='tight')
            plt.close()
            
    # --- 2. Scaling Law Analysis 
    plot_scaling_laws(results_df)

    # --- 3. Sequence Length Analysis 
    plot_sequence_analysis(results_df)

    # Statistical Summary
    available_metrics = [m for m in ['train_time', 'DL', 'JW', 'epochs', 'memory', 'model_size', 'time_per_epoch', 'test_loss', 'test_accuracy', 'acc_per_M', 'time_per_M'] if m in results_df.columns]
    if available_metrics:
        summary = results_df.groupby(['model', 'layers', 'units', 'optimizer', 'lr', 'wd', 'batch_size']).agg({
            metric: ['median', 'std'] for metric in available_metrics
        }).round(5)
        print("\nSummary Statistics (Grouped by Full Configuration):")
        print(summary)
        summary.to_csv('figures_low/summary_statistics_detailed.csv')

def get_unique_model_configs(model_names_to_run, layers, units, d_models):
    """
    Generates all unique (model_name, layer, unit, d_model) combinations
    and pre-calculates the model size for dynamic sequence length calculation.
    """
    model_configs = []
    
    # Create a dummy unique_symbols length for initialization (max possible: 52)
    dummy_input_size = 52 
    dummy_output_size = 52

    
    # Iterate over all hyperparameter combinations
    all_combinations = list(product(model_names_to_run, layers, units, d_models))

    for model_name, layer, unit, d_model in all_combinations:
        # Skip redundant runs for RNNs which only use 'units' (hidden_size)
        if model_name in ['LSTM', 'GRU', 'minGRU', 'minLSTM'] and d_model != d_models[0]:
            continue 

        try:
            # Instantiate model to get parameter count (model size)
            model = get_model(model_name, dummy_input_size, unit, dummy_output_size, layer, d_model=d_model)
            model_size_params = sum(p.numel() for p in model.parameters())
            model_size_M = model_size_params / 1e6
            
            model_configs.append({
                'model_name': model_name,
                'layer': layer,
                'unit': unit,
                'd_model': d_model,
                'model_size_M': model_size_M
            })

        except Exception as e:
            logging.error(f"Error calculating size for {model_name} L{layer} U{unit} D{d_model}: {str(e)}")
            continue

    return model_configs


# --- MAIN EXPERIMENT FUNCTION (MODIFIED AND FIXED) ---

def run_experiments():
    all_results = []
    model_names_to_run = ['LSTM', 'GRU', 'minGRU', 'minLSTM', 'Transformer', 'BERT', 'GPT', 'LinearAttention', 'Performer', 'RWKV']

    # 1. Pre-calculate all unique model configurations and their sizes
    unique_model_configs = get_unique_model_configs(model_names_to_run, layers, units, d_models)
    
    if not unique_model_configs:
        logging.error("No valid model configurations generated. Exiting.")
        return

    # 2. Start iterating over sequence complexity parameters
    for symbols in symbols_list:
        logging.info(f"\nProcessing symbols: {symbols}")
        try:
            df_strings = generate_strings(symbols, complexities)
            
            for complexity in complexities:
                logging.info(f"Processing complexity: {complexity}")
                
                complexity_df = df_strings[df_strings['LZW_complexity'] == complexity]
                if complexity_df.empty:
                    logging.warning(f"No strings available for symbols={symbols}, complexity={complexity}")
                    continue
                
                for _, row in complexity_df.iterrows():
                    seed_string = str(row['string'])
                    
                    # 3. Iterate over pre-calculated model configurations
                    for config in unique_model_configs:
                        model_name = config['model_name']
                        layer = config['layer']
                        unit = config['unit']
                        d_model = config['d_model']
                        model_size_M = config['model_size_M']
                        
                        # --- DYNAMIC SEQUENCE LENGTH CALCULATION ---
                        dynamic_length = int(SCALING_TOKENS_PER_MILLION_PARAMS * model_size_M) 
                        
                        # Enforce T meets the absolute minimum requirement for data splitting
                        target_length = max(dynamic_length, MIN_TOTAL_SEQUENCE_LENGTH) 
                        logging.info(f"Calculated target_length for {model_name} (Size={model_size_M:.3f}M): {target_length} (Min enforced: {MIN_TOTAL_SEQUENCE_LENGTH})")
                        # -------------------------------------------
                        
                        # Outer TRY block for data preparation and inner loops (This starts the sequence-specific part)
                        try:
                            # DATA PREPARATION: (runs only once per T)
                            X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, unique_symbols = prepare_data(seed_string, window_size, validation_length, target_length)
                            
                            for params in tuning_params:
                                for batch_size in batch_sizes: 
                                    for run in range(num_runs):
                                        print(f"===================================================",
                                                f"\nModel: {model_name}, L{layer}, U{unit}, D{d_model}, BS{batch_size}, Run {run}. T={target_length}",
                                                f"\n===================================================")

                                        # Inner TRY block for model training (THE MOST INNER LOOP)
                                        try:
                                            # Re-instantiate model using the correct input_size (len(unique_symbols))
                                            model = get_model(model_name, len(unique_symbols), unit, len(unique_symbols), layer, d_model=d_model)
                                            
                                            train_time, dl, jw, epochs, memory, _, time_per_epoch, test_loss, test_accuracy = train_and_evaluate(
                                                model, X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, unique_symbols, params, model_size_M, batch_size)
                                            
                                            results = {
                                                'run': run,
                                                'model': model_name,
                                                'complexity': complexity,
                                                'symbols': symbols,
                                                'sequence_length': target_length, 
                                                'layers': layer,
                                                'units': unit,
                                                'd_model': d_model,
                                                'optimizer': params['optim'],
                                                'lr': params['lr'],
                                                'wd': params['wd'],
                                                'batch_size': batch_size,
                                                'model_size': model_size_M,
                                                'train_time': train_time,
                                                'epochs': epochs,
                                                'time_per_epoch': time_per_epoch,
                                                'memory': memory,
                                                'test_loss': test_loss,
                                                'test_accuracy': test_accuracy,
                                                'DL': dl,
                                                'JW': jw
                                            }

                                            all_results.append(results)
                                            print(
                                                f"Results: T={target_length}, loss={test_loss:.3f}, "
                                                f"acc={test_accuracy:.3f}, "
                                                f"DL={dl:.3f}, "
                                                f"Time={train_time:.1f}s, "
                                                f"Size={model_size_M:.3f}M"
                                            )
                                        except Exception as e:
                                            logging.error(f"Run failed for {model_name} L{layer} U{unit} D{d_model} T{target_length}: {str(e)}")
                                            continue # Continue to the next run
                                    
                                    print("---------------------------------------------")
                                    
                        # Exception handling for the Outer TRY block (Data preparation or Inner Loops failure)
                        except ValueError as e:
                            logging.error(f"Data prep failed for complexity {complexity} (T={target_length}): {str(e)}")
                        except Exception as e:
                            logging.error(f"General error in model configuration loop (Model: {model_name}): {str(e)}")
        
        except Exception as e:
            logging.error(f"Error generating or processing strings for symbols {symbols}: {str(e)}")

    # 4. Final Processing and Visualization (outside all loops and try blocks)
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('results_low_scaling_law.csv', index=False)
        generate_visualizations(results_df)
    else:
        logging.warning("No successful experiments were completed.")

if __name__ == '__main__':
    run_experiments()
