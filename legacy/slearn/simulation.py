import numpy as np
import pandas as pd
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .sgenerate import lzw_string_seeds
from .dmetric import (normalized_damerau_levenshtein_distance, normalized_jaro_winkler_distance)
from .deep_models import (LSTMModel, GRUModel, TransformerModel, GPTLikeModel)
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import os
import shutil
import logging
import scipy.stats as stats

import random

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow warnings


random.seed(42) # Set random seeds for reproducibility
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True
np.random.seed(2)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3407)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device configuration
# logging.info(f"Using device: {device}")

# Create output directories
os.makedirs("figures", exist_ok=True)
os.makedirs("results_partial", exist_ok=True)

# Clear PyTorch extensions cache
cache_dir = os.path.expanduser("~/.cache/torch_extensions")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    logging.info(f"Cleared cache directory: {cache_dir}")


def generate_strings(symbols, complexities, max_strings_per_complexity=1000):
    all_strings = []
    target_num_strings = symbols * len(complexities)
    # strings_per_complexity = max(1, target_num_strings // len(complexities))
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



def prepare_data(seed_string, window_size, validation_length, target_length, train_split=0.8, valid_split=0.1):
    repeats = int(np.ceil(target_length / len(seed_string)))
    s = seed_string * repeats
    s = s[:target_length]
    print("validation_length:", validation_length)
    v = s[-validation_length:]
    train_test = s[:-validation_length]
    
    print("train_test, validation, :", len(train_test), len(v))
    X, y = [], []
    for i in range(len(train_test) - window_size):
        X.append(list(train_test[i:i + window_size]))
        y.append(train_test[i + window_size])
    
    symbols = sorted(list(set(seed_string)))
    enc = OneHotEncoder(sparse_output=False, categories=[symbols] * window_size)
    X_encoded = enc.fit_transform(X).reshape(len(X), window_size, -1)
    y_encoded = OneHotEncoder(sparse_output=False, categories=[symbols]).fit_transform(np.array(y).reshape(-1, 1))
    
    train_size = int(train_split * len(X_encoded))
    valid_size = int(valid_split * len(X_encoded))
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
    
    logging.debug(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_valid={X_valid.shape}, y_valid={y_valid.shape}, X_test={X_test.shape}, y_test={y_test.shape}")
    
    if X_valid.shape[0] == 0 or y_valid.shape[0] == 0:
        logging.error("Empty validation data generated")
        raise ValueError("Validation data is empty")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, symbols



def train_and_evaluate(model, X_train, y_train, X_valid, y_valid, X_test, y_test, 
                    validation_string, enc, symbols, learning_rate, batch_size, window_size, validation_length,
                    max_epochs, stopping_loss):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    if len(valid_loader) == 0:
        logging.error("Valid loader is empty, cannot train model")
        raise ValueError("Empty validation loader")
    if len(test_loader) == 0:
        logging.error("Test loader is empty, cannot evaluate model")
        raise ValueError("Empty test loader")
    
    start_time = time.time()
    best_loss = float('inf')
    patience = 10
    epochs_used = 0
    
    try:
        for epoch in range(max_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                if isinstance(model, (TransformerModel)):
                    tgt = torch.cat([X_batch[:, 1:, :], y_batch.unsqueeze(1)], dim=1)
                    outputs = model(X_batch, tgt)
                elif isinstance(model, GPTLikeModel):
                    outputs = model(X_batch)
                else:
                    outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"Invalid training loss: {loss.item()}")
                    raise ValueError("Invalid training loss")
                
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    if isinstance(model, (TransformerModel)):
                        tgt = torch.cat([X_batch[:, 1:, :], y_batch.unsqueeze(1)], dim=1)
                        outputs = model(X_batch, tgt)
                    elif isinstance(model, GPTLikeModel):
                        outputs = model(X_batch)
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

    # Test evaluation
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if isinstance(model, (TransformerModel)):
                tgt = torch.cat([X_batch[:, 1:, :], y_batch.unsqueeze(1)], dim=1)
                outputs = model(X_batch, tgt)
            elif isinstance(model, GPTLikeModel):
                outputs = model(X_batch)
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
    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    
    current_seq = list(validation_string[-window_size:])
    forecast = []
    
    with torch.no_grad():
        for _ in range(validation_length):
            seq_encoded = enc.transform(np.array(current_seq).reshape(1, -1)).reshape(1, window_size, -1)
            seq_tensor = torch.tensor(seq_encoded, dtype=torch.float32).to(device)
            if isinstance(model, (TransformerModel)):
                tgt = seq_tensor[:, -1:, :]
                pred = model(seq_tensor, tgt)
            elif isinstance(model, GPTLikeModel):
                pred = model(seq_tensor)
            else:
                pred = model(seq_tensor)
            next_symbol = symbols[torch.argmax(pred, dim=-1).item()]
            forecast.append(next_symbol)
            current_seq = current_seq[1:] + [next_symbol]
    
    forecast_str = ''.join(forecast)
    dl_dist = normalized_damerau_levenshtein_distance(validation_string, forecast_str)
    jw_dist = normalized_jaro_winkler_distance(validation_string, forecast_str)
    
    return training_time, dl_dist, jw_dist, epochs_used, memory_usage, model_size, time_per_epoch, test_loss, test_accuracy


def bootstrap_ci(data, n_boot=1000, ci=95):
    bootstraps = [np.median(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(bootstraps, (100 - ci) / 2)
    upper = np.percentile(bootstraps, 100 - (100 - ci) / 2)
    return lower, upper


def generate_visualizations(results_df):
    metrics = [
        ('train_time', 'Training Time by Model and Layers', 'Time (seconds)', 'time_violin.png', True, 'violin'),
        ('DL', 'Damerau-Levenshtein Distance by Model and Layers', 'DL Distance', 'dl_violin.png', False, 'violin'),
        ('JW', 'Jaro-Winkler Distance by Model and Layers', 'JW Distance', 'jw_violin.png', False, 'violin'),
        ('epochs', 'Epochs to Convergence by Model and Layers', 'Epochs', 'epochs_violin.png', False, 'violin'),
        ('memory', 'Memory Usage by Model and Layers', 'Memory (MB)', 'memory_violin.png', False, 'violin'),
        ('model_size', 'Model Size by Model and Layers', 'Parameters (Millions)', 'model_size_bar.png', False, 'bar'),
        ('time_per_epoch', 'Training Time per Epoch by Model and Layers', 'Time/Epoch (seconds)', 'time_per_epoch_violin.png', True, 'violin'),
        ('test_loss', 'Test Loss by Model and Layers', 'Loss', 'test_loss_violin.png', False, 'violin'),
        ('test_accuracy', 'Test Accuracy by Model and Layers', 'Accuracy', 'test_accuracy_violin.png', False, 'violin')
    ]

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 16, 'axes.titlesize': 16})

    for metric, title, ylabel, filename, log_scale, plot_type in metrics:
        if metric not in results_df.columns:
            logging.warning(f"Metric '{metric}' not found in results_df. Skipping visualization.")
            continue
        plt.figure(figsize=(12, 6))

        if plot_type == 'violin':
            palette = sns.color_palette("Set2", n_colors=len(results_df['layers'].unique()))
            sns.violinplot(
                x='model',
                y=metric,
                hue='layers',
                data=results_df,
                split=True,
                palette=palette,
                inner='quartile',
                linewidth=1.5,
                alpha=0.6
            )

            plt.legend(title='Layers', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
            
        else:  
            palette = sns.color_palette("Set2", n_colors=len(results_df['layers'].unique()))
            g = sns.catplot(
                x='model',
                y=metric,
                hue='layers',
                data=results_df,
                kind='bar',
                palette=palette,
                height=6,
                aspect=2,
                dodge=True, 
                errorbar=('ci', 95)  
            )
          
            g._legend.remove()

            g.ax.legend(
                fontsize=16,
                title='layers',
                title_fontsize=16,
                loc='upper center',
                bbox_to_anchor=(1, 1.15),
                ncol=1
            )

        plt.title(title, pad=15)
        plt.ylabel(ylabel)
        plt.xlabel('Model')
        if log_scale:
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'figures/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

    for metric in ['DL', 'JW', 'test_loss', 'test_accuracy']:
        if metric not in results_df.columns:
            logging.warning(f"Metric '{metric}' not found in results_df. Skipping heatmap.")
            continue

        plt.figure(figsize=(12, 8))
        pivot = results_df.pivot_table(values=metric, index='layers', columns='units', aggfunc='mean')
        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.4f')
        plt.title(f'Mean {metric} by Layers and Units')
        plt.tight_layout()
        plt.savefig(f'figures/{metric.lower()}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    available_metrics = [m for m in ['train_time', 'DL', 'JW', 'epochs', 'memory', 'model_size', 
                                    'time_per_epoch', 'test_loss', 'test_accuracy'] if m in results_df.columns]
    stats_results = []
    for metric in ['DL', 'JW', 'test_accuracy']:
        if metric in results_df.columns:
            groups = [results_df[results_df['model'] == model][metric].values for model in results_df['model'].unique()]
            if all(len(g) > 0 for g in groups):
                stat, p = stats.kruskal(*groups)
                stats_results.append({'Metric': metric, 'Kruskal-Wallis H': stat, 'p-value': p})
    
    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        print("\nStatistical Tests:")
        print(stats_df)
        stats_df.to_csv('figures/statistical_tests.csv', index=False)

    if available_metrics:
        summary = results_df.groupby(['model', 'layers']).agg({
            metric: [
                'median',
                'std',
                lambda x: bootstrap_ci(x)[0],
                lambda x: bootstrap_ci(x)[1]
            ] for metric in available_metrics
        }).round(3)
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        print("\nSummary Statistics (with 95% CI):")
        print(summary)
        summary.to_csv('figures/summary_statistics.csv')
    else:
        logging.error("No valid metrics found in results_df for summary statistics.")



def run_simulations(models=None, symbols_list=[2, 4, 6, 8], 
                    complexities=[210, 230, 250, 270, 290], learning_rates=[1e-3, 1e-4], 
                    max_strings_per_complexity=1000, sequence_lengths=[3500], window_size=100, validation_length=100, 
                    units=[128], layers=[1, 2, 3], num_runs=5, train_split=0.8, valid_split=0.1, batch_size=256, stopping_loss=0.1, max_epochs=999):

    if models is None:
        print("No specific models to run, exit early")
        return #

    for symbols in symbols_list:
        logging.info(f"\nProcessing symbols: {symbols}")
        try:
            df_strings = generate_strings(symbols, complexities, max_strings_per_complexity)
            for complexity in complexities:
                result_file = f"symbols_{symbols}_complexity_{complexity}.csv"
                if os.path.exists(result_file):
                    print(f"Skipping symbols: {symbols}, complexity: {complexity} (results already exist)")
                    continue
                
                logging.info(f"Processing complexity: {complexity}")
                results = []
                complexity_df = df_strings[df_strings['LZW_complexity'] == complexity]
                if complexity_df.empty:
                    logging.warning(f"No strings available for symbols={symbols}, complexity={complexity}")
                    continue
                
                for _, row in complexity_df.iterrows():
                    seed_string = str(row['string'])
                    for target_length in sequence_lengths:
                        try:
                            X_train, y_train, X_valid, y_valid, X_test, y_test, v, enc, unique_symbols = prepare_data(seed_string, window_size, 
                                                                                                                      validation_length, target_length, 
                                                                                                                      train_split, valid_split
                                                                                                                )

                            logging.info(f"Input shape: X_train={X_train.shape}, y_train={y_train.shape}, X_valid={X_valid.shape}, "
                                         f"y_valid={y_valid.shape}, X_test={X_test.shape}, y_test={y_test.shape}, unique_symbols={len(unique_symbols)}")
                            for layer in layers:
                                for unit in units:
                                    for learning_rate in learning_rates:
                                        for run in range(num_runs):
                                            print()

                                            for i in range(len(models)):
                                                try:
                                                    logging.debug(f"Initializing model_{i} with layers={layer}, units={unit}, run={run}, learning_rate={learning_rate}")
                                                    
                                                    Model = models[i]
                                                    model = Model(len(unique_symbols), unit, len(unique_symbols), num_layers=layer)

                                                    train_time, dl, jw, epochs, memory, model_size, time_per_epoch, test_loss, test_accuracy = train_and_evaluate(
                                                                     model, X_train, y_train, X_valid, y_valid, X_test, y_test,
                                                                     v, enc, unique_symbols, learning_rate, batch_size, window_size, validation_length,
                                                                     max_epochs, stopping_loss)
                                                
                                                    results.append({
                                                        'symbols': symbols,
                                                        'complexity': complexity,
                                                        'sequence_length': target_length,
                                                        'learning_rate': learning_rate,
                                                        'layers': layer,
                                                        'units': unit,
                                                        'run': run,
                                                        'model': f"model_{i}",
                                                        'train_time': train_time,
                                                        'DL': dl,
                                                        'JW': jw,
                                                        'epochs': epochs,
                                                        'memory': memory,
                                                        'model_size': model_size,
                                                        'time_per_epoch': time_per_epoch,
                                                        'test_loss': test_loss,
                                                        'test_accuracy': test_accuracy
                                                    })
                                                    print(f"******** model:{i}, DL:{dl}, JW:{jw}, Test Loss:{test_loss}, Test Accuracy:{test_accuracy}")

                                                except Exception as e:
                                                    logging.error(f"Error in model={i}, layers={layer}, units={unit}, run={run}: {str(e)}")
                                                    continue

                        except Exception as e:
                            logging.error(f"Error processing seed_string for symbols={symbols}, complexity={complexity}, sequence_length={target_length}: {str(e)}")
                            continue
                
                if results:
                    partial_df = pd.DataFrame(results)
                    partial_df.to_csv(result_file, index=False)
                    logging.info(f"Saved results for symbols: {symbols}, complexity: {complexity}")
                else:
                    logging.warning(f"No results generated for symbols: {symbols}, complexity: {complexity}")
       
        except ValueError as e:
            logging.error(f"Error generating strings for symbols={symbols}: {str(e)}")
            continue


def merge_results(symbols_list, complexities):
    all_results = []
    required_columns = ['symbols', 'complexity', 'sequence_length', 'learning_rate', 'layers', 'units',
                     'run', 'model', 'train_time', 'DL', 'JW', 'epochs', 'memory', 'model_size', 'time_per_epoch', 
                     'test_loss', 'test_accuracy']

    for symbols in symbols_list:
        for complexity in complexities:
            result_file = f"symbols_{symbols}_complexity_{complexity}.csv"
            if os.path.exists(result_file):
                partial_df = pd.read_csv(result_file)
                if not all(col in partial_df.columns for col in required_columns):
                    logging.warning(f"Result file {result_file} missing required columns. Regenerate this file.")
                    continue
                all_results.append(partial_df)
            else:
                logging.warning(f"Missing results for symbols: {symbols}, complexity: {complexity}")

    if not all_results:
        raise ValueError("No valid partial results found to merge.")
        
    return pd.concat(all_results, ignore_index=True)

  

def benchmark_models(model_list, # Experiment hyperparameters
                     symbols_list=[2, 4, 6, 8], # the number of unique symbols in the strings
                     complexities=[210, 230, 250, 270, 290],
                     sequence_lengths=[3500],
                     window_size=100,
                     validation_length=100,
                     stopping_loss=0.1,
                     max_epochs=999,
                     num_runs=5,
                     units=[128],
                     layers=[1, 2, 3],
                     batch_size=256,
                     max_strings_per_complexity=1000,
                     learning_rates=[1e-3, 1e-4]):
    #model_list = [
    #    lambda vocab_size, units, output_size, num_layers: TransformerModel(vocab_size, units, output_size, num_layers),
    #    lambda vocab_size, units, output_size, num_layers: GPTLikeModel(vocab_size, units, output_size, num_layers),
    #    lambda vocab_size, units, output_size, num_layers: SimpleRNNModel(vocab_size, units, output_size),
    #    lambda vocab_size, units, output_size, num_layers: LSTMModel(vocab_size, units, output_size),
    #    lambda vocab_size, units, output_size, num_layers: GRUModel(vocab_size, units, output_size)
    #]

    run_simulations(model_list, symbols_list, complexities, learning_rates, 
                    max_strings_per_complexity, sequence_lengths, window_size, validation_length, 
                    units, layers, num_runs)

    results_df = merge_results(symbols_list, complexities)
    generate_visualizations(results_df)

    return 




if __name__ == "__main__":
    
    model_list = [LSTMModel, GRUModel, TransformerModel, GPTLikeModel]  
    
    benchmark_models(model_list, 
                     symbols_list=[2, 4, 6, 8], 
                     complexities=[210, 230, 250, 270, 290],
                     sequence_lengths=[3500],
                     window_size=100,
                     validation_length=100,
                     stopping_loss=0.1,
                     max_epochs=999,
                     num_runs=5,
                     units=[128],
                     layers=[1, 2, 3],
                     batch_size=256,
                     max_strings_per_complexity=1000,
                     learning_rates=[1e-3, 1e-4]
                )
