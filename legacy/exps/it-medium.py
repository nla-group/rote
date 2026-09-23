import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from scipy.stats import trim_mean
from itertools import combinations
import logging
import os
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


symbols_list = [2, 4, 6, 8]
complexities = [110, 130, 150, 170, 190]

def bootstrap_ci(data, n_boot=1000, ci=95):
    """Compute bootstrap confidence interval for median."""
    try:
        if len(data) == 0:
            raise ValueError("Empty data")
        bootstraps = [np.median(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
        lower = np.percentile(bootstraps, (100 - ci) / 2)
        upper = np.percentile(bootstraps, 100 - (100 - ci) / 2)
        return lower, upper
    except Exception as e:
        logging.warning(f"Bootstrap CI failed: {e}")
        return np.nan, np.nan

def merge_results():
    """Merge all partial result files into a single DataFrame."""
    all_results = []
    required_columns = ['symbols', 'complexity', 'sequence_length', 'learning_rate', 'layers', 'units', 'run', 'model', 'train_time', 'DL', 'JW', 'epochs', 'memory', 'model_size', 'time_per_epoch', 'test_loss', 'test_accuracy']
    for symbols in symbols_list:
        for complexity in complexities:
            result_file = f"results_partial_medium/symbols_{symbols}_complexity_{complexity}.csv"
            if os.path.exists(result_file):
                partial_df = pd.read_csv(result_file)
                if not all(col in partial_df.columns for col in required_columns):
                    logging.warning(f"Result file {result_file} missing required columns: {set(required_columns) - set(partial_df.columns)}. Regenerate this file.")
                    continue
                all_results.append(partial_df)
            else:
                logging.warning(f"Missing results for symbols: {symbols}, complexity: {complexity}")
    if not all_results:
        raise ValueError("No valid partial results found to merge.")
    
    return pd.concat(all_results, ignore_index=True)

def compute_rank_biserial(data1, data2):
    """Compute rank-biserial correlation for Mann-Whitney U test."""
    try:
        u_stat, _ = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        n1, n2 = len(data1), len(data2)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)  # Rank-biserial correlation
        return r_rb
    except Exception as e:
        logging.warning(f"Rank-biserial calculation failed: {e}")
        return np.nan

def extended_pairwise_mannwhitney(results_df, metric):
    """Perform pairwise Mann-Whitney U tests with rank-biserial correlation."""
    models = results_df['model'].unique()
    pairwise_results = []
    for model1, model2 in combinations(models, 2):
        data1 = results_df[results_df['model'] == model1][metric].dropna().values
        data2 = results_df[results_df['model'] == model2][metric].dropna().values
        if len(data1) > 0 and len(data2) > 0:
            try:
                stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                r_rb = compute_rank_biserial(data1, data2)
                pairwise_results.append({
                    'Metric': metric,
                    'Model1': model1,
                    'Model2': model2,
                    'MannWhitney_stat': stat,
                    'p_value': p,
                    'Rank_Biserial': r_rb
                })
            except ValueError as e:
                logging.warning(f"Mann-Whitney U test failed for {model1} vs {model2} on {metric}: {e}")
    return pd.DataFrame(pairwise_results)

def extended_friedman_test(results_df, metric, factor):
    """Perform Friedman test to compare models for each factor value."""
    friedman_results = []
    for val in results_df[factor].unique():
        data = [results_df[(results_df['model'] == model) & (results_df[factor] == val)][metric].dropna().values
                for model in results_df['model'].unique()]
        data = [g for g in data if len(g) > 0]
        if len(data) >= 2 and all(len(g) == len(data[0]) for g in data):
            try:
                stat, p = stats.friedmanchisquare(*data)
                friedman_results.append({
                    'Metric': metric,
                    'Factor': factor,
                    'Factor_Value': val,
                    'Friedman_stat': stat,
                    'p_value': p
                })
            except ValueError as e:
                logging.warning(f"Friedman test failed for {metric} on {factor}={val}: {e}")
    return pd.DataFrame(friedman_results)

def extended_fit_lmm(results_df, metric):
    """
    Fit a linear mixed-effects model with 'model' as random effect group
    and 'model' and 'layers' as fixed effects, using 'Transformer' as reference.

    Parameters:
    - results_df: pandas.DataFrame containing experimental results
    - metric: str, the name of the dependent variable column to model

    Returns:
    - dict with the metric name and LMM result summary, or None if fitting fails
    """
    try:
        # Set 'model' as an ordered categorical variable with 'Transformer' as reference
        results_df['model'] = pd.Categorical(
            results_df['model'],
            categories=['Transformer', 'GRU', 'LSTM', 'xLSTM', 'BERT', 'GPT'],
            ordered=True
        )

        # Log-transform metrics where appropriate
        if metric in ['train_time', 'memory', 'time_per_epoch', 'epochs', 'model_size']:
            results_df[f'log_{metric}'] = np.log1p(results_df[metric])
            formula = f"log_{metric} ~ model + layers"
        else:
            formula = f"{metric} ~ model + layers"

        # Mixed effects model with random intercept for model
        lmm = smf.mixedlm(formula, results_df, groups=results_df['model'])

        # OPTIONAL: use random slopes (uncomment if needed)
        # lmm = smf.mixedlm(formula, results_df, groups=results_df['model'], re_formula="~layers")

        result = lmm.fit(reml=True)  # Use REML for variance component estimation

        return {
            'Metric': metric,
            'LMM_Result': result.summary()
        }

    except Exception as e:
        logging.warning(f"LMM failed for {metric}: {e}")
        return None

def extended_correlation_analysis(results_df, metric_pairs):
    """Compute Spearman correlations between metric pairs."""
    correlations = []
    for metric1, metric2 in metric_pairs:
        if metric1 in results_df.columns and metric2 in results_df.columns:
            try:
                data = results_df[[metric1, metric2]].dropna()
                if len(data) > 0:
                    corr, p = stats.spearmanr(data[metric1], data[metric2])
                    correlations.append({
                        'Metric1': metric1,
                        'Metric2': metric2,
                        'Spearman_Corr': corr,
                        'p_value': p
                    })
            except ValueError as e:
                logging.warning(f"Spearman correlation failed for {metric1} vs {metric2}: {e}")
    return pd.DataFrame(correlations)

def extended_statistical_analysis(results_df):
    """Perform extended statistical tests independently of generate_visualizations."""
    os.makedirs("figures_medium/extended_stats", exist_ok=True)
    
    # Generate string_id based on unique seed strings
    results_df['string_id'] = results_df.groupby(['symbols', 'complexity', 'sequence_length']).ngroup()
    
    # Metrics to analyze
    metrics = ['DL', 'JW', 'train_time', 'epochs', 'memory', 'time_per_epoch', 'test_accuracy']
    
    # Pairwise Mann-Whitney U tests with rank-biserial correlation
    pairwise_results = []
    for metric in metrics:
        if metric in results_df.columns:
            pairwise_df = extended_pairwise_mannwhitney(results_df, metric)
            if not pairwise_df.empty:
                pairwise_df['p_adjusted'] = multipletests(pairwise_df['p_value'], method='bonferroni')[1]
                pairwise_results.append(pairwise_df)
    if pairwise_results:
        pairwise_df = pd.concat(pairwise_results, ignore_index=True)
        pairwise_df.to_csv('figures_medium/extended_stats/pairwise_mannwhitney_tests.csv', index=False)
        logging.info("Saved pairwise Mann-Whitney U tests to figures_medium/extended_stats/pairwise_mannwhitney_tests.csv")
    else:
        pairwise_df = pd.DataFrame()
        logging.warning("No pairwise Mann-Whitney U tests generated.")
    
    # Friedman tests for interactions
    friedman_results = []
    for metric in ['DL', 'JW', 'train_time', 'test_accuracy']:
        for factor in ['layers', 'symbols', 'complexity']:
            friedman_df = extended_friedman_test(results_df, metric, factor)
            if not friedman_df.empty:
                friedman_results.append(friedman_df)
    if friedman_results:
        friedman_df = pd.concat(friedman_results, ignore_index=True)
        friedman_df.to_csv('figures_medium/extended_stats/friedman_tests.csv', index=False)
        logging.info("Saved Friedman tests to figures_medium/extended_stats/friedman_tests.csv")
    else:
        friedman_df = pd.DataFrame()
        logging.warning("No Friedman tests generated.")
    
    # Linear Mixed-Effects Models
    lmm_results = []
    for metric in ['DL', 'JW', 'train_time', 'test_accuracy']:
        result = extended_fit_lmm(results_df, metric)
        if result:
            lmm_results.append(result)
    with open('figures_medium/extended_stats/lmm_results.txt', 'w') as f:
        for result in lmm_results:
            f.write(f"Metric: {result['Metric']}\n{result['LMM_Result']}\n\n")
    logging.info("Saved LMM results to figures_medium/extended_stats/lmm_results.txt")
    
    # Correlation analysis
    metric_pairs = [
        ('DL', 'train_time'), ('JW', 'train_time'), ('DL', 'memory'), ('model_size', 'epochs'),
        ('test_accuracy', 'train_time'), ('JW', 'test_accuracy')
    ]
    corr_df = extended_correlation_analysis(results_df, metric_pairs)
    if not corr_df.empty:
        corr_df.to_csv('figures_medium/extended_stats/correlation_analysis.csv', index=False)
        logging.info("Saved correlation analysis to figures_medium/extended_stats/correlation_analysis.csv")
    else:
        corr_df = pd.DataFrame()
        logging.warning("No correlation analysis generated.")
    
    # Summary statistics with trimmed means
    available_metrics = [m for m in metrics if m in results_df.columns]
    if available_metrics:
        summary = results_df.groupby(['model', 'layers']).agg({
            metric: [
                'median',
                'std',
                lambda x: trim_mean(x, proportiontocut=0.1),
                lambda x: bootstrap_ci(x)[0],
                lambda x: bootstrap_ci(x)[1]
            ] for metric in available_metrics
        }).round(3)
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary.to_csv('figures_medium/extended_stats/summary_statistics_with_trimmed.csv')
        logging.info("Saved summary statistics to figures_medium/extended_stats/summary_statistics_with_trimmed.csv")
    else:
        summary = pd.DataFrame()
        logging.warning("No summary statistics generated.")
    
    return summary, pairwise_df, friedman_df, lmm_results, corr_df

if __name__ == "__main__":
    results_df = merge_results()
    summary, pairwise_df, friedman_df, lmm_results, corr_df = extended_statistical_analysis(results_df)
    print("\nExtended Summary Statistics (with 95% CI and Trimmed Mean):")
    print(summary)
    print("\nExtended Pairwise Mann-Whitney U Tests:")
    print(pairwise_df)
    print("\nExtended Friedman Tests:")
    print(friedman_df)
    print("\nExtended Correlation Analysis:")
    print(corr_df)