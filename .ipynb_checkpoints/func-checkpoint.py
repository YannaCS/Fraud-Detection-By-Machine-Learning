import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from IPython.display import display


# Calculate individual feature's FDR (Fraud Detection Rate)
def get_FDR(df):
    """
    Compute the max FDR between top 3% and bottom 3% values of each feature

    Returns:
    --------
    pandas DataFrame: (index named feature, column named FDR)
    """
    num_bads = df['fraud_label'].sum()
    cut = int(round(len(df) * 0.03))
    results = {}

    for col in df.columns:
        if col != 'fraud_label':
            try:
                # instead of using sort_values().head()
                top_idx = df[col].nlargest(cut).index
                bottom_idx = df[col].nsmallest(cut).index
                
                FDR1 = df.loc[top_idx, 'fraud_label'].sum() / num_bads
                FDR2 = df.loc[bottom_idx, 'fraud_label'].sum() / num_bads
                results[col] = max(FDR1, FDR2)
            except Exception as e:
                print(f"Skipping {col}: {e}")

    results = pd.DataFrame.from_dict(results, orient='index', columns=['FDR'])
    results = results.reset_index().rename(columns={'index': 'feature'})
    results.sort_values(by='FDR', ascending=False, inplace=True)
    return results
    
# Calculate model's FDR (Fraud Detection Rate)
def calculate_fdr(X_data, predictions, labels, top_percent):
    X_copy = X_data.copy() if hasattr(X_data, 'copy') else pd.DataFrame(X_data).copy()
    X_copy['prediction'] = predictions
    X_copy['fraud_label'] = labels
    top_rows = int(round(X_copy.shape[0] * top_percent))
    sorted_top_rows = X_copy.sort_values('prediction', ascending=False).head(top_rows)
    fdr = sum(sorted_top_rows['fraud_label']) / sum(X_copy['fraud_label'])
    return fdr


def optimize_dtypes(df):
    """
    Reduce memory usage by optimizing data types
    """
    print("\nOPTIMIZING DATA TYPES")
    print("-" * 40)
    
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Starting memory usage: {start_mem:.2f} MB")
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
            elif col_max < 4294967295:
                df[col] = df[col].astype(np.uint32)
        else:
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    
    # Optimize float columns
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert string columns to category if low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Ending memory usage: {end_mem:.2f} MB")
    print(f"Memory reduction: {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df

def plot_model_comparison(baseline_df, figsize=(18, 18), figsave=False, figtitle='Comprehensive Model Comparison', top_n=3, 
                         weights={'f1': 0.3, 'recall': 0.25, 'precision': 0.2, 
                                   'roc_auc': 0.15, 'avg_precision': 0.1}):
    """
    Create comprehensive visualizations for model performance comparison.
    
    Parameters:
    -----------
    baseline_df : DataFrame
        DataFrame containing model performance metrics
    figsize : tuple
        Figure size (width, height)
    figsave : Boolean
        To save fig or not (default: False)
    figtitle: string
        The final saved plot's name if figsave is True
    top_n : int
        Number of top models to select (default: 3)
    weights : dict
        Weights for each metric (should sum to 1.0)
        Default weights prioritize F1 and Recall for fraud detection
    """
    
    # Make a copy to avoid modifying the original
    baseline_df = baseline_df.copy()
    
    # Ensure all numeric columns are actually numeric
    numeric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision', 'train_time']
    for col in numeric_cols:
        if col in baseline_df.columns:
            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
    
    # Create consistent color mapping for each model ONCE
    models = baseline_df.index.tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    color_map = dict(zip(models, colors))
    
    # Create relative values for some plots
    numeric_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision']
    relative_df = baseline_df[numeric_metrics].apply(lambda x: (x - x.min()) / (x.max()-x.min()), axis=0)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # 1. Performance Metrics Bar Chart
    ax1 = plt.subplot(3, 3, 1)
    metrics = ['recall', 'f1', 'precision', 'roc_auc', 'accuracy']
    available_metrics = [m for m in metrics if m in baseline_df.columns]
    
    x = np.arange(len(available_metrics))
    width = 0.8 / len(models)
    
    # Sort by recall for display but maintain color consistency
    sorted_df = baseline_df.sort_values(by='recall', ascending=False)
    
    for i, (model_name, row) in enumerate(sorted_df.iterrows()):
        values = [row[metric] for metric in available_metrics]
        ax1.bar(x + i * width, values, width, label=model_name, 
                color=color_map[model_name], alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_ylim(min(baseline_df[available_metrics].min())-0.05, 1.0)
    ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(available_metrics)
    ax1.yaxis.grid(True, color="lightgrey")
    
    # 2. Enhanced Radar Chart
    ax2 = plt.subplot(3, 3, 2, projection='polar')
    
    radar_metrics = ['recall', 'f1', 'precision', 'roc_auc', 'accuracy']
    num_vars = len(radar_metrics)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot each model with original order to maintain consistency
    for idx, (model_name, row) in enumerate(relative_df.iterrows()):
        values = row[radar_metrics].tolist()
        values += values[:1]
        
        # Add slight offset to make overlapping lines visible
        offset = idx * 0.01
        values_offset = [v + offset for v in values]
        
        ax2.plot(angles, values_offset, 'o-', linewidth=2, label=model_name, 
                color=color_map[model_name], markersize=8, alpha=0.8)
        ax2.fill(angles, values_offset, alpha=0.1, color=color_map[model_name])
    
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_metrics, fontsize=10)
    ax2.set_title('Radar of Multi-Metric Comparison (MinMax Scaled)', fontsize=14, fontweight='bold', y=1.08)
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Time Comparison
    ax3 = plt.subplot(3, 3, 3)
    if 'train_time' in baseline_df.columns:
        # Sort by training time for this plot
        time_sorted_df = baseline_df.sort_values(by='train_time', ascending=False)
        
        for i, (model_name, row) in enumerate(time_sorted_df.iterrows()):
            value = row['train_time']
            ax3.barh(i, value, color=color_map[model_name], alpha=0.8, height=0.6)
            # Add value labels
            ax3.text(value + 0.5, i, f'{value:.2f}s', va='center', fontsize=9)
        
        ax3.set_yticks(range(len(models)))
        ax3.set_yticklabels(time_sorted_df.index)
        ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time (seconds)')
    
    # 4. Heatmap of All Metrics
    ax6 = plt.subplot(3, 3, 4)
    
    # Use original order for heatmap
    heatmap_metrics = ['recall', 'f1', 'precision', 'roc_auc', 'accuracy', 'avg_precision']
    heatmap_data = relative_df[heatmap_metrics].T
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='summer', 
                cbar_kws={'label': 'Score'}, ax=ax6, 
                linewidths=0.5, linecolor='gray')
    ax6.set_title('Performance Heatmap (MinMax Scaled)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Models')
    ax6.set_ylabel('Metrics')
    ax6.set_xticklabels([m.split()[0] for m in models], rotation=45, ha='right')
    
    # 5. Precision-Recall Trade-off
    ax4 = plt.subplot(3, 3, 5)
    
    # Use original data without sorting
    for model_name, row in baseline_df.iterrows():
        ax4.scatter(row['recall'], row['precision'], 
                    s=200, alpha=0.8,
                    color=color_map[model_name], 
                    edgecolors='black', linewidth=1,
                    label=model_name)
        ax4.annotate(model_name.split()[0], (row['recall'], row['precision']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 6. F1 Score and ROC-AUC Comparison
    ax5 = plt.subplot(3, 3, 6)
    
    # Sort by F1 for this specific plot
    f1_sorted_df = baseline_df.sort_values(by='f1', ascending=False)
    metrics_to_plot = ['f1', 'roc_auc']
    x = np.arange(len(models))
    width = 0.35
    
    for i, (model_name, row) in enumerate(f1_sorted_df.iterrows()):
        ax5.bar(i - width/2, row['f1'], width, color=color_map[model_name], 
                alpha=0.8, edgecolor='black', linewidth=1)
        ax5.bar(i + width/2, row['roc_auc'], width, color=color_map[model_name], 
                alpha=0.4, edgecolor='black', linewidth=1, hatch='//')
        
        # Add value labels
        ax5.text(i - width/2, row['f1'] + 0.005, f'{row["f1"]:.3f}', 
                ha='center', va='bottom', fontsize=8)
        ax5.text(i + width/2, row['roc_auc'] + 0.005, f'{row["roc_auc"]:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Add custom legend for metrics
    from matplotlib.patches import Patch
    metric_patches = [Patch(facecolor='gray', alpha=0.8, label='F1 Score'),
                     Patch(facecolor='gray', alpha=0.4, hatch='//', label='ROC-AUC')]
    ax5.legend(handles=metric_patches, loc='upper right')
    
    ax5.set_xlabel('Models')
    ax5.set_ylabel('Score')
    ax5.set_title('F1 Score vs ROC-AUC', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([m.split()[0] for m in f1_sorted_df.index], rotation=45, ha='right')
    ax5.yaxis.grid(True, color="lightgrey")
    ax5.set_ylim(round(baseline_df.f1.min(),3)-0.005, round(baseline_df.roc_auc.max()+0.05, 2))

    # 7. Cost-Benefit Analysis
    ax7 = plt.subplot(3, 3, 7)
    
    cost_ratio = 10
    # Calculate cost score without modifying original df
    cost_df = baseline_df.copy()
    cost_df['false_positive_rate'] = 1 - cost_df['precision']
    cost_df['cost_score'] = (cost_df['recall'] * cost_ratio - cost_df['false_positive_rate'])
    cost_sorted_df = cost_df.sort_values(by='cost_score', ascending=False)
    
    for i, (model_name, row) in enumerate(cost_sorted_df.iterrows()):
        value = row['cost_score']
        ax7.bar(i, value, color=color_map[model_name], alpha=0.8, 
                edgecolor='black', linewidth=1)
        # Add value labels
        ax7.text(i, value + 0.05, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax7.set_title(f'Cost-Benefit Score\n(Missing fraud costs {cost_ratio}x false alarm)', 
                  fontsize=12, fontweight='bold')
    ax7.set_xlabel('Models')
    ax7.set_ylabel('Cost Score (higher is better)')
    ax7.set_xticks(range(len(models)))
    ax7.set_xticklabels([m.split()[0] for m in cost_sorted_df.index], rotation=45, ha='right')
    ax7.set_ylim(3.5, 5.0)
    
    # 8. Performance vs Efficiency (FIXED)
    ax8 = plt.subplot(3, 3, 8)
    
    if 'train_time' in baseline_df.columns:
        # Use original baseline_df to maintain correct associations
        for model_name, row in baseline_df.iterrows():
            # Get the correct relative recall value for this model
            bubble_size = relative_df.loc[model_name, 'recall'] * 1000 + 100
            
            ax8.scatter(row['train_time'], row['f1'], 
                       s=bubble_size, 
                       alpha=0.8, color=color_map[model_name],
                       edgecolors='black', linewidth=1)
            ax8.annotate(model_name.split()[0], 
                        (row['train_time'], row['f1']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax8.set_xlabel('Training Time (seconds)')
        ax8.set_ylabel('F1 Score')
        ax8.set_title('Performance vs Efficiency\n(Bubble size = Recall)', 
                      fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.set_xscale('log')  # Log scale for better visualization

    # 9. Weighted Score
    ax9 = plt.subplot(3, 3, 9)
    
    # Calculate weighted scores using original data
    top_models, weighted_scores = select_best_fraud_models(baseline_df, top_n=3, weights=weights)
    
    # Create a new dataframe for weighted scores to avoid modifying original
    weighted_df = pd.DataFrame({'weighted_score': weighted_scores}, index=baseline_df.index)
    weighted_sorted = weighted_df.sort_values(by='weighted_score', ascending=False)
    
    for i, (model_name, row) in enumerate(weighted_sorted.iterrows()):
        value = row['weighted_score']
        ax9.bar(i, value, color=color_map[model_name], alpha=0.8, 
                edgecolor='black', linewidth=1)
        # Add value labels
        ax9.text(i, value + 0.005, f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax9.set_title(f'Overall Weighted Score\n[used in select_best_fraud_models()]', 
                  fontsize=12, fontweight='bold')
    ax9.set_xlabel('Models')
    ax9.set_ylabel('Weighted Score')
    ax9.set_xticks(range(len(models)))
    ax9.set_xticklabels([m.split()[0] for m in weighted_sorted.index], rotation=45, ha='right')
    
    # Final Step: Add single legend for all models at the bottom
    handles = [plt.Rectangle((0,0),1,1, fc=color_map[model], alpha=0.8) 
               for model in models]
    fig.legend(handles, models, loc='lower center', ncol=len(models), 
              frameon=True, fancybox=True, shadow=True, 
              bbox_to_anchor=(0.5, 0.0))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07, top=0.93)
    plt.suptitle(figtitle, fontsize=14, fontweight='bold')
    
    if figsave:
        plt.savefig(figtitle, dpi=300, bbox_inches='tight')
    plt.show()


def select_best_fraud_models(baseline_df, top_n=3, 
                            weights={'f1': 0.3, 'recall': 0.25, 'precision': 0.2, 
                                   'roc_auc': 0.15, 'avg_precision': 0.1}):
    """
    Select the best models for fraud detection based on weighted scoring.
    
    Parameters:
    -----------
    baseline_df : DataFrame
        DataFrame containing model performance metrics
    top_n : int
        Number of top models to select (default: 3)
    weights : dict
        Weights for each metric (should sum to 1.0)
        Default weights prioritize F1 and Recall for fraud detection
    
    Returns:
    --------
    tuple: (list of model names, Series of weighted scores)
    """
    
    # Create a copy to avoid modifying original
    df = baseline_df.copy()
    
    # Normalize metrics to 0-1 scale for fair comparison
    metrics_to_normalize = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision']
    
    for metric in metrics_to_normalize:
        if metric in df.columns:
            # Min-max normalization
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                df[f'{metric}_normalized'] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df[f'{metric}_normalized'] = 1.0
    
    # Calculate weighted score
    df['weighted_score'] = 0.0  # Initialize as float
    for metric, weight in weights.items():
        if f'{metric}_normalized' in df.columns:
            df['weighted_score'] = df['weighted_score'] + (df[f'{metric}_normalized'] * weight)
    
    # Ensure weighted_score is numeric
    df['weighted_score'] = pd.to_numeric(df['weighted_score'], errors='coerce')
    
    # Sort by weighted score and get top N model names
    top_model_names = df.nlargest(top_n, 'weighted_score').index.tolist()
    
    # Return scores with original index order
    return top_model_names, df['weighted_score']

def generate_summary(comparison_df, metrics, weights):
    """Generate detailed summary statistics."""
    print("\nPERFORMANCE IMPROVEMENT SUMMARY")
    print("="*60)
    
    # Average improvements
    print("\nAverage improvements across optimized models:")
    print("-" * 50)
    for metric in metrics:
        baseline_mean = comparison_df[f'{metric}_baseline'].mean()
        optimized_mean = comparison_df[f'{metric}_optimized'].mean()
        avg_improvement = optimized_mean - baseline_mean
        pct_improvement = (avg_improvement / baseline_mean) * 100
        print(f"{metric.capitalize():12s}: {avg_improvement:+.6f} ({pct_improvement:+.3f}%)")
    
    # Weighted score improvements
    print("\nWeighted Score Analysis:")
    print("-" * 50)
    print(f"Weights used: {weights}")
    
    weighted_improvements = []
    for _, row in comparison_df.iterrows():
        baseline_score = sum(row[f'{m}_baseline'] * weights.get(m, 0) for m in metrics if f'{m}_baseline' in row)
        optimized_score = sum(row[f'{m}_optimized'] * weights.get(m, 0) for m in metrics if f'{m}_optimized' in row)
        improvement = optimized_score - baseline_score
        pct_improvement = (improvement / baseline_score) * 100
        weighted_improvements.append({
            'model': row['model'],
            'baseline': baseline_score,
            'optimized': optimized_score,
            'improvement': improvement,
            'pct_improvement': pct_improvement
        })
    
    weighted_df = pd.DataFrame(weighted_improvements)
    for _, row in weighted_df.iterrows():
        print(f"{row['model']:20s}: {row['baseline']:.4f} → {row['optimized']:.4f} "
              f"({row['improvement']:+.4f}, {row['pct_improvement']:+.2f}%)")
    
    print(f"\nAverage weighted score improvement: {weighted_df['improvement'].mean():+.4f} "
          f"({weighted_df['pct_improvement'].mean():+.2f}%)")
    
    # Training time changes
    print("\nTraining time changes:")
    print("-" * 50)
    for _, row in comparison_df.iterrows():
        time_change = row['train_time_optimized'] - row['train_time_baseline']
        time_ratio = row['train_time_optimized'] / row['train_time_baseline']
        pct_change = (time_change / row['train_time_baseline']) * 100
        print(f"{row['model']:20s}: {time_change:+8.2f}s ({time_ratio:6.2f}x, {pct_change:+.1f}%)")
    
    # Best and worst changes by metric
    print("\nBest improvements by metric:")
    print("-" * 50)
    for metric in metrics:
        improvements = comparison_df[f'{metric}_optimized'] - comparison_df[f'{metric}_baseline']
        best_idx = improvements.idxmax()
        worst_idx = improvements.idxmin()
        best_model = comparison_df.loc[best_idx, 'model']
        worst_model = comparison_df.loc[worst_idx, 'model']
        best_improvement = improvements[best_idx]
        worst_improvement = improvements[worst_idx]
        
        print(f"{metric.capitalize():12s}: Best: {best_model:15s} ({best_improvement:+.6f})")
        if worst_improvement < 0:
            print(f"{'':12s}  Worst: {worst_model:15s} ({worst_improvement:+.6f})")
    
    # Create summary DataFrame
    summary_data = []
    for _, row in comparison_df.iterrows():
        for metric in metrics + ['train_time']:
            baseline_val = row[f'{metric}_baseline']
            optimized_val = row[f'{metric}_optimized']
            change = optimized_val - baseline_val
            pct_change = (change / baseline_val) * 100 if baseline_val != 0 else 0
            
            summary_data.append({
                'Model': row['model'],
                'Metric': metric,
                'Baseline': baseline_val,
                'Optimized': optimized_val,
                'Change': change,
                'Change %': pct_change
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*60)
    print("DETAILED COMPARISON TABLE")
    print("="*60)
    
    # Print summary by model
    for model in comparison_df['model'].unique():
        print(f"\n{model}:")
        print("-" * 50)
        model_data = summary_df[summary_df['Model'] == model]
        for _, row in model_data.iterrows():
            print(f"  {row['Metric']:12s}: {row['Baseline']:8.4f} → {row['Optimized']:8.4f} "
                  f"({row['Change']:+.4f}, {row['Change %']:+.2f}%)")
        
        # Add weighted score for this model
        weighted_row = weighted_df[weighted_df['model'] == model].iloc[0]
        print(f"  {'Weighted':12s}: {weighted_row['baseline']:8.4f} → {weighted_row['optimized']:8.4f} "
              f"({weighted_row['improvement']:+.4f}, {weighted_row['pct_improvement']:+.2f}%)")
    
    return summary_df


def compare_performance(
    baseline_df, optimized_df,
    metrics: Optional[List[str]] = None,
    figure_size: Tuple[int, int] = (22, 12),
    save_path: Optional[str] = None,
    show_plots: bool = True,
    return_summary: bool = False
) -> Optional[pd.DataFrame]:
    """
    Compare baseline and optimized model performance with comprehensive visualizations.
    
    Parameters:
    -----------
    baseline_df : pd.DataFrame
        DataFrame with baseline model results
    optimized_df : pd.DataFrame
        DataFrame with optimized model results
    metrics_to_plot : list, optional
        List of metrics to include in plots. If None, uses all available metrics
    figure_size : tuple, optional
        Figure size as (width, height). Default is (18, 12)   
    save_path : str, optional
        Path to save the figure. If None, figure is not saved    
    show_plots : bool, optional
        Whether to display the plots. Default is True   
    return_summary : bool, optional
        Whether to return a summary DataFrame. Default is False    
    Returns:
    --------
    pd.DataFrame or None
        Summary DataFrame if return_summary is True, otherwise None
    """
    
    # Define default metrics if not specified
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc','avg_precision']
    # make a copy of the original df
    baseline_df = baseline_df.copy()
    optimized_df = optimized_df.copy()
    
    # Ensure all numeric columns are actually numeric
    numeric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision', 'train_time']
    for col in numeric_cols:
        if col in baseline_df.columns:
            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
        if col in optimized_df.columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], errors='coerce')
    
    # Get models that exist in both dataframes
    common_models = baseline_df.index.intersection(optimized_df.index)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figure_size,
                                   gridspec_kw={'wspace': 0.25, 
                                                'width_ratios': [3, 1]
                                               })


    # Dumbbell plot
    # Flatten data for plotting
    plot_data = []
    for model in common_models:
        for metric in metrics:
            plot_data.append({
                'model': model,
                'metric': metric,
                'baseline': baseline_df.loc[model, metric],
                'optimized': optimized_df.loc[model, metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Calculate improvement percentage
    plot_df['improvement'] = ((plot_df['optimized'] - plot_df['baseline']) / plot_df['baseline']) * 100
    plot_df['abs_improvement'] = plot_df['optimized'] - plot_df['baseline']
    
    # Create labels
    plot_df['label'] = plot_df['model'] + ' - ' + plot_df['metric'].str.replace('_', ' ').str.title()
    
    # Sort by model first, then by improvement within each model
    available_models = list(common_models)
    plot_df['model_order'] = pd.Categorical(plot_df['model'], categories=available_models)
    plot_df = plot_df.sort_values(['model_order', 'improvement'], ascending=[True, False])
    
    # Create y positions with grouping
    models = available_models
    y_positions = []
    y_labels = []
    y_pos = 0
    model_boundaries = []
    
    for model in models:
        model_data = plot_df[plot_df['model'] == model]
        model_boundaries.append(y_pos)
        
        for _, row in model_data.iterrows():
            y_positions.append(y_pos)
            y_labels.append(row['label'])
            y_pos += 1
        
        # Add spacing between models
        y_pos += 0.5
    
    # Plot horizontal lines with extensions
    for i, row in enumerate(plot_df.itertuples()):
        y = y_positions[i]
        
        # Color lines based on improvement
        line_color = '#2ecc71' if row.abs_improvement >= 0 else '#e74c3c'
        
        # Calculate line extension for visibility
        min_val = min(row.baseline, row.optimized)
        max_val = max(row.baseline, row.optimized)
        line_length = max_val - min_val
        
        # Ensure minimum line length for visibility
        min_line_length = 0.02
        if line_length < min_line_length:
            midpoint = (min_val + max_val) / 2
            min_val = midpoint - min_line_length / 2
            max_val = midpoint + min_line_length / 2
        
        # Draw main connecting line
        ax1.plot([min_val, max_val], [y, y], 
                color=line_color, alpha=0.8, linewidth=4, 
                solid_capstyle='round')
        
        # Add subtle extended guide line
        extend_factor = 0.015
        ax1.plot([min_val - extend_factor, max_val + extend_factor], [y, y], 
                color=line_color, alpha=0.2, linewidth=2, 
                linestyle='--')
    
    # Plot dots
    baseline_scatter = ax1.scatter(plot_df['baseline'], y_positions, 
                                 color='#3498db', label='Baseline', s=100, 
                                 edgecolors='white', linewidth=2, zorder=3)
    optimized_scatter = ax1.scatter(plot_df['optimized'], y_positions, 
                                  color='#2ecc71', label='Optimized', s=100,
                                  edgecolors='white', linewidth=2, zorder=3)
    
    # Add directional arrows for very small changes
    for i, row in enumerate(plot_df.itertuples()):
        y = y_positions[i]
        if abs(row.abs_improvement) < 0.005:
            if row.abs_improvement > 0:
                arrow_props = dict(arrowstyle='->', color='#2ecc71', lw=2, alpha=0.6)
                ax1.annotate('', xy=(row.optimized + 0.01, y), 
                           xytext=(row.baseline, y),
                           arrowprops=arrow_props)
            elif row.abs_improvement < 0:
                arrow_props = dict(arrowstyle='->', color='#e74c3c', lw=2, alpha=0.6)
                ax1.annotate('', xy=(row.optimized - 0.01, y), 
                           xytext=(row.baseline, y),
                           arrowprops=arrow_props)
    
    # Add annotations
    for i, row in enumerate(plot_df.itertuples()):
        y = y_positions[i]
        delta = row.abs_improvement
        pct_change = row.improvement
        
        # Position text
        if delta >= 0:
            x_pos = max(row.baseline, row.optimized) + 0.015
            ha = 'left'
        else:
            x_pos = min(row.baseline, row.optimized) - 0.015
            ha = 'right'
        
        # Format annotation
        if abs(pct_change) >= 1:
            annotation = f"{delta:+.3f}\n({pct_change:+.1f}%)"
            fontsize = 10
        else:
            annotation = f"{delta:+.3f}"
            fontsize = 9
        
        if abs(delta) < 0.005:
            fontsize = 11
            fontweight = 'heavy'
        else:
            fontweight = 'bold'
        
        #line_color = '#2ecc71' if delta >= 0 else '#e74c3c'
        line_color = None if delta >= 0 else '#e74c3c'
        ax1.text(x_pos, y, annotation, 
                va='center', ha=ha,
                color=line_color, 
                fontsize=fontsize, fontweight=fontweight,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor=line_color, linewidth=1.5, alpha=0.9))
    
    # Add visual emphasis for significant changes
    for i, row in enumerate(plot_df.itertuples()):
        y = y_positions[i]
        if abs(row.improvement) >= 2:
            if row.improvement > 0:
                ax1.scatter(row.optimized, y, s=150, marker='*', 
                          color='gold', edgecolor='#2ecc71', linewidth=2, zorder=4)
            else:
                ax1.scatter(row.optimized, y, s=150, marker='X', 
                          color='orange', edgecolor='#e74c3c', linewidth=2, zorder=4)
    
    # Add model group labels
    for i, (model, boundary) in enumerate(zip(models, model_boundaries)):
        ax1.text(-0.06, boundary, model, transform=ax1.get_yaxis_transform(),
                fontsize=11, fontweight='bold', ha='right', va='top',
                color='#34495e')
    
    # Customize axes
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([label.split(' - ')[1] for label in y_labels], fontsize=10)
    ax1.invert_yaxis()
    
    # Add vertical lines at thresholds
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ax1.axvline(threshold, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    
    # Labels and title
    ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
    #ax1.set_ylabel('Metrics', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison: Baseline vs Optimized\n', 
                 fontsize=16, fontweight='bold', color='#2c3e50')
    
    # Add subtitle
    total_metrics = len(plot_df)
    improved = len(plot_df[plot_df['abs_improvement'] > 0])
    degraded = len(plot_df[plot_df['abs_improvement'] < 0])
    ax1.text(0.5, 1.02, f'Improved: {improved}/{total_metrics} | Degraded: {degraded}/{total_metrics}', 
            transform=ax1.transAxes, ha='center', fontsize=10, color='#7f8c8d')
    
    # Legend
    legend = ax1.legend(loc='lower right', frameon=True, shadow=True, 
                       fancybox=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    
    # Grid
    ax1.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
    ax1.set_axisbelow(True)
    
    # Set x-axis limits
    x_min = min(plot_df['baseline'].min(), plot_df['optimized'].min()) - 0.05
    x_max = max(plot_df['baseline'].max(), plot_df['optimized'].max()) + 0.15
    ax1.set_xlim(x_min, x_max)
    
    # Add background shading
    for i in range(len(model_boundaries)):
        if i < len(model_boundaries) - 1:
            y_start = model_boundaries[i] - 0.5
            y_end = model_boundaries[i+1] - 1
        else:
            y_start = model_boundaries[i] - 0.5
            y_end = y_positions[-1] + 0.5
        
        if i % 2 == 0:
            ax1.axhspan(y_start, y_end, facecolor='gray', alpha=0.05)
    
    # Remove spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right subplot for the heatmap
    # Calculate percentage changes directly
    heatmap_data = {}
    for model in common_models:
        model_changes = []
        for metric in metrics:
            baseline_val = baseline_df.loc[model, metric]
            optimized_val = optimized_df.loc[model, metric]
            pct_change = ((optimized_val - baseline_val) / baseline_val) * 100
            model_changes.append(pct_change)
        heatmap_data[model] = model_changes
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=[m.replace('_', ' ').title() for m in metrics]).T
    heatmap_df = heatmap_df.loc[available_models]  # Keep same order as left plot
    
    # Create custom colormap
    max_abs_change = max(abs(heatmap_df.min().min()), abs(heatmap_df.max().max()))
    cmap = sns.diverging_palette(10, 130, as_cmap=True)  # Red to Green
    
    # Create heatmap
    sns.heatmap(heatmap_df.T, 
                annot=True, 
                fmt='.2f',
                cmap=cmap,
                center=0,
                vmin=-max_abs_change,
                vmax=max_abs_change,
                cbar_kws={'label': 'Percentage Change (%)', 'pad': 0.01, 'shrink': 0.7},
                linewidths=0.5,
                linecolor='gray',
                square=True,
                ax=ax2,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    # Customize heatmap
    ax2.set_title('Performance Change Heatmap\n(% Change from Baseline)', 
                  fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    #ax2.set_ylabel('Metrics',fontsize=12, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    
    # Rotate labels
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    # Add borders for significant changes
    for i, model in enumerate(heatmap_df.index):
        for j, metric in enumerate(heatmap_df.columns):
            value = heatmap_df.loc[model, metric]
            if abs(value) >= 2:
                rect = plt.Rectangle((j, i), 1, 1, fill=False, 
                                   edgecolor='yellow' if value > 0 else 'orange', 
                                   linewidth=3, zorder=3)
                ax2.add_patch(rect)
    
    # Add summary statistics
    avg_change_per_model = heatmap_df.mean(axis=1)
    best_model = avg_change_per_model.idxmax()
    worst_model = avg_change_per_model.idxmin()
    
    summary_text = f"Best Overall: {best_model} ({avg_change_per_model[best_model]:.2f}%)\n"
    summary_text += f"Worst Overall: {worst_model} ({avg_change_per_model[worst_model]:.2f}%)"
    ax2.text(0.5, -0.25, summary_text, transform=ax2.transAxes, 
             ha='center', va='top', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    
    
    # Create a detailed summary table
    summary_data = []
    for model in available_models:
        model_df = plot_df[plot_df['model'] == model]
        summary_data.append({
            'Model': model,
            'Metrics Improved': len(model_df[model_df['abs_improvement'] > 0]),
            'Metrics Degraded': len(model_df[model_df['abs_improvement'] < 0]),
            'Avg Change': f"{model_df['improvement'].mean():.2f}%",
            'Best Improvement': f"{model_df.loc[model_df['improvement'].idxmax(), 'metric']} "
                               f"(+{model_df['improvement'].max():.2f}%)",
            'Worst Change': f"{model_df.loc[model_df['improvement'].idxmin(), 'metric']} "
                           f"({model_df['improvement'].min():.2f}%)"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nModel Performance Summary:")
    print(summary_df.to_string(index=False))
    if return_summary:
        return summary_df









def evaluate_model_goodness(
    model, whole_dataset, model_name="Model",
    test_size=0.2, 
    top_percent=0.03, 
    niter_max=5, 
    display_confusion_matrices=False
):
    """
    Evaluate a model's performance across multiple iterations with train/test/oot splits.
    
    Parameters:
    -----------
    model : sklearn classifier
        Fitted or unfitted sklearn classifier with fit() and predict_proba() methods
    X : pd.DataFrame or np.array
        Feature matrix for training/testing
    y : pd.Series or np.array
        Target labels for training/testing
    oot : pd.DataFrame or np.array
        Out-of-time feature matrix
    oot_label : pd.Series or np.array
        Out-of-time target labels
    model_name : str
        Name of the model for display purposes
    test_size : float
        Proportion of data to use for test set
    top_percent : float
        Top percentage for FDR calculation (default 3%)
    niter_max : int
        Number of iterations to run
    display_confusion_matrices : bool
        Whether to display confusion matrices for each iteration
    
    Returns:
    --------
    pd.DataFrame : Summary of model goodness metrics averaged across iterations
    dict : Detailed metrics for each iteration
    """
    
    # Initialize storage for metrics
    metrics_list = ['FDR', 'KS', 'AUC', 'THR', 'ACC', 'MIS', 'FPR', 'TPR', 'TNR', 'PRE']
    splits = ['train', 'test', 'oot']

    metrics = {
        metric: {split: np.zeros(niter_max) for split in splits}
        for metric in metrics_list
    }
    # Split data
    total = int(round(whole_dataset.shape[0]*0.8))
    X = whole_dataset.drop(['fraud_label'], axis=1)
    y = whole_dataset.fraud_label
    oot = X[total:]
    oot_label = y[total:]
    X = X[0:total]
    y = y[0:total]
    
    print(f"Evaluating {model_name} performance over {niter_max} iterations...\n")
    
    for niter in range(niter_max):     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=niter)
        
        # Clone model to avoid modifying the original
        from sklearn.base import clone
        model_iter = clone(model)
        
        # Fit model
        model_iter.fit(X_train, y_train)
        
        # Get predictions
        prediction_train = model_iter.predict_proba(X_train)[:, 1]
        prediction_test = model_iter.predict_proba(X_test)[:, 1]
        prediction_oot = model_iter.predict_proba(oot)[:, 1]
        
        pre_train = model_iter.predict(X_train)
        pre_test = model_iter.predict(X_test)
        pre_oot = model_iter.predict(oot)
        
        # Get model FDR
        metrics['FDR']['train'][niter] = calculate_fdr(X_train, prediction_train, y_train, top_percent)
        metrics['FDR']['test'][niter] = calculate_fdr(X_test, prediction_test, y_test, top_percent)
        metrics['FDR']['oot'][niter] = calculate_fdr(oot, prediction_oot, oot_label, top_percent)
        
        # Calculate confusion matrix metrics
        def calculate_cm_metrics(y_true, y_pred, dataset_name):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            if display_confusion_matrices:
                cm = pd.DataFrame([[tn, fp], [fn, tp]], 
                                index=['Actual Good', 'Actual Bad'],
                                columns=['Predicted Good', 'Predicted Bad'])
                display(f'Confusion Matrix for {dataset_name} set iteration {niter}:')
                display(cm)
            
            return {
                'ACC': (tp + tn) / (tp + tn + fp + fn),
                'MIS': (fp + fn) / (tp + tn + fp + fn),
                'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'TPR': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'TNR': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'PRE': tp / (tp + fp) if (tp + fp) > 0 else 0
            }
        
        # Calculate metrics for each dataset
        for dataset, y_true, y_pred in [('train', y_train, pre_train), 
                                        ('test', y_test, pre_test), 
                                        ('oot', oot_label, pre_oot)]:
            cm_metrics = calculate_cm_metrics(y_true, y_pred, dataset)
            for metric, value in cm_metrics.items():
                metrics[metric][dataset][niter] = value
        
        # Calculate ROC/AUC and KS
        def calculate_roc_ks_metrics(y_true, y_pred_proba):
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            auc = roc_auc_score(y_true, y_pred_proba)
            
            # Calculate KS (Kolmogorov-Smirnov)
            ks_values = tpr - fpr
            ks_max_idx = np.argmax(ks_values)
            ks_max = ks_values[ks_max_idx]
            best_threshold = thresholds[ks_max_idx]
            
            return auc, ks_max, best_threshold
        
        # for each dataset
        for dataset, y_true, y_pred_proba in [('train', y_train, prediction_train),
                                              ('test', y_test, prediction_test),
                                              ('oot', oot_label, prediction_oot)]:
            auc, ks, threshold = calculate_roc_ks_metrics(y_true, y_pred_proba)
            metrics['AUC'][dataset][niter] = round(auc, 4)
            metrics['KS'][dataset][niter] = round(ks, 4)
            metrics['THR'][dataset][niter] = round(threshold, 4)
    
    # Calculate average metrics
    goodness_df = pd.DataFrame(
        index=['FDR', 'KS', 'AUC', 'Thresholds', 'Accuracy', 'Misclassification', 
               'False Positive Rate', 'True Positive Rate', 'True Negative Rate', 'Precision'],
        columns=['train', 'test', 'oot']
    )
    
    metric_mapping = {
        'FDR': 'FDR',
        'KS': 'KS', 
        'AUC': 'AUC',
        'Thresholds': 'THR',
        'Accuracy': 'ACC',
        'Misclassification': 'MIS',
        'False Positive Rate': 'FPR',
        'True Positive Rate': 'TPR', # aka sensitivity/recall
        'True Negative Rate': 'TNR', # aka specificity/selectivity
        'Precision': 'PRE'
    }
    
    for display_name, metric_key in metric_mapping.items():
        goodness_df.loc[display_name] = [
            round(metrics[metric_key]['train'].mean(), 4),
            round(metrics[metric_key]['test'].mean(), 4),
            round(metrics[metric_key]['oot'].mean(), 4)
        ]
    
    print(f"\n{model_name} Goodness Summary (Average over {niter_max} iterations):")
    display(goodness_df)
    
    # Add standard deviations for key metrics
    print("\nKey Metrics Stability (Standard Deviation):")
    stability_metrics = ['AUC', 'KS', 'FDR', 'ACC', 'PRE']
    stability_df = pd.DataFrame(
        index=stability_metrics,
        columns=['train_std', 'test_std', 'oot_std']
    )
    
    for metric in stability_metrics:
        stability_df.loc[metric] = [
            round(metrics[metric]['train'].std(), 4),
            round(metrics[metric]['test'].std(), 4),
            round(metrics[metric]['oot'].std(), 4)
        ]
    
    display(stability_df)
    
    return goodness_df, metrics
# Example usage:
"""
# Define your model
model = GradientBoostingClassifier(...)

# Evaluate model
goodness_summary, detailed_metrics = evaluate_model_goodness(
    model=model, whole_dataset, model_name="Gradient Boosting",
    test_size=0.2, top_percent=0.03, niter_max=5,
    display_confusion_matrices=False
)

# Access specific metrics if needed
print(f"Average test AUC: {detailed_metrics['AUC']['test'].mean():.4f}")
print(f"Test AUC range: [{detailed_metrics['AUC']['test'].min():.4f}, {detailed_metrics['AUC']['test'].max():.4f}]")
"""




def calculate_performance_forms(model, whole_dataset, n_bins=20, model_name="Model"):
    """
    Calculate performance forms (train/test/oot) with binned predictions and create visualizations.
    
    Parameters:
    -----------
    model : sklearn classifier
        Fitted model with predict_proba method
    X_train, y_train : array-like
        Training data and labels
    X_test, y_test : array-like
        Test data and labels
    X_oot, y_oot : array-like
        Out-of-time data and labels
    n_bins : int
        Number of bins for population segmentation (default: 20)
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    dict : Dictionary containing train_form, test_form, oot_form DataFrames
    """
    total = int(round(whole_dataset.shape[0]*0.8))
    X = whole_dataset.drop(['fraud_label'], axis=1)
    y = whole_dataset.fraud_label
    X_oot = X[total:]
    y_oot = y[total:]
    X = X[0:total]
    y = y[0:total]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train)
             
    # Initialize forms
    columns = ['pop_bin', 'bin_records', 'bin_goods', 'bin_bads', 'bin_%good', 'bin_%bad',
               'cum_records', 'cum_goods', 'cum_bads', 'cum_%good', 'cum_%bad_FDR', 'KS', 'FPR']
    
    train_form = pd.DataFrame(np.zeros((n_bins, len(columns))), columns=columns)
    test_form = pd.DataFrame(np.zeros((n_bins, len(columns))), columns=columns)
    oot_form = pd.DataFrame(np.zeros((n_bins, len(columns))), columns=columns)
    
    # Get record counts
    trn_rcd = len(y_train)
    tst_rcd = len(y_test)
    oot_rcd = len(y_oot)
    
    # Get predictions
    trn_preprob = model.predict_proba(X_train)[:, 1]
    tst_preprob = model.predict_proba(X_test)[:, 1]
    oot_preprob = model.predict_proba(X_oot)[:, 1]
    
    # Calculate goods and bads
    trn_bads = sum(y_train)
    trn_goods = trn_rcd - trn_bads
    tst_bads = sum(y_test)
    tst_goods = tst_rcd - tst_bads
    oot_bads = sum(y_oot)
    oot_goods = oot_rcd - oot_bads
    
    # Calculate fraud rates
    trn_fraud_rate = trn_bads / trn_rcd
    tst_fraud_rate = tst_bads / tst_rcd
    oot_fraud_rate = oot_bads / oot_rcd
    
    print(f"\n{model_name} - Data Summary:")
    print(f"Train: {trn_goods:,} goods, {trn_bads:,} bads, fraud rate: {trn_fraud_rate:.2%}")
    print(f"Test:  {tst_goods:,} goods, {tst_bads:,} bads, fraud rate: {tst_fraud_rate:.2%}")
    print(f"OOT:   {oot_goods:,} goods, {oot_bads:,} bads, fraud rate: {oot_fraud_rate:.2%}")
    
    # Calculate bin sizes (using integer division and handling remainder)
    trn_bin_rcd = (trn_rcd + n_bins - 1) // n_bins  # Equivalent to ceil(trn_rcd / n_bins)
    tst_bin_rcd = (tst_rcd + n_bins - 1) // n_bins
    oot_bin_rcd = (oot_rcd + n_bins - 1) // n_bins
    
    
    # Create sorted datasets
    TRAIN = pd.DataFrame({
        'prediction_probability': trn_preprob,
        'label': y_train
    }).sort_values('prediction_probability', ascending=False)
    
    TEST = pd.DataFrame({
        'prediction_probability': tst_preprob,
        'label': y_test
    }).sort_values('prediction_probability', ascending=False)
    
    OOT = pd.DataFrame({
        'prediction_probability': oot_preprob,
        'label': y_oot
    }).sort_values('prediction_probability', ascending=False)
    
    # Fill forms
    for i in range(n_bins):
        # Calculate top rows for each bin
        trn_top_rows = min(trn_bin_rcd * (i + 1), trn_rcd)
        tst_top_rows = min(tst_bin_rcd * (i + 1), tst_rcd)
        oot_top_rows = min(oot_bin_rcd * (i + 1), oot_rcd)
        
        # Train form
        train_form.loc[i, 'pop_bin'] = i + 1
        train_form.loc[i, 'bin_records'] = trn_bin_rcd if i < n_bins - 1 else trn_rcd - trn_bin_rcd * (n_bins - 1)
        train_form.loc[i, 'cum_records'] = trn_top_rows
        train_form.loc[i, 'cum_bads'] = TRAIN['label'].head(trn_top_rows).sum()
        train_form.loc[i, 'cum_goods'] = trn_top_rows - train_form.loc[i, 'cum_bads']
        train_form.loc[i, 'cum_%good'] = round(train_form.loc[i, 'cum_goods'] / trn_goods, 4)
        train_form.loc[i, 'cum_%bad_FDR'] = round(train_form.loc[i, 'cum_bads'] / trn_bads, 4)
        train_form.loc[i, 'KS'] = (train_form.loc[i, 'cum_%bad_FDR'] - train_form.loc[i, 'cum_%good']) * 100
        train_form.loc[i, 'FPR'] = round(train_form.loc[i, 'cum_goods'] / train_form.loc[i, 'cum_bads'], 2) if train_form.loc[i, 'cum_bads'] > 0 else 0
        
        if i == 0:
            train_form.loc[i, 'bin_goods'] = train_form.loc[i, 'cum_goods']
            train_form.loc[i, 'bin_bads'] = train_form.loc[i, 'cum_bads']
        else:
            train_form.loc[i, 'bin_goods'] = train_form.loc[i, 'cum_goods'] - train_form.loc[i-1, 'cum_goods']
            train_form.loc[i, 'bin_bads'] = train_form.loc[i, 'cum_bads'] - train_form.loc[i-1, 'cum_bads']
        
        train_form.loc[i, 'bin_%good'] = round(train_form.loc[i, 'bin_goods'] / train_form.loc[i, 'bin_records'], 4)
        train_form.loc[i, 'bin_%bad'] = round(train_form.loc[i, 'bin_bads'] / train_form.loc[i, 'bin_records'], 4)
        
        # Test form (similar logic)
        test_form.loc[i, 'pop_bin'] = i + 1
        test_form.loc[i, 'bin_records'] = tst_bin_rcd if i < n_bins - 1 else tst_rcd - tst_bin_rcd * (n_bins - 1)
        test_form.loc[i, 'cum_records'] = tst_top_rows
        test_form.loc[i, 'cum_bads'] = TEST['label'].head(tst_top_rows).sum()
        test_form.loc[i, 'cum_goods'] = tst_top_rows - test_form.loc[i, 'cum_bads']
        test_form.loc[i, 'cum_%good'] = round(test_form.loc[i, 'cum_goods'] / tst_goods, 4)
        test_form.loc[i, 'cum_%bad_FDR'] = round(test_form.loc[i, 'cum_bads'] / tst_bads, 4)
        test_form.loc[i, 'KS'] = (test_form.loc[i, 'cum_%bad_FDR'] - test_form.loc[i, 'cum_%good']) * 100
        test_form.loc[i, 'FPR'] = round(test_form.loc[i, 'cum_goods'] / test_form.loc[i, 'cum_bads'], 2) if test_form.loc[i, 'cum_bads'] > 0 else 0
        
        if i == 0:
            test_form.loc[i, 'bin_goods'] = test_form.loc[i, 'cum_goods']
            test_form.loc[i, 'bin_bads'] = test_form.loc[i, 'cum_bads']
        else:
            test_form.loc[i, 'bin_goods'] = test_form.loc[i, 'cum_goods'] - test_form.loc[i-1, 'cum_goods']
            test_form.loc[i, 'bin_bads'] = test_form.loc[i, 'cum_bads'] - test_form.loc[i-1, 'cum_bads']
        
        test_form.loc[i, 'bin_%good'] = round(test_form.loc[i, 'bin_goods'] / test_form.loc[i, 'bin_records'], 4)
        test_form.loc[i, 'bin_%bad'] = round(test_form.loc[i, 'bin_bads'] / test_form.loc[i, 'bin_records'], 4)
        
        # OOT form (similar logic)
        oot_form.loc[i, 'pop_bin'] = i + 1
        oot_form.loc[i, 'bin_records'] = oot_bin_rcd if i < n_bins - 1 else oot_rcd - oot_bin_rcd * (n_bins - 1)
        oot_form.loc[i, 'cum_records'] = oot_top_rows
        oot_form.loc[i, 'cum_bads'] = OOT['label'].head(oot_top_rows).sum()
        oot_form.loc[i, 'cum_goods'] = oot_top_rows - oot_form.loc[i, 'cum_bads']
        oot_form.loc[i, 'cum_%good'] = round(oot_form.loc[i, 'cum_goods'] / oot_goods, 4)
        oot_form.loc[i, 'cum_%bad_FDR'] = round(oot_form.loc[i, 'cum_bads'] / oot_bads, 4)
        oot_form.loc[i, 'KS'] = (oot_form.loc[i, 'cum_%bad_FDR'] - oot_form.loc[i, 'cum_%good']) * 100
        oot_form.loc[i, 'FPR'] = round(oot_form.loc[i, 'cum_goods'] / oot_form.loc[i, 'cum_bads'], 2) if oot_form.loc[i, 'cum_bads'] > 0 else 0
        
        if i == 0:
            oot_form.loc[i, 'bin_goods'] = oot_form.loc[i, 'cum_goods']
            oot_form.loc[i, 'bin_bads'] = oot_form.loc[i, 'cum_bads']
        else:
            oot_form.loc[i, 'bin_goods'] = oot_form.loc[i, 'cum_goods'] - oot_form.loc[i-1, 'cum_goods']
            oot_form.loc[i, 'bin_bads'] = oot_form.loc[i, 'cum_bads'] - oot_form.loc[i-1, 'cum_bads']
        
        oot_form.loc[i, 'bin_%good'] = round(oot_form.loc[i, 'bin_goods'] / oot_form.loc[i, 'bin_records'], 4)
        oot_form.loc[i, 'bin_%bad'] = round(oot_form.loc[i, 'bin_bads'] / oot_form.loc[i, 'bin_records'], 4)
    
    # Convert to appropriate data types
    for form in [train_form, test_form, oot_form]:
        form['pop_bin'] = form['pop_bin'].astype(int)
        form['bin_records'] = form['bin_records'].astype(int)
        form['bin_goods'] = form['bin_goods'].astype(int)
        form['bin_bads'] = form['bin_bads'].astype(int)
        form['cum_records'] = form['cum_records'].astype(int)
        form['cum_goods'] = form['cum_goods'].astype(int)
        form['cum_bads'] = form['cum_bads'].astype(int)
    
    return {
        'train_form': train_form,
        'test_form': test_form,
        'oot_form': oot_form,
        'fraud_rates': {
            'train': trn_fraud_rate,
            'test': tst_fraud_rate,
            'oot': oot_fraud_rate
        }
    }


def plot_performance_metrics(forms_dict, model_name="Model"):
    """
    Create comprehensive visualizations for model performance metrics.
    
    Parameters:
    -----------
    forms_dict : dict
        Dictionary containing train_form, test_form, oot_form from calculate_performance_forms
    model_name : str
        Name of the model for display
    """
    
    train_form = forms_dict['train_form']
    test_form = forms_dict['test_form']
    oot_form = forms_dict['oot_form']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Cumulative Fraud Detection Rate (FDR)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_form['pop_bin'], train_form['cum_%bad_FDR'] * 100, 'b-o', label='Train', linewidth=2)
    ax1.plot(test_form['pop_bin'], test_form['cum_%bad_FDR'] * 100, 'g-s', label='Test', linewidth=2)
    ax1.plot(oot_form['pop_bin'], oot_form['cum_%bad_FDR'] * 100, 'r-^', label='OOT', linewidth=2)
    ax1.set_xlabel('Population Bin (5% each)')
    ax1.set_ylabel('Cumulative % of Frauds Captured')
    ax1.set_title('Cumulative Fraud Detection Rate (FDR)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 21))
    
    # 2. KS Statistic
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(train_form['pop_bin'], train_form['KS'], 'b-o', label='Train', linewidth=2)
    ax2.plot(test_form['pop_bin'], test_form['KS'], 'g-s', label='Test', linewidth=2)
    ax2.plot(oot_form['pop_bin'], oot_form['KS'], 'r-^', label='OOT', linewidth=2)
    ax2.set_xlabel('Population Bin (5% each)')
    ax2.set_ylabel('KS (%)')
    ax2.set_title('Kolmogorov-Smirnov (KS) Statistic')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 21))
    
    # 3. Bin-wise Fraud Rate
    ax3 = plt.subplot(3, 3, 3)
    x = np.arange(1, 21)
    width = 0.25
    ax3.bar(x - width, train_form['bin_%bad'], width, label='Train', alpha=0.8)
    ax3.bar(x, test_form['bin_%bad'], width, label='Test', alpha=0.8)
    ax3.bar(x + width, oot_form['bin_%bad'], width, label='OOT', alpha=0.8)
    ax3.set_xlabel('Population Bin')
    ax3.set_ylabel('Fraud Rate in Bin')
    ax3.set_title('Bin-wise Fraud Rate Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(range(1, 21, 2))
    
    # 4. Cumulative Good vs Bad Distribution
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(train_form['cum_%good'] * 100, train_form['cum_%bad_FDR'] * 100, 'b-', label='Train', linewidth=2)
    ax4.plot(test_form['cum_%good'] * 100, test_form['cum_%bad_FDR'] * 100, 'g-', label='Test', linewidth=2)
    ax4.plot(oot_form['cum_%good'] * 100, oot_form['cum_%bad_FDR'] * 100, 'r-', label='OOT', linewidth=2)
    ax4.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random')
    ax4.set_xlabel('Cumulative % of Goods')
    ax4.set_ylabel('Cumulative % of Bads (Frauds)')
    ax4.set_title('Lorenz Curve / CAP Curve')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    
    # 5. Lift Chart
    ax5 = plt.subplot(3, 3, 5)
    baseline_fraud_rate = forms_dict['fraud_rates']['train']
    train_lift = (train_form['bin_%bad'] / baseline_fraud_rate).replace([np.inf, -np.inf], 0)
    test_lift = (test_form['bin_%bad'] / forms_dict['fraud_rates']['test']).replace([np.inf, -np.inf], 0)
    oot_lift = (oot_form['bin_%bad'] / forms_dict['fraud_rates']['oot']).replace([np.inf, -np.inf], 0)
    
    ax5.plot(train_form['pop_bin'], train_lift, 'b-o', label='Train', linewidth=2)
    ax5.plot(test_form['pop_bin'], test_lift, 'g-s', label='Test', linewidth=2)
    ax5.plot(oot_form['pop_bin'], oot_lift, 'r-^', label='OOT', linewidth=2)
    ax5.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Population Bin (5% each)')
    ax5.set_ylabel('Lift')
    ax5.set_title('Lift Chart (vs Baseline Fraud Rate)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(range(1, 21))
    
    # 6. Capture Rate at Different Cutoffs
    ax6 = plt.subplot(3, 3, 6)
    cutoffs = [1, 3, 5, 10, 20]  # Top x bins (5% each)
    train_capture = [train_form.loc[c-1, 'cum_%bad_FDR'] * 100 for c in cutoffs]
    test_capture = [test_form.loc[c-1, 'cum_%bad_FDR'] * 100 for c in cutoffs]
    oot_capture = [oot_form.loc[c-1, 'cum_%bad_FDR'] * 100 for c in cutoffs]
    
    x_pos = np.arange(len(cutoffs))
    width = 0.25
    ax6.bar(x_pos - width, train_capture, width, label='Train', alpha=0.8)
    ax6.bar(x_pos, test_capture, width, label='Test', alpha=0.8)
    ax6.bar(x_pos + width, oot_capture, width, label='OOT', alpha=0.8)
    ax6.set_xlabel('Top % of Population')
    ax6.set_ylabel('% of Frauds Captured')
    ax6.set_title('Fraud Capture Rate at Different Cutoffs')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'{c*5}%' for c in cutoffs])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(train_capture):
        ax6.text(i - width, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(test_capture):
        ax6.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(oot_capture):
        ax6.text(i + width, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 7. Precision (PPV) by Bin
    ax7 = plt.subplot(3, 3, 7)
    train_precision = train_form['bin_bads'] / train_form['bin_records']
    test_precision = test_form['bin_bads'] / test_form['bin_records']
    oot_precision = oot_form['bin_bads'] / oot_form['bin_records']
    
    ax7.plot(train_form['pop_bin'], train_precision * 100, 'b-o', label='Train', linewidth=2)
    ax7.plot(test_form['pop_bin'], test_precision * 100, 'g-s', label='Test', linewidth=2)
    ax7.plot(oot_form['pop_bin'], oot_precision * 100, 'r-^', label='OOT', linewidth=2)
    
    # Add baseline fraud rates
    ax7.axhline(y=forms_dict['fraud_rates']['train'] * 100, color='b', linestyle='--', alpha=0.5)
    ax7.axhline(y=forms_dict['fraud_rates']['test'] * 100, color='g', linestyle='--', alpha=0.5)
    ax7.axhline(y=forms_dict['fraud_rates']['oot'] * 100, color='r', linestyle='--', alpha=0.5)
    
    ax7.set_xlabel('Population Bin (5% each)')
    ax7.set_ylabel('Precision (%)')
    ax7.set_title('Precision (PPV) by Population Bin')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xticks(range(1, 21))
    
    # 8. Summary Statistics Table
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('tight')
    ax8.axis('off')
    
    # Calculate summary metrics
    summary_data = []
    for name, form, fraud_rate in [('Train', train_form, forms_dict['fraud_rates']['train']),
                                   ('Test', test_form, forms_dict['fraud_rates']['test']),
                                   ('OOT', oot_form, forms_dict['fraud_rates']['oot'])]:
        max_ks = form['KS'].max()
        max_ks_bin = form.loc[form['KS'].idxmax(), 'pop_bin']
        fdr_at_5 = form.loc[0, 'cum_%bad_FDR'] * 100  # Top 5%
        fdr_at_10 = form.loc[1, 'cum_%bad_FDR'] * 100  # Top 10%
        fdr_at_20 = form.loc[3, 'cum_%bad_FDR'] * 100  # Top 20%
        
        summary_data.append([name, f'{fraud_rate:.2%}', f'{max_ks:.2f}%', f'{max_ks_bin}',
                           f'{fdr_at_5:.1f}%', f'{fdr_at_10:.1f}%', f'{fdr_at_20:.1f}%'])
    
    table = ax8.table(cellText=summary_data,
                     colLabels=['Dataset', 'Fraud Rate', 'Max KS', 'Max KS Bin', 
                               'FDR@5%', 'FDR@10%', 'FDR@20%'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax8.set_title('Summary Statistics', fontsize=12, pad=20)
    
    # 9. Model Stability Analysis
    ax9 = plt.subplot(3, 3, 9)
    metrics = ['Max KS', 'FDR@5%', 'FDR@10%', 'FDR@20%']
    train_values = [train_form['KS'].max(), 
                   train_form.loc[0, 'cum_%bad_FDR'] * 100,
                   train_form.loc[1, 'cum_%bad_FDR'] * 100,
                   train_form.loc[3, 'cum_%bad_FDR'] * 100]
    test_values = [test_form['KS'].max(), 
                  test_form.loc[0, 'cum_%bad_FDR'] * 100,
                  test_form.loc[1, 'cum_%bad_FDR'] * 100,
                  test_form.loc[3, 'cum_%bad_FDR'] * 100]
    oot_values = [oot_form['KS'].max(), 
                 oot_form.loc[0, 'cum_%bad_FDR'] * 100,
                 oot_form.loc[1, 'cum_%bad_FDR'] * 100,
                 oot_form.loc[3, 'cum_%bad_FDR'] * 100]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    ax9.bar(x - width, train_values, width, label='Train', alpha=0.8)
    ax9.bar(x, test_values, width, label='Test', alpha=0.8)
    ax9.bar(x + width, oot_values, width, label='OOT', alpha=0.8)
    
    ax9.set_ylabel('Value (%)')
    ax9.set_title('Model Stability Across Datasets')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics)
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(train_values):
        ax9.text(i - width, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(test_values):
        ax9.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(oot_values):
        ax9.text(i + width, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'{model_name} - Performance Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Display forms
    print(f"\n{model_name} - Train Form (Top 5 bins):")
    display(train_form.head())
    
    print(f"\n{model_name} - Test Form (Top 5 bins):")
    display(test_form.head())
    
    print(f"\n{model_name} - OOT Form (Top 5 bins):")
    display(oot_form.head())


# Example usage
"""
# Assuming you have a fitted model and data
forms_dict = calculate_performance_forms(
    model=your_fitted_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_oot=X_oot,
    y_oot=y_oot,
    n_bins=20,
    model_name="Gradient Boosting"
)

# Plot the results
plot_performance_metrics(forms_dict, model_name="Gradient Boosting")

# Access individual forms if needed
train_form = forms_dict['train_form']
test_form = forms_dict['test_form']
oot_form = forms_dict['oot_form']
"""