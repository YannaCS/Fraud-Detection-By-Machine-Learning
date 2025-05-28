import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from typing import List, Optional, Tuple

def get_FDR(df):
    """
    Compute the max FDR between top 3% and bottom 3% values of each feature
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


def compare_model(
    baseline_df, optimized_df,
    metrics_to_plot: Optional[List[str]] = None,
    weights: Optional[dict] = None,
    figure_size: Tuple[int, int] = (18, 8),
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
    weights : dict, optional
        Weights for each metric for weighted score calculation
        Default: {'f1': 0.3, 'recall': 0.25, 'precision': 0.2, 'roc_auc': 0.15, 'accuracy': 0.1}
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
    if metrics_to_plot is None:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Define default weights if not specified
    if weights is None:
        weights = {'f1': 0.3, 'recall': 0.25, 'precision': 0.2, 
                  'roc_auc': 0.15, 'accuracy': 0.1}
    
    # Ensure all numeric columns are actually numeric
    numeric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision', 'train_time']
    for col in numeric_cols:
        if col in baseline_df.columns:
            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
        if col in optimized_df.columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], errors='coerce')
    
    # Merge for comparison
    comparison_df = pd.merge(baseline_df, optimized_df, 
                           left_index=True, right_index=True, 
                           how='inner', 
                           suffixes=('_baseline', '_optimized'))
    comparison_df = comparison_df.reset_index().rename(columns={'index': 'model'})
    
    # Create consistent color mapping using Set3 colormap (same as plot_model_comparison)
    models = comparison_df['model'].tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    color_map = dict(zip(models, colors))
    
    # Calculate number of subplots needed
    n_metrics = len(metrics_to_plot)
    n_cols = 3
    # Total plots: n_metrics + train_time + weighted_score + heatmap
    total_plots = n_metrics + 3
    n_rows = (total_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Calculate appropriate figure height - about 3.5 inches per row
    row_height = 3.5  # inches per row
    adjusted_height = n_rows * row_height
    
    # Create figure with gridspec for better control
    fig = plt.figure(figsize=(figure_size[0], adjusted_height))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    
    # Create axes using gridspec
    axes = []
    for i in range(n_rows):
        for j in range(n_cols):
            if i * n_cols + j < total_plots:
                axes.append(fig.add_subplot(gs[i, j]))
    
    # Ensure we have enough axes
    if len(axes) < total_plots:
        raise ValueError(f"Not enough subplots. Need {total_plots}, but only have {len(axes)}")
    
    fig.suptitle('Model Performance Changes: Baseline vs Optimized', fontsize=12, fontweight='bold', y=0.98)
    
    # Plot change bars for each metric
    plot_idx = 0
    for metric in metrics_to_plot:
        ax = axes[plot_idx]
        
        # Get data for this metric
        baseline_values = comparison_df[f'{metric}_baseline']
        optimized_values = comparison_df[f'{metric}_optimized']
        changes = optimized_values - baseline_values
        
        x_pos = np.arange(len(models))
        width = 0.35
        
        # Plot baseline and optimized bars with model-specific colors
        for i, model in enumerate(models):
            ax.bar(i - width/2, baseline_values.iloc[i], width, 
                   color=color_map[model], alpha=0.6, 
                   edgecolor='black', linewidth=1)
            ax.bar(i + width/2, optimized_values.iloc[i], width, 
                   color=color_map[model], alpha=0.9, 
                   edgecolor='black', linewidth=1, hatch='///')
        
        # Add change values as text above bars
        for i, (base, opt, change) in enumerate(zip(baseline_values, optimized_values, changes)):
            y_pos = max(base, opt) + 0.002
            change_color = '#27ae60' if change > 0 else '#e74c3c' if change < 0 else '#7f8c8d'
            ax.text(i, y_pos, f'{change:+.4f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=change_color)
        
        #ax.set_xlabel('Model')
        ax.set_ylabel(f'{metric.capitalize()} Score')
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=0, ha='center')
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis limits with some padding
        y_min = min(baseline_values.min(), optimized_values.min()) * 0.95
        y_max = max(baseline_values.max(), optimized_values.max()) * 1.05
        ax.set_ylim(y_min, y_max)
        
        plot_idx += 1
    
    # Plot training time change
    ax = axes[plot_idx]
    baseline_times = comparison_df['train_time_baseline']
    optimized_times = comparison_df['train_time_optimized']
    time_changes = optimized_times - baseline_times
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    # Plot baseline and optimized bars with model-specific colors
    for i, model in enumerate(models):
        ax.bar(i - width/2, baseline_times.iloc[i], width, 
               color=color_map[model], alpha=0.6, 
               edgecolor='black', linewidth=1)
        ax.bar(i + width/2, optimized_times.iloc[i], width, 
               color=color_map[model], alpha=0.9, 
               edgecolor='black', linewidth=1, hatch='///')
    
    # Add change values and time labels
    for i, (base, opt, change) in enumerate(zip(baseline_times, optimized_times, time_changes)):
        # Add time values on bars
        ax.text(x_pos[i] - width/2, base, f'{base:.1f}s', 
                ha='center', va='bottom', fontsize=8)
        ax.text(x_pos[i] + width/2, opt, f'{opt:.1f}s', 
                ha='center', va='bottom', fontsize=8)
        
        # Add change value above
        y_pos = max(base, opt) * 1.1
        change_color = '#e74c3c' if change > 0 else '#27ae60'
        ax.text(i, y_pos, f'{change:+.1f}s', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=change_color)
    
    #ax.set_xlabel('Model')
    ax.set_ylabel('Training Time (seconds, log scale)')
    ax.set_title('Training Time Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=0, ha='center')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, which='both')
    plot_idx += 1
    
    # Plot weighted score change
    ax = axes[plot_idx]
    
    # Calculate weighted scores for baseline and optimized
    baseline_scores = []
    optimized_scores = []
    
    for _, row in comparison_df.iterrows():
        baseline_score = 0
        optimized_score = 0
        
        for metric, weight in weights.items():
            if f'{metric}_baseline' in row and f'{metric}_optimized' in row:
                baseline_score += row[f'{metric}_baseline'] * weight
                optimized_score += row[f'{metric}_optimized'] * weight
        
        baseline_scores.append(baseline_score)
        optimized_scores.append(optimized_score)
    
    baseline_scores = np.array(baseline_scores)
    optimized_scores = np.array(optimized_scores)
    changes = optimized_scores - baseline_scores
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    # Plot baseline and optimized bars with model-specific colors
    for i, model in enumerate(models):
        ax.bar(i - width/2, baseline_scores[i], width, 
               color=color_map[model], alpha=0.6, 
               edgecolor='black', linewidth=1)
        ax.bar(i + width/2, optimized_scores[i], width, 
               color=color_map[model], alpha=0.9, 
               edgecolor='black', linewidth=1, hatch='///')
    
    # Add change values as text above bars
    for i, (base, opt, change) in enumerate(zip(baseline_scores, optimized_scores, changes)):
        y_pos = max(base, opt) + 0.002
        change_color = '#27ae60' if change > 0 else '#e74c3c' if change < 0 else '#7f8c8d'
        ax.text(i, y_pos, f'{change:+.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=change_color)
        
        # Add score values on bars
        ax.text(x_pos[i] - width/2, base/2, f'{base:.3f}', 
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        ax.text(x_pos[i] + width/2, opt/2, f'{opt:.3f}', 
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Weighted Score')
    ax.set_title('Overall Weighted Score Comparison',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis limits with some padding
    y_min = min(baseline_scores.min(), optimized_scores.min()) * 0.95
    y_max = max(baseline_scores.max(), optimized_scores.max()) * 1.05
    ax.set_ylim(y_min, y_max)
    plot_idx += 1
    
    # Plot performance heatmap
    ax = axes[plot_idx]
    
    # Calculate percentage changes
    changes = []
    model_names = comparison_df['model'].tolist()
    
    # Include training time in the heatmap
    for metric in metrics_to_plot:
        metric_changes = []
        for _, row in comparison_df.iterrows():
            baseline_val = row[f'{metric}_baseline']
            optimized_val = row[f'{metric}_optimized']
            if baseline_val != 0:
                pct_change = ((optimized_val - baseline_val) / baseline_val) * 100
            else:
                pct_change = 0
            metric_changes.append(pct_change)
        changes.append(metric_changes)
    
    # Create heatmap
    changes_array = np.array(changes)
    
    # Use diverging colormap centered at 0
    vmax = np.abs(changes_array).max()
    im = ax.imshow(changes_array, cmap='RdYlGn', aspect='auto', 
                   alpha=0.8, vmin=-vmax, vmax=vmax)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(metrics_to_plot)))
    ax.set_xticklabels(model_names, rotation=0, ha='center')
    ax.set_yticklabels([m.replace('_', ' ').capitalize() for m in metrics_to_plot])
    
    # Add text annotations
    for i in range(len(metrics_to_plot)):
        for j in range(len(model_names)):
            value = changes_array[i, j]
            text_color = 'white' if abs(value) > vmax * 0.6 else 'black'
            text = ax.text(j, i, f'{value:.1f}%',
                          ha='center', va='center', fontsize=9, 
                          fontweight='bold', color=text_color)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('% Change', rotation=270, labelpad=15)
    
    ax.set_title('Performance Changes Heatmap (%)')
    ax.grid(False)
    
    # Add both legends in a single row at the bottom
    from matplotlib.patches import Patch
    
    # Combine model rectangles and baseline/optimized patches
    all_handles = []
    all_labels = []
    
    # Add model color rectangles
    for model in models:
        all_handles.append(plt.Rectangle((0,0),1,1, fc=color_map[model], alpha=0.8))
        all_labels.append(model)
    
    # Add separator
    all_handles.append(plt.Rectangle((0,0),0,0, alpha=0))  # Invisible spacer
    all_labels.append('')
    
    # Add baseline/optimized indicators
    all_handles.extend([Patch(facecolor='gray', alpha=0.6, label='Baseline'),
                       Patch(facecolor='gray', alpha=0.9, hatch='///', label='Optimized')])
    all_labels.extend(['Baseline', 'Optimized'])
    
    # Create single combined legend at the very bottom of the figure
    fig.legend(all_handles, all_labels, loc='lower center', 
              ncol=len(models) + 3,  # models + spacer + 2 indicators
              frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.5, 0.02))
    
    # Adjust to ensure we use the full figure
    # plt.subplots_adjust(top=0.93, bottom=0.08)
    # Use tight_layout with rect parameter to leave space for legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Show plots if requested
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Generate and optionally return summary
    if return_summary:
        summary_df = generate_summary(comparison_df, metrics_to_plot, weights)
        return summary_df
    
    #return None


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