import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

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
    list of model names
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
    
    return top_model_names, df['weighted_score']



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
    
    # Ensure all numeric columns are actually numeric
    numeric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision', 'train_time']
    for col in numeric_cols:
        if col in baseline_df.columns:
            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
    baseline_df = baseline_df.sort_values(by='recall', ascending=False)
    # create relative values, use it instead of original in some plots
    relative_df = baseline_df.drop(['model', 'train_time'], axis=1).apply(lambda x: (x - x.min()) / (x.max()-x.min()), axis=0)
    # Set style and create color mapping
    #plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create consistent color mapping for each model
    models = baseline_df.index.tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    color_map = dict(zip(models, colors))
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # 1. Performance Metrics Bar Chart
    ax1 = plt.subplot(3, 3, 1)
    metrics = ['recall', 'f1', 'precision',  'roc_auc', 'accuracy']
    available_metrics = [m for m in metrics if m in baseline_df.columns]
    
    x = np.arange(len(available_metrics))
    width = 0.8 / len(models)
    
    for i, (model_name, row) in enumerate(baseline_df.iterrows()):
        values = [row[metric] for metric in available_metrics]
        ax1.bar(x + i * width, values, width, label=model_name, 
                color=color_map[model_name], alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_ylim(min(baseline_df[available_metrics].min())-0.05, 1.0)
    ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(available_metrics)
    ax1.yaxis.grid(True, color ="lightgrey")
    
    # 2. Enhanced Radar Chart with better visibility
    ax2 = plt.subplot(3, 3, 2, projection='polar')
    
    # Select metrics for radar chart
    radar_metrics = ['recall', 'f1', 'precision',  'roc_auc', 'accuracy']
    num_vars = len(radar_metrics)
    # use relative values instead of original 
    used_df = relative_df[radar_metrics]
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Find min and max for better scaling
    all_values = []
    for metric in radar_metrics:
        if metric in used_df.columns:
            all_values.extend(used_df[metric].values)
    
    # Set radar chart limits with some padding
    min_val = min(all_values) * 0.9
    max_val = max(all_values) * 1.2
    
    # Plot each model with offset for better visibility
    for idx, (model_name, row) in enumerate(used_df.iterrows()):
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
    ax2.set_ylim(min_val, max_val)
    ax2.set_title('Radar of Multi-Metric Comparison (MinMax Scaled)', fontsize=14, fontweight='bold', y=1.08)
    ax2.grid(True, alpha=0.3)
    
    # Add radial labels for better readability
    ax2.set_rlabel_position(45)
    yticks = np.linspace(min_val, max_val, 5)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f'{y:.2f}' for y in yticks], fontsize=8)
    
    # 3. Training Time Comparison
    ax3 = plt.subplot(3, 3, 3)
    baseline_df = baseline_df.sort_values(by='train_time', ascending=False)
    if 'train_time' in baseline_df.columns:
        bars = []
        for i, (model_name, value) in enumerate(baseline_df['train_time'].items()):
            bar = ax3.barh(i, value, color=color_map[model_name], alpha=0.8, height=0.6)
            bars.append(bar)
            # Add value labels
            ax3.text(value + 0.5, i, f'{value:.2f}s', va='center', fontsize=9)
        
        ax3.set_yticks(range(len(models)))
        ax3.set_yticklabels(models)
        ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time (seconds)')
    
    # 4. Heatmap of All Metrics
    ax6 = plt.subplot(3, 3, 4)
    
    # Prepare data for heatmap
    heatmap_metrics = ['recall', 'f1', 'precision',  'roc_auc', 'accuracy', 'avg_precision']
    heatmap_data = relative_df[heatmap_metrics].T
    
    # Create heatmap with custom colormap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='summer', 
                cbar_kws={'label': 'Score'}, ax=ax6, 
                linewidths=0.5, linecolor='gray')
    ax6.set_title('Performance Heatmap (MinMax Scaled)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Models')
    ax6.set_ylabel('Metrics')
    ax6.set_xticklabels([m.split()[0] for m in models], rotation=45, ha='right')
    
    # 5. Precision-Recall Trade-off
    ax4 = plt.subplot(3, 3, 5)
    ax4.scatter(x=baseline_df['recall'], y=baseline_df['precision'], 
                s=200, alpha=0.8,
                color=[color_map[idx] for idx in baseline_df.index], 
                edgecolors='black', linewidth=1
               )
    for model_name, row in baseline_df.iterrows():
        #Iterate over DataFrame rows as (index, Series) pairs
        ax4.annotate(model_name.split()[0], (row['recall'], row['precision']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 6. F1 Score and ROC-AUC Comparison
    ax5 = plt.subplot(3, 3, 6)
    baseline_df = baseline_df.sort_values(by='f1', ascending=False)
    metrics_to_plot = ['f1', 'roc_auc']
    x = np.arange(len(models))
    width = 0.35
    
    for i, (model_name, row) in enumerate(baseline_df.iterrows()):
        ax5.bar(i - width/2, row['f1'], width, color=color_map[model_name], 
                alpha=0.8, edgecolor='black', linewidth=1)
        ax5.bar(i + width/2, row['roc_auc'], width, color=color_map[model_name], 
                alpha=0.4, edgecolor='black', linewidth=1, hatch='//')
    
    # Add custom legend for metrics
    from matplotlib.patches import Patch
    metric_patches = [Patch(facecolor='gray', alpha=0.8, label='F1 Score'),
                     Patch(facecolor='gray', alpha=0.4, hatch='//', label='ROC-AUC')]
    ax5.legend(handles=metric_patches, loc='upper right')
    
    ax5.set_xlabel('Models')
    ax5.set_ylabel('Score')
    ax5.set_ylim(min(baseline_df[metrics_to_plot].min())-0.05, max(baseline_df[metrics_to_plot].max())+0.05)
    ax5.set_title('F1 Score vs ROC-AUC', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([m.split()[0] for m in models], rotation=45, ha='right')
    
    # Add value labels
    for i, (model_name, row) in enumerate(baseline_df.iterrows()):
        ax5.text(i - width/2, row['f1'] + 0.005, f'{row["f1"]:.3f}', 
                ha='center', va='bottom', fontsize=8)
        ax5.text(i + width/2, row['roc_auc'] + 0.005, f'{row["roc_auc"]:.3f}', 
                ha='center', va='bottom', fontsize=8)

    ax5.yaxis.grid(True, color ="lightgrey")
    ax5.set_ylim(round(baseline_df.f1.min(),3)-0.005, round(baseline_df.roc_auc.max()+0.05, 2))

    #7. Cost-Benefit Analysis
    ax7 = plt.subplot(3, 3, 7)
    
    cost_ratio = 10
    baseline_df['false_positive_rate'] = 1 - baseline_df['precision']
    baseline_df['cost_score'] = (baseline_df['recall'] * cost_ratio - 
                                 baseline_df['false_positive_rate'])
    baseline_df = baseline_df.sort_values(by='cost_score', ascending=False)
    bars = []
    for i, (model_name, value) in enumerate(baseline_df['cost_score'].items()):
        bar = ax7.bar(i, value, color=color_map[model_name], alpha=0.8, 
                      edgecolor='black', linewidth=1)
        bars.append(bar)
        # Add value labels
        ax7.text(i, value + 0.05, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax7.set_title(f'Cost-Benefit Score\n(Missing fraud costs {cost_ratio}x false alarm)', 
                  fontsize=12, fontweight='bold')
    ax7.set_xlabel('Models')
    ax7.set_ylabel('Cost Score (higher is better)')
    ax7.set_xticks(range(len(models)))
    ax7.set_xticklabels([m.split()[0] for m in models], rotation=45, ha='right')
    ax7.set_ylim(3.5, 5.0)
    
    #8. Performance vs Efficiency
    ax8 = plt.subplot(3, 3, 8)
    
    if 'train_time' in baseline_df.columns:
        for model_name, row in baseline_df.iterrows():
            scatter = ax8.scatter(row['train_time'], row['f1'], 
                                s=relative_df.loc[model_name,'recall']*1000+100, 
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
        ax8.set_ylim(round(baseline_df.f1.min()-0.005, 2), round(baseline_df.f1.max(), 2))

    #9. Weighted Score by Func select_best_fraud_models()
    ax9 = plt.subplot(3, 3, 9)
    top_models, weighted_score = select_best_fraud_models(baseline_df, top_n=3, weights=weights)
    baseline_df['weighted_score'] = weighted_score
    baseline_df = baseline_df.sort_values(by='weighted_score', ascending=False)
    bars = []
    for i, (model_name, value) in enumerate(baseline_df['weighted_score'].items()):
        bar = ax9.bar(i, value, color=color_map[model_name], alpha=0.8, 
                      edgecolor='black', linewidth=1)
        bars.append(bar)
        # Add value labels
        ax9.text(i, value + 0.005, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax9.set_title(f'Overall Weighted Score\n[used in select_best_fraud_models()]', 
                  fontsize=12, fontweight='bold')
    ax9.set_xlabel('Models')
    ax9.set_ylabel('Weighted Score')
    ax9.set_xticks(range(len(models)))
    ax9.set_xticklabels([m.split()[0] for m in models], rotation=45, ha='right')
    #ax9.set_ylim(, )
    
    # Final Step: Add single legend for all models at the bottom
    handles = [plt.Rectangle((0,0),1,1, fc=color_map[model], alpha=0.8) 
               for model in models]
    fig.legend(handles, models, loc='lower center', ncol=len(models), 
              frameon=True, fancybox=True, shadow=True, 
              bbox_to_anchor=(0.5, 0.0))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07, top=0.93)
    plt.suptitle(figtitle,
                fontsize=14, fontweight='bold')
    if figsave==True: 
        plt.savefig(figtitle, dpi=300, bbox_inches='tight')
    plt.show()
    