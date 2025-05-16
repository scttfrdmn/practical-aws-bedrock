# Visualization configuration for AWS Bedrock Inference project
# All visualizations should be SVG format for optimal clarity and scalability

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional

# Set consistent style for all visualizations
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette for model families (colorblind-friendly)
MODEL_FAMILY_COLORS = {
    'anthropic': '#1f77b4',  # Blue
    'meta': '#ff7f0e',       # Orange
    'amazon': '#2ca02c',     # Green
    'stability': '#d62728',  # Red
    'ai21': '#9467bd',       # Purple
    'cohere': '#8c564b'      # Brown
}

# Default SVG export parameters
SVG_CONFIG = {
    'format': 'svg',
    'dpi': 300,
    'bbox_inches': 'tight',
    'transparent': True
}

def setup_visualization_dir():
    """Create directories for visualization outputs if they don't exist"""
    viz_dirs = [
        'docs/images',
        'benchmarks/results/images',
        'tutorials/basic/images',
        'tutorials/intermediate/images',
        'tutorials/advanced/images'
    ]
    
    for dir_path in viz_dirs:
        os.makedirs(dir_path, exist_ok=True)

def get_model_color(model_id: str) -> str:
    """Get the color for a specific model based on its family"""
    for family, color in MODEL_FAMILY_COLORS.items():
        if family in model_id.lower():
            return color
    # Default color if no match
    return '#17becf'  # Cyan

def create_comparison_chart(
    data: Dict[str, Union[float, int]], 
    title: str,
    ylabel: str,
    filename: str,
    output_dir: str = 'docs/images',
    sort_values: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> str:
    """
    Create and save an SVG bar chart comparing values across models
    
    Args:
        data: Dictionary of model_ids and their values
        title: Chart title
        ylabel: Y-axis label
        filename: Output filename (without extension)
        output_dir: Directory to save the SVG file
        sort_values: Whether to sort bars by value
        figsize: Figure dimensions
        
    Returns:
        Path to the saved SVG file
    """
    # Convert dictionary to dataframe
    df = pd.DataFrame(list(data.items()), columns=['Model', 'Value'])
    
    # Extract model family for coloring
    df['Family'] = df['Model'].apply(lambda x: x.split('.')[0] if '.' in x else x)
    
    # Sort if requested
    if sort_values:
        df = df.sort_values('Value', ascending=False)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart with colors based on model family
    bars = ax.bar(
        df['Model'], 
        df['Value'],
        color=[get_model_color(model) for model in df['Model']]
    )
    
    # Add labels and styling
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02 * max(df['Value']),
            f'{height:.2f}',
            ha='center', 
            va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as SVG
    output_path = os.path.join(output_dir, f"{filename}.svg")
    plt.savefig(output_path, **SVG_CONFIG)
    plt.close()
    
    return output_path

def create_time_series_chart(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    output_dir: str = 'docs/images',
    figsize: Tuple[int, int] = (12, 6)
) -> str:
    """
    Create and save an SVG line chart for time series data
    
    Args:
        data: DataFrame containing the time series data
        x_column: Column name for x-axis values
        y_columns: List of column names for different lines to plot
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Output filename (without extension)
        output_dir: Directory to save the SVG file
        figsize: Figure dimensions
        
    Returns:
        Path to the saved SVG file
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each column as a separate line
    for i, column in enumerate(y_columns):
        if column in data.columns:
            # Use model family colors if the column name contains a model family
            color = None
            for family, family_color in MODEL_FAMILY_COLORS.items():
                if family in column.lower():
                    color = family_color
                    break
            
            # If no model family match, use default color cycling
            if color:
                ax.plot(data[x_column], data[column], label=column, color=color)
            else:
                ax.plot(data[x_column], data[column], label=column)
    
    # Add labels and styling
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as SVG
    output_path = os.path.join(output_dir, f"{filename}.svg")
    plt.savefig(output_path, **SVG_CONFIG)
    plt.close()
    
    return output_path

# Add more specialized visualization functions below as needed