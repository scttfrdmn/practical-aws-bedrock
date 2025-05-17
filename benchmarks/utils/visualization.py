"""
Visualization utilities for AWS Bedrock benchmarks.

This module provides tools for visualizing benchmark results with
various chart types, including line charts, bar charts, and heatmaps.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from .metrics import BenchmarkResult, analyze_benchmark, compare_benchmarks

# Check if matplotlib and other visualization libraries are available
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    from datetime import datetime
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


def create_throughput_chart(
    result: Union[BenchmarkResult, str],
    output_path: str = None,
    title: Optional[str] = None,
    window_size: int = 10,
    show_tokens: bool = True,
    show_chart: bool = True
) -> Optional[str]:
    """
    Create a throughput over time chart from benchmark results.
    
    Args:
        result: BenchmarkResult object or path to JSON result file
        output_path: Optional path to save the chart
        title: Optional chart title
        window_size: Window size (in seconds) for calculating throughput
        show_tokens: Whether to show token throughput
        show_chart: Whether to display the chart
        
    Returns:
        Path to saved chart if output_path is provided, None otherwise
    """
    if not HAS_VISUALIZATION:
        raise ImportError(
            "Visualization libraries not available. "
            "Install matplotlib with: pip install matplotlib"
        )
    
    # Load result if path provided
    if isinstance(result, str):
        with open(result, 'r') as f:
            data = json.load(f)
            # Convert to BenchmarkResult
            benchmark_result = BenchmarkResult(
                benchmark_id=data["benchmark_id"],
                model_id=data["model_id"],
                inference_type=data["inference_type"],
                start_time=data["start_time"],
                end_time=data["end_time"],
                description=data.get("description"),
                config=data.get("config", {})
            )
            
            # Add metrics if available
            if "metrics" in data:
                for metric_data in data["metrics"]:
                    from .metrics import InferenceMetrics
                    
                    metric = InferenceMetrics(
                        request_id=metric_data["request_id"],
                        model_id=benchmark_result.model_id,
                        inference_type=benchmark_result.inference_type,
                        timestamp=metric_data["timestamp"],
                        input_tokens=metric_data.get("input_tokens", 0),
                        output_tokens=metric_data.get("output_tokens", 0),
                        total_tokens=metric_data.get("total_tokens", 0),
                        request_time=metric_data.get("request_time", 0.0),
                        success=metric_data.get("success", True),
                        error=metric_data.get("error"),
                        retry_count=metric_data.get("retry_count", 0),
                        was_throttled=metric_data.get("was_throttled", False),
                        backoff_time=metric_data.get("backoff_time", 0.0)
                    )
                    
                    if "first_token_time" in metric_data:
                        metric.first_token_time = metric_data["first_token_time"]
                    
                    benchmark_result.add_metric(metric)
    else:
        benchmark_result = result
    
    # Analyze benchmark with specified window size
    analysis = analyze_benchmark(benchmark_result, throughput_window=window_size)
    
    # Check if time series data is available
    if "time_series" not in analysis or not analysis["time_series"]:
        raise ValueError(
            "No time series data available for visualization. "
            "Ensure the benchmark contains multiple requests over time."
        )
    
    # Extract time series data
    timestamps = analysis["time_series"]["timestamps"]
    rps_values = analysis["time_series"]["requests_per_second"]
    tps_values = analysis["time_series"]["tokens_per_second"]
    
    # Convert Unix timestamps to datetime objects
    datetime_labels = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot requests per second
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Requests per Second', color=color)
    ax1.plot(datetime_labels, rps_values, marker='o', linestyle='-', color=color, label='Requests/sec')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    # Only show token throughput if requested
    if show_tokens:
        # Create second y-axis for tokens per second
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Tokens per Second', color=color)
        ax2.plot(datetime_labels, tps_values, marker='s', linestyle='-', color=color, label='Tokens/sec')
        ax2.tick_params(axis='y', labelcolor=color)
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f"Throughput Analysis - {benchmark_result.model_id} ({benchmark_result.inference_type})")
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if show_tokens:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Tight layout
    fig.tight_layout()
    
    # Save chart if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")
    
    # Show chart if requested
    if show_chart:
        plt.show()
    else:
        plt.close()
    
    return output_path if output_path else None


def create_comparison_chart(
    results: Union[List[BenchmarkResult], str],
    metrics: List[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_chart: bool = True
) -> Optional[str]:
    """
    Create a comparison chart for multiple benchmark results.
    
    Args:
        results: List of BenchmarkResult objects or path to directory with JSON results
        metrics: List of metrics to compare (default: RPM, latency, tokens per second)
        output_path: Optional path to save the chart
        title: Optional chart title
        show_chart: Whether to display the chart
        
    Returns:
        Path to saved chart if output_path is provided, None otherwise
    """
    if not HAS_VISUALIZATION:
        raise ImportError(
            "Visualization libraries not available. "
            "Install matplotlib with: pip install matplotlib"
        )
    
    # Default metrics to compare
    if metrics is None:
        metrics = [
            "estimated_rpm",
            "avg_request_time",
            "avg_tokens_per_second"
        ]
    
    # Load results if directory path provided
    if isinstance(results, str):
        # Check if it's a directory
        if os.path.isdir(results):
            loaded_results = []
            
            # Find JSON files in directory
            for filename in os.listdir(results):
                if filename.endswith('.json'):
                    file_path = os.path.join(results, filename)
                    
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                            # Convert to BenchmarkResult
                            benchmark_result = BenchmarkResult(
                                benchmark_id=data["benchmark_id"],
                                model_id=data["model_id"],
                                inference_type=data["inference_type"],
                                start_time=data["start_time"],
                                end_time=data["end_time"],
                                description=data.get("description"),
                                config=data.get("config", {})
                            )
                            
                            # Set summary if available
                            if "summary" in data:
                                benchmark_result._summary = data["summary"]
                            
                            loaded_results.append(benchmark_result)
                    except Exception as e:
                        print(f"Error loading result from {file_path}: {str(e)}")
            
            results = loaded_results
        else:
            raise ValueError(f"Path {results} is not a directory")
    
    # Ensure we have results to compare
    if not results:
        raise ValueError("No benchmark results provided for comparison")
    
    # Compare benchmarks
    comparison = compare_benchmarks(results, metrics=metrics)
    
    # Extract data for visualization
    benchmark_ids = comparison["benchmark_ids"]
    model_ids = comparison["model_ids"]
    inference_types = comparison["inference_types"]
    
    # Create labels combining model and inference type
    labels = [
        f"{model}_{infer_type}"
        for model, infer_type in zip(model_ids, inference_types)
    ]
    
    # Set up the figure with subplots for each metric
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(15, 6))
    
    # Handle case where only one metric is provided
    if num_metrics == 1:
        axes = [axes]
    
    # Create a bar chart for each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = comparison["metrics"][metric]
        
        # Create bar chart
        bars = ax.bar(range(len(labels)), values, width=0.7)
        
        # Set title and labels
        metric_name = metric.replace('_', ' ').title()
        ax.set_title(metric_name)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom'
            )
        
        # Grid lines
        ax.grid(True, alpha=0.3, axis='y')
        
        # Y-axis starts at 0
        ax.set_ylim(bottom=0)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle("Benchmark Comparison", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add additional space for the overall title
    fig.subplots_adjust(top=0.85)
    
    # Save chart if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")
    
    # Show chart if requested
    if show_chart:
        plt.show()
    else:
        plt.close()
    
    return output_path if output_path else None


def create_latency_distribution(
    result: Union[BenchmarkResult, str],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_percentiles: bool = True,
    show_chart: bool = True
) -> Optional[str]:
    """
    Create a latency distribution chart from benchmark results.
    
    Args:
        result: BenchmarkResult object or path to JSON result file
        output_path: Optional path to save the chart
        title: Optional chart title
        show_percentiles: Whether to show percentile lines
        show_chart: Whether to display the chart
        
    Returns:
        Path to saved chart if output_path is provided, None otherwise
    """
    if not HAS_VISUALIZATION:
        raise ImportError(
            "Visualization libraries not available. "
            "Install matplotlib with: pip install matplotlib"
        )
    
    # Load result if path provided
    if isinstance(result, str):
        with open(result, 'r') as f:
            data = json.load(f)
            # If this is a pre-analyzed result with latency data
            if "latency" in data and "values" in data["latency"]:
                latency_values = data["latency"]["values"]
                percentiles = data["latency"]["percentiles"] if "percentiles" in data["latency"] else None
                
                # Create a basic BenchmarkResult for labels
                benchmark_result = BenchmarkResult(
                    benchmark_id=data["benchmark_id"],
                    model_id=data["model_id"],
                    inference_type=data["inference_type"],
                    start_time=data.get("start_time", 0),
                    description=data.get("description")
                )
            else:
                # Load the BenchmarkResult and analyze
                benchmark_result = BenchmarkResult(
                    benchmark_id=data["benchmark_id"],
                    model_id=data["model_id"],
                    inference_type=data["inference_type"],
                    start_time=data["start_time"],
                    end_time=data["end_time"],
                    description=data.get("description"),
                    config=data.get("config", {})
                )
                
                # Add metrics if available
                if "metrics" in data:
                    for metric_data in data["metrics"]:
                        from .metrics import InferenceMetrics
                        
                        metric = InferenceMetrics(
                            request_id=metric_data["request_id"],
                            model_id=benchmark_result.model_id,
                            inference_type=benchmark_result.inference_type,
                            timestamp=metric_data["timestamp"],
                            input_tokens=metric_data.get("input_tokens", 0),
                            output_tokens=metric_data.get("output_tokens", 0),
                            total_tokens=metric_data.get("total_tokens", 0),
                            request_time=metric_data.get("request_time", 0.0),
                            success=metric_data.get("success", True),
                            error=metric_data.get("error"),
                            retry_count=metric_data.get("retry_count", 0),
                            was_throttled=metric_data.get("was_throttled", False),
                            backoff_time=metric_data.get("backoff_time", 0.0)
                        )
                        
                        if "first_token_time" in metric_data:
                            metric.first_token_time = metric_data["first_token_time"]
                        
                        benchmark_result.add_metric(metric)
                
                # Analyze to get latency distribution
                analysis = analyze_benchmark(benchmark_result)
                
                if "latency" not in analysis:
                    raise ValueError(
                        "No latency data available for visualization. "
                        "Ensure the benchmark contains successful requests."
                    )
                
                latency_values = analysis["latency"]["values"]
                percentiles = analysis["latency"]["percentiles"]
    else:
        benchmark_result = result
        
        # Analyze benchmark to get latency distribution
        analysis = analyze_benchmark(benchmark_result)
        
        if "latency" not in analysis:
            raise ValueError(
                "No latency data available for visualization. "
                "Ensure the benchmark contains successful requests."
            )
        
        latency_values = analysis["latency"]["values"]
        percentiles = analysis["latency"]["percentiles"]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(
        latency_values, 
        bins=30, 
        alpha=0.7, 
        color='tab:blue',
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add percentile lines if requested
    if show_percentiles and percentiles:
        percentile_colors = {
            "p50": "green",
            "p75": "orange",
            "p90": "red",
            "p95": "purple",
            "p99": "brown"
        }
        
        for p_name, p_value in percentiles.items():
            ax.axvline(
                x=p_value, 
                color=percentile_colors.get(p_name, 'gray'),
                linestyle='--', 
                linewidth=2,
                label=f"{p_name}: {p_value:.3f}s"
            )
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Latency Distribution - {benchmark_result.model_id} "
            f"({benchmark_result.inference_type})"
        )
    
    # Set labels
    ax.set_xlabel('Request Time (seconds)')
    ax.set_ylabel('Frequency')
    
    # Add mean and median lines
    mean_latency = sum(latency_values) / len(latency_values)
    median_latency = sorted(latency_values)[len(latency_values) // 2]
    
    ax.axvline(
        x=mean_latency, 
        color='red',
        linestyle='-', 
        linewidth=2,
        label=f"Mean: {mean_latency:.3f}s"
    )
    
    ax.axvline(
        x=median_latency, 
        color='green',
        linestyle='-', 
        linewidth=2,
        label=f"Median: {median_latency:.3f}s"
    )
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    fig.tight_layout()
    
    # Save chart if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")
    
    # Show chart if requested
    if show_chart:
        plt.show()
    else:
        plt.close()
    
    return output_path if output_path else None


def create_throughput_vs_quota_chart(
    results: Union[List[BenchmarkResult], str],
    quota_limits: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_chart: bool = True
) -> Optional[str]:
    """
    Create a chart showing throughput relative to quota limits.
    
    Args:
        results: List of BenchmarkResult objects or path to directory with JSON results
        quota_limits: Dictionary mapping model IDs to RPM and TPM limits
        output_path: Optional path to save the chart
        title: Optional chart title
        show_chart: Whether to display the chart
        
    Returns:
        Path to saved chart if output_path is provided, None otherwise
    """
    if not HAS_VISUALIZATION:
        raise ImportError(
            "Visualization libraries not available. "
            "Install matplotlib with: pip install matplotlib"
        )
    
    # Load results if directory path provided
    if isinstance(results, str):
        # Check if it's a directory
        if os.path.isdir(results):
            loaded_results = []
            
            # Find JSON files in directory
            for filename in os.listdir(results):
                if filename.endswith('.json'):
                    file_path = os.path.join(results, filename)
                    
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                            # Convert to BenchmarkResult
                            benchmark_result = BenchmarkResult(
                                benchmark_id=data["benchmark_id"],
                                model_id=data["model_id"],
                                inference_type=data["inference_type"],
                                start_time=data["start_time"],
                                end_time=data["end_time"],
                                description=data.get("description"),
                                config=data.get("config", {})
                            )
                            
                            # Set summary if available
                            if "summary" in data:
                                benchmark_result._summary = data["summary"]
                            
                            loaded_results.append(benchmark_result)
                    except Exception as e:
                        print(f"Error loading result from {file_path}: {str(e)}")
            
            results = loaded_results
        else:
            raise ValueError(f"Path {results} is not a directory")
    
    # Ensure we have results to compare
    if not results:
        raise ValueError("No benchmark results provided for visualization")
    
    # Extract data for visualization
    labels = []
    rpm_values = []
    rpm_quotas = []
    tpm_values = []
    tpm_quotas = []
    
    for result in results:
        # Get summary
        summary = result.get_summary()
        
        # Create label
        label = f"{result.model_id}_{result.inference_type}"
        labels.append(label)
        
        # Get throughput values
        rpm_values.append(summary["estimated_rpm"])
        tpm_values.append(summary["estimated_tpm"])
        
        # Get quota limits
        model_quotas = quota_limits.get(result.model_id, {})
        rpm_quotas.append(model_quotas.get("rpm", 0))
        tpm_quotas.append(model_quotas.get("tpm", 0))
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot RPM vs Quota
    x = np.arange(len(labels))
    width = 0.35
    
    # RPM bars
    rpm_bars = ax1.bar(x - width/2, rpm_values, width, label='Actual RPM')
    quota_bars = ax1.bar(x + width/2, rpm_quotas, width, label='Quota Limit')
    
    # RPM chart formatting
    ax1.set_title('Requests Per Minute vs Quota')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add utilization percentage
    for i, (rpm, quota) in enumerate(zip(rpm_values, rpm_quotas)):
        if quota > 0:
            util_pct = (rpm / quota) * 100
            ax1.annotate(
                f"{util_pct:.1f}%",
                xy=(x[i], max(rpm, quota) + 5),
                ha='center'
            )
    
    # Plot TPM vs Quota
    tpm_bars = ax2.bar(x - width/2, tpm_values, width, label='Actual TPM')
    tpm_quota_bars = ax2.bar(x + width/2, tpm_quotas, width, label='Quota Limit')
    
    # TPM chart formatting
    ax2.set_title('Tokens Per Minute vs Quota')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add utilization percentage
    for i, (tpm, quota) in enumerate(zip(tpm_values, tpm_quotas)):
        if quota > 0:
            util_pct = (tpm / quota) * 100
            ax2.annotate(
                f"{util_pct:.1f}%",
                xy=(x[i], max(tpm, quota) + 5),
                ha='center'
            )
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle("Throughput vs Quota Limits", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add additional space for the overall title
    fig.subplots_adjust(top=0.85)
    
    # Save chart if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")
    
    # Show chart if requested
    if show_chart:
        plt.show()
    else:
        plt.close()
    
    return output_path if output_path else None