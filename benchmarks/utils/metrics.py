"""
Metrics collection and analysis utilities for AWS Bedrock benchmarks.

This module provides tools for collecting and analyzing performance metrics
for AWS Bedrock inference methods, including latency, throughput, tokens per
second, and quota utilization.
"""

import time
import json
import statistics
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class InferenceMetrics:
    """Class for storing inference metrics for a single request."""
    
    # Request metadata
    request_id: str
    model_id: str
    inference_type: str  # 'sync', 'stream', 'async'
    timestamp: float
    input_text: Optional[str] = None
    
    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Timing metrics
    request_time: float = 0.0  # Total request time
    first_token_time: Optional[float] = None  # Time to first token (for streaming)
    tokens_per_second: float = 0.0
    
    # Status
    success: bool = True
    error: Optional[str] = None
    retry_count: int = 0
    
    # Quota metrics
    was_throttled: bool = False
    backoff_time: float = 0.0


@dataclass
class BenchmarkResult:
    """Class for storing benchmark results for multiple requests."""
    
    # Benchmark metadata
    benchmark_id: str
    model_id: str
    inference_type: str
    start_time: float
    end_time: Optional[float] = None
    description: Optional[str] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    metrics: List[InferenceMetrics] = field(default_factory=list)
    
    # Summary statistics (calculated when requested)
    _summary: Optional[Dict[str, Any]] = None
    
    def add_metric(self, metric: InferenceMetrics) -> None:
        """Add a metric to the benchmark results."""
        self.metrics.append(metric)
        # Invalidate summary when new data is added
        self._summary = None
    
    def end_benchmark(self) -> None:
        """Mark the benchmark as complete."""
        self.end_time = time.time()
        # Invalidate summary
        self._summary = None
    
    def get_summary(self, recalculate: bool = False) -> Dict[str, Any]:
        """
        Get summary statistics for the benchmark.
        
        Args:
            recalculate: Force recalculation even if summary exists
            
        Returns:
            Dictionary with summary statistics
        """
        if self._summary is not None and not recalculate:
            return self._summary
        
        # Basic counts
        total_requests = len(self.metrics)
        successful_requests = sum(1 for m in self.metrics if m.success)
        failed_requests = total_requests - successful_requests
        throttled_requests = sum(1 for m in self.metrics if m.was_throttled)
        
        # Timing metrics (only for successful requests)
        successful_metrics = [m for m in self.metrics if m.success]
        
        if successful_metrics:
            request_times = [m.request_time for m in successful_metrics]
            
            # Calculate tokens
            total_input_tokens = sum(m.input_tokens for m in successful_metrics)
            total_output_tokens = sum(m.output_tokens for m in successful_metrics)
            total_tokens = total_input_tokens + total_output_tokens
            
            # Calculate timing stats
            avg_request_time = statistics.mean(request_times)
            median_request_time = statistics.median(request_times)
            min_request_time = min(request_times)
            max_request_time = max(request_times)
            
            # Only include non-zero tps values for average
            tps_values = [m.tokens_per_second for m in successful_metrics if m.tokens_per_second > 0]
            avg_tokens_per_second = statistics.mean(tps_values) if tps_values else 0
            
            # First token times (for streaming)
            first_token_times = [m.first_token_time for m in successful_metrics if m.first_token_time is not None]
            avg_first_token_time = statistics.mean(first_token_times) if first_token_times else None
            
            # Retries
            total_retries = sum(m.retry_count for m in self.metrics)
            avg_retries = total_retries / total_requests if total_requests > 0 else 0
            
            # Throughput
            duration = (self.end_time or time.time()) - self.start_time
            rps = successful_requests / duration if duration > 0 else 0
            
            # Estimated RPM and TPM
            rpm = rps * 60
            tpm = (total_tokens / duration) * 60 if duration > 0 else 0
            
        else:
            # No successful requests
            avg_request_time = 0
            median_request_time = 0
            min_request_time = 0
            max_request_time = 0
            avg_tokens_per_second = 0
            total_input_tokens = 0
            total_output_tokens = 0
            total_tokens = 0
            avg_first_token_time = None
            total_retries = sum(m.retry_count for m in self.metrics)
            avg_retries = total_retries / total_requests if total_requests > 0 else 0
            duration = (self.end_time or time.time()) - self.start_time
            rps = 0
            rpm = 0
            tpm = 0
        
        # Build summary
        self._summary = {
            "benchmark_id": self.benchmark_id,
            "model_id": self.model_id,
            "inference_type": self.inference_type,
            "duration_seconds": duration,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "throttled_requests": throttled_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "avg_request_time": avg_request_time,
            "median_request_time": median_request_time,
            "min_request_time": min_request_time,
            "max_request_time": max_request_time,
            "avg_tokens_per_second": avg_tokens_per_second,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "avg_tokens_per_request": total_tokens / successful_requests if successful_requests > 0 else 0,
            "avg_first_token_time": avg_first_token_time,
            "total_retries": total_retries,
            "avg_retries": avg_retries,
            "requests_per_second": rps,
            "estimated_rpm": rpm,
            "estimated_tpm": tpm
        }
        
        return self._summary
    
    def to_json(self, include_metrics: bool = True) -> str:
        """
        Convert the benchmark result to JSON.
        
        Args:
            include_metrics: Whether to include individual metrics
            
        Returns:
            JSON string representation
        """
        # Get summary
        summary = self.get_summary()
        
        # Create result object
        result = {
            "benchmark_id": self.benchmark_id,
            "model_id": self.model_id,
            "inference_type": self.inference_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "description": self.description,
            "config": self.config,
            "summary": summary
        }
        
        # Add individual metrics if requested
        if include_metrics:
            metrics_list = []
            
            for metric in self.metrics:
                metric_dict = {
                    "request_id": metric.request_id,
                    "timestamp": metric.timestamp,
                    "success": metric.success,
                    "input_tokens": metric.input_tokens,
                    "output_tokens": metric.output_tokens,
                    "total_tokens": metric.total_tokens,
                    "request_time": metric.request_time,
                    "tokens_per_second": metric.tokens_per_second,
                    "retry_count": metric.retry_count,
                    "was_throttled": metric.was_throttled,
                    "backoff_time": metric.backoff_time
                }
                
                # Add optional fields
                if metric.error:
                    metric_dict["error"] = metric.error
                
                if metric.first_token_time is not None:
                    metric_dict["first_token_time"] = metric.first_token_time
                
                metrics_list.append(metric_dict)
            
            result["metrics"] = metrics_list
        
        return json.dumps(result, indent=2)
    
    def save_to_file(self, file_path: str, include_metrics: bool = True) -> None:
        """
        Save the benchmark result to a file.
        
        Args:
            file_path: Path to save the result
            include_metrics: Whether to include individual metrics
        """
        with open(file_path, 'w') as f:
            f.write(self.to_json(include_metrics=include_metrics))


class MetricsCollector:
    """Utility for collecting metrics during benchmark runs."""
    
    def __init__(
        self,
        benchmark_id: str,
        model_id: str,
        inference_type: str,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize a metrics collector.
        
        Args:
            benchmark_id: Unique identifier for the benchmark
            model_id: Model ID being benchmarked
            inference_type: Type of inference ('sync', 'stream', 'async')
            description: Optional description of the benchmark
            config: Optional configuration parameters
            logger: Optional logger instance
        """
        self.benchmark_id = benchmark_id
        self.model_id = model_id
        self.inference_type = inference_type
        self.description = description
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize benchmark result
        self.result = BenchmarkResult(
            benchmark_id=benchmark_id,
            model_id=model_id,
            inference_type=inference_type,
            start_time=time.time(),
            description=description,
            config=self.config
        )
        
        # Request counter for generating request IDs
        self.request_counter = 0
    
    def start_request(
        self,
        input_text: Optional[str] = None,
        custom_request_id: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Start tracking a new request.
        
        Args:
            input_text: Optional input text for the request
            custom_request_id: Optional custom request ID
            
        Returns:
            Tuple of (request_id, start_time)
        """
        # Generate request ID if not provided
        self.request_counter += 1
        request_id = custom_request_id or f"{self.benchmark_id}-req-{self.request_counter}"
        
        # Record start time
        start_time = time.time()
        
        self.logger.debug(f"Starting request {request_id}")
        
        return request_id, start_time
    
    def record_metric(
        self,
        request_id: str,
        start_time: float,
        end_time: float,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
        error: Optional[str] = None,
        first_token_time: Optional[float] = None,
        retry_count: int = 0,
        was_throttled: bool = False,
        backoff_time: float = 0.0,
        input_text: Optional[str] = None
    ) -> None:
        """
        Record metrics for a request.
        
        Args:
            request_id: Request identifier
            start_time: Request start time
            end_time: Request end time
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            success: Whether the request was successful
            error: Error message if request failed
            first_token_time: Time to first token (for streaming)
            retry_count: Number of retries
            was_throttled: Whether the request was throttled
            backoff_time: Total backoff time
            input_text: Optional input text
        """
        request_time = end_time - start_time
        total_tokens = input_tokens + output_tokens
        
        # Calculate tokens per second
        if success and request_time > 0 and output_tokens > 0:
            tokens_per_second = output_tokens / request_time
        else:
            tokens_per_second = 0
        
        # Create metric
        metric = InferenceMetrics(
            request_id=request_id,
            model_id=self.model_id,
            inference_type=self.inference_type,
            timestamp=start_time,
            input_text=input_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            request_time=request_time,
            first_token_time=first_token_time,
            tokens_per_second=tokens_per_second,
            success=success,
            error=error,
            retry_count=retry_count,
            was_throttled=was_throttled,
            backoff_time=backoff_time
        )
        
        # Add to results
        self.result.add_metric(metric)
        
        # Log metric
        log_level = logging.INFO if success else logging.WARNING
        self.logger.log(
            log_level,
            f"Request {request_id}: {'Success' if success else 'Failed'} "
            f"in {request_time:.3f}s, {input_tokens}/{output_tokens} tokens, "
            f"{tokens_per_second:.1f} tokens/sec"
        )
    
    def end_benchmark(self) -> BenchmarkResult:
        """
        End the benchmark and return results.
        
        Returns:
            BenchmarkResult object
        """
        self.result.end_benchmark()
        summary = self.result.get_summary()
        
        # Log summary
        self.logger.info(
            f"Benchmark {self.benchmark_id} completed: "
            f"{summary['successful_requests']}/{summary['total_requests']} requests succeeded, "
            f"{summary['avg_request_time']:.3f}s avg, "
            f"{summary['total_tokens']} total tokens, "
            f"{summary['estimated_rpm']:.1f} RPM, "
            f"{summary['estimated_tpm']:.1f} TPM"
        )
        
        return self.result


def analyze_benchmark(
    result: BenchmarkResult,
    throughput_window: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform extended analysis on benchmark results.
    
    Args:
        result: BenchmarkResult to analyze
        throughput_window: Window size (in seconds) for throughput analysis
        
    Returns:
        Dictionary with analysis results
    """
    # Get basic summary
    summary = result.get_summary(recalculate=True)
    
    # Prepare analysis
    analysis = {
        "summary": summary,
        "time_series": {}
    }
    
    # Only analyze if we have metrics
    if not result.metrics:
        return analysis
    
    # Time series data - if window specified, calculate moving throughput
    if throughput_window:
        # Sort metrics by timestamp
        sorted_metrics = sorted(result.metrics, key=lambda m: m.timestamp)
        
        # Prepare time series data
        timestamps = []
        rps_values = []
        tps_values = []
        
        # Use sliding window to calculate throughput
        window_start = sorted_metrics[0].timestamp
        window_end = window_start + throughput_window
        
        while window_start < sorted_metrics[-1].timestamp:
            # Count requests and tokens in this window
            window_requests = 0
            window_tokens = 0
            
            for metric in sorted_metrics:
                if window_start <= metric.timestamp < window_end and metric.success:
                    window_requests += 1
                    window_tokens += metric.total_tokens
            
            # Calculate throughput
            rps = window_requests / throughput_window
            tps = window_tokens / throughput_window
            
            # Add to time series
            timestamps.append(window_start)
            rps_values.append(rps)
            tps_values.append(tps)
            
            # Move window
            window_start += throughput_window
            window_end = window_start + throughput_window
        
        # Add time series to analysis
        analysis["time_series"] = {
            "timestamps": timestamps,
            "requests_per_second": rps_values,
            "tokens_per_second": tps_values
        }
    
    # Latency distribution - generate histogram data
    request_times = [m.request_time for m in result.metrics if m.success]
    
    if request_times:
        # Calculate latency percentiles
        percentiles = [50, 75, 90, 95, 99]
        latency_percentiles = {}
        
        for p in percentiles:
            latency_percentiles[f"p{p}"] = statistics.quantiles(request_times, n=100)[p-1]
        
        # Add to analysis
        analysis["latency"] = {
            "percentiles": latency_percentiles,
            "values": request_times
        }
    
    # Calculate token efficiency (tokens per request)
    token_counts = [m.total_tokens for m in result.metrics if m.success]
    
    if token_counts:
        analysis["tokens"] = {
            "avg_per_request": statistics.mean(token_counts),
            "median_per_request": statistics.median(token_counts),
            "min_per_request": min(token_counts),
            "max_per_request": max(token_counts)
        }
    
    # Error analysis
    errors = {}
    for m in result.metrics:
        if not m.success and m.error:
            error_type = m.error.split(':')[0] if ':' in m.error else m.error
            errors[error_type] = errors.get(error_type, 0) + 1
    
    if errors:
        analysis["errors"] = errors
    
    return analysis


def compare_benchmarks(
    results: List[BenchmarkResult],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare multiple benchmark results.
    
    Args:
        results: List of BenchmarkResult objects to compare
        metrics: Optional list of specific metrics to compare
        
    Returns:
        Dictionary with comparison results
    """
    if not results:
        return {}
    
    # Default metrics to compare
    if metrics is None:
        metrics = [
            "success_rate",
            "avg_request_time",
            "median_request_time",
            "avg_tokens_per_second",
            "requests_per_second",
            "estimated_rpm",
            "estimated_tpm"
        ]
    
    # Prepare comparison
    comparison = {
        "benchmark_ids": [result.benchmark_id for result in results],
        "model_ids": [result.model_id for result in results],
        "inference_types": [result.inference_type for result in results],
        "metrics": {}
    }
    
    # Extract and compare each metric
    for metric in metrics:
        values = []
        
        for result in results:
            summary = result.get_summary()
            values.append(summary.get(metric, 0))
        
        comparison["metrics"][metric] = values
    
    return comparison