"""
Quota optimization benchmark for AWS Bedrock.

This script benchmarks different strategies for optimizing throughput
within AWS Bedrock quota limits for RPM (Requests Per Minute) and
TPM (Tokens Per Minute).
"""

import os
import time
import json
import logging
import argparse
import random
import concurrent.futures
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.synchronous.basic_client import BedrockClient
from src.inference.synchronous.quota_aware import QuotaAwareBedrockClient

from benchmarks.utils.metrics import MetricsCollector, BenchmarkResult


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bedrock-quota-benchmark")


# Sample prompts of varying input token sizes
SAMPLE_PROMPTS = {
    "small": [
        "Summarize AWS Bedrock in 10 words.",
        "What is a transformer model?",
        "Explain API quotas briefly.",
        "What is TPM?",
        "Define foundation models."
    ],
    "medium": [
        "Explain AWS Bedrock quota management and how it affects application throughput. Include examples of TPM and RPM limits.",
        "Write a brief comparison of synchronous, streaming, and asynchronous APIs in AWS Bedrock. Explain when each should be used.",
        "Describe three strategies for optimizing throughput when working with foundation models. Include considerations for response latency.",
        "Explain the token economy of large language models and how token limitations impact application design. Give examples.",
        "Compare and contrast the performance characteristics of different foundation model families available in AWS Bedrock."
    ]
}


def run_naive_throughput_benchmark(
    model_id: str,
    concurrent_requests: int,
    duration_seconds: int,
    prompt_size: str = "small",
    max_tokens: int = 150,
    temperature: float = 0.7,
    output_dir: Optional[str] = None
) -> BenchmarkResult:
    """
    Run a naive (non-quota-aware) throughput benchmark.
    
    This benchmark makes concurrent requests without specific quota management.
    
    Args:
        model_id: The Bedrock model ID
        concurrent_requests: Number of concurrent requests
        duration_seconds: Benchmark duration in seconds
        prompt_size: Prompt size category ('small' or 'medium')
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation
        output_dir: Optional directory to save results
        
    Returns:
        BenchmarkResult object with benchmark results
    """
    # Create client
    client = BedrockClient(model_id=model_id)
    
    # Create metrics collector
    collector = MetricsCollector(
        benchmark_id=f"naive-{model_id.split('.')[-1]}-c{concurrent_requests}",
        model_id=model_id,
        inference_type="sync-naive",
        description=f"Naive throughput benchmark with {concurrent_requests} concurrent requests",
        config={
            "concurrent_requests": concurrent_requests,
            "duration_seconds": duration_seconds,
            "prompt_size": prompt_size,
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        logger=logger
    )
    
    # Get prompts for this size
    prompts = SAMPLE_PROMPTS.get(prompt_size, SAMPLE_PROMPTS["small"])
    
    # Run benchmark
    logger.info(f"Starting naive throughput benchmark for {model_id} with {concurrent_requests} concurrent requests")
    logger.info(f"Using {prompt_size} prompts with max_tokens={max_tokens}, temperature={temperature}")
    logger.info(f"Benchmark will run for {duration_seconds} seconds")
    
    # Create a thread pool for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        # Calculate end time
        end_time = time.time() + duration_seconds
        
        # Function to process a single request
        def process_request():
            # Select a random prompt
            prompt = random.choice(prompts)
            
            # Start tracking request
            request_id, start_time = collector.start_request(input_text=prompt)
            
            try:
                # Make the request
                response = client.invoke(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Record successful result
                collector.record_metric(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=time.time(),
                    input_tokens=response.get("input_tokens", 0),
                    output_tokens=response.get("output_tokens", 0),
                    success=True,
                    input_text=prompt
                )
                
                return True
                
            except Exception as e:
                # Check for throttling
                was_throttled = "ThrottlingException" in str(e)
                
                # Record error
                logger.error(f"Error in request {request_id}: {str(e)}")
                collector.record_metric(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=time.time(),
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error=str(e),
                    was_throttled=was_throttled,
                    input_text=prompt
                )
                
                return False
        
        # Submit tasks until duration is reached
        futures = []
        
        while time.time() < end_time:
            # Submit new tasks to replace completed ones
            while len(futures) < concurrent_requests and time.time() < end_time:
                futures.append(executor.submit(process_request))
            
            # Wait for at least one task to complete
            done, not_done = concurrent.futures.wait(
                futures, 
                timeout=0.1,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            # Remove completed futures
            futures = list(not_done)
            
            # Small delay to avoid CPU spinning
            time.sleep(0.01)
    
    # End benchmark and get results
    result = collector.end_benchmark()
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{result.benchmark_id}.json")
        result.save_to_file(output_path)
        logger.info(f"Results saved to {output_path}")
    
    return result


def run_quota_aware_benchmark(
    model_id: str,
    max_rpm: Optional[int],
    max_tpm: Optional[int],
    duration_seconds: int,
    prompt_size: str = "small",
    max_tokens: int = 150,
    temperature: float = 0.7,
    token_smoothing: bool = True,
    output_dir: Optional[str] = None
) -> BenchmarkResult:
    """
    Run a quota-aware throughput benchmark.
    
    This benchmark uses the QuotaAwareBedrockClient to optimize throughput.
    
    Args:
        model_id: The Bedrock model ID
        max_rpm: Maximum requests per minute (model's quota)
        max_tpm: Maximum tokens per minute (model's quota)
        duration_seconds: Benchmark duration in seconds
        prompt_size: Prompt size category ('small' or 'medium')
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation
        token_smoothing: Whether to use token bucket smoothing
        output_dir: Optional directory to save results
        
    Returns:
        BenchmarkResult object with benchmark results
    """
    # Create client
    client = QuotaAwareBedrockClient(
        model_id=model_id,
        max_rpm=max_rpm,
        max_tpm=max_tpm,
        token_bucket_smoothing=token_smoothing
    )
    
    # Determine optimal concurrency based on quota limits
    if max_rpm:
        # Set concurrency based on RPM
        optimal_concurrency = max(1, (max_rpm // 60) * 3)  # 3x the per-second limit
    else:
        # Default concurrency
        optimal_concurrency = 10
    
    # Create metrics collector
    collector = MetricsCollector(
        benchmark_id=f"quota-{model_id.split('.')[-1]}-rpm{max_rpm or 0}-tpm{max_tpm or 0}",
        model_id=model_id,
        inference_type="sync-quota",
        description=f"Quota-aware throughput benchmark with RPM={max_rpm}, TPM={max_tpm}",
        config={
            "max_rpm": max_rpm,
            "max_tpm": max_tpm,
            "token_smoothing": token_smoothing,
            "optimal_concurrency": optimal_concurrency,
            "duration_seconds": duration_seconds,
            "prompt_size": prompt_size,
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        logger=logger
    )
    
    # Get prompts for this size
    prompts = SAMPLE_PROMPTS.get(prompt_size, SAMPLE_PROMPTS["small"])
    
    # Run benchmark
    logger.info(f"Starting quota-aware throughput benchmark for {model_id}")
    logger.info(f"Using quota limits: RPM={max_rpm}, TPM={max_tpm}, smoothing={token_smoothing}")
    logger.info(f"Using {prompt_size} prompts with max_tokens={max_tokens}, temperature={temperature}")
    logger.info(f"Benchmark will run for {duration_seconds} seconds with {optimal_concurrency} concurrent requests")
    
    # Create a thread pool for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_concurrency) as executor:
        # Calculate end time
        end_time = time.time() + duration_seconds
        
        # Function to process a single request
        def process_request():
            # Select a random prompt
            prompt = random.choice(prompts)
            
            # Start tracking request
            request_id, start_time = collector.start_request(input_text=prompt)
            
            try:
                # Make the request (with quota management)
                response = client.invoke(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    wait_for_quota=True,  # Wait for quota if needed
                    quota_timeout=30       # Maximum wait time for quota
                )
                
                # Record successful result
                backoff_time = getattr(response, "quota_wait_time", 0)
                
                collector.record_metric(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=time.time(),
                    input_tokens=response.get("input_tokens", 0),
                    output_tokens=response.get("output_tokens", 0),
                    success=True,
                    backoff_time=backoff_time,
                    was_throttled=backoff_time > 0,
                    input_text=prompt
                )
                
                return True
                
            except Exception as e:
                # Check for throttling
                was_throttled = "ThrottlingException" in str(e) or "QuotaExceededException" in str(e)
                
                # Record error
                logger.error(f"Error in request {request_id}: {str(e)}")
                collector.record_metric(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=time.time(),
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error=str(e),
                    was_throttled=was_throttled,
                    input_text=prompt
                )
                
                return False
        
        # Submit tasks until duration is reached
        futures = []
        
        while time.time() < end_time:
            # Submit new tasks to replace completed ones
            while len(futures) < optimal_concurrency and time.time() < end_time:
                futures.append(executor.submit(process_request))
            
            # Wait for at least one task to complete
            done, not_done = concurrent.futures.wait(
                futures, 
                timeout=0.1,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            # Remove completed futures
            futures = list(not_done)
            
            # Small delay to avoid CPU spinning
            time.sleep(0.01)
    
    # End benchmark and get results
    result = collector.end_benchmark()
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{result.benchmark_id}.json")
        result.save_to_file(output_path)
        logger.info(f"Results saved to {output_path}")
    
    return result


def compare_quota_strategies(
    model_id: str,
    max_rpm: Optional[int],
    max_tpm: Optional[int],
    duration_seconds: int = 60,
    output_dir: Optional[str] = None,
    visualize: bool = False
) -> Dict[str, BenchmarkResult]:
    """
    Compare different quota management strategies.
    
    Args:
        model_id: The Bedrock model ID
        max_rpm: Maximum requests per minute (model's quota)
        max_tpm: Maximum tokens per minute (model's quota)
        duration_seconds: Benchmark duration in seconds
        output_dir: Optional directory to save results
        visualize: Whether to generate visualization charts
        
    Returns:
        Dictionary mapping strategies to benchmark results
    """
    # Run benchmarks for each strategy
    results = {}
    
    # 1. Naive approach
    logger.info("Running naive throughput benchmark...")
    naive_result = run_naive_throughput_benchmark(
        model_id=model_id,
        concurrent_requests=5,  # Start with a moderate concurrency
        duration_seconds=duration_seconds,
        output_dir=output_dir
    )
    results["naive"] = naive_result
    
    # 2. Quota-aware without smoothing
    logger.info("Running quota-aware benchmark without token smoothing...")
    quota_no_smoothing_result = run_quota_aware_benchmark(
        model_id=model_id,
        max_rpm=max_rpm,
        max_tpm=max_tpm,
        duration_seconds=duration_seconds,
        token_smoothing=False,
        output_dir=output_dir
    )
    results["quota_no_smoothing"] = quota_no_smoothing_result
    
    # 3. Quota-aware with smoothing
    logger.info("Running quota-aware benchmark with token smoothing...")
    quota_smoothing_result = run_quota_aware_benchmark(
        model_id=model_id,
        max_rpm=max_rpm,
        max_tpm=max_tpm,
        duration_seconds=duration_seconds,
        token_smoothing=True,
        output_dir=output_dir
    )
    results["quota_smoothing"] = quota_smoothing_result
    
    # Generate comparison chart if requested
    if visualize:
        try:
            from benchmarks.utils.visualization import create_comparison_chart
            
            # Create output path for chart
            chart_path = os.path.join(output_dir, f"quota-comparison-{model_id.split('.')[-1]}.png")
            
            # Create chart
            create_comparison_chart(
                results=list(results.values()),
                metrics=["estimated_rpm", "estimated_tpm", "throttled_requests"],
                output_path=chart_path,
                title=f"Quota Management Strategy Comparison - {model_id}"
            )
            
            logger.info(f"Comparison chart saved to {chart_path}")
            
            # Generate throughput chart for each strategy
            for strategy, result in results.items():
                from benchmarks.utils.visualization import create_throughput_chart
                
                # Create output path for chart
                chart_path = os.path.join(output_dir, f"throughput-{strategy}-{model_id.split('.')[-1]}.png")
                
                # Create chart
                create_throughput_chart(
                    result=result,
                    output_path=chart_path,
                    window_size=5,  # 5-second window for smoother visualization
                    title=f"Throughput Over Time - {strategy.replace('_', ' ').title()} - {model_id}"
                )
                
                logger.info(f"Throughput chart for {strategy} saved to {chart_path}")
            
        except ImportError:
            logger.warning("Visualization libraries not available; skipping chart generation")
    
    return results


def main():
    """Main function for running the benchmark from command line."""
    parser = argparse.ArgumentParser(description="AWS Bedrock Quota Optimization Benchmark")
    
    parser.add_argument("--model", "-m", required=True,
                        help="Bedrock model ID (e.g., anthropic.claude-v2)")
    
    parser.add_argument("--rpm", type=int,
                        help="Requests per minute quota")
    
    parser.add_argument("--tpm", type=int,
                        help="Tokens per minute quota")
    
    parser.add_argument("--strategy", choices=["naive", "quota", "quota-smoothing", "all"],
                        default="all", help="Quota strategy to benchmark")
    
    parser.add_argument("--duration", type=int, default=60,
                        help="Benchmark duration in seconds")
    
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of concurrent requests (for naive strategy)")
    
    parser.add_argument("--prompt-size", choices=["small", "medium"],
                        default="small", help="Prompt size to use")
    
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Maximum tokens to generate")
    
    parser.add_argument("--output-dir", "-o", default="./benchmark_results",
                        help="Directory to save results")
    
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Generate visualization charts")
    
    args = parser.parse_args()
    
    # Run benchmark based on strategy
    if args.strategy == "all":
        results = compare_quota_strategies(
            model_id=args.model,
            max_rpm=args.rpm,
            max_tpm=args.tpm,
            duration_seconds=args.duration,
            output_dir=args.output_dir,
            visualize=args.visualize
        )
        
        # Print summary
        print("\nBenchmark Results Summary:")
        for strategy, result in results.items():
            summary = result.get_summary()
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Estimated RPM: {summary['estimated_rpm']:.1f}")
            print(f"  Estimated TPM: {summary['estimated_tpm']:.1f}")
            print(f"  Throttled Requests: {summary['throttled_requests']} ({summary['throttled_requests']/summary['total_requests']*100:.1f}%)")
            print(f"  Success Rate: {summary['success_rate'] * 100:.1f}%")
            
    elif args.strategy == "naive":
        result = run_naive_throughput_benchmark(
            model_id=args.model,
            concurrent_requests=args.concurrency,
            duration_seconds=args.duration,
            prompt_size=args.prompt_size,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir
        )
        
        # Generate visualization if requested
        if args.visualize:
            try:
                from benchmarks.utils.visualization import create_throughput_chart
                
                # Create output path for chart
                chart_path = os.path.join(args.output_dir, f"throughput-naive-{args.model.split('.')[-1]}.png")
                
                # Create chart
                create_throughput_chart(
                    result=result,
                    output_path=chart_path,
                    title=f"Naive Throughput Over Time - {args.model}"
                )
                
                logger.info(f"Throughput chart saved to {chart_path}")
                
            except ImportError:
                logger.warning("Visualization libraries not available; skipping chart generation")
                
    elif args.strategy == "quota":
        if not args.rpm and not args.tpm:
            parser.error("Either --rpm or --tpm must be specified for quota strategy")
            
        result = run_quota_aware_benchmark(
            model_id=args.model,
            max_rpm=args.rpm,
            max_tpm=args.tpm,
            duration_seconds=args.duration,
            prompt_size=args.prompt_size,
            max_tokens=args.max_tokens,
            token_smoothing=False,
            output_dir=args.output_dir
        )
        
        # Generate visualization if requested
        if args.visualize:
            try:
                from benchmarks.utils.visualization import create_throughput_chart
                
                # Create output path for chart
                chart_path = os.path.join(args.output_dir, f"throughput-quota-{args.model.split('.')[-1]}.png")
                
                # Create chart
                create_throughput_chart(
                    result=result,
                    output_path=chart_path,
                    title=f"Quota-Aware Throughput Over Time - {args.model}"
                )
                
                logger.info(f"Throughput chart saved to {chart_path}")
                
            except ImportError:
                logger.warning("Visualization libraries not available; skipping chart generation")
                
    elif args.strategy == "quota-smoothing":
        if not args.rpm and not args.tpm:
            parser.error("Either --rpm or --tpm must be specified for quota-smoothing strategy")
            
        result = run_quota_aware_benchmark(
            model_id=args.model,
            max_rpm=args.rpm,
            max_tpm=args.tpm,
            duration_seconds=args.duration,
            prompt_size=args.prompt_size,
            max_tokens=args.max_tokens,
            token_smoothing=True,
            output_dir=args.output_dir
        )
        
        # Generate visualization if requested
        if args.visualize:
            try:
                from benchmarks.utils.visualization import create_throughput_chart
                
                # Create output path for chart
                chart_path = os.path.join(args.output_dir, f"throughput-quota-smoothing-{args.model.split('.')[-1]}.png")
                
                # Create chart
                create_throughput_chart(
                    result=result,
                    output_path=chart_path,
                    title=f"Quota-Aware (Smoothing) Throughput Over Time - {args.model}"
                )
                
                logger.info(f"Throughput chart saved to {chart_path}")
                
            except ImportError:
                logger.warning("Visualization libraries not available; skipping chart generation")


if __name__ == "__main__":
    main()