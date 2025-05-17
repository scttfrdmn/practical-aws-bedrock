"""
Batch optimization benchmark for AWS Bedrock.

This script benchmarks batch processing strategies for optimizing throughput
with asynchronous jobs in AWS Bedrock.
"""

import os
import time
import json
import logging
import argparse
import random
from typing import Dict, Any, List, Optional, Union, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.asynchronous.job_client import BedrockJobClient
from src.inference.asynchronous.batch_processor import BedrockBatchProcessor

from benchmarks.utils.metrics import MetricsCollector, BenchmarkResult


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bedrock-batch-benchmark")


# Sample prompts for batch processing
SAMPLE_PROMPTS = [
    "Summarize the key features of AWS Bedrock.",
    "Explain the difference between synchronous and asynchronous APIs.",
    "Describe the main benefits of foundation models for developers.",
    "List three strategies for optimizing inference throughput.",
    "Explain how token quotas work in AWS Bedrock.",
    "Compare Claude and Titan model families in AWS Bedrock.",
    "Describe a use case for streaming inference APIs.",
    "Explain the concept of prompt engineering.",
    "List the key components of a generative AI application.",
    "Describe the benefits of using managed foundation models.",
    "Explain how to handle errors in AWS Bedrock APIs.",
    "Compare text generation and text embedding use cases.",
    "Explain the concept of token economy in large language models.",
    "Describe how to optimize API requests for cost efficiency.",
    "Explain the benefits of asynchronous processing for batch workloads.",
    "List considerations for production deployment of foundation models.",
    "Explain how to monitor and log foundation model API usage.",
    "Describe strategies for managing API rate limits.",
    "Explain the concept of request batching for throughput optimization.",
    "Describe architectural patterns for high-throughput AI applications."
]


def run_sequential_batch_benchmark(
    model_id: str,
    batch_size: int,
    num_batches: int,
    max_tokens: int = 150,
    temperature: float = 0.7,
    output_s3_uri: str = None,
    output_dir: Optional[str] = None
) -> BenchmarkResult:
    """
    Run a sequential batch processing benchmark.
    
    This benchmark processes batches sequentially without parallelism.
    
    Args:
        model_id: The Bedrock model ID
        batch_size: Number of prompts per batch
        num_batches: Number of batches to process
        max_tokens: Maximum tokens to generate per prompt
        temperature: Temperature for generation
        output_s3_uri: S3 URI for job outputs
        output_dir: Optional directory to save benchmark results
        
    Returns:
        BenchmarkResult object with benchmark results
    """
    # Check if output S3 URI is provided
    if not output_s3_uri:
        raise ValueError("output_s3_uri is required for batch processing benchmark")
    
    # Create job client for creating and tracking individual jobs
    job_client = BedrockJobClient(
        model_id=model_id,
        output_s3_uri=output_s3_uri
    )
    
    # Create metrics collector
    collector = MetricsCollector(
        benchmark_id=f"batch-seq-{model_id.split('.')[-1]}-b{batch_size}",
        model_id=model_id,
        inference_type="async-sequential",
        description=f"Sequential batch processing benchmark with batch_size={batch_size}",
        config={
            "batch_size": batch_size,
            "num_batches": num_batches,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "output_s3_uri": output_s3_uri
        },
        logger=logger
    )
    
    # Run benchmark
    logger.info(f"Starting sequential batch benchmark for {model_id}")
    logger.info(f"Processing {num_batches} batches of {batch_size} prompts each")
    
    # Process batches sequentially
    for batch_num in range(num_batches):
        logger.info(f"Processing batch {batch_num + 1}/{num_batches}")
        
        # Generate batch of prompts
        batch_prompts = random.sample(SAMPLE_PROMPTS, min(batch_size, len(SAMPLE_PROMPTS)))
        if len(batch_prompts) < batch_size:
            # If we need more prompts than available in the sample, repeat some
            additional_prompts = random.choices(
                SAMPLE_PROMPTS, 
                k=batch_size - len(batch_prompts)
            )
            batch_prompts.extend(additional_prompts)
        
        # Process each prompt in the batch
        batch_start_time = time.time()
        job_ids = []
        request_ids = []
        start_times = []
        
        # Submit all jobs in this batch
        for i, prompt in enumerate(batch_prompts):
            # Start tracking request
            request_id, start_time = collector.start_request(input_text=prompt)
            
            try:
                # Create the job
                job_id = job_client.create_job(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    job_name=f"batch-seq-{batch_num}-{i}"
                )
                
                # Track job info
                job_ids.append(job_id)
                request_ids.append(request_id)
                start_times.append(start_time)
                
                logger.debug(f"Submitted job {i+1}/{batch_size} in batch {batch_num+1} with ID: {job_id}")
                
            except Exception as e:
                # Record submission error
                logger.error(f"Error submitting job for prompt {i+1} in batch {batch_num+1}: {str(e)}")
                collector.record_metric(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=time.time(),
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error=f"Job submission error: {str(e)}",
                    input_text=prompt
                )
        
        # Wait for all jobs in this batch to complete
        for i, job_id in enumerate(job_ids):
            try:
                # Wait for job completion
                job_result = job_client.wait_for_job(job_id)
                end_time = time.time()
                
                # Extract job details
                request_id = request_ids[i]
                start_time = start_times[i]
                prompt = batch_prompts[i] if i < len(batch_prompts) else "Unknown prompt"
                
                # If job completed successfully, estimate token counts
                if job_result["status"] == "COMPLETED":
                    # Rough estimation of token counts
                    input_tokens = len(prompt.split()) * 4 // 3  # ~4/3 tokens per word
                    
                    # If token counts available in output, use them
                    if "output_tokens" in job_result:
                        output_tokens = job_result["output_tokens"]
                    else:
                        # Otherwise make a rough estimate based on output length
                        if "output" in job_result and job_result["output"]:
                            output_tokens = len(job_result["output"].split()) * 4 // 3  # ~4/3 tokens per word
                        else:
                            output_tokens = 0
                    
                    # Record successful result
                    collector.record_metric(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        success=True,
                        input_text=prompt
                    )
                else:
                    # Record failure
                    collector.record_metric(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        input_tokens=0,
                        output_tokens=0,
                        success=False,
                        error=f"Job failed with status: {job_result['status']}",
                        input_text=prompt
                    )
                
            except Exception as e:
                # Record job waiting error
                request_id = request_ids[i] if i < len(request_ids) else f"unknown-{i}"
                start_time = start_times[i] if i < len(start_times) else time.time() - 60
                prompt = batch_prompts[i] if i < len(batch_prompts) else "Unknown prompt"
                
                logger.error(f"Error waiting for job {job_id}: {str(e)}")
                collector.record_metric(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=time.time(),
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error=f"Job waiting error: {str(e)}",
                    input_text=prompt
                )
        
        # Log batch completion
        batch_duration = time.time() - batch_start_time
        logger.info(f"Completed batch {batch_num + 1}/{num_batches} in {batch_duration:.2f}s")
    
    # End benchmark and get results
    result = collector.end_benchmark()
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{result.benchmark_id}.json")
        result.save_to_file(output_path)
        logger.info(f"Results saved to {output_path}")
    
    return result


def run_parallel_batch_benchmark(
    model_id: str,
    batch_size: int,
    max_concurrent_jobs: int,
    num_batches: int,
    max_tokens: int = 150,
    temperature: float = 0.7,
    output_s3_uri: str = None,
    output_dir: Optional[str] = None
) -> BenchmarkResult:
    """
    Run a parallel batch processing benchmark using BatchProcessor.
    
    This benchmark processes batches using parallel job execution.
    
    Args:
        model_id: The Bedrock model ID
        batch_size: Number of prompts per batch
        max_concurrent_jobs: Maximum number of concurrent jobs
        num_batches: Number of batches to process
        max_tokens: Maximum tokens to generate per prompt
        temperature: Temperature for generation
        output_s3_uri: S3 URI for job outputs
        output_dir: Optional directory to save benchmark results
        
    Returns:
        BenchmarkResult object with benchmark results
    """
    # Check if output S3 URI is provided
    if not output_s3_uri:
        raise ValueError("output_s3_uri is required for batch processing benchmark")
    
    # Create batch processor
    batch_processor = BedrockBatchProcessor(
        model_id=model_id,
        output_s3_uri=output_s3_uri,
        max_concurrent_jobs=max_concurrent_jobs
    )
    
    # Create metrics collector
    collector = MetricsCollector(
        benchmark_id=f"batch-par-{model_id.split('.')[-1]}-b{batch_size}-c{max_concurrent_jobs}",
        model_id=model_id,
        inference_type="async-parallel",
        description=f"Parallel batch processing benchmark with batch_size={batch_size}, max_concurrent_jobs={max_concurrent_jobs}",
        config={
            "batch_size": batch_size,
            "max_concurrent_jobs": max_concurrent_jobs,
            "num_batches": num_batches,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "output_s3_uri": output_s3_uri
        },
        logger=logger
    )
    
    # Run benchmark
    logger.info(f"Starting parallel batch benchmark for {model_id}")
    logger.info(f"Processing {num_batches} batches of {batch_size} prompts each")
    logger.info(f"Using max_concurrent_jobs={max_concurrent_jobs}")
    
    # Process batches
    for batch_num in range(num_batches):
        logger.info(f"Processing batch {batch_num + 1}/{num_batches}")
        
        # Generate batch of prompts
        batch_prompts = random.sample(SAMPLE_PROMPTS, min(batch_size, len(SAMPLE_PROMPTS)))
        if len(batch_prompts) < batch_size:
            # If we need more prompts than available in the sample, repeat some
            additional_prompts = random.choices(
                SAMPLE_PROMPTS, 
                k=batch_size - len(batch_prompts)
            )
            batch_prompts.extend(additional_prompts)
        
        # Start tracking this batch
        batch_start_time = time.time()
        request_ids = []
        start_times = []
        
        # Create a request ID and start time for each prompt
        for i, prompt in enumerate(batch_prompts):
            request_id, start_time = collector.start_request(input_text=prompt)
            request_ids.append(request_id)
            start_times.append(start_time)
        
        # Define a progress callback to track individual job completions
        def progress_callback(index, result, error=None):
            # Get the corresponding request info
            if 0 <= index < len(request_ids):
                request_id = request_ids[index]
                start_time = start_times[index]
                prompt = batch_prompts[index]
                
                if error:
                    # Record error
                    logger.error(f"Error in job {index}: {str(error)}")
                    collector.record_metric(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=time.time(),
                        input_tokens=0,
                        output_tokens=0,
                        success=False,
                        error=str(error),
                        input_text=prompt
                    )
                else:
                    # Extract token counts (estimated)
                    input_tokens = len(prompt.split()) * 4 // 3  # ~4/3 tokens per word
                    
                    # If token counts available in output, use them
                    if "token_count" in result:
                        output_tokens = result["token_count"]
                    elif "output" in result:
                        # Estimate based on output length
                        output_tokens = len(result["output"].split()) * 4 // 3  # ~4/3 tokens per word
                    else:
                        output_tokens = 0
                    
                    # Record success
                    collector.record_metric(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=time.time(),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        success=True,
                        input_text=prompt
                    )
        
        try:
            # Process the batch with the BatchProcessor
            batch_results = batch_processor.process_batch(
                inputs=batch_prompts,
                job_name_prefix=f"batch-par-{batch_num}",
                max_tokens=max_tokens,
                temperature=temperature,
                progress_callback=progress_callback
            )
            
            # Check for any prompts that didn't get processed by the callback
            for i, prompt in enumerate(batch_prompts):
                # Skip if already recorded by callback
                request_id = request_ids[i]
                if any(m.request_id == request_id for m in collector.result.metrics):
                    continue
                
                # Record as failure (something went wrong with tracking)
                collector.record_metric(
                    request_id=request_id,
                    start_time=start_times[i],
                    end_time=time.time(),
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error="Job tracking error: No result recorded",
                    input_text=prompt
                )
            
        except Exception as e:
            # Record batch processing error for all prompts
            logger.error(f"Error processing batch {batch_num + 1}: {str(e)}")
            
            for i, prompt in enumerate(batch_prompts):
                # Skip if already recorded by callback
                request_id = request_ids[i]
                if any(m.request_id == request_id for m in collector.result.metrics):
                    continue
                
                collector.record_metric(
                    request_id=request_id,
                    start_time=start_times[i],
                    end_time=time.time(),
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error=f"Batch processing error: {str(e)}",
                    input_text=prompt
                )
        
        # Log batch completion
        batch_duration = time.time() - batch_start_time
        logger.info(f"Completed batch {batch_num + 1}/{num_batches} in {batch_duration:.2f}s")
    
    # End benchmark and get results
    result = collector.end_benchmark()
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{result.benchmark_id}.json")
        result.save_to_file(output_path)
        logger.info(f"Results saved to {output_path}")
    
    return result


def compare_batch_strategies(
    model_id: str,
    batch_size: int = 10,
    concurrency_levels: List[int] = [1, 5, 10, 20],
    num_batches: int = 3,
    max_tokens: int = 150,
    output_s3_uri: str = None,
    output_dir: Optional[str] = None,
    visualize: bool = False
) -> Dict[str, BenchmarkResult]:
    """
    Compare different batch processing strategies.
    
    Args:
        model_id: The Bedrock model ID
        batch_size: Number of prompts per batch
        concurrency_levels: List of concurrency levels to test
        num_batches: Number of batches to process
        max_tokens: Maximum tokens to generate per prompt
        output_s3_uri: S3 URI for job outputs
        output_dir: Optional directory to save results
        visualize: Whether to generate visualization charts
        
    Returns:
        Dictionary mapping strategies to benchmark results
    """
    # Check if output S3 URI is provided
    if not output_s3_uri:
        raise ValueError("output_s3_uri is required for batch processing benchmarks")
    
    # Run benchmarks for each strategy
    results = {}
    
    # 1. Sequential batch processing
    logger.info("Running sequential batch processing benchmark...")
    sequential_result = run_sequential_batch_benchmark(
        model_id=model_id,
        batch_size=batch_size,
        num_batches=num_batches,
        max_tokens=max_tokens,
        output_s3_uri=output_s3_uri,
        output_dir=output_dir
    )
    results["sequential"] = sequential_result
    
    # 2. Parallel batch processing with different concurrency levels
    for concurrency in concurrency_levels:
        logger.info(f"Running parallel batch processing benchmark with concurrency={concurrency}...")
        parallel_result = run_parallel_batch_benchmark(
            model_id=model_id,
            batch_size=batch_size,
            max_concurrent_jobs=concurrency,
            num_batches=num_batches,
            max_tokens=max_tokens,
            output_s3_uri=output_s3_uri,
            output_dir=output_dir
        )
        results[f"parallel_c{concurrency}"] = parallel_result
    
    # Generate comparison chart if requested
    if visualize:
        try:
            from benchmarks.utils.visualization import create_comparison_chart
            
            # Create output path for chart
            chart_path = os.path.join(output_dir, f"batch-comparison-{model_id.split('.')[-1]}.png")
            
            # Create chart
            create_comparison_chart(
                results=list(results.values()),
                metrics=["avg_request_time", "requests_per_second", "success_rate"],
                output_path=chart_path,
                title=f"Batch Processing Strategy Comparison - {model_id}"
            )
            
            logger.info(f"Comparison chart saved to {chart_path}")
            
        except ImportError:
            logger.warning("Visualization libraries not available; skipping chart generation")
    
    return results


def main():
    """Main function for running the benchmark from command line."""
    parser = argparse.ArgumentParser(description="AWS Bedrock Batch Processing Benchmark")
    
    parser.add_argument("--model", "-m", required=True,
                        help="Bedrock model ID (e.g., anthropic.claude-v2)")
    
    parser.add_argument("--strategy", choices=["sequential", "parallel", "compare"],
                        default="compare", help="Batch strategy to benchmark")
    
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="Number of prompts per batch")
    
    parser.add_argument("--concurrency", "-c", type=int, default=5,
                        help="Maximum number of concurrent jobs (for parallel strategy)")
    
    parser.add_argument("--num-batches", "-n", type=int, default=3,
                        help="Number of batches to process")
    
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Maximum tokens to generate per prompt")
    
    parser.add_argument("--s3-uri", required=True,
                        help="S3 URI for job outputs")
    
    parser.add_argument("--output-dir", "-o", default="./benchmark_results",
                        help="Directory to save results")
    
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Generate visualization charts")
    
    args = parser.parse_args()
    
    # Run benchmark based on strategy
    if args.strategy == "compare":
        results = compare_batch_strategies(
            model_id=args.model,
            batch_size=args.batch_size,
            concurrency_levels=[1, args.concurrency, args.concurrency * 2],
            num_batches=args.num_batches,
            max_tokens=args.max_tokens,
            output_s3_uri=args.s3_uri,
            output_dir=args.output_dir,
            visualize=args.visualize
        )
        
        # Print summary
        print("\nBenchmark Results Summary:")
        for strategy, result in results.items():
            summary = result.get_summary()
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Average Request Time: {summary['avg_request_time']:.3f}s")
            print(f"  Requests Per Second: {summary['requests_per_second']:.2f}")
            print(f"  Success Rate: {summary['success_rate'] * 100:.1f}%")
            
    elif args.strategy == "sequential":
        result = run_sequential_batch_benchmark(
            model_id=args.model,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            max_tokens=args.max_tokens,
            output_s3_uri=args.s3_uri,
            output_dir=args.output_dir
        )
        
        # Print summary
        summary = result.get_summary()
        print("\nSequential Batch Benchmark Results:")
        print(f"  Average Request Time: {summary['avg_request_time']:.3f}s")
        print(f"  Requests Per Second: {summary['requests_per_second']:.2f}")
        print(f"  Success Rate: {summary['success_rate'] * 100:.1f}%")
        
    elif args.strategy == "parallel":
        result = run_parallel_batch_benchmark(
            model_id=args.model,
            batch_size=args.batch_size,
            max_concurrent_jobs=args.concurrency,
            num_batches=args.num_batches,
            max_tokens=args.max_tokens,
            output_s3_uri=args.s3_uri,
            output_dir=args.output_dir
        )
        
        # Print summary
        summary = result.get_summary()
        print("\nParallel Batch Benchmark Results:")
        print(f"  Average Request Time: {summary['avg_request_time']:.3f}s")
        print(f"  Requests Per Second: {summary['requests_per_second']:.2f}")
        print(f"  Success Rate: {summary['success_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()