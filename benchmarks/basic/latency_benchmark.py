"""
Latency benchmark for AWS Bedrock inference methods.

This script measures the latency characteristics of different AWS Bedrock
models and inference methods, including synchronous, streaming, and asynchronous.
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

from src.inference.synchronous.basic_client import BedrockClient
from src.inference.streaming.basic_client import BedrockStreamingClient
from src.inference.asynchronous.job_client import BedrockJobClient

from benchmarks.utils.metrics import MetricsCollector, BenchmarkResult


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bedrock-latency-benchmark")


# Sample prompts of varying lengths
SAMPLE_PROMPTS = {
    "short": [
        "What is AWS Bedrock?",
        "Explain cloud computing in one sentence.",
        "What is a foundation model?",
        "How does machine learning work?",
        "What is the difference between AI and ML?"
    ],
    "medium": [
        "Explain the key benefits of using AWS Bedrock for generative AI applications and how it differs from building your own models.",
        "Write a short paragraph explaining how transformer models work and why they've become so popular for NLP tasks.",
        "Describe the main architectural components of a cloud-native application and how they interact with each other.",
        "Compare and contrast synchronous and asynchronous APIs in the context of large language model inference.",
        "Explain the concept of token economy in large language models and how it affects performance and cost."
    ],
    "long": [
        "Write a detailed technical explanation of how AWS Bedrock handles quota management for different foundation models. Include information about Tokens Per Minute (TPM) and Requests Per Minute (RPM) limits, and strategies for optimizing throughput within these constraints.",
        
        "Provide a comprehensive overview of the evolution of large language models from early neural networks to modern transformer-based models like GPT, Claude, and Llama. Discuss key architectural innovations, training methodologies, and performance improvements over time.",
        
        "Explain the challenges and best practices for deploying foundation models in production environments. Address concerns like latency, reliability, cost optimization, and scaling strategies. Include specific examples of architectural patterns that work well for high-throughput scenarios.",
        
        "Write a detailed comparison of different inference methods available in AWS Bedrock, including synchronous requests, streaming, and asynchronous batch processing. Explain the trade-offs between these methods and when each one should be preferred based on specific use cases and requirements.",
        
        "Describe the process of fine-tuning foundation models for specific domains or tasks. Explain concepts like continued pre-training, instruction tuning, and RLHF (Reinforcement Learning from Human Feedback). Discuss how these techniques can improve model performance for specialized applications."
    ]
}


def run_synchronous_benchmark(
    model_id: str,
    num_requests: int,
    prompt_length: str = "medium",
    max_tokens: int = 500,
    temperature: float = 0.7,
    output_dir: Optional[str] = None
) -> BenchmarkResult:
    """
    Run a synchronous inference latency benchmark.
    
    Args:
        model_id: The Bedrock model ID
        num_requests: Number of requests to make
        prompt_length: Prompt length category ('short', 'medium', 'long')
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
        benchmark_id=f"sync-{model_id.split('.')[-1]}-{prompt_length}",
        model_id=model_id,
        inference_type="sync",
        description=f"Synchronous inference latency benchmark with {prompt_length} prompts",
        config={
            "num_requests": num_requests,
            "prompt_length": prompt_length,
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        logger=logger
    )
    
    # Get prompts for this length
    prompts = SAMPLE_PROMPTS.get(prompt_length, SAMPLE_PROMPTS["medium"])
    
    # Run benchmark
    logger.info(f"Starting synchronous benchmark for {model_id} with {num_requests} requests")
    logger.info(f"Using {prompt_length} prompts with max_tokens={max_tokens}, temperature={temperature}")
    
    for i in range(num_requests):
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
            
        except Exception as e:
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
                input_text=prompt
            )
        
        # Add a small delay between requests to avoid throttling
        time.sleep(0.5)
    
    # End benchmark and get results
    result = collector.end_benchmark()
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{result.benchmark_id}.json")
        result.save_to_file(output_path)
        logger.info(f"Results saved to {output_path}")
    
    return result


def run_streaming_benchmark(
    model_id: str,
    num_requests: int,
    prompt_length: str = "medium",
    max_tokens: int = 500,
    temperature: float = 0.7,
    output_dir: Optional[str] = None
) -> BenchmarkResult:
    """
    Run a streaming inference latency benchmark.
    
    Args:
        model_id: The Bedrock model ID
        num_requests: Number of requests to make
        prompt_length: Prompt length category ('short', 'medium', 'long')
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation
        output_dir: Optional directory to save results
        
    Returns:
        BenchmarkResult object with benchmark results
    """
    # Create client
    client = BedrockStreamingClient(model_id=model_id)
    
    # Create metrics collector
    collector = MetricsCollector(
        benchmark_id=f"stream-{model_id.split('.')[-1]}-{prompt_length}",
        model_id=model_id,
        inference_type="stream",
        description=f"Streaming inference latency benchmark with {prompt_length} prompts",
        config={
            "num_requests": num_requests,
            "prompt_length": prompt_length,
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        logger=logger
    )
    
    # Get prompts for this length
    prompts = SAMPLE_PROMPTS.get(prompt_length, SAMPLE_PROMPTS["medium"])
    
    # Run benchmark
    logger.info(f"Starting streaming benchmark for {model_id} with {num_requests} requests")
    logger.info(f"Using {prompt_length} prompts with max_tokens={max_tokens}, temperature={temperature}")
    
    for i in range(num_requests):
        # Select a random prompt
        prompt = random.choice(prompts)
        
        # Start tracking request
        request_id, start_time = collector.start_request(input_text=prompt)
        
        try:
            # Track first token time
            first_token_received = False
            first_token_time = None
            
            # Make the streaming request and collect chunks
            full_response = ""
            token_count = 0
            
            for chunk in client.invoke_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                # Handle first token timing
                if not first_token_received:
                    first_token_time = time.time() - start_time
                    first_token_received = True
                
                # Accumulate response
                full_response += chunk
                token_count += 1
            
            # Estimate input tokens (not directly available from streaming)
            # Rough estimate based on typical token sizes
            input_tokens = len(prompt.split()) * 4 // 3  # ~4/3 tokens per word
            
            # Record successful result
            collector.record_metric(
                request_id=request_id,
                start_time=start_time,
                end_time=time.time(),
                input_tokens=input_tokens,
                output_tokens=token_count,
                success=True,
                first_token_time=first_token_time,
                input_text=prompt
            )
            
        except Exception as e:
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
                input_text=prompt
            )
        
        # Add a small delay between requests to avoid throttling
        time.sleep(0.5)
    
    # End benchmark and get results
    result = collector.end_benchmark()
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{result.benchmark_id}.json")
        result.save_to_file(output_path)
        logger.info(f"Results saved to {output_path}")
    
    return result


def run_async_benchmark(
    model_id: str,
    num_requests: int,
    prompt_length: str = "medium",
    max_tokens: int = 500,
    temperature: float = 0.7,
    output_s3_uri: Optional[str] = None,
    output_dir: Optional[str] = None
) -> BenchmarkResult:
    """
    Run an asynchronous inference latency benchmark.
    
    Args:
        model_id: The Bedrock model ID
        num_requests: Number of requests to make
        prompt_length: Prompt length category ('short', 'medium', 'long')
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation
        output_s3_uri: S3 URI for async job outputs
        output_dir: Optional directory to save results
        
    Returns:
        BenchmarkResult object with benchmark results
    """
    # Create client
    client = BedrockJobClient(
        model_id=model_id,
        output_s3_uri=output_s3_uri
    )
    
    # Create metrics collector
    collector = MetricsCollector(
        benchmark_id=f"async-{model_id.split('.')[-1]}-{prompt_length}",
        model_id=model_id,
        inference_type="async",
        description=f"Asynchronous inference latency benchmark with {prompt_length} prompts",
        config={
            "num_requests": num_requests,
            "prompt_length": prompt_length,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "output_s3_uri": output_s3_uri
        },
        logger=logger
    )
    
    # Get prompts for this length
    prompts = SAMPLE_PROMPTS.get(prompt_length, SAMPLE_PROMPTS["medium"])
    
    # Run benchmark
    logger.info(f"Starting asynchronous benchmark for {model_id} with {num_requests} requests")
    logger.info(f"Using {prompt_length} prompts with max_tokens={max_tokens}, temperature={temperature}")
    
    # Track job IDs and start times
    jobs = []
    
    # Submit all jobs first
    for i in range(num_requests):
        # Select a random prompt
        prompt = random.choice(prompts)
        
        # Start tracking request
        request_id, start_time = collector.start_request(input_text=prompt)
        
        try:
            # Create the job
            job_id = client.create_job(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                job_name=f"latency-benchmark-job-{i}"
            )
            
            # Track job info
            jobs.append({
                "job_id": job_id,
                "request_id": request_id,
                "start_time": start_time,
                "prompt": prompt
            })
            
            logger.info(f"Submitted job {i+1}/{num_requests} with ID: {job_id}")
            
        except Exception as e:
            # Record submission error
            logger.error(f"Error submitting job {request_id}: {str(e)}")
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
        
        # Add a small delay between submissions to avoid throttling
        time.sleep(0.5)
    
    # Wait for all jobs to complete
    logger.info(f"Waiting for {len(jobs)} jobs to complete...")
    
    for job in jobs:
        try:
            # Wait for job completion
            job_result = client.wait_for_job(job["job_id"])
            end_time = time.time()
            
            # Extract job details
            output_tokens = 0
            
            # If job completed successfully, estimate token counts
            if job_result["status"] == "COMPLETED":
                # Rough estimation of token counts
                input_tokens = len(job["prompt"].split()) * 4 // 3  # ~4/3 tokens per word
                
                # If token counts available in output, use them
                if "output_tokens" in job_result:
                    output_tokens = job_result["output_tokens"]
                else:
                    # Otherwise make a rough estimate based on output length
                    if "output" in job_result and job_result["output"]:
                        output_tokens = len(job_result["output"].split()) * 4 // 3  # ~4/3 tokens per word
                
                # Record successful result
                collector.record_metric(
                    request_id=job["request_id"],
                    start_time=job["start_time"],
                    end_time=end_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                    input_text=job["prompt"]
                )
            else:
                # Record failure
                collector.record_metric(
                    request_id=job["request_id"],
                    start_time=job["start_time"],
                    end_time=end_time,
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error=f"Job failed with status: {job_result['status']}",
                    input_text=job["prompt"]
                )
            
        except Exception as e:
            # Record job waiting error
            logger.error(f"Error waiting for job {job['job_id']}: {str(e)}")
            collector.record_metric(
                request_id=job["request_id"],
                start_time=job["start_time"],
                end_time=time.time(),
                input_tokens=0,
                output_tokens=0,
                success=False,
                error=f"Job waiting error: {str(e)}",
                input_text=job["prompt"]
            )
    
    # End benchmark and get results
    result = collector.end_benchmark()
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{result.benchmark_id}.json")
        result.save_to_file(output_path)
        logger.info(f"Results saved to {output_path}")
    
    return result


def compare_inference_methods(
    model_id: str,
    num_requests: int = 10,
    prompt_length: str = "medium",
    output_dir: Optional[str] = None,
    visualize: bool = False
) -> Dict[str, BenchmarkResult]:
    """
    Compare latency across different inference methods.
    
    Args:
        model_id: The Bedrock model ID
        num_requests: Number of requests per method
        prompt_length: Prompt length to use
        output_dir: Optional directory to save results
        visualize: Whether to generate visualization charts
        
    Returns:
        Dictionary mapping methods to benchmark results
    """
    # Run benchmarks for each method
    results = {}
    
    # Synchronous inference
    logger.info("Running synchronous inference benchmark...")
    sync_result = run_synchronous_benchmark(
        model_id=model_id,
        num_requests=num_requests,
        prompt_length=prompt_length,
        output_dir=output_dir
    )
    results["sync"] = sync_result
    
    # Streaming inference
    logger.info("Running streaming inference benchmark...")
    stream_result = run_streaming_benchmark(
        model_id=model_id,
        num_requests=num_requests,
        prompt_length=prompt_length,
        output_dir=output_dir
    )
    results["stream"] = stream_result
    
    # Asynchronous inference (if output S3 URI is provided)
    if "S3_OUTPUT_URI" in os.environ:
        output_s3_uri = os.environ["S3_OUTPUT_URI"]
        logger.info("Running asynchronous inference benchmark...")
        async_result = run_async_benchmark(
            model_id=model_id,
            num_requests=num_requests,
            prompt_length=prompt_length,
            output_s3_uri=output_s3_uri,
            output_dir=output_dir
        )
        results["async"] = async_result
    else:
        logger.warning("Skipping asynchronous benchmark; S3_OUTPUT_URI not set")
    
    # Generate comparison chart if requested
    if visualize:
        try:
            from benchmarks.utils.visualization import create_comparison_chart
            
            # Create output path for chart
            chart_path = os.path.join(output_dir, f"comparison-{model_id.split('.')[-1]}-{prompt_length}.png")
            
            # Create chart
            create_comparison_chart(
                results=list(results.values()),
                metrics=["avg_request_time", "estimated_rpm", "avg_tokens_per_second"],
                output_path=chart_path,
                title=f"Inference Method Comparison - {model_id}"
            )
            
            logger.info(f"Comparison chart saved to {chart_path}")
            
        except ImportError:
            logger.warning("Visualization libraries not available; skipping chart generation")
    
    return results


def main():
    """Main function for running the benchmark from command line."""
    parser = argparse.ArgumentParser(description="AWS Bedrock Inference Latency Benchmark")
    
    parser.add_argument("--model", "-m", required=True,
                        help="Bedrock model ID (e.g., anthropic.claude-v2)")
    
    parser.add_argument("--method", choices=["sync", "stream", "async", "all"],
                        default="all", help="Inference method to benchmark")
    
    parser.add_argument("--requests", "-n", type=int, default=10,
                        help="Number of requests to make")
    
    parser.add_argument("--prompt-length", choices=["short", "medium", "long"],
                        default="medium", help="Prompt length to use")
    
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Maximum tokens to generate")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    
    parser.add_argument("--s3-uri", help="S3 URI for async job outputs")
    
    parser.add_argument("--output-dir", "-o", default="./benchmark_results",
                        help="Directory to save results")
    
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Generate visualization charts")
    
    args = parser.parse_args()
    
    # Set S3 output URI environment variable if provided
    if args.s3_uri:
        os.environ["S3_OUTPUT_URI"] = args.s3_uri
    
    # Run benchmark based on method
    if args.method == "all":
        results = compare_inference_methods(
            model_id=args.model,
            num_requests=args.requests,
            prompt_length=args.prompt_length,
            output_dir=args.output_dir,
            visualize=args.visualize
        )
        
        # Print summary
        print("\nBenchmark Results Summary:")
        for method, result in results.items():
            summary = result.get_summary()
            print(f"\n{method.upper()} Method:")
            print(f"  Average Request Time: {summary['avg_request_time']:.3f}s")
            print(f"  Estimated RPM: {summary['estimated_rpm']:.1f}")
            print(f"  Average Tokens/Sec: {summary['avg_tokens_per_second']:.1f}")
            print(f"  Success Rate: {summary['success_rate'] * 100:.1f}%")
            
    elif args.method == "sync":
        result = run_synchronous_benchmark(
            model_id=args.model,
            num_requests=args.requests,
            prompt_length=args.prompt_length,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            output_dir=args.output_dir
        )
        
        # Generate visualization if requested
        if args.visualize:
            try:
                from benchmarks.utils.visualization import create_latency_distribution
                
                # Create output path for chart
                chart_path = os.path.join(args.output_dir, f"latency-{args.model.split('.')[-1]}-{args.prompt_length}.png")
                
                # Create chart
                create_latency_distribution(
                    result=result,
                    output_path=chart_path,
                    title=f"Synchronous Inference Latency - {args.model}"
                )
                
                logger.info(f"Latency chart saved to {chart_path}")
                
            except ImportError:
                logger.warning("Visualization libraries not available; skipping chart generation")
                
    elif args.method == "stream":
        result = run_streaming_benchmark(
            model_id=args.model,
            num_requests=args.requests,
            prompt_length=args.prompt_length,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            output_dir=args.output_dir
        )
        
        # Generate visualization if requested
        if args.visualize:
            try:
                from benchmarks.utils.visualization import create_latency_distribution
                
                # Create output path for chart
                chart_path = os.path.join(args.output_dir, f"latency-stream-{args.model.split('.')[-1]}-{args.prompt_length}.png")
                
                # Create chart
                create_latency_distribution(
                    result=result,
                    output_path=chart_path,
                    title=f"Streaming Inference Latency - {args.model}"
                )
                
                logger.info(f"Latency chart saved to {chart_path}")
                
            except ImportError:
                logger.warning("Visualization libraries not available; skipping chart generation")
                
    elif args.method == "async":
        if not args.s3_uri:
            parser.error("--s3-uri is required for async benchmarks")
            
        result = run_async_benchmark(
            model_id=args.model,
            num_requests=args.requests,
            prompt_length=args.prompt_length,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            output_s3_uri=args.s3_uri,
            output_dir=args.output_dir
        )
        
        # Generate visualization if requested
        if args.visualize:
            try:
                from benchmarks.utils.visualization import create_latency_distribution
                
                # Create output path for chart
                chart_path = os.path.join(args.output_dir, f"latency-async-{args.model.split('.')[-1]}-{args.prompt_length}.png")
                
                # Create chart
                create_latency_distribution(
                    result=result,
                    output_path=chart_path,
                    title=f"Asynchronous Inference Latency - {args.model}"
                )
                
                logger.info(f"Latency chart saved to {chart_path}")
                
            except ImportError:
                logger.warning("Visualization libraries not available; skipping chart generation")


if __name__ == "__main__":
    main()