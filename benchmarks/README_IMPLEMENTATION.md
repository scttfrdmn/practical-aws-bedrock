# AWS Bedrock Benchmarking Implementation Guide

This document provides implementation guidelines for creating benchmarking scripts to test AWS Bedrock inference methods.

## Benchmark Objectives

The benchmarking suite should measure:

1. **Performance metrics**
   - Latency (time to first token, total time)
   - Throughput (tokens per second, requests per minute)
   - Success/error rates under different loads

2. **Quota utilization**
   - TPM (tokens per minute) efficiency
   - RPM (requests per minute) maximization
   - Throttling behavior analysis

3. **Optimization techniques**
   - Impact of prompt structure on throughput
   - Effectiveness of batching strategies
   - Parallelization efficiency

## Required Benchmarks

### Basic Performance Benchmarks

1. **Model Response Time**
   - Measure response time across different models
   - Test with varying input lengths
   - Measure time to first token for streaming

2. **Throughput Measurements**
   - Maximum sustainable RPM without throttling
   - Maximum token throughput (TPM)
   - Impact of different payload sizes

3. **Quota Limit Testing**
   - Controlled throttling tests
   - Recovery time measurements
   - Quota consumption patterns

### Comparative Benchmarks

1. **Inference Method Comparison**
   - Synchronous vs Streaming vs Asynchronous
   - Performance under different workloads
   - Resource utilization differences

2. **Model Family Comparison**
   - Performance across different model families
   - Token efficiency comparisons
   - Cost-performance tradeoffs

3. **Optimization Strategy Comparison**
   - Prompt optimization effectiveness
   - Batching vs single requests
   - Parallel vs sequential processing

## Implementation Structure

```
benchmarks/
├── basic/                      # Basic performance benchmarks
│   ├── response_time.py        # Response time measurements
│   ├── throughput.py           # Throughput measurements
│   └── quota_limits.py         # Quota limit testing
│
├── comparative/                # Comparative benchmarks
│   ├── inference_methods.py    # Compare different inference methods
│   ├── model_families.py       # Compare different model families
│   └── optimization.py         # Compare optimization strategies
│
├── utils/                      # Benchmark utilities
│   ├── metrics.py              # Metrics collection utilities
│   ├── load_generator.py       # Generate controlled load patterns
│   └── visualizers.py          # Result visualization tools
│
├── scenarios/                  # Real-world benchmark scenarios
│   ├── chat_simulation.py      # Simulate chat traffic patterns
│   ├── batch_processing.py     # Batch processing scenarios
│   └── mixed_workload.py       # Mixed workload patterns
│
└── results/                    # Results storage (gitignored)
    ├── basic/                  # Basic benchmark results
    ├── comparative/            # Comparative benchmark results
    └── scenarios/              # Scenario benchmark results
```

## Implementation Guidelines

### 1. Metrics Collection

Each benchmark should collect:

- **Timestamps** - Record start and end of each request/operation
- **Token counts** - Track input and output token counts
- **Success/failure** - Track success rates and error types
- **Quota impact** - Monitor quota consumption and throttling

### 2. Test Methodology

Follow these best practices:

- **Warm-up phase** - Initial requests to warm up connections
- **Measurement phase** - Collect metrics during stable operation
- **Cool-down phase** - Gradual reduction in load
- **Multiple runs** - Average results across multiple runs
- **Controlled conditions** - Test at the same time of day if possible

### 3. Visualization Requirements

All benchmark results should be visualized with:

- **Line charts** for time-series data
- **Bar charts** for comparative metrics
- **Heatmaps** for parameter optimization
- **SVG format** for high-quality visuals
- **Standardized color schemes** from visualization_config.py

## Example Implementation: Response Time Benchmark

```python
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from utils.profile_manager import get_profile
from utils.visualization_config import SVG_CONFIG

def measure_response_time(
    model_id,
    prompt_text,
    num_iterations=10,
    output_file="response_time_results.csv"
):
    """
    Measure response time for a given model and prompt.
    
    Args:
        model_id: The model identifier to test
        prompt_text: The prompt to send
        num_iterations: Number of measurements to make
        output_file: Where to save the results
    """
    import boto3
    
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime')
    
    # Prepare the proper request format based on model family
    model_family = model_id.split('.')[0].lower()
    
    if "anthropic" in model_family:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt_text}]
        }
    elif "meta" in model_family:
        payload = {
            "prompt": f"<s>[INST] {prompt_text} [/INST]",
            "max_gen_len": 100
        }
    else:
        payload = {
            "inputText": prompt_text,
            "textGenerationConfig": {"maxTokenCount": 100}
        }
    
    results = []
    
    # Warm-up phase (not measured)
    print(f"Warming up with {model_id}...")
    try:
        bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(payload)
        )
        time.sleep(1)  # Brief pause after warm-up
    except Exception as e:
        print(f"Warm-up error: {str(e)}")
    
    # Measurement phase
    print(f"Running {num_iterations} measurements...")
    for i in range(num_iterations):
        try:
            # Record start time
            start_time = time.time()
            
            # Invoke model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(payload)
            )
            
            # Record end time
            end_time = time.time()
            
            # Extract token counts if available
            response_body = json.loads(response['body'].read())
            
            if "anthropic" in model_family:
                input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
                output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
            else:
                # Estimate for models that don't return token counts
                input_tokens = len(prompt_text.split()) * 1.3
                output_text = str(response_body)
                output_tokens = len(output_text.split()) * 1.3
            
            # Calculate metrics
            latency = end_time - start_time
            tokens_per_second = output_tokens / latency if latency > 0 else 0
            
            # Store results
            results.append({
                "iteration": i + 1,
                "model_id": model_id,
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "tokens_per_second": tokens_per_second,
                "status": "success"
            })
            
            print(f"Iteration {i+1}/{num_iterations}: Latency = {latency:.2f}s")
            
            # Pause between requests (to avoid throttling)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {str(e)}")
            results.append({
                "iteration": i + 1,
                "model_id": model_id,
                "latency_seconds": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "tokens_per_second": None,
                "status": f"error: {str(e)}"
            })
            time.sleep(1)  # Longer pause after error
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Create summary statistics
    successful_results = df[df['status'] == 'success']
    if len(successful_results) > 0:
        avg_latency = successful_results['latency_seconds'].mean()
        avg_tokens_per_second = successful_results['tokens_per_second'].mean()
        min_latency = successful_results['latency_seconds'].min()
        max_latency = successful_results['latency_seconds'].max()
        
        print(f"\nResults Summary for {model_id}:")
        print(f"Average Latency: {avg_latency:.2f} seconds")
        print(f"Average Tokens/Second: {avg_tokens_per_second:.2f}")
        print(f"Min Latency: {min_latency:.2f} seconds")
        print(f"Max Latency: {max_latency:.2f} seconds")
        print(f"Success Rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
    
    return df

def visualize_response_time(results_file, output_file="response_time_visualization.svg"):
    """Create a visualization of response time results"""
    df = pd.read_csv(results_file)
    
    # Filter to only successful results
    df = df[df['status'] == 'success']
    
    if len(df) == 0:
        print("No successful results to visualize")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot latency over iterations
    ax1.plot(df['iteration'], df['latency_seconds'], marker='o', linestyle='-', color='#1f77b4')
    ax1.set_title(f'Response Latency - {df["model_id"].iloc[0]}', fontsize=14)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Latency (seconds)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot tokens per second over iterations
    ax2.plot(df['iteration'], df['tokens_per_second'], marker='o', linestyle='-', color='#2ca02c')
    ax2.set_title(f'Processing Speed - {df["model_id"].iloc[0]}', fontsize=14)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Tokens per Second', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save as SVG
    plt.savefig(output_file, **SVG_CONFIG)
    plt.close()
    
    print(f"Visualization saved to {output_file}")
    return output_file

def run_model_comparison(models, prompt, iterations_per_model=10):
    """Run comparison across multiple models"""
    results = {}
    
    for model_id in models:
        print(f"\nBenchmarking {model_id}...")
        output_file = f"results/basic/{model_id.replace('.', '_')}_response_time.csv"
        results[model_id] = measure_response_time(
            model_id=model_id,
            prompt_text=prompt,
            num_iterations=iterations_per_model,
            output_file=output_file
        )
        
        # Visualize individual model results
        visualize_response_time(
            output_file,
            f"results/basic/{model_id.replace('.', '_')}_response_time.svg"
        )
    
    # Create comparative visualization
    create_comparative_visualization(models, results)

def create_comparative_visualization(models, results):
    """Create a visualization comparing multiple models"""
    # Implementation details for the comparative visualization
    pass

if __name__ == "__main__":
    # Example usage
    models_to_test = [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "meta.llama2-13b-chat-v1",
        "amazon.titan-text-express-v1"
    ]
    
    test_prompt = "Explain the concept of machine learning in a paragraph."
    
    run_model_comparison(models_to_test, test_prompt, iterations_per_model=5)
```

## Implementation Schedule

1. **Week 1: Core Infrastructure**
   - Create benchmark utilities
   - Implement metrics collection
   - Set up visualization framework

2. **Week 2: Basic Benchmarks**
   - Implement response time benchmarks
   - Create throughput measurements
   - Develop quota limit testing

3. **Week 3: Comparative Benchmarks**
   - Build inference method comparisons
   - Implement model family comparisons
   - Create optimization strategy tests

4. **Week 4: Real-world Scenarios**
   - Develop chat simulation benchmarks
   - Create batch processing scenarios
   - Implement mixed workload tests

5. **Week 5: Documentation and Refinement**
   - Complete documentation
   - Refine visualizations
   - Create summary reports

## Deliverables

For each benchmark, deliver:

1. Python implementation files
2. CSV results files 
3. SVG visualizations
4. Markdown documentation with analysis
5. Summary report with findings and recommendations