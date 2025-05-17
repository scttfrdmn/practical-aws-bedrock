"""
Example demonstrating usage of the asynchronous inference clients.

This example shows how to use the BedrockJobClient and BedrockBatchProcessor
for asynchronous inference requests to AWS Bedrock models.
"""

import os
import json
import time
import logging
from src.inference import BedrockJobClient, BedrockBatchProcessor


def single_job_example(s3_output_uri):
    """Demonstrate basic job client usage."""
    print("\n=== Single Job Example ===\n")
    
    # Create a job client for Claude
    client = BedrockJobClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        output_s3_uri=s3_output_uri
    )
    
    # Create a job
    try:
        prompt = "Explain quantum computing in simple terms, covering key concepts and potential applications."
        
        print(f"Creating job with prompt: {prompt[:50]}...")
        
        job_id = client.create_job(
            prompt=prompt,
            max_tokens=800,
            job_name="Quantum-Computing-Explanation"
        )
        
        print(f"Created job: {job_id}")
        
        # Wait for job to complete and get result
        print("Waiting for job to complete...")
        result = client.wait_for_job(job_id)
        
        print("\nJob Result:")
        output = result.get("output", "No output")
        print(output[:500] + "..." if len(output) > 500 else output)
        
        # Print token usage if available
        if "total_tokens" in result:
            print(f"\nToken Usage:")
            print(f"Input tokens: {result.get('input_tokens', 'unknown')}")
            print(f"Output tokens: {result.get('output_tokens', 'unknown')}")
            print(f"Total tokens: {result.get('total_tokens', 'unknown')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # List recent jobs
    try:
        print("\nRecent Jobs:")
        jobs = client.list_jobs(max_results=5)
        
        for job in jobs:
            print(f"Job ID: {job['job_id']}")
            print(f"Status: {job['status']}")
            print(f"Created: {job['created_at']}")
            print(f"Model: {job['model_id']}")
            if job.get('job_name'):
                print(f"Name: {job['job_name']}")
            print("---")
            
    except Exception as e:
        print(f"Error listing jobs: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    print(f"Job count: {metrics['job_count']}")
    print(f"Completed job count: {metrics['completed_job_count']}")
    print(f"Failed job count: {metrics['failed_job_count']}")


def batch_processing_example(s3_output_uri):
    """Demonstrate batch processor usage."""
    print("\n=== Batch Processing Example ===\n")
    
    # Create a batch processor
    processor = BedrockBatchProcessor(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        output_s3_uri=s3_output_uri,
        max_concurrent_jobs=3
    )
    
    # Sample inputs
    inputs = [
        {"prompt": "Explain quantum computing in simple terms."},
        {"prompt": "What are the key differences between machine learning and deep learning?"},
        {"prompt": "How does cloud computing work?", "system_prompt": "Keep your response under 3 paragraphs."},
        {"prompt": "What are the ethical considerations in artificial intelligence?"},
        {"prompt": "Explain how blockchain technology works."}
    ]
    
    # Define progress callback
    def print_progress(current, total):
        print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
    
    try:
        # Process batch
        print(f"Processing batch of {len(inputs)} inputs...")
        results = processor.process_batch(
            inputs=inputs,
            job_name_prefix="Example-Batch",
            progress_callback=print_progress
        )
        
        # Print results
        print("\nResults:")
        for i, result in enumerate(results):
            print(f"\nInput {i+1}:")
            if result.get("status") == "completed":
                output = result.get("output", "")
                print(f"Output: {output[:200]}..." if len(output) > 200 else output)
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}")
        
        # Print metrics
        metrics = processor.get_metrics()
        print("\nProcessing Metrics:")
        print(f"Total inputs processed: {metrics['total_inputs_processed']}")
        print(f"Successful inputs: {metrics['successful_inputs']}")
        print(f"Failed inputs: {metrics['failed_inputs']}")
        print(f"Total processing time: {metrics['total_processing_time']:.2f} seconds")
        if "avg_processing_time" in metrics:
            print(f"Average processing time: {metrics['avg_processing_time']:.2f} seconds per input")
        if "success_rate" in metrics:
            print(f"Success rate: {metrics['success_rate']:.1f}%")
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")


def file_processing_example(s3_output_uri):
    """Demonstrate file-based batch processing."""
    print("\n=== File Processing Example ===\n")
    
    # Create a temporary input file
    input_file = "batch_inputs.json"
    output_file = "batch_results.json"
    
    # Sample inputs
    inputs = [
        {"prompt": "Write a short blog post introduction about cloud computing."},
        {"prompt": "Create a product description for a smart home device that controls lighting."},
        {"prompt": "Write a short technical guide explaining how to set up a basic web server."}
    ]
    
    # Write inputs to file
    with open(input_file, 'w') as f:
        json.dump(inputs, f, indent=2)
    
    print(f"Created input file with {len(inputs)} prompts.")
    
    # Create a batch processor
    processor = BedrockBatchProcessor(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        output_s3_uri=s3_output_uri,
        max_concurrent_jobs=3
    )
    
    try:
        # Process from file
        print(f"Processing inputs from file...")
        results = processor.process_file(
            input_file=input_file,
            output_file=output_file,
            job_name_prefix="File-Batch"
        )
        
        print(f"Processed {len(results)} inputs.")
        print(f"Results saved to {output_file}")
        
        # Print first result as example
        if results and len(results) > 0:
            print("\nSample Result:")
            result = results[0]
            if result.get("status") == "completed":
                output = result.get("output", "")
                print(f"Output: {output[:200]}..." if len(output) > 200 else output)
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}")
        
        # Clean up temporary files
        os.remove(input_file)
        print(f"Removed temporary input file: {input_file}")
        
    except Exception as e:
        print(f"Error in file processing: {str(e)}")


def system_prompt_batch_example(s3_output_uri):
    """Demonstrate batch processing with system prompts."""
    print("\n=== System Prompt Batch Example ===\n")
    
    # Create a batch processor
    processor = BedrockBatchProcessor(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        output_s3_uri=s3_output_uri,
        max_concurrent_jobs=3
    )
    
    # Sample inputs with different system prompts
    inputs = [
        {
            "prompt": "Write about the future of AI.",
            "system_prompt": "You are a technical expert. Provide detailed, technical explanations."
        },
        {
            "prompt": "Write about the future of AI.",
            "system_prompt": "You are a business consultant. Focus on business implications and strategies."
        },
        {
            "prompt": "Write about the future of AI.",
            "system_prompt": "You are a science fiction writer. Create an imaginative, creative narrative."
        }
    ]
    
    try:
        # Process batch
        print(f"Processing batch with different system prompts...")
        results = processor.process_batch(
            inputs=inputs,
            job_name_prefix="System-Prompt-Batch"
        )
        
        # Print results to compare different system prompts
        print("\nResults with Different System Prompts:")
        for i, result in enumerate(results):
            system_prompt = inputs[i].get("system_prompt", "None")
            print(f"\nResult {i+1} (System Prompt: {system_prompt[:50]}...):")
            
            if result.get("status") == "completed":
                output = result.get("output", "")
                print(f"Output: {output[:200]}..." if len(output) > 200 else output)
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"Error in system prompt batch processing: {str(e)}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Provide your S3 output URI
    # This is required for asynchronous jobs
    s3_output_uri = None  # "s3://your-bucket/bedrock-outputs/"
    
    if not s3_output_uri:
        print("WARNING: You must provide a valid S3 output URI to run these examples.")
        print("Please update the s3_output_uri variable in this script.")
        print("For example: s3://your-bucket/bedrock-outputs/")
        exit(1)
    
    # Run examples
    single_job_example(s3_output_uri)
    batch_processing_example(s3_output_uri)
    file_processing_example(s3_output_uri)
    system_prompt_batch_example(s3_output_uri)
    
    print("\nAll examples completed.")