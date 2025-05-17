"""
Batch processing for AWS Bedrock inference.

This module provides a batch processor for AWS Bedrock models, allowing for
efficient processing of multiple inputs using asynchronous job processing.
"""

import os
import json
import time
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

from .job_client import BedrockJobClient


class BedrockBatchProcessor:
    """
    A batch processor for AWS Bedrock models using asynchronous jobs.
    
    This processor allows for efficient processing of large numbers of
    inputs by leveraging AWS Bedrock's asynchronous job processing.
    """
    
    def __init__(
        self,
        model_id: str,
        output_s3_uri: str,
        max_concurrent_jobs: int = 5,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        max_retries: int = 3,
        job_poll_interval: float = 5.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the batch processor.
        
        Args:
            model_id: The Bedrock model identifier
            output_s3_uri: S3 URI for storing job outputs
            max_concurrent_jobs: Maximum number of concurrent jobs
            profile_name: AWS profile name
            region_name: AWS region name
            max_retries: Maximum number of retry attempts
            job_poll_interval: Interval for polling job status
            logger: Optional logger instance
        """
        self.model_id = model_id
        self.output_s3_uri = output_s3_uri
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Create job client
        self.job_client = BedrockJobClient(
            model_id=model_id,
            profile_name=profile_name,
            region_name=region_name,
            max_retries=max_retries,
            output_s3_uri=output_s3_uri,
            default_poll_interval=job_poll_interval,
            logger=self.logger
        )
        
        # Processing metrics
        self.total_inputs_processed = 0
        self.successful_inputs = 0
        self.failed_inputs = 0
        self.total_processing_time = 0.0
    
    def process_batch(
        self,
        inputs: List[Dict[str, Any]],
        job_name_prefix: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of inputs using asynchronous jobs.
        
        Args:
            inputs: List of input dictionaries with 'prompt' and optional 'system_prompt'
            job_name_prefix: Prefix for job names
            tags: Optional tags for jobs
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results corresponding to inputs
        """
        start_time = time.time()
        
        batch_size = len(inputs)
        self.logger.info(f"Processing batch of {batch_size} inputs using model {self.model_id}")
        
        # Create a thread pool executor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_jobs) as executor:
            # Submit jobs
            self.logger.info(f"Submitting jobs (max concurrency: {self.max_concurrent_jobs})")
            
            futures = []
            for i, input_data in enumerate(inputs):
                # Generate job name if prefix provided
                job_name = f"{job_name_prefix}-{i}" if job_name_prefix else None
                
                # Extract prompt and system prompt
                prompt = input_data.get("prompt")
                system_prompt = input_data.get("system_prompt")
                
                if not prompt:
                    self.logger.warning(f"Skipping input {i}: no prompt provided")
                    futures.append(executor.submit(lambda: {"error": "No prompt provided", "status": "failed"}))
                    continue
                
                # Get any model-specific parameters
                other_params = {}
                for key, value in input_data.items():
                    if key not in ["prompt", "system_prompt"]:
                        other_params[key] = value
                
                # Submit the job
                future = executor.submit(
                    self._process_single_input,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    job_name=job_name,
                    tags=tags,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    other_params=other_params,
                    input_index=i
                )
                
                futures.append(future)
            
            # Collect results
            results = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update counts
                    self.total_inputs_processed += 1
                    if result.get("status") == "completed":
                        self.successful_inputs += 1
                    else:
                        self.failed_inputs += 1
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(i + 1, batch_size)
                    
                except Exception as e:
                    self.logger.error(f"Error processing input: {str(e)}")
                    results.append({"error": str(e), "status": "failed"})
                    self.total_inputs_processed += 1
                    self.failed_inputs += 1
        
        # Sort results to match input order
        sorted_results = [None] * batch_size
        for result in results:
            if "input_index" in result:
                index = result.pop("input_index")
                sorted_results[index] = result
        
        # Filter out any None values (should not happen)
        sorted_results = [r for r in sorted_results if r is not None]
        
        # Update processing time
        self.total_processing_time += time.time() - start_time
        
        self.logger.info(
            f"Batch processing complete: {self.successful_inputs}/{batch_size} successful, "
            f"{self.failed_inputs}/{batch_size} failed"
        )
        
        return sorted_results
    
    def _process_single_input(
        self,
        prompt: str,
        system_prompt: Optional[str],
        job_name: Optional[str],
        tags: Optional[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        other_params: Dict[str, Any],
        input_index: int
    ) -> Dict[str, Any]:
        """
        Process a single input using an asynchronous job.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            job_name: Optional job name
            tags: Optional tags
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            other_params: Additional model-specific parameters
            input_index: Index of the input in the original batch
            
        Returns:
            Job result
        """
        try:
            self.logger.debug(f"Processing input {input_index} with prompt: {prompt[:50]}...")
            
            # Create job
            job_id = self.job_client.create_job(
                prompt=prompt,
                system_prompt=system_prompt,
                job_name=job_name,
                max_tokens=max_tokens,
                temperature=temperature,
                tags=tags,
                other_params=other_params
            )
            
            self.logger.debug(f"Created job {job_id} for input {input_index}")
            
            # Wait for job to complete
            result = self.job_client.wait_for_job(job_id)
            
            # Add input index and status
            result["input_index"] = input_index
            result["status"] = "completed"
            result["job_id"] = job_id
            
            self.logger.debug(f"Job {job_id} for input {input_index} completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing input {input_index}: {str(e)}")
            
            return {
                "input_index": input_index,
                "status": "failed",
                "error": str(e),
                "output": "",
                "job_id": job_id if 'job_id' in locals() else None
            }
    
    def process_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        job_name_prefix: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process inputs from a file.
        
        The input file should be a JSON file containing a list of objects,
        each with a 'prompt' field and optional 'system_prompt' field.
        
        Args:
            input_file: Path to input JSON file
            output_file: Optional path to save results
            job_name_prefix: Prefix for job names
            tags: Optional tags for jobs
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results
        """
        # Load inputs from file
        with open(input_file, 'r') as f:
            inputs = json.load(f)
        
        if not isinstance(inputs, list):
            raise ValueError("Input file must contain a JSON array of objects")
        
        # Process the batch
        results = self.process_batch(
            inputs=inputs,
            job_name_prefix=job_name_prefix,
            tags=tags,
            max_tokens=max_tokens,
            temperature=temperature,
            progress_callback=progress_callback
        )
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        file_pattern: str = "*.txt",
        prompt_template: str = "{content}",
        system_prompt: Optional[str] = None,
        job_name_prefix: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process text files from a directory.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Optional directory to save results
            file_pattern: Glob pattern for matching files
            prompt_template: Template for creating prompts from file content
            system_prompt: Optional system prompt to use for all files
            job_name_prefix: Prefix for job names
            tags: Optional tags for jobs
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results
        """
        import glob
        
        # Find input files
        file_paths = glob.glob(os.path.join(input_dir, file_pattern))
        
        if not file_paths:
            self.logger.warning(f"No files matching pattern {file_pattern} found in {input_dir}")
            return []
        
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Prepare inputs
        inputs = []
        for file_path in file_paths:
            try:
                # Read file content
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Create prompt using template
                prompt = prompt_template.format(content=content)
                
                # Add to inputs
                inputs.append({
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path)
                })
                
            except Exception as e:
                self.logger.error(f"Error reading file {file_path}: {str(e)}")
        
        # Process the batch
        results = self.process_batch(
            inputs=inputs,
            job_name_prefix=job_name_prefix,
            tags=tags,
            max_tokens=max_tokens,
            temperature=temperature,
            progress_callback=progress_callback
        )
        
        # Save individual results if output directory specified
        if output_dir:
            for result, input_data in zip(results, inputs):
                file_name = input_data.get("file_name")
                base_name, _ = os.path.splitext(file_name)
                output_path = os.path.join(output_dir, f"{base_name}.json")
                
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get processing metrics.
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            "total_inputs_processed": self.total_inputs_processed,
            "successful_inputs": self.successful_inputs,
            "failed_inputs": self.failed_inputs,
            "total_processing_time": self.total_processing_time
        }
        
        # Calculate average processing time per input if inputs processed
        if self.total_inputs_processed > 0:
            metrics["avg_processing_time"] = self.total_processing_time / self.total_inputs_processed
            metrics["success_rate"] = (self.successful_inputs / self.total_inputs_processed) * 100
        
        # Include job client metrics
        metrics.update(self.job_client.get_metrics())
        
        return metrics


# Example usage (requires valid S3 bucket with appropriate permissions)
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a batch processor
    # Note: You need a valid S3 bucket with appropriate permissions
    s3_output_uri = "s3://your-bucket/bedrock-outputs/"
    
    processor = BedrockBatchProcessor(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        output_s3_uri=s3_output_uri,
        max_concurrent_jobs=3
    )
    
    # Sample inputs
    inputs = [
        {"prompt": "Explain quantum computing in simple terms."},
        {"prompt": "What are the key differences between machine learning and deep learning?"},
        {"prompt": "How does cloud computing work?", "system_prompt": "Keep your response under 3 paragraphs."}
    ]
    
    # Define progress callback
    def print_progress(current, total):
        print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
    
    try:
        # Process batch
        print("Processing batch...")
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