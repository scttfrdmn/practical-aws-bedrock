---
layout: chapter
title: Asynchronous Processing with AWS Bedrock
difficulty: intermediate
time-estimate: 40 minutes
---

# Asynchronous Processing with AWS Bedrock

> "You need to process thousands of documents with AI, but synchronous APIs are too slow and unreliable. Let's solve this with AWS Bedrock's asynchronous processing capabilities."

## The Problem

---

**Scenario**: You're a developer at a legal tech company that needs to analyze thousands of legal contracts daily. Each contract ranges from 20-100 pages and requires AI analysis to extract key clauses, obligations, and risks.

You've tried using synchronous API calls to AWS Bedrock, but you're encountering several problems:

1. Long-running requests frequently timeout before completion
2. Processing large documents exceeds the synchronous API request size limits
3. Your system can't handle the backlog during peak upload times
4. There's no way to track progress for users waiting for results
5. Failed requests require manual intervention to restart

You need a robust solution that:
- Reliably processes documents of any size
- Handles thousands of documents efficiently
- Provides status tracking and notifications
- Automatically retries failed processing
- Scales to meet fluctuating demand

---

## Key Concepts Explained

### Understanding Asynchronous Processing

Asynchronous processing is a pattern where:
1. You submit a job to be processed
2. The system immediately returns a job ID (not the results)
3. Processing happens in the background
4. You check for completion or receive a notification when done
5. You retrieve the results when processing is complete

Think of it like dropping your car off at a mechanic:
- You don't wait at the shop while they work
- They give you a ticket to identify your car
- You can check on progress by calling
- They notify you when it's ready
- You return to pick up your car when it's done

### AWS Bedrock's Asynchronous APIs

AWS Bedrock provides dedicated APIs for asynchronous processing:

1. **CreateModelInvocationJob**: Submit a job to be processed
2. **GetModelInvocationJob**: Check the status of a running job
3. **ListModelInvocationJobs**: View all your submitted jobs
4. **DeleteModelInvocationJob**: Cancel a job (if possible)

These APIs are separate from the synchronous `InvokeModel` and streaming `InvokeModelWithResponseStream` APIs we've covered in previous chapters.

### Anatomy of an Asynchronous Job

An asynchronous job in AWS Bedrock consists of:

- **Job ID**: Unique identifier for tracking
- **Model ID**: The foundation model to use
- **Input Data**: The prompts/data to process
- **Input/Output Locations**: S3 buckets for data exchange
- **Status**: Current state (IN_PROGRESS, COMPLETED, FAILED, etc.)
- **Configuration**: Job-specific settings and parameters

This structure allows for processing much larger inputs and handling long-running tasks reliably.

## Step-by-Step Implementation

Let's build a complete solution for asynchronous document processing with AWS Bedrock.

### 1. Setting Up Your Environment

First, we need to set up the AWS SDK and required permissions:

```bash
# Install required packages
pip install boto3 pandas tqdm
```

You'll need these IAM permissions:
- `bedrock:CreateModelInvocationJob`
- `bedrock:GetModelInvocationJob`
- `bedrock:ListModelInvocationJobs`
- `s3:PutObject`
- `s3:GetObject`
- `s3:ListBucket`

### 2. Creating the Asynchronous Processing Client

Let's create a robust client for asynchronous processing:

```python
import json
import time
import logging
from typing import Dict, Any, Optional, List

import boto3
from botocore.exceptions import ClientError

from utils.profile_manager import get_profile, get_region


class BedrockJobClient:
    """
    A client for creating and managing asynchronous AWS Bedrock inference jobs.
    
    This client follows the AWS profile conventions specified in CLAUDE.md and
    provides functionality for creating, monitoring, and retrieving results from
    asynchronous inference jobs.
    """
    
    def __init__(
        self, 
        model_id: str,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        output_s3_uri: Optional[str] = None,
        default_poll_interval: float = 5.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Bedrock asynchronous job client.
        
        Args:
            model_id: The Bedrock model identifier
            profile_name: AWS profile name (defaults to value from get_profile())
            region_name: AWS region name (defaults to value from get_region())
            max_retries: Maximum number of retry attempts for recoverable errors
            base_backoff: Base backoff time (in seconds) for exponential backoff
            output_s3_uri: S3 URI for storing job outputs
            default_poll_interval: Default interval for polling job status
            logger: Optional logger instance
        """
        self.model_id = model_id
        self.profile_name = profile_name or get_profile()
        self.region_name = region_name or get_region()
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.output_s3_uri = output_s3_uri
        self.default_poll_interval = default_poll_interval
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Create AWS session with profile
        self.session = boto3.Session(
            profile_name=self.profile_name, 
            region_name=self.region_name
        )
        
        # Create Bedrock client for jobs
        self.client = self.session.client('bedrock')
        
        # Create S3 client for output retrieval
        self.s3_client = self.session.client('s3')
        
        # Track metrics
        self.job_count = 0
        self.completed_job_count = 0
        self.failed_job_count = 0
    
    def create_job(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7, 
        system_prompt: Optional[str] = None,
        job_name: Optional[str] = None,
        output_s3_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        other_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an asynchronous inference job.
        
        Args:
            prompt: The user prompt or instruction
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            system_prompt: Optional system prompt for models that support it
            job_name: Optional name for the job
            output_s3_uri: S3 URI for storing job outputs (overrides instance default)
            tags: Optional tags for the job
            other_params: Additional model-specific parameters
            
        Returns:
            The job ID
            
        Raises:
            ValueError: For invalid input parameters
            RuntimeError: For unrecoverable errors after retries
        """
        # Validate inputs
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        
        # Determine output S3 URI
        s3_uri = output_s3_uri or self.output_s3_uri
        if not s3_uri:
            raise ValueError("Output S3 URI must be provided")
        
        # Create model-specific request body
        request_body = self._create_request_body(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            other_params=other_params or {}
        )
        
        # Prepare job configuration
        job_config = {
            "modelId": self.model_id,
            "contentType": "application/json",
            "accept": "application/json",
            "inputData": json.dumps(request_body),
            "outputDataConfig": {
                "s3Uri": s3_uri
            }
        }
        
        # Add optional job name
        if job_name:
            job_config["jobName"] = job_name
        
        # Add tags if provided
        if tags:
            formatted_tags = [{"key": k, "value": v} for k, v in tags.items()]
            job_config["tags"] = formatted_tags
        
        # Make the request with retries
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Creating job for model {self.model_id} (attempt {attempt + 1})")
                
                response = self.client.create_model_invocation_job(**job_config)
                
                job_id = response.get("jobArn").split("/")[-1]
                
                self.logger.info(f"Created job {job_id} for model {self.model_id}")
                
                # Update metrics
                self.job_count += 1
                
                return job_id
                
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_message = e.response["Error"]["Message"]
                
                self.logger.warning(
                    f"Error creating job (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{error_code} - {error_message}"
                )
                
                # Check if the error is recoverable
                if error_code in ["ThrottlingException", "ServiceUnavailableException", "InternalServerException"]:
                    if attempt < self.max_retries:
                        # Calculate backoff time with exponential backoff and jitter
                        backoff_time = self._calculate_backoff(attempt)
                        self.logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                        time.sleep(backoff_time)
                        continue
                
                # If we've exhausted retries or the error is not recoverable, raise
                raise RuntimeError(f"Failed to create job after {attempt + 1} attempts: {error_code} - {error_message}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of an asynchronous inference job.
        
        Args:
            job_id: The job ID
            
        Returns:
            Dictionary with job status information
            
        Raises:
            RuntimeError: For unrecoverable errors after retries
        """
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Getting status for job {job_id} (attempt {attempt + 1})")
                
                response = self.client.get_model_invocation_job(
                    jobIdentifier=job_id
                )
                
                status = {
                    "job_id": job_id,
                    "status": response.get("status"),
                    "created_at": response.get("creationTime"),
                    "completed_at": response.get("endTime"),
                    "model_id": response.get("modelId"),
                    "output_s3_uri": response.get("outputDataConfig", {}).get("s3Uri")
                }
                
                # Add error information if available
                if "failureMessage" in response:
                    status["error"] = response["failureMessage"]
                
                return status
                
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_message = e.response["Error"]["Message"]
                
                self.logger.warning(
                    f"Error getting job status (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{error_code} - {error_message}"
                )
                
                # Check if the error is recoverable
                if error_code in ["ThrottlingException", "ServiceUnavailableException", "InternalServerException"]:
                    if attempt < self.max_retries:
                        # Calculate backoff time with exponential backoff and jitter
                        backoff_time = self._calculate_backoff(attempt)
                        self.logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                        time.sleep(backoff_time)
                        continue
                
                # If we've exhausted retries or the error is not recoverable, raise
                raise RuntimeError(f"Failed to get job status after {attempt + 1} attempts: {error_code} - {error_message}")
    
    def wait_for_job(
        self, 
        job_id: str, 
        poll_interval: Optional[float] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete and return the result.
        
        Args:
            job_id: The job ID
            poll_interval: Interval between status checks (in seconds)
            timeout: Maximum time to wait (in seconds)
            
        Returns:
            Dictionary with job result
            
        Raises:
            RuntimeError: If the job fails or times out
        """
        interval = poll_interval or self.default_poll_interval
        start_time = time.time()
        
        self.logger.info(f"Waiting for job {job_id} to complete...")
        
        while True:
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                raise RuntimeError(f"Timeout waiting for job {job_id} to complete")
            
            # Get job status
            status = self.get_job_status(job_id)
            job_status = status.get("status")
            
            if job_status == "COMPLETED":
                self.logger.info(f"Job {job_id} completed successfully")
                self.completed_job_count += 1
                
                # Retrieve and return result
                return self.get_job_result(job_id)
                
            elif job_status in ["FAILED", "STOPPED"]:
                error_message = status.get("error", "Unknown error")
                self.logger.error(f"Job {job_id} failed: {error_message}")
                self.failed_job_count += 1
                
                raise RuntimeError(f"Job {job_id} failed: {error_message}")
                
            elif job_status in ["IN_PROGRESS", "STARTING", "QUEUED"]:
                self.logger.debug(f"Job {job_id} is {job_status}, waiting...")
                time.sleep(interval)
                
            else:
                self.logger.warning(f"Job {job_id} has unknown status: {job_status}")
                time.sleep(interval)
    
    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed job.
        
        Args:
            job_id: The job ID
            
        Returns:
            Dictionary with job result
            
        Raises:
            RuntimeError: If the job is not completed or result cannot be retrieved
        """
        # Get job status to verify it's completed and get S3 URI
        status = self.get_job_status(job_id)
        
        if status.get("status") != "COMPLETED":
            raise RuntimeError(f"Cannot get result for job {job_id} with status {status.get('status')}")
        
        # Extract S3 URI
        s3_uri = status.get("output_s3_uri")
        if not s3_uri:
            raise RuntimeError(f"No output S3 URI found for job {job_id}")
        
        # Parse S3 URI
        s3_uri_parts = s3_uri.replace("s3://", "").split("/")
        bucket = s3_uri_parts[0]
        
        # The key might have job ID appended by the service
        prefix = "/".join(s3_uri_parts[1:])
        
        try:
            # List objects to find the result file
            self.logger.debug(f"Listing objects in bucket {bucket} with prefix {prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            if "Contents" not in response or len(response["Contents"]) == 0:
                raise RuntimeError(f"No result files found for job {job_id}")
            
            # Get the result file
            result_key = response["Contents"][0]["Key"]
            
            self.logger.debug(f"Getting result file {result_key} from bucket {bucket}")
            
            response = self.s3_client.get_object(
                Bucket=bucket,
                Key=result_key
            )
            
            # Read and parse the result
            result_content = response["Body"].read().decode("utf-8")
            result_json = json.loads(result_content)
            
            # Parse model-specific result format
            parsed_result = self._parse_job_result(result_json)
            
            # Add job metadata
            parsed_result.update({
                "job_id": job_id,
                "model_id": self.model_id,
                "created_at": status.get("created_at"),
                "completed_at": status.get("completed_at")
            })
            
            return parsed_result
            
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            
            raise RuntimeError(f"Failed to retrieve job result: {error_code} - {error_message}")
```

### 3. Using the Batch Processor for Document Processing

Now let's create a document processing application that uses our asynchronous batch processor:

```python
import os
import json
import time
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List, Callable

from inference.asynchronous.job_client import BedrockJobClient


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
```

### 4. Processing Documents from a Directory

Here's an example of using our batch processor to process files from a directory:

```python
import logging
import time
import json
from datetime import datetime

from inference.asynchronous.batch_processor import BedrockBatchProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("legal_doc_processor")

def process_legal_contracts():
    """Process a batch of legal contracts to extract key clauses."""
    # Initialize processor (requires valid S3 bucket with appropriate permissions)
    output_s3_uri = "s3://your-legal-analysis-bucket/outputs/"
    
    processor = BedrockBatchProcessor(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        output_s3_uri=output_s3_uri,
        max_concurrent_jobs=5
    )
    
    # Process directory of legal contracts
    logger.info("Processing legal contracts directory...")
    start_time = time.time()
    
    # Define system prompt to specify the extraction task
    system_prompt = """You are a legal contract analyzer. Extract key information from contracts into a structured format.
    Respond with a JSON object containing the requested fields."""
    
    # Define template for creating prompts from file content
    prompt_template = """Please analyze this legal contract and extract the following information:

1. Parties involved (names and roles)
2. Key dates (execution, effective, termination)
3. Payment terms and amounts
4. Key obligations for each party
5. Termination conditions
6. Governing law
7. Any unusual or potentially risky clauses

The contract text is below:

{content}"""
    
    # Process the directory
    results = processor.process_directory(
        input_dir="contracts/",
        output_dir="results/",
        file_pattern="*.txt",  # Process all text files
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        job_name_prefix="legal-analysis",
        max_tokens=4000,
        temperature=0.2
    )
    
    elapsed = time.time() - start_time
    
    # Get metrics
    metrics = processor.get_metrics()
    
    # Print summary statistics
    logger.info(f"Processing complete! Stats:")
    logger.info(f"- Total documents: {metrics['total_inputs_processed']}")
    logger.info(f"- Successfully processed: {metrics['successful_inputs']}")
    logger.info(f"- Failed: {metrics['failed_inputs']}")
    logger.info(f"- Total time: {elapsed:.2f} seconds")
    if metrics['total_inputs_processed'] > 0:
        logger.info(f"- Average time per document: {metrics['avg_processing_time']:.2f} seconds")
        logger.info(f"- Success rate: {metrics['success_rate']:.1f}%")
    
    # Save combined results
    output_path = f"results/legal_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Combined results saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    process_legal_contracts()
```

### 5. Implementing a Job Monitoring Dashboard

For production use, you'll want a monitoring dashboard. Here's a simple CLI-based monitor using our BedrockJobClient:

```python
import os
import json
import logging
import time
from datetime import datetime
import threading

from inference.asynchronous.job_client import BedrockJobClient

def monitor_bedrock_jobs():
    """Monitor all running Bedrock asynchronous jobs."""
    # Initialize client
    client = BedrockJobClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # Model ID doesn't matter for listing jobs
        output_s3_uri="s3://your-output-bucket/outputs/"
    )
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("bedrock_monitor")
    
    # Print header
    print("\n==== AWS Bedrock Job Monitor ====\n")
    
    try:
        while True:
            # Clear screen (works on Windows and Unix)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Print header with timestamp
            print(f"\n==== AWS Bedrock Job Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")
            
            # Get active jobs
            active_jobs = client.list_jobs(status_filter="IN_PROGRESS")
            completed_jobs = client.list_jobs(status_filter="COMPLETED", max_results=5)
            failed_jobs = client.list_jobs(status_filter="FAILED", max_results=5)
            
            # Print active jobs
            print(f"Active Jobs ({len(active_jobs)}):")
            if active_jobs:
                # Create a table view
                headers = ["Job ID", "Job Name", "Model", "Created At", "Status"]
                rows = []
                
                for job in active_jobs:
                    job_id_short = job["job_id"][-8:]  # Show last 8 chars for readability
                    created_at = job.get("created_at", "").replace("T", " ").split(".")[0]
                    
                    rows.append([
                        job_id_short, 
                        job.get("job_name", "N/A")[:30],  # Truncate long names
                        job["model_id"].split(".")[-1][:20],  # Show only model name
                        created_at,
                        job["status"]
                    ])
                
                # Print table
                print_table(headers, rows)
            else:
                print("  No active jobs\n")
            
            # Print recently completed jobs
            print(f"\nRecently Completed Jobs ({len(completed_jobs)}):")
            if completed_jobs:
                headers = ["Job ID", "Job Name", "Model", "Completed At"]
                rows = []
                
                for job in completed_jobs:
                    job_id_short = job["job_id"][-8:]
                    completed_at = job.get("completed_at", "").replace("T", " ").split(".")[0]
                    
                    rows.append([
                        job_id_short, 
                        job.get("job_name", "N/A")[:30],
                        job["model_id"].split(".")[-1][:20],
                        completed_at
                    ])
                
                print_table(headers, rows)
            else:
                print("  No recently completed jobs\n")
            
            # Print recently failed jobs
            print(f"\nRecently Failed Jobs ({len(failed_jobs)}):")
            if failed_jobs:
                headers = ["Job ID", "Job Name", "Model", "Failed At"]
                rows = []
                
                for job in failed_jobs:
                    job_id_short = job["job_id"][-8:]
                    failed_at = job.get("completed_at", "").replace("T", " ").split(".")[0]
                    
                    rows.append([
                        job_id_short, 
                        job.get("job_name", "N/A")[:30],
                        job["model_id"].split(".")[-1][:20],
                        failed_at
                    ])
                
                print_table(headers, rows)
            else:
                print("  No recently failed jobs\n")
            
            # Display options
            print("\nOptions:")
            print("  r: Refresh")
            print("  d <job_id>: View job details")
            print("  c <job_id>: Cancel job")
            print("  q: Quit")
            
            # Get user input with timeout
            command = input_with_timeout("\nCommand: ", timeout=10)
            
            if command.lower() == 'q':
                break
            elif command.lower() == 'r':
                continue  # Just refresh
            elif command.lower().startswith('d '):
                # View job details
                job_id = command[2:].strip()
                display_job_details(client, job_id)
                input("\nPress Enter to continue...")
            elif command.lower().startswith('c '):
                # Cancel job
                job_id = command[2:].strip()
                try:
                    result = client.cancel_job(job_id)
                    print(f"\nJob {job_id} cancellation requested.")
                    print(f"Status: {result}")
                    input("\nPress Enter to continue...")
                except Exception as e:
                    print(f"\nError canceling job: {str(e)}")
                    input("\nPress Enter to continue...")
            
    except KeyboardInterrupt:
        print("\nExiting job monitor...")

def print_table(headers, rows):
    """Print a formatted table."""
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print headers
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(row_line)

def input_with_timeout(prompt, timeout=10):
    """Get user input with timeout."""
    result = [None]
    
    def get_input():
        result[0] = input(prompt)
    
    # Start input thread
    thread = threading.Thread(target=get_input)
    thread.daemon = True
    thread.start()
    
    # Wait for input or timeout
    thread.join(timeout)
    
    if thread.is_alive():
        # Thread still running, no input received
        return 'r'  # Default to refresh
    
    return result[0]

def display_job_details(client, job_id):
    """Display detailed information about a job."""
    try:
        # Get job status
        status_info = client.get_job_status(job_id)
        
        # Display details
        print("\n=== Job Details ===")
        print(f"Job ID: {job_id}")
        print(f"Status: {status_info['status']}")
        print(f"Model: {status_info['model_id']}")
        print(f"Created: {status_info.get('created_at', 'N/A')}")
        
        if status_info.get('completed_at'):
            print(f"Completed: {status_info['completed_at']}")
        
        print(f"Output Location: {status_info.get('output_s3_uri', 'N/A')}")
        
        if "error" in status_info:
            print(f"Error: {status_info['error']}")
        
        # For completed jobs, ask if user wants to retrieve results
        if status_info['status'] == "COMPLETED":
            retrieve = input("\nRetrieve results? (y/n): ")
            
            if retrieve.lower() == 'y':
                # Get results
                results = client.get_job_result(job_id)
                
                # Display summary
                print("\n=== Job Results ===")
                if "output" in results:
                    print(f"\nOutput:")
                    output = results["output"]
                    if len(output) > 500:
                        print(f"{output[:500]}...")
                    else:
                        print(output)
                    
                    if "input_tokens" in results:
                        print(f"\nToken Usage:")
                        print(f"Input tokens: {results.get('input_tokens', 'unknown')}")
                        print(f"Output tokens: {results.get('output_tokens', 'unknown')}")
                        print(f"Total tokens: {results.get('total_tokens', 'unknown')}")
                else:
                    print("\nRaw response:")
                    print(json.dumps(results.get("raw_response", {}), indent=2)[:500] + "...")
                    
    except Exception as e:
        print(f"Error retrieving job details: {str(e)}")
```

## Common Pitfalls and Troubleshooting

### Pitfall #1: Not Handling S3 Permissions Correctly

**Problem**: Jobs fail with access denied errors when reading from or writing to S3.

**Solution**: Ensure proper IAM permissions and bucket policies:

```python
import boto3
import uuid
from typing import Dict

def check_s3_permissions(bucket_name: str) -> Dict[str, bool]:
    """
    Check if you have the necessary S3 permissions.
    
    Returns a dictionary with permission status.
    """
    s3 = boto3.client('s3')
    results = {
        "bucket_exists": False,
        "can_list": False,
        "can_get": False,
        "can_put": False
    }
    
    try:
        # Check if bucket exists
        s3.head_bucket(Bucket=bucket_name)
        results["bucket_exists"] = True
        
        # Check list permission
        s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        results["can_list"] = True
        
        # Check get permission (if bucket has objects)
        try:
            response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            if "Contents" in response and response["Contents"]:
                test_key = response["Contents"][0]["Key"]
                s3.head_object(Bucket=bucket_name, Key=test_key)
                results["can_get"] = True
        except Exception:
            # Can't test get if no objects or no permission
            pass
        
        # Check put permission
        test_key = f"permissions-test-{uuid.uuid4()}.txt"
        s3.put_object(Bucket=bucket_name, Key=test_key, Body="Test write permission")
        results["can_put"] = True
        
        # Clean up test object
        s3.delete_object(Bucket=bucket_name, Key=test_key)
        
    except Exception as e:
        print(f"Error checking S3 permissions: {str(e)}")
    
    return results
```

You also need to ensure your Bedrock IAM role has access to these S3 buckets:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-input-bucket",
                "arn:aws:s3:::your-input-bucket/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-output-bucket",
                "arn:aws:s3:::your-output-bucket/*"
            ]
        }
    ]
}
```

### Pitfall #2: Not Properly Handling Large Documents

**Problem**: Very large documents cause memory issues or API limits.

**Solution**: Implement document chunking for large files and process them in parallel:

```python
import os
import json
import time
from typing import List, Dict, Any, Optional

from inference.asynchronous.batch_processor import BedrockBatchProcessor

def chunk_large_document(document_path: str, max_chunk_size: int = 100000, overlap: int = 5000) -> List[str]:
    """
    Split a large document into overlapping chunks for processing.
    
    Args:
        document_path: Path to document
        max_chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of temporary files with chunks
    """
    # Read the document
    with open(document_path, 'r') as f:
        content = f.read()
    
    # If document is small enough, return original
    if len(content) <= max_chunk_size:
        return [document_path]
    
    # Split into chunks
    chunks = []
    chunk_files = []
    
    for i in range(0, len(content), max_chunk_size - overlap):
        # Get chunk with overlap
        chunk = content[i:i + max_chunk_size]
        chunks.append(chunk)
        
        # Create temporary file for this chunk
        base_name = os.path.basename(document_path)
        temp_file = f"tmp_{base_name}_chunk{len(chunks)}.txt"
        
        with open(temp_file, 'w') as f:
            f.write(chunk)
        
        chunk_files.append(temp_file)
    
    return chunk_files

def process_chunks_with_batch_processor(
    chunk_files: List[str],
    processor: BedrockBatchProcessor,
    system_prompt: str,
    task_prompt: str,
    max_tokens: int = 4000
) -> Dict[str, Any]:
    """
    Process document chunks using the batch processor.
    
    Args:
        chunk_files: List of chunk file paths
        processor: BedrockBatchProcessor instance
        system_prompt: System prompt for model
        task_prompt: Task instructions
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with combined results
    """
    # Prepare inputs for batch processor
    inputs = []
    
    for i, chunk_file in enumerate(chunk_files):
        # Read chunk content
        with open(chunk_file, 'r') as f:
            chunk_content = f.read()
        
        # Create full prompt with task and chunk content
        full_prompt = f"{task_prompt}\n\nNOTE: This is chunk {i+1} of {len(chunk_files)} from a larger document.\n\n{chunk_content}"
        
        # Add to inputs
        inputs.append({
            "prompt": full_prompt,
            "system_prompt": system_prompt,
            "chunk_index": i,
            "total_chunks": len(chunk_files)
        })
    
    # Process all chunks in parallel
    print(f"Processing {len(chunk_files)} chunks in parallel...")
    chunk_results = processor.process_batch(
        inputs=inputs,
        job_name_prefix="chunk-processing",
        max_tokens=max_tokens,
        temperature=0.2  # Lower temperature for more consistent results
    )
    
    # Clean up temporary files
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    
    return {
        "chunk_results": chunk_results,
        "num_chunks": len(chunk_files)
    }

def combine_chunk_results(chunk_results: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
    """
    Combine results from document chunks based on task type.
    
    Args:
        chunk_results: List of results from individual chunks
        task_type: Type of task (e.g., "extract_entities", "summarize")
        
    Returns:
        Combined result
    """
    # Example: Combining entity extraction results
    if "extract" in task_type.lower() and "entities" in task_type.lower():
        # Collect all entities
        all_entities = {}
        
        for result in chunk_results:
            if "output" in result and isinstance(result["output"], str):
                # Try to parse output as JSON
                try:
                    data = json.loads(result["output"])
                    if "entities" in data and isinstance(data["entities"], list):
                        for entity in data["entities"]:
                            # Use entity name or ID as key to remove duplicates
                            entity_key = entity.get("id", entity.get("name", str(entity)))
                            all_entities[entity_key] = entity
                except json.JSONDecodeError:
                    # If not valid JSON, try basic extraction using regex (not shown)
                    pass
        
        return {
            "task_type": "entity_extraction",
            "entities": list(all_entities.values()),
            "entity_count": len(all_entities)
        }
    
    # Example: Combining summarization results
    elif "summarize" in task_type.lower() or "summary" in task_type.lower():
        # Extract summaries from each chunk
        summaries = []
        for result in chunk_results:
            if "output" in result:
                summaries.append(result["output"])
        
        combined_text = "\n\n".join(summaries)
        
        return {
            "task_type": "summarization",
            "combined_summaries": combined_text,
            "chunk_count": len(summaries)
        }
    
    # Default approach for other task types
    else:
        return {
            "task_type": "unknown",
            "chunk_results": chunk_results,
            "chunk_count": len(chunk_results)
        }
```

### Pitfall #3: Dealing with Request Timeouts and Throttling

**Problem**: AWS Bedrock API may throttle requests or timeout during high loads.

**Solution**: Implement backoff, jitter, and concurrency control:

```python
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any

def execute_with_backoff(func: Callable, max_retries: int = 5, base_delay: float = 1.0) -> Any:
    """
    Execute a function with exponential backoff and jitter.
    
    Args:
        func: The function to execute
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        
    Returns:
        Function result
        
    Raises:
        Exception: The last exception encountered after max retries
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # Attempt to execute the function
            return func()
            
        except Exception as e:
            last_exception = e
            
            # Check if this is a throttling-related error
            error_msg = str(e).lower()
            is_throttling = any(x in error_msg for x in ["throttl", "rate", "limit", "capacity"])
            
            # Give up on last attempt or non-throttling errors
            if attempt == max_retries or not is_throttling:
                raise
            
            # Calculate backoff time with exponential backoff and jitter
            backoff = base_delay * (2 ** attempt)
            jitter = backoff * 0.2
            delay = backoff + random.uniform(-jitter, jitter)
            
            logging.warning(f"Request throttled (attempt {attempt+1}/{max_retries+1}). Retrying in {delay:.2f}s")
            time.sleep(delay)
    
    # This should never happen due to the raise above, but just in case
    raise last_exception

class RequestRateLimiter:
    """
    A rate limiter for API requests.
    """
    
    def __init__(self, requests_per_second: float = 5.0):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
    
    def wait(self):
        """
        Wait if necessary to maintain the rate limit.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            # Need to wait to respect rate limit
            wait_time = self.min_interval - time_since_last
            time.sleep(wait_time)
        
        # Update last request time
        self.last_request_time = time.time()
```

## Try It Yourself Challenge

### Challenge: Build a Document Processing Pipeline with SQS Integration

Create a comprehensive document processing pipeline using AWS Bedrock asynchronous processing and Amazon SQS for job management.

**Starting Code**:

```python
import boto3
import json
import os
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading

from inference.asynchronous.job_client import BedrockJobClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bedrock_pipeline")

class DocumentProcessingPipeline:
    """
    A pipeline for processing documents asynchronously with AWS Bedrock and SQS.
    
    This pipeline integrates SQS for job management, allowing for:
    1. Job persistence across application restarts
    2. Distributed processing across multiple instances
    3. Automatic retries of failed jobs
    4. Dead-letter queues for persistently failing jobs
    """
    
    def __init__(
        self,
        model_id: str,
        output_s3_uri: str,
        region_name: str = "us-west-2",
        profile_name: Optional[str] = None,
        queue_name_prefix: str = "bedrock-processing",
        max_retries: int = 3,
        use_dlq: bool = True
    ):
        """Initialize the document processing pipeline."""
        # TODO: Initialize Bedrock job client
        
        # TODO: Initialize SQS client and create/get queues
        
        # TODO: Set up job tracking
        pass
    
    def setup_queues(self, queue_name_prefix: str, use_dlq: bool) -> Dict[str, str]:
        """Set up SQS queues for job management."""
        # TODO: Create main processing queue
        
        # TODO: Create dead-letter queue if requested
        
        # TODO: Set up redrive policy if using DLQ
        pass
    
    def submit_document_for_processing(
        self,
        document_path: str,
        task_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        priority: str = "normal"  # "high", "normal", "low"
    ) -> Dict[str, Any]:
        """Submit a document for processing."""
        # TODO: Prepare document (handle local vs S3 paths)
        
        # TODO: Create SQS message with processing details
        
        # TODO: Add message to queue with appropriate priority
        pass
    
    def start_processing_worker(
        self,
        worker_id: str,
        polling_interval: float = 5.0,
        batch_size: int = 5,
        shutdown_event: Optional[threading.Event] = None
    ) -> None:
        """Start a worker to process documents from the queue."""
        # TODO: Implement queue polling and processing
        
        # TODO: Handle job processing and result storage
        
        # TODO: Implement proper message deletion on success
        
        # TODO: Handle failed jobs appropriately
        pass
    
    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get the result of a completed job."""
        # TODO: Retrieve job result from the Bedrock client
        pass
    
    def process_failed_jobs(
        self,
        max_jobs: int = 10,
        reprocess: bool = False
    ) -> Dict[str, Any]:
        """Process or analyze failed jobs from the dead letter queue."""
        # TODO: Retrieve messages from DLQ
        
        # TODO: Analyze failure patterns
        
        # TODO: Optionally reprocess jobs
        pass
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get statistics about the queues."""
        # TODO: Get queue attributes and return statistics
        pass

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DocumentProcessingPipeline(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        output_s3_uri="s3://your-output-bucket/bedrock-results/",
        queue_name_prefix="legal-doc-processing"
    )
    
    # Start processing workers
    shutdown_event = threading.Event()
    
    worker_threads = []
    for i in range(3):  # Start 3 worker threads
        worker_id = f"worker-{i+1}"
        worker = threading.Thread(
            target=pipeline.start_processing_worker,
            args=(worker_id, 5.0, 5, shutdown_event)
        )
        worker.daemon = True
        worker.start()
        worker_threads.append(worker)
    
    try:
        # Submit some documents for processing
        doc_paths = [
            "documents/contract1.txt",
            "documents/contract2.txt",
            "s3://your-input-bucket/documents/large_contract.pdf"
        ]
        
        for doc_path in doc_paths:
            result = pipeline.submit_document_for_processing(
                document_path=doc_path,
                task_prompt="Extract all legal entities from this document",
                system_prompt="You are a legal document analyzer. Extract entities in JSON format.",
                max_tokens=4000
            )
            print(f"Submitted document {doc_path} with job ID: {result.get('job_id')}")
        
        # Wait for processing to complete (in a real application, this would be handled differently)
        print("Processing documents. Press Ctrl+C to stop...")
        while True:
            stats = pipeline.get_queue_statistics()
            print(f"Queue stats: {stats}")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nShutting down workers...")
        shutdown_event.set()
        
        for worker in worker_threads:
            worker.join(timeout=2.0)
            
        print("Workers stopped. Exiting.")
```

**Expected Outcome**: A robust document processing pipeline that:
1. Creates and manages SQS queues for job tracking
2. Processes documents asynchronously with multiple worker threads
3. Handles job failures with retries and dead-letter queues
4. Provides job status tracking and statistics
5. Efficiently manages resources with proper cleanup

## Beyond the Basics

Once you've mastered asynchronous processing, consider these advanced techniques:

### 1. Implementing SQS for Job Queue Management

For production workloads, use Amazon SQS to manage processing queues:

```python
def create_processing_queue():
    """Create an SQS queue for document processing jobs."""
    sqs = boto3.client('sqs')
    
    # Create queue with settings for reliable message delivery
    response = sqs.create_queue(
        QueueName="bedrock-document-processing-queue",
        Attributes={
            'VisibilityTimeout': '3600',  # 1 hour
            'MessageRetentionPeriod': '1209600',  # 14 days
            'DelaySeconds': '0',
            'ReceiveMessageWaitTimeSeconds': '20'  # Long polling
        }
    )
    
    return response['QueueUrl']

def submit_document_to_queue(queue_url, document_path, task_type):
    """Submit a document to the processing queue."""
    sqs = boto3.client('sqs')
    
    # Create message with document information
    message = {
        "document_path": document_path,
        "task_type": task_type,
        "submitted_at": datetime.now().isoformat()
    }
    
    # Send message to queue
    response = sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message)
    )
    
    return {
        "message_id": response['MessageId'],
        "document_path": document_path,
        "task_type": task_type
    }

def process_queue_documents(queue_url, max_documents=10):
    """Process documents from the queue."""
    sqs = boto3.client('sqs')
    
    # Initialize processor
    processor = DocumentProcessor(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        input_bucket="your-input-bucket",
        output_bucket="your-output-bucket"
    )
    
    # Receive messages from queue
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=max_documents,
        WaitTimeSeconds=20,  # Long polling
        AttributeNames=['All'],
        MessageAttributeNames=['All']
    )
    
    if 'Messages' not in response:
        logger.info("No messages in queue")
        return []
    
    # Process documents
    documents = []
    receipt_handles = {}
    
    for message in response['Messages']:
        try:
            # Parse message
            message_body = json.loads(message['Body'])
            document_path = message_body['document_path']
            task_type = message_body['task_type']
            
            # Add to processing batch
            documents.append(document_path)
            
            # Store receipt handle for later deletion
            receipt_handles[document_path] = message['ReceiptHandle']
            
        except Exception as e:
            logger.error(f"Error parsing message: {str(e)}")
    
    # Create task prompt based on task type
    task_prompt = get_task_prompt(next(iter(set(d['task_type'] for d in documents))))
    
    # Process documents
    results = processor.process_document_batch(
        documents=documents,
        task_prompt=task_prompt,
        max_tokens=4000,
        batch_size=max_documents
    )
    
    # Delete successfully processed messages from queue
    for doc_path, result in results['results'].items():
        if doc_path in receipt_handles:
            # Delete message
            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handles[doc_path]
            )
    
    # For failed documents, visibility timeout will expire and they'll be reprocessed
    
    return results
```

### 2. Creating a Dead Letter Queue for Failed Jobs

Implement a dead letter queue for handling persistently failing jobs:

```python
def setup_processing_queues():
    """Set up main queue and dead letter queue for document processing."""
    sqs = boto3.client('sqs')
    
    # Create dead letter queue first
    dlq_response = sqs.create_queue(
        QueueName="bedrock-document-processing-dlq",
        Attributes={
            'MessageRetentionPeriod': '1209600'  # 14 days retention
        }
    )
    dlq_url = dlq_response['QueueUrl']
    
    # Get DLQ ARN
    dlq_attrs = sqs.get_queue_attributes(
        QueueUrl=dlq_url,
        AttributeNames=['QueueArn']
    )
    dlq_arn = dlq_attrs['Attributes']['QueueArn']
    
    # Create main queue with redrive policy
    main_response = sqs.create_queue(
        QueueName="bedrock-document-processing-queue",
        Attributes={
            'VisibilityTimeout': '3600',  # 1 hour
            'MessageRetentionPeriod': '259200',  # 3 days
            'DelaySeconds': '0',
            'ReceiveMessageWaitTimeSeconds': '20',  # Long polling
            'RedrivePolicy': json.dumps({
                'deadLetterTargetArn': dlq_arn,
                'maxReceiveCount': '5'  # Move to DLQ after 5 failed attempts
            })
        }
    )
    main_queue_url = main_response['QueueUrl']
    
    return {
        "main_queue": main_queue_url,
        "dlq": dlq_url
    }

def process_dlq_messages(dlq_url):
    """Process messages from dead letter queue with manual intervention."""
    sqs = boto3.client('sqs')
    
    # Receive messages from DLQ
    response = sqs.receive_message(
        QueueUrl=dlq_url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=1,
        AttributeNames=['All'],
        MessageAttributeNames=['All']
    )
    
    if 'Messages' not in response:
        logger.info("No messages in DLQ")
        return []
    
    # Process failed messages
    failed_docs = []
    
    for message in response['Messages']:
        try:
            # Parse message
            message_body = json.loads(message['Body'])
            document_path = message_body['document_path']
            task_type = message_body['task_type']
            submitted_at = message_body.get('submitted_at', 'unknown')
            
            # Add to list with receipt handle for possible retry
            failed_docs.append({
                "document_path": document_path,
                "task_type": task_type,
                "submitted_at": submitted_at,
                "receipt_handle": message['ReceiptHandle'],
                "failure_count": int(message.get('Attributes', {}).get('ApproximateReceiveCount', '0'))
            })
            
        except Exception as e:
            logger.error(f"Error parsing DLQ message: {str(e)}")
    
    return failed_docs

def retry_failed_document(dlq_url, document_info, main_queue_url):
    """Retry a failed document by moving it from DLQ back to main queue."""
    sqs = boto3.client('sqs')
    
    # Create message with updated document information
    message = {
        "document_path": document_info["document_path"],
        "task_type": document_info["task_type"],
        "submitted_at": document_info["submitted_at"],
        "retried_at": datetime.now().isoformat(),
        "previous_failures": document_info["failure_count"]
    }
    
    # Send message to main queue
    response = sqs.send_message(
        QueueUrl=main_queue_url,
        MessageBody=json.dumps(message)
    )
    
    # Delete from DLQ
    sqs.delete_message(
        QueueUrl=dlq_url,
        ReceiptHandle=document_info["receipt_handle"]
    )
    
    return {
        "message_id": response['MessageId'],
        "document_path": document_info["document_path"],
        "task_type": document_info["task_type"],
        "previous_failures": document_info["failure_count"]
    }
```

### 3. Implementing a Progress Notification System

Keep users informed about job progress:

```python
def setup_job_notifications(sns_topic_name="bedrock-job-notifications"):
    """Set up SNS topic for job notifications."""
    sns = boto3.client('sns')
    
    # Create SNS topic
    response = sns.create_topic(Name=sns_topic_name)
    topic_arn = response['TopicArn']
    
    return topic_arn

def subscribe_to_notifications(topic_arn, email=None, sms=None):
    """Subscribe to job notifications via email or SMS."""
    sns = boto3.client('sns')
    
    if email:
        # Subscribe via email
        response = sns.subscribe(
            TopicArn=topic_arn,
            Protocol='email',
            Endpoint=email
        )
        return {
            "subscription_arn": response['SubscriptionArn'],
            "endpoint": email,
            "protocol": "email"
        }
    
    if sms:
        # Subscribe via SMS
        response = sns.subscribe(
            TopicArn=topic_arn,
            Protocol='sms',
            Endpoint=sms
        )
        return {
            "subscription_arn": response['SubscriptionArn'],
            "endpoint": sms,
            "protocol": "sms"
        }
    
    raise ValueError("Either email or sms must be provided")

def send_job_notification(topic_arn, job_id, status, details=None):
    """Send a notification about job status change."""
    sns = boto3.client('sns')
    
    # Create message
    message = {
        "job_id": job_id,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    
    if details:
        message["details"] = details
    
    # Define subject based on status
    if status == "COMPLETED":
        subject = f"Job {job_id} completed successfully"
    elif status == "FAILED":
        subject = f"Job {job_id} failed"
    else:
        subject = f"Job {job_id} status update: {status}"
    
    # Send notification
    response = sns.publish(
        TopicArn=topic_arn,
        Message=json.dumps(message, indent=2),
        Subject=subject
    )
    
    return response['MessageId']
```

## Key Takeaways

- Asynchronous processing is essential for handling large documents and high-volume workloads
- AWS Bedrock's asynchronous APIs use S3 for input and output data exchange
- Proper job monitoring and status tracking are crucial for production systems
- S3 permissions and IAM roles must be configured correctly for AWS Bedrock to access your data
- Batch processing with threading enables controlled concurrent execution
- Implementing robust error handling with exponential backoff makes your pipeline resilient
- Proper metrics tracking helps monitor system performance

---

**Next Steps**: Now that you understand asynchronous processing, learn about the [Converse API](/chapters/apis/converse/) for building conversational experiences with AWS Bedrock.

---

© 2025 Scott Friedman. Licensed under CC BY-NC-ND 4.0