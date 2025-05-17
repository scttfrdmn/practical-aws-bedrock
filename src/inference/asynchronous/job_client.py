"""
Asynchronous job client for AWS Bedrock inference.

This module provides a client for working with AWS Bedrock's asynchronous
job processing capabilities, allowing for longer running inference jobs and
higher throughput for batch processing.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple

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
    
    def list_jobs(
        self, 
        status_filter: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List model invocation jobs.
        
        Args:
            status_filter: Optional status to filter by
            max_results: Maximum number of results to return
            
        Returns:
            List of job information dictionaries
        """
        try:
            # Prepare request parameters
            request_params = {
                "maxResults": max_results
            }
            
            if status_filter:
                request_params["statusEquals"] = status_filter
            
            # Make request
            response = self.client.list_model_invocation_jobs(**request_params)
            
            # Parse results
            jobs = []
            for job in response.get("modelInvocationJobs", []):
                job_info = {
                    "job_id": job.get("jobArn").split("/")[-1],
                    "status": job.get("status"),
                    "created_at": job.get("creationTime"),
                    "model_id": job.get("modelId"),
                    "job_name": job.get("jobName", "")
                }
                
                jobs.append(job_info)
            
            return jobs
            
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            
            raise RuntimeError(f"Failed to list jobs: {error_code} - {error_message}")
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job that is in progress.
        
        Args:
            job_id: The job ID
            
        Returns:
            True if successful, False if failed
        """
        try:
            self.client.stop_model_invocation_job(
                jobIdentifier=job_id
            )
            
            self.logger.info(f"Cancelled job {job_id}")
            return True
            
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            
            self.logger.warning(f"Failed to cancel job {job_id}: {error_code} - {error_message}")
            return False
    
    def _create_request_body(
        self, 
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        other_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create the appropriate request body for the model family.
        
        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            system_prompt: Optional system prompt
            other_params: Additional model-specific parameters
            
        Returns:
            Model-specific request body
        """
        # Extract model family from model ID
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        if "anthropic" in model_family:
            # Claude models
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
        elif "meta" in model_family or "llama" in model_family:
            # Llama models
            if system_prompt:
                formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            else:
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            body = {
                "prompt": formatted_prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature
            }
            
        elif "cohere" in model_family:
            # Cohere models
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            if system_prompt:
                body["preamble"] = system_prompt
                
        elif "ai21" in model_family:
            # AI21 models
            body = {
                "prompt": prompt,
                "maxTokens": max_tokens,
                "temperature": temperature
            }
            
        else:
            # Default to Amazon Titan format
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            }
        
        # Add any additional parameters
        body.update(other_params)
        
        return body
    
    def _parse_job_result(self, result_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the job result based on model family.
        
        Args:
            result_json: The raw JSON result from S3
            
        Returns:
            Parsed result dictionary
        """
        # Extract model family from model ID
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        # Initialize result with raw response
        parsed_result = {
            "raw_response": result_json
        }
        
        if "anthropic" in model_family:
            # Claude models
            if 'content' in result_json and len(result_json['content']) > 0:
                parsed_result["output"] = result_json['content'][0]['text']
            else:
                parsed_result["output"] = ""
                
            if 'usage' in result_json:
                parsed_result["input_tokens"] = result_json['usage'].get('input_tokens', 0)
                parsed_result["output_tokens"] = result_json['usage'].get('output_tokens', 0)
                parsed_result["total_tokens"] = parsed_result["input_tokens"] + parsed_result["output_tokens"]
            
        elif "meta" in model_family or "llama" in model_family:
            # Llama models
            parsed_result["output"] = result_json.get('generation', '')
            
        elif "cohere" in model_family:
            # Cohere models
            parsed_result["output"] = result_json.get('text', '')
            
            if 'meta' in result_json:
                parsed_result["input_tokens"] = result_json['meta'].get('prompt_tokens', 0)
                parsed_result["output_tokens"] = result_json['meta'].get('response_tokens', 0)
                parsed_result["total_tokens"] = parsed_result["input_tokens"] + parsed_result["output_tokens"]
            
        elif "ai21" in model_family:
            # AI21 models
            if 'completions' in result_json and len(result_json['completions']) > 0:
                parsed_result["output"] = result_json['completions'][0].get('data', {}).get('text', '')
            else:
                parsed_result["output"] = ""
                
            if 'usage' in result_json:
                parsed_result["input_tokens"] = result_json['usage'].get('input_tokens', 0)
                parsed_result["output_tokens"] = result_json['usage'].get('output_tokens', 0)
                parsed_result["total_tokens"] = parsed_result["input_tokens"] + parsed_result["output_tokens"]
            
        else:
            # Default extraction (Amazon Titan)
            if 'results' in result_json and len(result_json['results']) > 0:
                parsed_result["output"] = result_json['results'][0].get('outputText', '')
            else:
                parsed_result["output"] = ""
        
        return parsed_result
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff time using exponential backoff with jitter.
        
        Args:
            attempt: The current attempt number (0-indexed)
            
        Returns:
            Backoff time in seconds
        """
        import random
        
        # Calculate exponential backoff: base * 2^attempt
        backoff = self.base_backoff * (2 ** attempt)
        
        # Add jitter (±20%)
        jitter = backoff * 0.2
        backoff = backoff + random.uniform(-jitter, jitter)
        
        return backoff
    
    def get_metrics(self) -> Dict[str, int]:
        """
        Get usage metrics for this client instance.
        
        Returns:
            Dictionary with usage metrics
        """
        return {
            "job_count": self.job_count,
            "completed_job_count": self.completed_job_count,
            "failed_job_count": self.failed_job_count
        }


# Example usage (requires valid S3 bucket with appropriate permissions)
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a job client for Claude
    # Note: You need a valid S3 bucket with appropriate permissions
    s3_output_uri = "s3://your-bucket/bedrock-outputs/"
    
    client = BedrockJobClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        output_s3_uri=s3_output_uri
    )
    
    try:
        # Create a job
        prompt = "Explain quantum computing in simple terms."
        
        job_id = client.create_job(
            prompt=prompt,
            max_tokens=500,
            job_name="Quantum-Computing-Explanation"
        )
        
        print(f"Created job: {job_id}")
        
        # Wait for job to complete and get result
        print("Waiting for job to complete...")
        result = client.wait_for_job(job_id)
        
        print("\nJob Result:")
        print(result.get("output", "No output"))
        
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