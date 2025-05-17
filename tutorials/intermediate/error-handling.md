---
layout: page
title: Error Handling and Retry Strategies for AWS Bedrock
difficulty: intermediate
time_required: 30 minutes
---

# Error Handling and Retry Strategies for AWS Bedrock

This tutorial explores robust error handling and retry strategies for AWS Bedrock, with a focus on managing throttling and quota limitations.

## Objective

By the end of this tutorial, you'll be able to:
1. Identify and handle different types of AWS Bedrock errors
2. Implement effective retry strategies with exponential backoff
3. Create a robust error handling framework for production applications
4. Optimize retry behavior around quota limitations

## Prerequisites

- Understanding of AWS Bedrock and its quota system
- Familiarity with Python error handling concepts
- Basic understanding of boto3 and AWS SDK error patterns

## Understanding AWS Bedrock Error Types

AWS Bedrock can return several types of errors that require different handling strategies:

### 1. Throttling Errors

These occur when you exceed your quota limits:

- **ThrottlingException** - You've exceeded RPM or TPM quotas
- **TooManyRequestsException** - Too many concurrent requests
- **ServiceQuotaExceededException** - Explicit quota limit exceeded

### 2. Validation Errors

These indicate issues with your request format:

- **ValidationException** - Invalid request structure or parameters
- **InvalidRequestException** - Malformed request
- **ModelNotReadyException** - Model is not ready for inference

### 3. Service Errors

These represent issues on the AWS side:

- **ServiceUnavailableException** - Temporary service unavailability
- **InternalServerException** - Internal error in the AWS service
- **ServiceException** - General service error

### 4. Authentication/Authorization Errors

These indicate permission issues:

- **AccessDeniedException** - Insufficient permissions
- **UnauthorizedException** - Invalid credentials
- **ResourceNotFoundException** - Specified resource does not exist

## Step 1: Basic Error Handling Structure

Let's start with a basic error handling structure for AWS Bedrock:

```python
import boto3
import json
import logging
from botocore.exceptions import ClientError
from utils.profile_manager import get_profile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def invoke_with_basic_error_handling(model_id, prompt_data):
    """
    Invoke a model with basic error handling.
    
    Args:
        model_id: The model identifier
        prompt_data: Dictionary with the prompt payload
        
    Returns:
        Model response or None if an error occurred
    """
    # Use the configured profile (defaults to 'aws' for local testing)
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime')
    
    try:
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(prompt_data)
        )
        
        # Process the response
        response_body = json.loads(response['body'].read())
        return response_body
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        if error_code == 'ThrottlingException':
            logger.warning(f"Request throttled: {error_message}")
        elif error_code == 'ValidationException':
            logger.error(f"Validation error: {error_message}")
        elif error_code == 'ServiceUnavailableException':
            logger.warning(f"Service unavailable: {error_message}")
        elif error_code == 'AccessDeniedException':
            logger.error(f"Access denied: {error_message}")
        else:
            logger.error(f"Error invoking model: {error_code} - {error_message}")
        
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None
```

## Step 2: Implementing Exponential Backoff with Jitter

For throttling errors, implement exponential backoff with jitter to spread out retry attempts:

```python
import random
import time

def invoke_with_backoff(model_id, prompt_data, max_retries=5, base_delay=1.0):
    """
    Invoke a model with exponential backoff retry strategy.
    
    Args:
        model_id: The model identifier
        prompt_data: Dictionary with the prompt payload
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        
    Returns:
        Model response or None if all retries failed
    """
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime')
    
    # Retry loop
    retries = 0
    while retries <= max_retries:
        try:
            # Invoke the model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(prompt_data)
            )
            
            # Process the response
            response_body = json.loads(response['body'].read())
            return response_body
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            # Only retry on throttling or temporary service errors
            if error_code in ['ThrottlingException', 'TooManyRequestsException', 
                             'ServiceUnavailableException', 'ServiceException']:
                
                retries += 1
                
                if retries > max_retries:
                    logger.warning(f"Maximum retries ({max_retries}) exceeded. Giving up.")
                    return None
                
                # Calculate backoff delay with jitter
                delay = base_delay * (2 ** (retries - 1))  # Exponential backoff
                jitter = delay * 0.2 * random.random()     # 20% jitter
                sleep_time = delay + jitter
                
                logger.info(f"Throttled, retrying in {sleep_time:.2f}s (attempt {retries}/{max_retries})")
                time.sleep(sleep_time)
                
            else:
                # Non-retryable error
                logger.error(f"Non-retryable error: {error_code} - {error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None
    
    return None  # Should not reach here, but just in case
```

## Step 3: Creating an Error Classification System

Let's create a more sophisticated error classification system:

```python
class BedrockErrorClassifier:
    """
    Classifies AWS Bedrock errors and provides handling recommendations.
    """
    
    # Error categories
    RETRYABLE = "retryable"       # Can be retried
    NON_RETRYABLE = "non-retryable"  # Should not be retried
    THROTTLING = "throttling"     # Throttling-specific errors
    VALIDATION = "validation"     # Input validation errors
    AUTHENTICATION = "authentication"  # Auth/permission errors
    SERVICE = "service"           # AWS service errors
    
    # Error codes mapped to categories
    ERROR_CATEGORIES = {
        # Throttling errors
        "ThrottlingException": THROTTLING,
        "TooManyRequestsException": THROTTLING,
        "ServiceQuotaExceededException": THROTTLING,
        
        # Validation errors
        "ValidationException": VALIDATION,
        "InvalidRequestException": VALIDATION,
        "ModelNotReadyException": VALIDATION,
        
        # Service errors
        "ServiceUnavailableException": SERVICE,
        "InternalServerException": SERVICE,
        "ServiceException": SERVICE,
        
        # Auth errors
        "AccessDeniedException": AUTHENTICATION,
        "UnauthorizedException": AUTHENTICATION,
        "ResourceNotFoundException": AUTHENTICATION
    }
    
    # Errors that should be retried
    RETRYABLE_CATEGORIES = {THROTTLING, SERVICE}
    
    @classmethod
    def classify(cls, error):
        """
        Classify a boto3 ClientError.
        
        Args:
            error: The boto3 ClientError
            
        Returns:
            Tuple of (error_code, category, is_retryable)
        """
        if not isinstance(error, ClientError):
            return "UnknownError", "unknown", False
        
        error_code = error.response['Error']['Code']
        category = cls.ERROR_CATEGORIES.get(error_code, "unknown")
        is_retryable = category in cls.RETRYABLE_CATEGORIES
        
        return error_code, category, is_retryable
    
    @classmethod
    def get_retry_strategy(cls, error, attempt=0):
        """
        Get recommended retry strategy for an error.
        
        Args:
            error: The boto3 ClientError
            attempt: Current retry attempt (0-based)
            
        Returns:
            Dictionary with retry recommendations
        """
        error_code, category, is_retryable = cls.classify(error)
        
        if not is_retryable:
            return {
                "should_retry": False,
                "reason": f"Non-retryable error category: {category}"
            }
        
        # Base delay for different categories
        if category == cls.THROTTLING:
            base_delay = 1.0  # Start with 1 second for throttling
        elif category == cls.SERVICE:
            base_delay = 2.0  # Start with 2 seconds for service errors
        else:
            base_delay = 0.5  # Default for other retryable errors
        
        # Calculate delay with exponential backoff and jitter
        delay = base_delay * (2 ** attempt)
        jitter = delay * 0.2 * random.random()  # 20% jitter
        retry_delay = min(delay + jitter, 60)  # Cap at 60 seconds
        
        return {
            "should_retry": True,
            "retry_delay": retry_delay,
            "category": category,
            "error_code": error_code
        }
```

## Step 4: Building a Comprehensive Retry Framework

Now let's create a comprehensive retry framework that's quota-aware:

```python
class BedrockRetryer:
    """
    A comprehensive retry framework for AWS Bedrock operations.
    """
    
    def __init__(self, max_retries=5, respect_quota=True):
        """
        Initialize the retriever.
        
        Args:
            max_retries: Maximum number of retry attempts
            respect_quota: Whether to be quota-aware in retry strategy
        """
        self.max_retries = max_retries
        self.respect_quota = respect_quota
        self.error_classifier = BedrockErrorClassifier
        
        # Quota tracking (simplified)
        self.throttling_events = 0
        self.last_throttle_time = 0
    
    def execute(self, operation, *args, **kwargs):
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Function to execute
            *args, **kwargs: Arguments to pass to the operation
            
        Returns:
            Result of the operation or None on failure
        """
        attempt = 0
        
        while attempt <= self.max_retries:
            try:
                # Check if we should apply quota-aware delay
                if self.respect_quota and self.throttling_events > 0:
                    self._apply_quota_aware_delay()
                
                # Execute the operation
                result = operation(*args, **kwargs)
                
                # Success - reset throttling counter
                self.throttling_events = max(0, self.throttling_events - 1)
                
                return result
                
            except ClientError as e:
                # Classify the error
                error_code, category, is_retryable = self.error_classifier.classify(e)
                
                # Log the error
                logger.warning(f"AWS error: {error_code} ({category}) - {e.response['Error']['Message']}")
                
                # Update throttling metrics if relevant
                if category == BedrockErrorClassifier.THROTTLING:
                    self.throttling_events += 1
                    self.last_throttle_time = time.time()
                
                # Check if we should retry
                attempt += 1
                if not is_retryable or attempt > self.max_retries:
                    logger.error(f"Not retrying: {'Max retries exceeded' if is_retryable else 'Non-retryable error'}")
                    return None
                
                # Get retry strategy
                retry_strategy = self.error_classifier.get_retry_strategy(e, attempt - 1)
                
                # Apply the retry delay
                delay = retry_strategy["retry_delay"]
                logger.info(f"Retrying in {delay:.2f}s (attempt {attempt}/{self.max_retries})")
                time.sleep(delay)
                
            except Exception as e:
                # Unexpected error
                logger.error(f"Unexpected error: {str(e)}")
                return None
        
        return None  # Should not reach here
    
    def _apply_quota_aware_delay(self):
        """Apply an additional delay based on recent throttling events"""
        now = time.time()
        time_since_last_throttle = now - self.last_throttle_time
        
        # If we've had throttling recently
        if time_since_last_throttle < 60 and self.throttling_events > 1:
            # Calculate a progressive delay based on throttling frequency
            adaptive_delay = min(1.0 * self.throttling_events, 5.0)
            logger.info(f"Adding quota-aware delay of {adaptive_delay:.2f}s")
            time.sleep(adaptive_delay)
```

## Step 5: Using the Framework with AWS Bedrock

Now let's put it all together with AWS Bedrock:

```python
def create_bedrock_client(profile_name=None):
    """Create a boto3 client for Bedrock with the specified profile"""
    profile = profile_name or get_profile()
    session = boto3.Session(profile_name=profile)
    return session.client('bedrock-runtime')

def invoke_model_with_retries(model_id, prompt_data, max_retries=5):
    """
    Invoke an AWS Bedrock model with robust retry handling.
    
    Args:
        model_id: The model identifier
        prompt_data: Dictionary with the prompt payload
        max_retries: Maximum retry attempts
        
    Returns:
        Model response or None on failure
    """
    # Create the client
    client = create_bedrock_client()
    
    # Create the retriever
    retriever = BedrockRetryer(max_retries=max_retries)
    
    # Define the operation to retry
    def invoke_operation():
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(prompt_data)
        )
        return json.loads(response['body'].read())
    
    # Execute with retry logic
    return retriever.execute(invoke_operation)
```

## Step 6: Handling Streaming Responses

Streaming responses require special error handling:

```python
def invoke_streaming_with_retries(model_id, prompt_data, max_retries=5):
    """
    Invoke a streaming AWS Bedrock model with retry handling.
    
    Args:
        model_id: The model identifier
        prompt_data: Dictionary with the prompt payload
        max_retries: Maximum retry attempts
        
    Returns:
        Generator yielding response chunks or None on failure
    """
    client = create_bedrock_client()
    retriever = BedrockRetryer(max_retries=max_retries)
    
    def start_stream():
        """Initiate the streaming response"""
        response = client.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(prompt_data)
        )
        return response.get('body')
    
    # Get the stream
    stream = retriever.execute(start_stream)
    
    if not stream:
        logger.error("Failed to initiate streaming response")
        return None
    
    # Process the stream with error handling
    try:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                yield json.loads(chunk.get('bytes').decode())
    except Exception as e:
        logger.error(f"Error processing stream: {str(e)}")
        # Stream errors are not retryable once the stream has started
        return None
```

## Step 7: Quota-Aware Request Batching

For high-throughput scenarios, implement quota-aware batching:

```python
class BedrockBatchProcessor:
    """
    Process batches of requests with quota awareness.
    """
    
    def __init__(self, model_id, requests_per_minute=60, tokens_per_minute=10000):
        """
        Initialize the batch processor.
        
        Args:
            model_id: The model identifier
            requests_per_minute: RPM quota limit
            tokens_per_minute: TPM quota limit
        """
        self.model_id = model_id
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.client = create_bedrock_client()
        self.retriever = BedrockRetryer(max_retries=3)
        
        # Token tracking
        self.estimated_tokens_used = 0
        self.requests_made = 0
        self.window_start_time = time.time()
    
    def process_batch(self, prompts, token_estimator=None):
        """
        Process a batch of prompts with quota awareness.
        
        Args:
            prompts: List of prompt payloads
            token_estimator: Optional function to estimate tokens in a prompt
            
        Returns:
            List of results (or None for failed requests)
        """
        results = []
        
        # Reset tracking at the start of a batch
        self._reset_if_window_expired()
        
        for prompt in prompts:
            # Check if we should wait before proceeding
            self._apply_rate_limiting(prompt, token_estimator)
            
            # Process the individual request
            result = self._process_single_request(prompt)
            results.append(result)
        
        return results
    
    def _process_single_request(self, prompt):
        """Process a single request with retries"""
        def invoke_operation():
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(prompt)
            )
            result = json.loads(response['body'].read())
            
            # Update token tracking
            if "anthropic" in self.model_id and "usage" in result:
                self.estimated_tokens_used += result["usage"]["input_tokens"]
                self.estimated_tokens_used += result["usage"]["output_tokens"]
            
            return result
        
        # Track the request
        self.requests_made += 1
        
        # Execute with retry logic
        return self.retriever.execute(invoke_operation)
    
    def _reset_if_window_expired(self):
        """Reset tracking if the current minute window has expired"""
        now = time.time()
        seconds_elapsed = now - self.window_start_time
        
        if seconds_elapsed >= 60:
            logger.info(f"Resetting quota window. Previous window: {self.requests_made} requests, "
                      f"~{self.estimated_tokens_used} tokens")
            self.window_start_time = now
            self.requests_made = 0
            self.estimated_tokens_used = 0
    
    def _apply_rate_limiting(self, prompt, token_estimator):
        """Apply rate limiting based on quota usage"""
        # Estimate tokens in the prompt
        estimated_request_tokens = 0
        if token_estimator:
            estimated_request_tokens = token_estimator(prompt)
        else:
            # Rough estimation if no estimator provided
            prompt_str = json.dumps(prompt)
            estimated_request_tokens = len(prompt_str.split()) * 1.3
        
        # Check if we're approaching RPM limit
        rpm_utilization = self.requests_made / self.rpm_limit
        
        # Check if we're approaching TPM limit
        tpm_utilization = (self.estimated_tokens_used + estimated_request_tokens) / self.tpm_limit
        
        # Use the higher utilization to determine delay
        utilization = max(rpm_utilization, tpm_utilization)
        
        if utilization > 0.9:
            # We're at >90% of quota, wait until next window
            seconds_in_window = time.time() - self.window_start_time
            seconds_to_wait = max(0, 60 - seconds_in_window)
            
            logger.info(f"Approaching quota limit ({utilization:.1%} utilized), "
                      f"waiting {seconds_to_wait:.1f}s for next window")
            
            if seconds_to_wait > 0:
                time.sleep(seconds_to_wait)
                self._reset_if_window_expired()
                
        elif utilization > 0.7:
            # We're at >70% of quota, add some delay to spread requests
            delay = utilization * 0.5  # Up to 0.5s delay at 100% utilization
            logger.info(f"Spreading requests ({utilization:.1%} utilized), adding {delay:.2f}s delay")
            time.sleep(delay)
```

## Step 8: Error Response Interpretation

Different models return errors in different formats. Let's handle this:

```python
def interpret_model_error(model_id, response):
    """
    Interpret model-specific error responses.
    
    Args:
        model_id: The model identifier
        response: The error response from the model
        
    Returns:
        Dictionary with error details
    """
    # Default error info
    error_info = {
        "error_type": "unknown",
        "message": "Unknown error",
        "is_retryable": False
    }
    
    try:
        if "anthropic" in model_id:
            if "error" in response:
                error = response["error"]
                error_info["error_type"] = error.get("type", "unknown")
                error_info["message"] = error.get("message", "Unknown error")
                
                # Anthropic-specific error types
                if error_info["error_type"] in ["rate_limit_exceeded", "service_unavailable"]:
                    error_info["is_retryable"] = True
        
        elif "meta" in model_id or "llama" in model_id:
            if "error" in response:
                error_info["error_type"] = "model_error"
                error_info["message"] = response["error"]
                
                # Check for retryable phrases
                retryable_phrases = ["rate limit", "capacity", "try again", "temporarily"]
                if any(phrase in response["error"].lower() for phrase in retryable_phrases):
                    error_info["is_retryable"] = True
        
        elif "ai21" in model_id:
            if "error" in response:
                error_info["error_type"] = response["error"].get("code", "unknown")
                error_info["message"] = response["error"].get("message", "Unknown error")
                
                # AI21-specific error types
                if error_info["error_type"] in ["throttling", "service_unavailable"]:
                    error_info["is_retryable"] = True
        
        # Add more model-specific error handling as needed
        
    except Exception as e:
        logger.error(f"Error interpreting model response: {str(e)}")
    
    return error_info
```

## Step 9: Comprehensive Error Monitoring

For production applications, implement monitoring:

```python
class BedrockErrorMonitor:
    """
    Monitor and track errors for AWS Bedrock operations.
    """
    
    def __init__(self):
        """Initialize the error monitor"""
        self.error_counts = {
            BedrockErrorClassifier.THROTTLING: 0,
            BedrockErrorClassifier.VALIDATION: 0,
            BedrockErrorClassifier.SERVICE: 0,
            BedrockErrorClassifier.AUTHENTICATION: 0,
            "unknown": 0
        }
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.retry_counts = []  # Number of retries needed for each request
        
        # Time tracking
        self.start_time = time.time()
    
    def record_request(self, success, retries=0):
        """Record a request result"""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.retry_counts.append(retries)
    
    def record_error(self, error):
        """Record an error"""
        _, category, _ = BedrockErrorClassifier.classify(error)
        self.error_counts[category] = self.error_counts.get(category, 0) + 1
    
    def get_stats(self):
        """Get current statistics"""
        elapsed_time = time.time() - self.start_time
        minutes = elapsed_time / 60
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "error_counts": self.error_counts,
            "requests_per_minute": self.total_requests / minutes if minutes > 0 else 0,
            "average_retries": sum(self.retry_counts) / len(self.retry_counts) if self.retry_counts else 0,
            "elapsed_minutes": minutes
        }
    
    def log_stats(self):
        """Log the current statistics"""
        stats = self.get_stats()
        
        logger.info(f"=== Bedrock Error Monitor Statistics ===")
        logger.info(f"Total requests: {stats['total_requests']}")
        logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"Requests per minute: {stats['requests_per_minute']:.1f}")
        logger.info(f"Average retries: {stats['average_retries']:.2f}")
        logger.info(f"Error counts:")
        
        for category, count in stats['error_counts'].items():
            if count > 0:
                logger.info(f"  - {category}: {count}")
        
        logger.info(f"========================================")
        
        return stats
```

## Error Handling Best Practices

### For Throttling Errors

1. **Implement exponential backoff** - Increase delay between retries exponentially
2. **Add jitter** - Randomize delay times to prevent retry storms
3. **Track throttling frequency** - Adjust strategy based on recent throttling history
4. **Pre-emptively rate limit** - Stay under quota limits by self-limiting
5. **Monitor TPM and RPM** - Track both metrics to identify the limiting factor

### For Validation Errors

1. **Validate requests client-side** - Check input before sending to the API
2. **Log validation errors in detail** - Include the specific validation issue
3. **Don't retry validation errors** - These generally won't succeed on retry
4. **Add unit tests for request formats** - Ensure your request format is valid

### For Service Errors

1. **Implement retries with increasing backoff** - Services often recover
2. **Add circuit breaker** - Stop retrying after persistent failures
3. **Log service errors for diagnosis** - Help identify patterns or regional issues
4. **Consider fallback services** - Have a backup plan for critical operations

### For Authentication Errors

1. **Validate credentials early** - Test authentication at startup
2. **Don't retry auth errors** - These generally require human intervention
3. **Implement secure credential handling** - Use AWS best practices
4. **Log auth errors at high priority** - These need immediate attention

## Conclusion

Effective error handling is essential for robust AWS Bedrock applications, especially when working within quota limits. By implementing proper error classification, intelligent retry strategies, and quota-aware processing, you can maximize throughput while gracefully handling temporary service limitations.

The approaches demonstrated in this tutorial can be combined with the quota optimization techniques from previous tutorials to create highly resilient AWS Bedrock applications that make the most of available resources.

## Next Steps

- Implement a complete error monitoring dashboard
- Integrate these strategies with the throughput optimization techniques
- Explore adaptive quota management based on error patterns
- Develop model-specific error handling strategies