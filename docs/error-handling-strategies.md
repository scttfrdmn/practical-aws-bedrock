# Error Handling and Retry Strategies for AWS Bedrock

Implementing robust error handling and retry mechanisms is critical when working with AWS Bedrock to ensure application reliability, maximize quota utilization, and provide a smooth user experience. This guide covers common error scenarios and best practices for handling them.

## Common AWS Bedrock Errors

| Error Type | Description | Common Causes | Retry? |
|------------|-------------|---------------|--------|
| `ThrottlingException` | Request rate exceeds quota limits | Exceeded TPM or RPM quota | Yes |
| `ValidationException` | Invalid request parameters | Malformed request, token limits exceeded | No |
| `AccessDeniedException` | Permission issues | IAM policy problems, invalid credentials | No |
| `ResourceNotFoundException` | Resource not found | Invalid model ID, job ID not found | No |
| `ServiceUnavailableException` | Temporary service issue | AWS service disruption | Yes |
| `InternalServerException` | Internal AWS error | Backend service failures | Yes |
| `ModelTimeoutException` | Model took too long to respond | Complex prompt, large generation | Yes |
| `ModelStreamErrorException` | Error during streaming | Stream interrupted | Sometimes |
| `ModelErrorException` | Model-specific errors | Content filtering, model limitations | No |
| `ConnectionError` | Network connectivity issues | Client network problems | Yes |

## Implementing Exponential Backoff with Jitter

Exponential backoff with jitter is the recommended approach for retrying recoverable errors:

```python
import random
import time

def retry_with_exponential_backoff(func, max_retries=3, base_delay=1.0, max_delay=60.0):
    """
    Execute a function with exponential backoff retry logic.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Result from the function
    """
    retries = 0
    
    while True:
        try:
            return func()
        except Exception as e:
            # Check if error is recoverable
            if not is_recoverable_error(e) or retries >= max_retries:
                raise  # Re-raise if not recoverable or max retries exceeded
            
            # Calculate backoff with jitter
            delay = min(max_delay, base_delay * (2 ** retries))
            jitter = delay * 0.2  # 20% jitter
            actual_delay = delay + random.uniform(-jitter, jitter)
            
            print(f"Retrying after error: {str(e)}. Attempt {retries+1}/{max_retries}. "
                  f"Waiting {actual_delay:.2f} seconds...")
            
            time.sleep(actual_delay)
            retries += 1

def is_recoverable_error(error):
    """Determine if an error is recoverable."""
    error_str = str(error)
    
    recoverable_errors = [
        "ThrottlingException",
        "ServiceUnavailableException", 
        "InternalServerException",
        "ModelTimeoutException",
        "ConnectionError",
        "RequestTimeout"
    ]
    
    return any(err_type in error_str for err_type in recoverable_errors)
```

## Token Bucket for Rate Limiting

Implementing a token bucket algorithm helps prevent throttling by controlling request rates:

```python
class TokenBucket:
    """Token bucket for rate limiting requests."""
    
    def __init__(self, capacity, refill_rate):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens the bucket can hold
            refill_rate: Tokens per second to refill
        """
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate)
        self.last_refill_time = time.time()
        self.lock = threading.RLock()
    
    def consume(self, tokens=1.0, block=True, timeout=None):
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            block: Whether to wait for tokens to be available
            timeout: Maximum time to wait
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            if not block:
                return False
            
            wait_time = (tokens - self.tokens) / self.refill_rate
            
            if timeout is not None and wait_time > timeout:
                return False
        
        # Wait outside the lock
        time.sleep(wait_time)
        
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        
        if elapsed > 0:
            new_tokens = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill_time = now
```

## Error Handling Patterns by Inference Type

### Synchronous Inference

```python
from aws_bedrock_inference import BedrockClient

client = BedrockClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

def invoke_with_retry(prompt, max_retries=3):
    def invoke_func():
        return client.invoke(prompt=prompt)
    
    try:
        # Use retry wrapper
        result = retry_with_exponential_backoff(
            invoke_func, 
            max_retries=max_retries
        )
        return result
    except Exception as e:
        # Handle non-recoverable errors
        if "ValidationException" in str(e):
            print(f"Invalid request: {str(e)}")
            # Maybe try with modified parameters
        elif "ModelErrorException" in str(e):
            print(f"Model error: {str(e)}")
            # Maybe try with different model
        else:
            print(f"Unexpected error: {str(e)}")
        return None
```

### Streaming Inference

```python
from aws_bedrock_inference import BedrockStreamingClient

client = BedrockStreamingClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

def stream_with_retry(prompt, max_retries=3):
    retries = 0
    
    while retries <= max_retries:
        content_buffer = []
        stream_complete = False
        
        try:
            for chunk in client.invoke_stream(prompt=prompt):
                content_buffer.append(chunk)
                yield chunk
            
            stream_complete = True
            break  # Success, exit retry loop
            
        except Exception as e:
            # Handle streaming-specific errors
            if not is_recoverable_error(e) or retries >= max_retries:
                # For non-recoverable or max retries, re-raise
                raise
            
            retries += 1
            delay = calculate_backoff(retries)
            
            print(f"Stream error: {str(e)}. Retrying ({retries}/{max_retries}) "
                  f"after {delay:.2f}s...")
            
            time.sleep(delay)
            
            # If we had partial content, inform caller
            if content_buffer:
                yield "\n[Stream was interrupted. Retrying...]\n"
    
    if not stream_complete and retries > max_retries:
        yield "\n[Maximum retry attempts reached. Stream incomplete.]\n"
```

### Asynchronous Processing

```python
from aws_bedrock_inference import BedrockJobClient

client = BedrockJobClient(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    output_s3_uri="s3://your-bucket/outputs/"
)

def create_job_with_retry(prompt, max_retries=3):
    def create_job_func():
        return client.create_job(prompt=prompt)
    
    try:
        # Use retry wrapper for job creation
        job_id = retry_with_exponential_backoff(
            create_job_func,
            max_retries=max_retries
        )
        
        # Wait for job with intelligent polling
        return wait_for_job_with_adaptive_polling(job_id)
        
    except Exception as e:
        print(f"Job creation failed: {str(e)}")
        return None

def wait_for_job_with_adaptive_polling(job_id, timeout=3600):
    """
    Wait for job with adaptive polling intervals.
    Starts with short intervals and increases them over time.
    """
    start_time = time.time()
    poll_interval = 1.0  # Start with 1 second
    max_interval = 30.0  # Max 30 seconds between polls
    
    while time.time() - start_time < timeout:
        try:
            status = client.get_job_status(job_id)
            
            if status["status"] == "COMPLETED":
                return client.get_job_result(job_id)
            elif status["status"] in ["FAILED", "STOPPED"]:
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            
            # Adaptive polling - gradually increase interval
            poll_interval = min(poll_interval * 1.5, max_interval)
            time.sleep(poll_interval)
            
        except Exception as e:
            if "ResourceNotFoundException" in str(e):
                raise  # Job not found, don't retry
            
            # For other errors, retry with backoff
            poll_interval = min(poll_interval * 2, max_interval)
            time.sleep(poll_interval)
    
    raise TimeoutError(f"Timeout waiting for job {job_id} to complete")
```

## Handling Content Filter Errors

AWS Bedrock models have content filtering that may reject certain prompts. Handle these gracefully:

```python
def safe_invoke(prompt, fallback_message=None):
    try:
        return client.invoke(prompt=prompt)
    except Exception as e:
        if "content filtering" in str(e).lower() or "content filter" in str(e).lower():
            # Handle content filtering case
            return {
                "output": fallback_message or "I'm unable to provide a response to that prompt due to content restrictions."
            }
        else:
            # Re-raise other errors
            raise
```

## Circuit Breaker Pattern

Implement a circuit breaker to prevent cascading failures:

```python
class CircuitBreaker:
    """
    Circuit breaker for API calls.
    Prevents repeated calls to failing services.
    """
    
    # States
    CLOSED = 'closed'  # Normal operation
    OPEN = 'open'      # Not allowing calls
    HALF_OPEN = 'half-open'  # Testing if service is healthy
    
    def __init__(self, failure_threshold=5, recovery_timeout=30, 
                 retry_timeout=60, failure_window=60):
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.retry_timeout = retry_timeout
        self.last_failure_time = 0
        self.last_retry_time = 0
        self.failure_window = failure_window
        self.lock = threading.RLock()
    
    def execute(self, func):
        """Execute function with circuit breaker protection."""
        with self.lock:
            # Clear old failures outside the window
            self._clear_old_failures()
            
            if self.state == self.OPEN:
                # Check if we should try to recover
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = self.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is open")
        
        # Execute in CLOSED or HALF_OPEN state
        try:
            result = func()
            
            # If successful in HALF_OPEN, close the circuit
            with self.lock:
                if self.state == self.HALF_OPEN:
                    self.state = self.CLOSED
                    self.failure_count = 0
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # If we hit threshold, open the circuit
                if (self.state == self.CLOSED and 
                    self.failure_count >= self.failure_threshold):
                    self.state = self.OPEN
                
                # If test call in HALF_OPEN fails, reopen the circuit
                if self.state == self.HALF_OPEN:
                    self.state = self.OPEN
            
            # Re-raise the original exception
            raise
    
    def _clear_old_failures(self):
        """Clear failures outside the window."""
        current_time = time.time()
        if (self.failure_count > 0 and 
            current_time - self.last_failure_time > self.failure_window):
            self.failure_count = 0

class CircuitBreakerOpenError(Exception):
    """Error raised when circuit is open."""
    pass
```

## Error Logging and Monitoring

Implement structured logging to track errors and identify patterns:

```python
import logging
import json

def setup_structured_logging():
    """Set up structured JSON logging."""
    logger = logging.getLogger("bedrock_client")
    logger.setLevel(logging.INFO)
    
    # Add console handler
    handler = logging.StreamHandler()
    
    # Define JSON formatter
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name
            }
            
            # Add exception info if present
            if record.exc_info:
                log_record["exception"] = self.formatException(record.exc_info)
            
            # Add extra fields
            if hasattr(record, 'data'):
                log_record.update(record.data)
                
            return json.dumps(log_record)
    
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    
    return logger

# Example usage
logger = setup_structured_logging()

def log_bedrock_error(error, context=None):
    """Log a Bedrock error with context."""
    error_type = "unknown_error"
    
    # Classify error
    error_str = str(error)
    if "ThrottlingException" in error_str:
        error_type = "throttling"
    elif "ValidationException" in error_str:
        error_type = "validation"
    # Add more classifications...
    
    # Create structured log
    log_data = {
        "error_type": error_type,
        "error_message": error_str,
        "service": "bedrock",
        "data": context or {}
    }
    
    # Log with structured data
    logger.error("Bedrock API error", extra={"data": log_data})
```

## Implementing Dead Letter Queues for Failed Requests

For production applications, implement a dead letter queue for failed requests:

```python
import boto3

class DeadLetterQueue:
    """
    Store failed requests for later analysis or reprocessing.
    """
    
    def __init__(self, table_name="BedrockFailedRequests"):
        """
        Initialize with DynamoDB table for storing failed requests.
        
        Args:
            table_name: DynamoDB table name
        """
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)
    
    def store_failed_request(self, request_data, error, metadata=None):
        """
        Store a failed request.
        
        Args:
            request_data: Original request data
            error: Error information
            metadata: Additional metadata
        """
        item = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'request': request_data,
            'error': str(error),
            'metadata': metadata or {}
        }
        
        self.table.put_item(Item=item)
        return item['id']
    
    def get_failed_requests(self, limit=100):
        """
        Retrieve failed requests.
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of failed request items
        """
        response = self.table.scan(Limit=limit)
        return response.get('Items', [])
    
    def retry_request(self, request_id, client):
        """
        Retry a failed request.
        
        Args:
            request_id: ID of the failed request
            client: Bedrock client to use
            
        Returns:
            Result of the retry attempt
        """
        response = self.table.get_item(Key={'id': request_id})
        if 'Item' not in response:
            raise ValueError(f"Request {request_id} not found")
        
        item = response['Item']
        request_data = item['request']
        
        # Retry the request
        result = client.invoke(**request_data)
        
        # Update the record with retry result
        self.table.update_item(
            Key={'id': request_id},
            UpdateExpression="SET retried = :retried, retry_result = :result",
            ExpressionAttributeValues={
                ':retried': datetime.datetime.utcnow().isoformat(),
                ':result': 'success'
            }
        )
        
        return result
```

## Error Handling Best Practices

1. **Categorize errors properly**: Distinguish between recoverable and non-recoverable errors
2. **Use exponential backoff**: Increase delay between retries
3. **Add jitter**: Prevent thundering herd problems with randomization
4. **Set maximum retries**: Don't retry indefinitely
5. **Implement circuit breakers**: Fail fast when a service is experiencing issues
6. **Rate limit requests**: Stay within quota limits
7. **Log detailed error information**: Include context for troubleshooting
8. **Implement dead letter queues**: Store failed requests for analysis
9. **Provide graceful degradation**: Have fallback options when possible
10. **Monitor error rates**: Set up alerting for unusual error patterns

## Quota Optimization During Error Handling

1. **Keep track of quota usage**: Monitor TPM and RPM consumption
2. **Prioritize requests**: Implement queue prioritization for critical operations
3. **Adaptive rate limiting**: Adjust request rates based on observed throttling
4. **Backoff when throttled**: Exponentially increase delays when hitting limits
5. **Cache common responses**: Avoid redundant API calls

## Next Steps

- Explore the [Quota Management](quota-management.md) guide for more on managing limits
- See [Implementation Examples](../src/examples) for complete code examples
- Learn about [Advanced Retry Patterns](advanced-retry-patterns.md) for complex scenarios