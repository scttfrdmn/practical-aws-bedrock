# Maximizing Throughput: Balancing Token and Request Quotas

> "You've built an amazing AI application with AWS Bedrock, but now it's failing under load. Let's fix that together."

## The Problem

---

**Scenario**: Your team has built a document processing system using AWS Bedrock to analyze thousands of legal contracts daily. The application was working perfectly in testing, but after deploying to production, users are reporting intermittent failures and slow response times during busy periods.

You check your CloudWatch metrics and see a pattern of `ThrottlingException` errors that correlate with peak usage hours. Some documents process quickly while others time out, creating an inconsistent user experience that's frustrating your customers.

After digging into the AWS Bedrock quotas for your account, you realize you're hitting both the **Requests Per Minute (RPM)** and **Tokens Per Minute (TPM)** limits, but in an uneven way. You need to find a solution that maximizes throughput while providing a consistent, reliable experience.

**Key Challenges**:
- How to predict and prevent throttling errors before they impact users
- How to make optimal use of both RPM and TPM quotas simultaneously
- How to handle documents of varying sizes efficiently
- How to prioritize critical processing when near quota limits

---

## Key Concepts Explained

Think of AWS Bedrock quotas like a highway with two different types of toll booths that every request must pass through:

1. **The RPM Booth**: Counts the number of vehicles (requests) passing through
2. **The TPM Booth**: Weighs the total cargo (tokens) being transported

Your traffic can get stopped at either booth if you exceed its limits. The tricky part? These limits are connected in ways that aren't always obvious.

### The Quota Relationship

Imagine you have:
- 100 RPM (requests per minute) quota
- 10,000 TPM (tokens per minute) quota

This means you could make:
- 100 small requests (100 tokens each)
- 10 large requests (1,000 tokens each)
- Or any combination that stays under both limits

The key insight is that **you need to optimize for both constraints simultaneously**. If your application only watches one quota, you'll hit the other and face throttling.

### Why Traditional Rate Limiting Falls Short

Most rate limiters focus on request count (RPM) but ignore token consumption (TPM). A naive approach might look like:

```python
# Naive approach - DON'T do this
requests_this_minute = 0
MAX_REQUESTS = 100

def can_make_request():
    global requests_this_minute
    if requests_this_minute < MAX_REQUESTS:
        requests_this_minute += 1
        return True
    return False
```

This completely ignores the token dimension, which means you could still hit TPM limits and face throttling errors.

## Step-by-Step Solution

Let's build a comprehensive solution that handles both constraints elegantly.

### 1. Identify Your Bottleneck

First, we need a diagnostic tool to determine whether RPM or TPM is your limiting factor:

```python
import boto3
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def analyze_quota_usage(model_id, sample_request, repeat_count=10):
    """Analyze whether you're RPM or TPM limited for a given model and request type."""
    
    # Create Bedrock client
    bedrock = boto3.client('bedrock-runtime')
    
    # Get approximate token counts for this model and request
    token_info = estimate_tokens(model_id, sample_request)
    avg_tokens_per_request = token_info['input_tokens'] + token_info['output_tokens']
    
    # Get account quota limits
    quota_client = boto3.client('service-quotas')
    
    # This is simplified - in practice, you'd need to map model IDs to specific quota codes
    # which requires some lookup logic
    rpm_quota = get_quota_for_model(quota_client, model_id, "RPM")
    tpm_quota = get_quota_for_model(quota_client, model_id, "TPM")
    
    if not rpm_quota or not tpm_quota:
        print("Could not retrieve quota information. Using estimates.")
        rpm_quota = 100  # Example default
        tpm_quota = 10000  # Example default
    
    # Calculate theoretical max requests given TPM limit
    theoretical_max_rpm = tpm_quota / avg_tokens_per_request if avg_tokens_per_request > 0 else float('inf')
    
    # Determine bottleneck
    if theoretical_max_rpm < rpm_quota:
        bottleneck = "TPM"
        limiting_factor = theoretical_max_rpm
    else:
        bottleneck = "RPM"
        limiting_factor = rpm_quota
    
    results = {
        "model_id": model_id,
        "avg_tokens_per_request": avg_tokens_per_request,
        "rpm_quota": rpm_quota,
        "tpm_quota": tpm_quota,
        "theoretical_max_rpm": theoretical_max_rpm,
        "bottleneck": bottleneck,
        "limiting_factor": limiting_factor
    }
    
    # Visualize the results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(["RPM Quota", "TPM-Limited RPM"], 
           [rpm_quota, theoretical_max_rpm],
           color=['blue', 'red'])
    
    ax.set_ylabel('Requests Per Minute')
    ax.set_title(f'Quota Analysis for {model_id}')
    
    ax.annotate(f"Your bottleneck is: {bottleneck}",
                xy=(0.5, 0.9), xycoords='axes fraction',
                ha='center', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{model_id.replace('.', '_')}_quota_analysis.png")
    plt.close()
    
    return results

def estimate_tokens(model_id, request_payload):
    """Estimate token counts for a given model and request.
    In practice, this would make a small test request to get actual counts."""
    # Simplified implementation - in practice, you'd use the actual API response
    # or a tokenizer library appropriate for the model
    input_text = json.dumps(request_payload)
    input_tokens = len(input_text.split()) * 1.3  # Rough estimate
    output_tokens = input_tokens * 1.5  # Rough estimate of response size
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

def get_quota_for_model(quota_client, model_id, quota_type):
    """Get quota limits for a specific model and quota type.
    In practice, this would involve mapping model IDs to quota codes."""
    # Simplified implementation - in practice, this requires proper quota code lookup
    try:
        # This is a placeholder - actual implementation would require
        # mapping model_id to the corresponding quota codes
        if "anthropic" in model_id and quota_type == "RPM":
            quota_code = "L-12345678"  # Placeholder
        elif "anthropic" in model_id and quota_type == "TPM":
            quota_code = "L-87654321"  # Placeholder
        else:
            return None
            
        response = quota_client.get_service_quota(
            ServiceCode='bedrock',
            QuotaCode=quota_code
        )
        
        return response.get('Quota', {}).get('Value')
    except Exception as e:
        print(f"Error retrieving quota: {str(e)}")
        return None
```

### 2. Implementing a Dual Token Bucket Algorithm

Now, let's create a rate limiter that respects both RPM and TPM limits:

```python
import time
import threading
from dataclasses import dataclass
import boto3
import random
import logging

logger = logging.getLogger("bedrock_rate_limiter")

@dataclass
class BedrockQuotaConfig:
    """Configuration for AWS Bedrock quota limits."""
    rpm_limit: float  # Requests per minute
    tpm_limit: float  # Tokens per minute
    burst_factor: float = 1.5  # Allow bursts up to this factor of the limit
    
    @classmethod
    def from_model_id(cls, model_id, service_quotas_client=None):
        """
        Attempt to retrieve actual quota limits for the model from AWS Service Quotas.
        Falls back to conservative defaults if unable to determine actual limits.
        """
        # If no service quotas client is provided, create one
        if service_quotas_client is None:
            try:
                service_quotas_client = boto3.client('service-quotas')
            except Exception as e:
                logger.warning(f"Could not create service-quotas client: {e}")
                # Return conservative defaults
                return cls(
                    rpm_limit=60,    # 1 request per second
                    tpm_limit=6000,  # 100 tokens per second
                )
        
        # Try to get actual limits (simplified - would need actual quota code mapping)
        rpm_limit = None
        tpm_limit = None
        
        try:
            # In practice, this would involve mapping model IDs to quota codes
            # This is just placeholder logic
            model_family = model_id.split('.')[0].lower()
            
            # List quotas for bedrock
            response = service_quotas_client.list_service_quotas(
                ServiceCode='bedrock'
            )
            
            # Look for relevant quotas
            for quota in response.get('Quotas', []):
                quota_name = quota.get('QuotaName', '').lower()
                if model_family in quota_name:
                    if 'requests per minute' in quota_name:
                        rpm_limit = quota.get('Value')
                    elif 'tokens per minute' in quota_name:
                        tpm_limit = quota.get('Value')
        except Exception as e:
            logger.warning(f"Error retrieving quota limits: {e}")
        
        # Fall back to defaults if needed
        if rpm_limit is None:
            rpm_limit = 60  # Conservative default
        if tpm_limit is None:
            tpm_limit = 6000  # Conservative default
            
        return cls(
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit
        )


class DualTokenBucketRateLimiter:
    """
    Rate limiter that respects both RPM and TPM limits for AWS Bedrock.
    Uses the token bucket algorithm for both limits simultaneously.
    """
    
    def __init__(self, config: BedrockQuotaConfig):
        """
        Initialize the rate limiter with quota configuration.
        
        Args:
            config: BedrockQuotaConfig with rpm_limit and tpm_limit
        """
        # Convert limits to per-second rates
        self.rps_limit = config.rpm_limit / 60
        self.tps_limit = config.tpm_limit / 60
        self.burst_factor = config.burst_factor
        
        # Initialize buckets at full capacity
        self.request_bucket = self.rps_limit * self.burst_factor
        self.token_bucket = self.tps_limit * self.burst_factor
        
        # Track last refill time
        self.last_refill = time.time()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Metrics
        self.metrics = {
            "allowed_requests": 0,
            "throttled_requests": 0,
            "total_tokens_processed": 0
        }
    
    def _refill_buckets(self):
        """Refill the token buckets based on elapsed time."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill request bucket
            self.request_bucket = min(
                self.rps_limit * self.burst_factor,
                self.request_bucket + (elapsed * self.rps_limit)
            )
            
            # Refill token bucket
            self.token_bucket = min(
                self.tps_limit * self.burst_factor,
                self.token_bucket + (elapsed * self.tps_limit)
            )
            
            self.last_refill = now
    
    def check_and_consume(self, estimated_tokens):
        """
        Check if a request can be processed and consume tokens if so.
        
        Args:
            estimated_tokens: Estimated token count for this request
            
        Returns:
            True if request is allowed, False if it should be throttled
        """
        with self.lock:
            self._refill_buckets()
            
            # Check if we have capacity for both one request and the token count
            if self.request_bucket >= 1 and self.token_bucket >= estimated_tokens:
                # If yes, consume from both buckets
                self.request_bucket -= 1
                self.token_bucket -= estimated_tokens
                
                # Update metrics
                self.metrics["allowed_requests"] += 1
                self.metrics["total_tokens_processed"] += estimated_tokens
                
                return True
            else:
                # Update metrics
                self.metrics["throttled_requests"] += 1
                
                return False
    
    def wait_time_for_next_request(self, estimated_tokens):
        """
        Calculate wait time until the next request can be processed.
        
        Args:
            estimated_tokens: Estimated token count for this request
            
        Returns:
            Time in seconds to wait
        """
        with self.lock:
            self._refill_buckets()
            
            # Calculate time needed to refill request bucket
            request_wait = 0
            if self.request_bucket < 1:
                request_wait = (1 - self.request_bucket) / self.rps_limit
            
            # Calculate time needed to refill token bucket
            token_wait = 0
            if self.token_bucket < estimated_tokens:
                token_wait = (estimated_tokens - self.token_bucket) / self.tps_limit
            
            # Return the longer wait time
            return max(request_wait, token_wait)
    
    def get_metrics(self):
        """Get current metrics."""
        with self.lock:
            return self.metrics.copy()
```

### 3. Creating a Complete Quota-Aware Client

Now, let's build a complete client that integrates with AWS Bedrock while respecting quotas:

```python
import boto3
import json
import time
import logging
import threading
import random
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger("bedrock_client")

class QuotaAwareBedrockClient:
    """
    A client for AWS Bedrock that manages quota limits intelligently.
    Provides automatic backoff, request prioritization, and optimal throughput.
    """
    
    def __init__(self, model_id: str, config: Optional[BedrockQuotaConfig] = None):
        """
        Initialize the quota-aware client.
        
        Args:
            model_id: The AWS Bedrock model identifier
            config: Optional quota configuration (will be auto-detected if not provided)
        """
        self.model_id = model_id
        self.bedrock = boto3.client('bedrock-runtime')
        
        # Set up rate limiter
        if config is None:
            # Auto-detect limits if not provided
            quotas_client = boto3.client('service-quotas')
            config = BedrockQuotaConfig.from_model_id(model_id, quotas_client)
            
        self.rate_limiter = DualTokenBucketRateLimiter(config)
        
        # Request queue for prioritization
        self.request_queue = []
        self.queue_lock = threading.Lock()
        
        # Start background worker thread for queued requests
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
    
    def estimate_tokens(self, request_payload: Dict[str, Any]) -> Dict[str, int]:
        """
        Estimate token count for a request. 
        
        This is a simplified implementation - in practice, you would use:
        1. The actual response from a previous call if available
        2. A proper tokenizer for the specific model
        3. Records of average token counts for similar requests
        """
        input_text = json.dumps(request_payload)
        
        # Very rough estimation based on text length
        # In production, use a proper tokenizer for your model
        input_tokens = len(input_text.split()) * 1.3
        
        # Estimate output based on model and input type
        # This is very model-specific - adjust for your use case
        if "max_tokens" in request_payload:
            output_tokens = request_payload["max_tokens"]
        elif "max_tokens_to_sample" in request_payload:
            output_tokens = request_payload["max_tokens_to_sample"]
        else:
            # Default fallback - assume output similar to input size
            output_tokens = input_tokens
        
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens + output_tokens)
        }
    
    def invoke_model(self, 
                    request_payload: Dict[str, Any], 
                    priority: str = "normal",
                    wait_if_throttled: bool = True,
                    timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Invoke the model with quota awareness.
        
        Args:
            request_payload: The request payload for the model
            priority: Request priority ('high', 'normal', 'low')
            wait_if_throttled: Whether to wait or fail immediately if throttled
            timeout: Maximum time to wait for a response
            
        Returns:
            Model response
            
        Raises:
            ThrottlingException: If request is throttled and wait_if_throttled is False
        """
        # Estimate token usage
        token_estimate = self.estimate_tokens(request_payload)
        total_tokens = token_estimate["total_tokens"]
        
        # Check if we can make this request now
        if self.rate_limiter.check_and_consume(total_tokens):
            # We have capacity - make the request right away
            start_time = time.time()
            
            try:
                response = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_payload)
                )
                
                response_body = json.loads(response["body"].read())
                return response_body
                
            except Exception as e:
                if "ThrottlingException" in str(e):
                    # Despite our best efforts, we got throttled by AWS
                    # Log this as it indicates our rate limiter might need adjustment
                    logger.warning(f"Unexpected throttling occurred: {str(e)}")
                    
                    if not wait_if_throttled:
                        raise
                    
                    # Calculate backoff time - exponential with jitter
                    backoff = min(10, 0.1 * (2 ** random.randint(0, 5)))
                    backoff += random.uniform(0, 0.1 * backoff)  # Add jitter
                    
                    logger.info(f"Backing off for {backoff:.2f} seconds")
                    time.sleep(backoff)
                    
                    # Try again with high priority
                    return self.invoke_model(
                        request_payload, 
                        priority="high",
                        wait_if_throttled=wait_if_throttled,
                        timeout=timeout
                    )
                else:
                    # For other errors, propagate them up
                    raise
        elif wait_if_throttled:
            # We're over quota limit, but client wants to wait
            wait_time = self.rate_limiter.wait_time_for_next_request(total_tokens)
            
            # Check if wait time exceeds timeout
            if timeout is not None and wait_time > timeout:
                raise TimeoutError(f"Wait time ({wait_time:.2f}s) exceeds timeout ({timeout}s)")
            
            logger.info(f"Quota limit reached. Waiting {wait_time:.2f} seconds before trying again.")
            time.sleep(wait_time)
            
            # Retry after waiting
            return self.invoke_model(
                request_payload,
                priority=priority,
                wait_if_throttled=wait_if_throttled,
                timeout=timeout
            )
        else:
            # We're over quota and client doesn't want to wait
            # Queue the request based on priority
            if priority in ("high", "normal"):
                # Add high and normal priority requests to the queue
                future = self._queue_request(request_payload, priority, timeout)
                return future.result(timeout=timeout)
            else:
                # Low priority requests just fail immediately when over quota
                raise Exception("Rate limit exceeded and request has low priority")
    
    def _queue_request(self, request_payload, priority, timeout):
        """Add a request to the priority queue."""
        from concurrent.futures import Future
        
        future = Future()
        
        with self.queue_lock:
            # Add to queue with priority
            priority_value = {"high": 0, "normal": 1, "low": 2}.get(priority, 1)
            self.request_queue.append({
                "payload": request_payload,
                "priority": priority_value,
                "future": future,
                "enqueue_time": time.time(),
                "timeout": timeout
            })
            
            # Sort queue by priority
            self.request_queue.sort(key=lambda x: x["priority"])
        
        return future
    
    def _process_queue(self):
        """Background thread to process queued requests."""
        while True:
            # Sleep a bit to avoid busy waiting
            time.sleep(0.1)
            
            # Check if queue is empty
            if not self.request_queue:
                continue
            
            # Get next request
            with self.queue_lock:
                if not self.request_queue:
                    continue
                
                # Get highest priority request
                request = self.request_queue[0]
                
                # Check if it's expired
                if request["timeout"] is not None:
                    elapsed = time.time() - request["enqueue_time"]
                    if elapsed > request["timeout"]:
                        # Remove expired request
                        self.request_queue.pop(0)
                        request["future"].set_exception(
                            TimeoutError("Request timed out in queue")
                        )
                        continue
            
            # Try to process the request
            payload = request["payload"]
            token_estimate = self.estimate_tokens(payload)
            total_tokens = token_estimate["total_tokens"]
            
            if self.rate_limiter.check_and_consume(total_tokens):
                # We have capacity - remove from queue
                with self.queue_lock:
                    # Make sure it's still at the front
                    if self.request_queue and self.request_queue[0] == request:
                        self.request_queue.pop(0)
                    else:
                        # Someone else took it - skip
                        continue
                
                # Process the request
                try:
                    response = self.bedrock.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(payload)
                    )
                    
                    response_body = json.loads(response["body"].read())
                    request["future"].set_result(response_body)
                except Exception as e:
                    request["future"].set_exception(e)
    
    def get_queue_length(self):
        """Get the current queue length."""
        with self.queue_lock:
            return len(self.request_queue)
    
    def get_metrics(self):
        """Get performance metrics."""
        metrics = self.rate_limiter.get_metrics()
        
        with self.queue_lock:
            metrics["queued_requests"] = len(self.request_queue)
        
        return metrics
```

### 4. Using the Quota-Aware Client in Your Application

Here's how to integrate this solution into your application:

```python
import logging
from pathlib import Path
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize quota-aware client
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Use your preferred model
client = QuotaAwareBedrockClient(model_id)

def process_document(document_path, prompt_template, priority="normal"):
    """Process a document with quota awareness."""
    try:
        # Read document content
        content = Path(document_path).read_text()
        
        # Prepare request payload - format depends on model
        request_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt_template.format(document=content)
                }
            ]
        }
        
        # Invoke model with quota awareness
        response = client.invoke_model(
            request_payload=request_payload,
            priority=priority,
            wait_if_throttled=True,
            timeout=300  # 5 minutes max wait
        )
        
        # Extract and return the results
        return {
            "document_path": document_path,
            "result": response["content"][0]["text"] if "content" in response else response,
            "status": "success"
        }
        
    except Exception as e:
        logging.error(f"Error processing {document_path}: {str(e)}")
        return {
            "document_path": document_path,
            "error": str(e),
            "status": "error"
        }

# Example of batch processing with priority
def batch_process_documents(document_paths, prompt_template, max_parallel=5):
    """Process multiple documents in parallel with quota management."""
    results = []
    
    # Set priorities based on document types or business rules
    def determine_priority(doc_path):
        # This is where you'd implement your business logic for priorities
        if "urgent" in doc_path.lower():
            return "high"
        elif "archive" in doc_path.lower():
            return "low"
        else:
            return "normal"
    
    # Process in parallel with controlled concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                process_document, 
                doc_path, 
                prompt_template, 
                determine_priority(doc_path)
            ): doc_path for doc_path in document_paths
        }
        
        for future in concurrent.futures.as_completed(futures):
            doc_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                logging.info(f"Processed {doc_path}: {result['status']}")
            except Exception as e:
                logging.error(f"Exception processing {doc_path}: {str(e)}")
                results.append({
                    "document_path": doc_path,
                    "error": str(e),
                    "status": "error"
                })
    
    # Summarize results
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count
    
    logging.info(f"Batch processing complete: {success_count} succeeded, {error_count} failed")
    logging.info(f"Client metrics: {client.get_metrics()}")
    
    return results
```

## Common Pitfalls and Troubleshooting

### Pitfall #1: Ignoring Token Estimation Accuracy

Many implementations use rough token estimates that can be wildly inaccurate, leading to unexpected throttling.

**Solution**: Improve your token estimation:

```python
def better_token_estimation(model_id, text):
    """More accurate token estimation for specific models."""
    
    if "anthropic.claude" in model_id:
        # For Claude models, use Anthropic's tokenizer if available
        try:
            from anthropic import Anthropic
            client = Anthropic()
            count = client.count_tokens(text)
            return count
        except ImportError:
            # Fall back to approximation if library not available
            return len(text.split()) * 1.3
    elif "meta.llama" in model_id:
        # For Llama models, use their tokenizer if available
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
            return len(tokenizer.encode(text))
        except (ImportError, OSError):
            # Fall back to approximation
            return len(text.split()) * 1.4
    else:
        # Generic fallback
        return len(text.split()) * 1.3
```

### Pitfall #2: Not Adapting to Changing Conditions

A static rate limiter doesn't account for changing request patterns or temporary quota adjustments.

**Solution**: Implement adaptive behavior:

```python
class AdaptiveRateLimiter(DualTokenBucketRateLimiter):
    """Rate limiter that adapts to changing conditions."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Track error rates
        self.request_history = []
        self.throttle_history = []
        self.history_window = 100  # Track last 100 requests
        
        # Adaptive settings
        self.safety_factor = 1.0  # Start at 100% of quota
        
    def update_safety_factor(self):
        """Adjust safety factor based on recent throttling."""
        if len(self.request_history) < 20:
            # Not enough data yet
            return
            
        # Calculate recent throttle rate
        recent_total = min(len(self.request_history), self.history_window)
        recent_throttles = sum(self.throttle_history[-recent_total:])
        throttle_rate = recent_throttles / recent_total if recent_total > 0 else 0
        
        # Adjust safety factor based on throttle rate
        if throttle_rate > 0.05:  # More than 5% of requests throttled
            # Increase safety margin (reduce usable quota)
            self.safety_factor = max(0.5, self.safety_factor * 0.9)
        elif throttle_rate < 0.01:  # Less than 1% throttled
            # Decrease safety margin (increase usable quota)
            self.safety_factor = min(1.0, self.safety_factor * 1.05)
    
    def check_and_consume(self, estimated_tokens):
        """Override to include adaptive behavior."""
        # Record request attempt
        self.request_history.append(time.time())
        if len(self.request_history) > self.history_window:
            self.request_history.pop(0)
        
        # Apply safety factor to available capacity
        with self.lock:
            self._refill_buckets()
            
            # Apply safety factor to current buckets
            adjusted_request_bucket = self.request_bucket * self.safety_factor
            adjusted_token_bucket = self.token_bucket * self.safety_factor
            
            if adjusted_request_bucket >= 1 and adjusted_token_bucket >= estimated_tokens:
                # We have capacity - consume from real buckets
                self.request_bucket -= 1
                self.token_bucket -= estimated_tokens
                
                # Update metrics
                self.metrics["allowed_requests"] += 1
                self.metrics["total_tokens_processed"] += estimated_tokens
                
                # Record no throttle
                self.throttle_history.append(0)
                if len(self.throttle_history) > self.history_window:
                    self.throttle_history.pop(0)
                
                # Update safety factor occasionally
                if self.metrics["allowed_requests"] % 10 == 0:
                    self.update_safety_factor()
                
                return True
            else:
                # Update metrics
                self.metrics["throttled_requests"] += 1
                
                # Record throttle
                self.throttle_history.append(1)
                if len(self.throttle_history) > self.history_window:
                    self.throttle_history.pop(0)
                
                # Update safety factor
                self.update_safety_factor()
                
                return False
```

### Common Error Messages and Solutions

**Error**: `{"message": "Rate exceeded", "code": "ThrottlingException"}`

**Solution**: 
1. Check which quota you're hitting (RPM or TPM)
2. If RPM-limited: Implement request batching or queuing
3. If TPM-limited: Optimize prompt size and output length
4. Implement exponential backoff with jitter for retries

**Error**: `{"message": "Token limit exceeded", "code": "ValidationException"}`

**Solution**:
1. Reduce the token size of your input
2. For large documents, implement a chunking strategy
3. Consider using the Converse API which handles token management

## Try It Yourself Challenge

### Challenge: Implement Priority-Based Document Processing

Create a document processing system that:

1. Processes documents of varying sizes efficiently
2. Prioritizes certain document types over others
3. Provides estimates of processing time to users
4. Automatically scales back token usage during peak times

**Starting Code**:

```python
# Your task is to extend the QuotaAwareBedrockClient with:
# 1. Document size-based queue priority
# 2. Processing time estimation
# 3. Dynamic token usage adjustment

class EnhancedBedrockClient(QuotaAwareBedrockClient):
    """Enhanced client with priority-based processing."""
    
    def __init__(self, model_id, config=None):
        super().__init__(model_id, config)
        # Add any additional instance variables here
    
    def estimate_processing_time(self, document_size, current_queue_status):
        """
        Estimate processing time for a document based on its size
        and current queue status.
        """
        # TODO: Implement this method
        pass
    
    def process_document_with_priority(self, document, document_type):
        """
        Process a document with priority based on its type.
        """
        # TODO: Implement this method
        pass
    
    def adapt_token_usage(self, peak_time_factor):
        """
        Dynamically adjust token usage based on time of day
        or other load factors.
        """
        # TODO: Implement this method
        pass
```

## Beyond the Basics

Once you've implemented the core quota management strategies, consider these advanced techniques:

### Multi-Region Distribution

AWS Bedrock quotas are per-region. You can effectively multiply your available throughput by distributing requests across regions:

```python
class MultiRegionBedrockClient:
    """Client that distributes requests across multiple AWS regions."""
    
    def __init__(self, model_id, regions=None):
        """Initialize with multiple regions."""
        self.model_id = model_id
        
        # Default to common Bedrock regions if none provided
        self.regions = regions or ["us-east-1", "us-west-2"]
        
        # Create a client and rate limiter for each region
        self.regional_clients = {}
        for region in self.regions:
            # Create regional clients
            bedrock = boto3.client('bedrock-runtime', region_name=region)
            
            # Detect quota limits for this region
            config = BedrockQuotaConfig.from_model_id(model_id, region=region)
            
            # Create rate limiter
            rate_limiter = DualTokenBucketRateLimiter(config)
            
            self.regional_clients[region] = {
                "client": bedrock,
                "rate_limiter": rate_limiter,
                "metrics": {"requests": 0, "throttles": 0}
            }
        
        # Round-robin counter for basic load balancing
        self.next_region_index = 0
    
    def invoke_model(self, request_payload):
        """Invoke model across multiple regions with quota awareness."""
        # Estimate tokens for this request
        token_estimate = self.estimate_tokens(request_payload)
        total_tokens = token_estimate["total_tokens"]
        
        # Try each region in turn
        for _ in range(len(self.regions)):
            # Get next region in round-robin fashion
            region = self.regions[self.next_region_index]
            self.next_region_index = (self.next_region_index + 1) % len(self.regions)
            
            regional_data = self.regional_clients[region]
            rate_limiter = regional_data["rate_limiter"]
            
            # Check if this region has capacity
            if rate_limiter.check_and_consume(total_tokens):
                try:
                    # Make the request in this region
                    response = regional_data["client"].invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(request_payload)
                    )
                    
                    # Update metrics
                    regional_data["metrics"]["requests"] += 1
                    
                    # Return the response
                    return json.loads(response["body"].read())
                    
                except Exception as e:
                    if "ThrottlingException" in str(e):
                        # Track throttling
                        regional_data["metrics"]["throttles"] += 1
                        # Try next region
                        continue
                    else:
                        # For non-throttling errors, raise
                        raise
        
        # If we get here, all regions are at capacity
        # Implement waiting, queuing, etc.
        raise Exception("All regions at capacity")
```

### Dynamic Prompt Optimization

Reduce token usage during peak times by dynamically adjusting prompt verbosity:

```python
class DynamicPromptOptimizer:
    """Dynamically optimize prompts based on current load."""
    
    def __init__(self, base_client):
        """Initialize with a base client."""
        self.client = base_client
        
        # Define different verbosity levels for prompts
        self.verbosity_levels = {
            "full": 1.0,       # Full prompt with examples and detailed instructions
            "standard": 0.7,   # Standard prompt with basic instructions
            "minimal": 0.4     # Minimal prompt with just the task
        }
        
        # Current verbosity setting
        self.current_verbosity = "standard"
    
    def adjust_verbosity(self, queue_length, metrics):
        """Adjust verbosity based on current load."""
        # Calculate load factors
        rpm_usage = metrics.get("allowed_requests", 0) / self.client.rate_limiter.rps_limit / 60
        queue_factor = min(1.0, queue_length / 10)  # Normalize queue length
        
        combined_load = (rpm_usage * 0.7) + (queue_factor * 0.3)  # Weighted combination
        
        # Adjust verbosity
        if combined_load > 0.8:
            # High load - reduce verbosity
            self.current_verbosity = "minimal"
        elif combined_load > 0.5:
            # Moderate load - standard verbosity
            self.current_verbosity = "standard"
        else:
            # Low load - full verbosity
            self.current_verbosity = "full"
        
        return self.current_verbosity
    
    def optimize_prompt(self, original_prompt, document_type):
        """Optimize a prompt based on current verbosity setting."""
        verbosity_factor = self.verbosity_levels[self.current_verbosity]
        
        # This is where you'd implement your prompt optimization logic
        # based on the document type and verbosity factor
        # ...
        
        # Example implementation - simplify prompt based on verbosity
        if self.current_verbosity == "minimal":
            # Most minimal version - just the core task
            return self._extract_core_instruction(original_prompt)
        elif self.current_verbosity == "standard":
            # Standard version - remove examples but keep instructions
            return self._remove_examples(original_prompt)
        else:
            # Full verbosity - use original
            return original_prompt
    
    def _extract_core_instruction(self, prompt):
        """Extract just the core instruction from a prompt."""
        # This is a simplified implementation
        # In practice, you'd have more sophisticated NLP
        lines = prompt.split('\n')
        for line in lines:
            if '?' in line or any(word in line.lower() for word in ['summarize', 'analyze', 'extract', 'identify']):
                return line
        
        # Fallback - return first non-empty line
        for line in lines:
            if line.strip():
                return line
        
        # Ultimate fallback
        return prompt
    
    def _remove_examples(self, prompt):
        """Remove examples but keep instructions."""
        # Simplified implementation
        if "Example:" in prompt or "Examples:" in prompt:
            parts = re.split(r'Example[s]?:', prompt, flags=re.IGNORECASE)
            return parts[0].strip()
        
        return prompt
```

## Key Takeaways

- **Understand your bottleneck**: Determine whether you're TPM-limited or RPM-limited to choose the right strategy
- **Implement dual token buckets**: Use a rate limiter that respects both RPM and TPM constraints
- **Prioritize critical requests**: Don't treat all requests equally - implement a priority system
- **Adapt dynamically**: Change your approach based on current conditions and time of day
- **Consider advanced strategies**: Multi-region distribution and dynamic prompt optimization can further increase throughput

---

**Next Steps**: Now that you've optimized your throughput, learn how to [implement multi-model orchestration](/docs/multi-model-orchestration.md) to balance workloads across different foundation models.

---

*Have questions or suggestions? Open an issue or contribute improvements!*