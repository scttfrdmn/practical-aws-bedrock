"""
Quota-aware synchronous client for AWS Bedrock inference.

This module extends the basic synchronous client with advanced quota management
capabilities to help maximize throughput while staying within quota limits.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Union, Tuple

from .basic_client import BedrockClient


class TokenBucket:
    """
    Token bucket for rate limiting.
    
    Implements the token bucket algorithm for rate limiting:
    - Tokens are added to the bucket at a constant rate
    - Each request consumes one or more tokens
    - If the bucket is empty, requests wait or are rejected
    """
    
    def __init__(
        self, 
        capacity: float, 
        refill_rate: float,
        initial_tokens: Optional[float] = None
    ):
        """
        Initialize the token bucket.
        
        Args:
            capacity: Maximum number of tokens the bucket can hold
            refill_rate: Tokens per second to add
            initial_tokens: Initial number of tokens (defaults to capacity)
        """
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)  # tokens per second
        self.tokens = float(initial_tokens if initial_tokens is not None else capacity)
        self.last_refill_time = time.time()
        self.lock = threading.RLock()
    
    def _refill(self) -> None:
        """Refill the token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        
        if elapsed > 0:
            new_tokens = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill_time = now
    
    def consume(self, tokens: float = 1.0, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            block: Whether to block until tokens are available
            timeout: Maximum time to wait (seconds)
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        start_time = time.time()
        
        with self.lock:
            # Refill tokens based on elapsed time
            self._refill()
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            # If not blocking, return False
            if not block:
                return False
            
            # Calculate wait time
            wait_time = (tokens - self.tokens) / self.refill_rate
            
            # Check timeout
            if timeout is not None and wait_time > timeout:
                return False
        
        # Wait outside the lock
        time.sleep(wait_time)
        
        # Try again after waiting
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            # Still not enough tokens
            return False
    
    def get_token_count(self) -> float:
        """Get the current token count."""
        with self.lock:
            self._refill()
            return self.tokens


class QuotaAwareBedrockClient(BedrockClient):
    """
    A client for AWS Bedrock that respects quota limits.
    
    This client extends the basic BedrockClient with:
    - Token bucket rate limiting for RPM (requests per minute)
    - Token bucket rate limiting for TPM (tokens per minute)
    - Dynamic backoff based on throttling responses
    - Quota utilization tracking
    """
    
    def __init__(
        self,
        model_id: str,
        max_rpm: Optional[float] = None,
        max_tpm: Optional[float] = None,
        token_estimate_per_request: Optional[float] = None,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the quota-aware Bedrock client.
        
        Args:
            model_id: The Bedrock model identifier
            max_rpm: Maximum requests per minute (if None, no RPM limiting)
            max_tpm: Maximum tokens per minute (if None, no TPM limiting)
            token_estimate_per_request: Estimated tokens per request (for TPM limiting)
            profile_name: AWS profile name (defaults to value from get_profile())
            region_name: AWS region name (defaults to value from get_region())
            max_retries: Maximum number of retry attempts for recoverable errors
            base_backoff: Base backoff time (in seconds) for exponential backoff
            logger: Optional logger instance
        """
        super().__init__(
            model_id=model_id,
            profile_name=profile_name,
            region_name=region_name,
            max_retries=max_retries,
            base_backoff=base_backoff,
            logger=logger
        )
        
        # Set up quota limiting
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.token_estimate_per_request = token_estimate_per_request or 1000
        
        # Create token buckets if limits are specified
        self.rpm_bucket = TokenBucket(
            capacity=max_rpm / 60.0 * 10,  # 10 second burst capacity
            refill_rate=max_rpm / 60.0  # tokens per second
        ) if max_rpm is not None else None
        
        self.tpm_bucket = TokenBucket(
            capacity=max_tpm / 60.0 * 10,  # 10 second burst capacity
            refill_rate=max_tpm / 60.0  # tokens per second
        ) if max_tpm is not None else None
        
        # Quota utilization tracking
        self.quota_limited_count = 0
        self.last_token_counts = {}
        
        # Dynamic backoff tracking
        self.consecutive_throttles = 0
        self.backoff_multiplier = 1.0
    
    def invoke(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        other_params: Optional[Dict[str, Any]] = None,
        wait_for_quota: bool = True,
        quota_timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Invoke the model with a prompt and return the response, respecting quota limits.
        
        Args:
            prompt: The user prompt or instruction
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            system_prompt: Optional system prompt for models that support it
            other_params: Additional model-specific parameters
            wait_for_quota: Whether to wait for quota to be available
            quota_timeout: Maximum time to wait for quota
            
        Returns:
            The parsed response from the model
            
        Raises:
            ValueError: For invalid input parameters
            RuntimeError: For unrecoverable errors after retries
            QuotaExceededException: If quota is exceeded and not waiting
        """
        # Estimate token usage for this request
        estimated_tokens = self._estimate_tokens(prompt, max_tokens)
        
        # Check RPM quota
        if self.rpm_bucket is not None:
            self.logger.debug("Checking RPM quota...")
            if not self.rpm_bucket.consume(1.0, block=wait_for_quota, timeout=quota_timeout):
                self.quota_limited_count += 1
                self.logger.warning("RPM quota limit reached")
                if not wait_for_quota:
                    raise QuotaExceededException("RPM quota exceeded")
        
        # Check TPM quota
        if self.tpm_bucket is not None:
            self.logger.debug(f"Checking TPM quota (estimated tokens: {estimated_tokens})...")
            if not self.tpm_bucket.consume(estimated_tokens, block=wait_for_quota, timeout=quota_timeout):
                self.quota_limited_count += 1
                self.logger.warning("TPM quota limit reached")
                if not wait_for_quota:
                    raise QuotaExceededException("TPM quota exceeded")
        
        try:
            # Call the parent invoke method
            result = super().invoke(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                other_params=other_params
            )
            
            # Reset consecutive throttles counter on success
            self.consecutive_throttles = 0
            self.backoff_multiplier = 1.0
            
            # Update last token counts
            self.last_token_counts = {
                "input_tokens": result.get("input_tokens", estimated_tokens * 0.3),
                "output_tokens": result.get("output_tokens", estimated_tokens * 0.7),
                "total_tokens": result.get("total_tokens", estimated_tokens)
            }
            
            return result
            
        except Exception as e:
            # Check if it's a throttling exception
            if "ThrottlingException" in str(e):
                # Increment consecutive throttles counter
                self.consecutive_throttles += 1
                
                # Increase backoff multiplier (up to a max of 16x)
                self.backoff_multiplier = min(self.backoff_multiplier * 2, 16.0)
                
                # Adjust token bucket consumption rate if we're getting throttled
                self._adjust_rate_limits()
            
            # Re-raise the exception
            raise
    
    def _estimate_tokens(self, prompt: str, max_tokens: int) -> float:
        """
        Estimate the number of tokens that will be used for this request.
        
        Args:
            prompt: The prompt text
            max_tokens: Maximum tokens to generate
            
        Returns:
            Estimated total tokens (input + output)
        """
        # Use previous token counts if available
        if self.last_token_counts and "total_tokens" in self.last_token_counts:
            return float(self.last_token_counts["total_tokens"])
        
        # Rough estimate based on prompt length and max_tokens
        input_estimate = len(self._tokenize_rough(prompt))
        return float(input_estimate + max_tokens)
    
    def _adjust_rate_limits(self) -> None:
        """
        Dynamically adjust rate limits based on throttling behavior.
        This reduces the effective rate when throttling occurs.
        """
        if self.consecutive_throttles > 0:
            # Reduce effective rate based on consecutive throttles
            reduction_factor = 1.0 / self.backoff_multiplier
            
            self.logger.warning(
                f"Adjusting rate limits due to throttling (factor: {reduction_factor:.2f}, "
                f"consecutive throttles: {self.consecutive_throttles})"
            )
            
            # Adjust RPM bucket
            if self.rpm_bucket is not None:
                new_rpm_rate = (self.max_rpm / 60.0) * reduction_factor
                self.rpm_bucket.refill_rate = new_rpm_rate
                self.logger.info(f"Adjusted RPM rate to {new_rpm_rate * 60:.2f} per minute")
            
            # Adjust TPM bucket
            if self.tpm_bucket is not None:
                new_tpm_rate = (self.max_tpm / 60.0) * reduction_factor
                self.tpm_bucket.refill_rate = new_tpm_rate
                self.logger.info(f"Adjusted TPM rate to {new_tpm_rate * 60:.2f} per minute")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get usage metrics for this client instance.
        
        Returns:
            Dictionary with usage metrics
        """
        metrics = super().get_metrics()
        
        # Add quota-specific metrics
        metrics.update({
            "quota_limited_count": self.quota_limited_count,
            "rpm_remaining": self.rpm_bucket.get_token_count() if self.rpm_bucket else None,
            "tpm_remaining": self.tpm_bucket.get_token_count() if self.tpm_bucket else None,
            "consecutive_throttles": self.consecutive_throttles,
            "backoff_multiplier": self.backoff_multiplier
        })
        
        return metrics
    
    def wait_for_quota_reset(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for quota buckets to refill to at least 75% capacity.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if quota was reset, False if timeout was reached
        """
        start_time = time.time()
        reset_complete = False
        
        while not reset_complete:
            # Check if timeout has been reached
            if timeout is not None and time.time() - start_time > timeout:
                self.logger.warning("Timeout reached waiting for quota reset")
                return False
            
            # Check if RPM bucket is mostly full
            rpm_ready = True
            if self.rpm_bucket is not None:
                rpm_capacity = self.rpm_bucket.capacity
                rpm_current = self.rpm_bucket.get_token_count()
                rpm_ready = rpm_current >= rpm_capacity * 0.75
            
            # Check if TPM bucket is mostly full
            tpm_ready = True
            if self.tpm_bucket is not None:
                tpm_capacity = self.tpm_bucket.capacity
                tpm_current = self.tpm_bucket.get_token_count()
                tpm_ready = tpm_current >= tpm_capacity * 0.75
            
            # If both are ready, we're done
            reset_complete = rpm_ready and tpm_ready
            
            # If not reset, sleep for a bit
            if not reset_complete:
                time.sleep(0.5)
        
        self.logger.info("Quota reset complete")
        return True


class QuotaExceededException(Exception):
    """Exception raised when quota limits are exceeded."""
    pass


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a quota-aware client for Claude
    client = QuotaAwareBedrockClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        max_rpm=60,  # 60 requests per minute
        max_tpm=100000  # 100K tokens per minute
    )
    
    # Define a test prompt
    test_prompt = "Explain quantum computing in simple terms."
    
    # Process multiple requests to demonstrate quota management
    num_requests = 10
    successful = 0
    throttled = 0
    
    print(f"Sending {num_requests} requests with quota management...")
    
    for i in range(num_requests):
        try:
            print(f"\nRequest {i+1}/{num_requests}:")
            
            # Check quota metrics before request
            metrics = client.get_metrics()
            print(f"RPM remaining: {metrics.get('rpm_remaining', 'N/A')}")
            print(f"TPM remaining: {metrics.get('tpm_remaining', 'N/A')}")
            
            # Invoke model (wait for quota if needed)
            response = client.invoke(
                prompt=test_prompt,
                max_tokens=100,
                wait_for_quota=True
            )
            
            # Print abbreviated response
            output = response["output"]
            print(f"Response: {output[:50]}..." if len(output) > 50 else output)
            print(f"Tokens: {response.get('total_tokens', 'unknown')}")
            
            successful += 1
            
        except QuotaExceededException:
            print("Quota exceeded, would wait for quota reset")
            throttled += 1
            
        except Exception as e:
            print(f"Error: {str(e)}")
            throttled += 1
        
        # Add a small delay between requests (just for the example)
        time.sleep(0.5)
    
    # Print summary
    print("\nExecution Summary:")
    print(f"Successful requests: {successful}/{num_requests}")
    print(f"Throttled requests: {throttled}/{num_requests}")
    
    # Print final metrics
    metrics = client.get_metrics()
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")