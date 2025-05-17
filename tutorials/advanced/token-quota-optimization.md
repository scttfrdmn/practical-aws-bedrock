# Optimizing AWS Bedrock Throughput Under Token Size and Request Limits

This tutorial provides strategies for maximizing inference throughput when facing constraints on token sizes and request limits. We'll explore how to balance these competing constraints to achieve optimal performance.

## Objective

By the end of this tutorial, you'll understand how to:
1. Identify your bottlenecks (token quota vs. request quota)
2. Optimize prompt and response sizes for maximum throughput
3. Implement request batching and parallelization strategies
4. Create adaptive systems that automatically balance throughput constraints

## Understanding the Constraints

AWS Bedrock imposes two primary quota types that affect throughput:

### 1. Tokens Per Minute (TPM)
- Limits the total input and output tokens processed per minute
- Applies across all requests to a specific model
- Scale varies by model (Claude models typically have higher TPM limits than Llama 2)

### 2. Requests Per Minute (RPM)
- Limits the number of API calls made to a specific model per minute
- Independent of token size per request
- Also varies by model family

The key insight is that these quotas interact in complex ways. The optimal strategy depends on which constraint is your bottleneck.

## Identifying Your Bottleneck

Let's start by creating a quota utilization analyzer:

```python
import boto3
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from utils.profile_manager import get_bedrock_client
from utils.visualization_config import SVG_CONFIG

class QuotaAnalyzer:
    """Analyze AWS Bedrock quota usage to identify bottlenecks."""
    
    def __init__(self, model_id):
        """Initialize with model ID to analyze."""
        self.model_id = model_id
        self.bedrock = get_bedrock_client()
        
        # Fetch quota limits (if available)
        self.quota_limits = self._get_quota_limits()
        
    def _get_quota_limits(self):
        """Attempt to retrieve quota limits from Service Quotas API."""
        try:
            session = boto3.Session(profile_name="aws")
            quotas = session.client('service-quotas')
            
            # Extract model family from ID
            model_family = self.model_id.split('.')[0].lower()
            
            # List all quotas for bedrock
            response = quotas.list_service_quotas(ServiceCode='bedrock')
            
            rpm_limit = None
            tpm_limit = None
            
            # Look for quota codes matching this model
            for quota in response['Quotas']:
                quota_name = quota['QuotaName'].lower()
                if model_family in quota_name:
                    if "requests per minute" in quota_name:
                        rpm_limit = quota['Value']
                    elif "tokens per minute" in quota_name:
                        tpm_limit = quota['Value']
            
            return {
                "rpm": rpm_limit,
                "tpm": tpm_limit
            }
        except Exception as e:
            print(f"Could not retrieve quota limits: {e}")
            # Return default values if Service Quotas API is unavailable
            return None
    
    def run_usage_test(self, prompt_sizes, response_sizes, requests_per_batch=1):
        """
        Test different combinations of prompt and response sizes.
        
        Args:
            prompt_sizes: List of prompt sizes to test (in approximate tokens)
            response_sizes: List of response sizes to request (in max tokens)
            requests_per_batch: Number of requests to send in each batch
            
        Returns:
            DataFrame with test results
        """
        results = []
        
        for prompt_size in prompt_sizes:
            for response_size in response_sizes:
                print(f"Testing prompt size: {prompt_size}, response size: {response_size}")
                
                # Generate a prompt of approximately this size
                prompt = self._generate_prompt_of_size(prompt_size)
                
                # Create the request payload
                payload = self._format_request(prompt, response_size)
                
                # Test throughput
                start_time = time.time()
                success_count = 0
                token_count = 0
                
                # Send multiple requests to measure throughput
                for _ in range(requests_per_batch):
                    try:
                        response = self.bedrock.invoke_model(
                            modelId=self.model_id,
                            body=json.dumps(payload)
                        )
                        
                        response_body = json.loads(response['body'].read())
                        
                        # Extract token counts based on model type
                        input_tokens, output_tokens = self._extract_token_counts(response_body)
                        
                        token_count += input_tokens + output_tokens
                        success_count += 1
                        
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    # Brief pause between requests
                    time.sleep(0.5)
                
                elapsed_time = time.time() - start_time
                
                # Calculate metrics
                if success_count > 0:
                    avg_tokens = token_count / success_count
                    avg_input_tokens = prompt_size
                    avg_output_tokens = avg_tokens - avg_input_tokens
                    
                    rpm = (success_count / elapsed_time) * 60
                    tpm = (token_count / elapsed_time) * 60
                    
                    # Calculate projected throughput based on quota limits
                    rpm_limited = self.quota_limits.get("rpm", float('inf')) if self.quota_limits else float('inf')
                    tpm_limited = self.quota_limits.get("tpm", float('inf')) if self.quota_limits else float('inf')
                    
                    # Calculate theoretical max RPM given TPM limit
                    if avg_tokens > 0:
                        max_rpm_under_tpm = tpm_limited / avg_tokens if self.quota_limits else float('inf')
                    else:
                        max_rpm_under_tpm = float('inf')
                    
                    # Determine bottleneck
                    if max_rpm_under_tpm < rpm_limited:
                        bottleneck = "TPM"
                        theoretical_max_rpm = max_rpm_under_tpm
                    else:
                        bottleneck = "RPM"
                        theoretical_max_rpm = rpm_limited
                    
                    results.append({
                        "prompt_size": prompt_size,
                        "response_size": response_size,
                        "avg_input_tokens": avg_input_tokens,
                        "avg_output_tokens": avg_output_tokens,
                        "avg_total_tokens": avg_tokens,
                        "rpm_achieved": rpm,
                        "tpm_achieved": tpm,
                        "theoretical_max_rpm": theoretical_max_rpm,
                        "bottleneck": bottleneck
                    })
        
        return pd.DataFrame(results)
    
    def _generate_prompt_of_size(self, approx_tokens):
        """Generate a prompt of approximately the specified token count."""
        # Simple approximation: 1 token ≈ 4 characters or 0.75 words
        words_needed = int(approx_tokens * 0.75)
        
        # Generate placeholder text
        if words_needed <= 10:
            return f"Summarize this in {approx_tokens} tokens."
        
        # Generate longer text for larger token counts
        base_text = "This is a test prompt to evaluate throughput. " * (words_needed // 8 + 1)
        return base_text[:words_needed * 5]  # Rough character count
    
    def _format_request(self, prompt, max_tokens):
        """Format the request based on model type."""
        model_family = self.model_id.split('.')[0].lower()
        
        if "anthropic" in model_family:
            if "claude-3" in self.model_id:
                # Claude 3 models
                return {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            else:
                # Claude 2 and earlier
                return {
                    "prompt": f"Human: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": max_tokens,
                    "temperature": 0.7
                }
        elif "llama" in model_family or "meta" in model_family:
            return {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_gen_len": max_tokens,
                "temperature": 0.7
            }
        elif "titan" in model_family or "amazon" in model_family:
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": 0.7
                }
            }
        else:
            # Generic fallback
            return {
                "prompt": prompt,
                "max_tokens": max_tokens
            }
    
    def _extract_token_counts(self, response_body):
        """Extract token counts from the response based on model family."""
        model_family = self.model_id.split('.')[0].lower()
        
        if "anthropic" in model_family:
            # Claude models provide usage info
            usage = response_body.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
        elif "ai21" in model_family:
            # AI21 also provides token counts
            usage = response_body.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
        else:
            # For other models, make rough estimates
            # For actual implementation, you'd want to use a tokenizer
            output_text = ""
            if "generation" in response_body:
                output_text = response_body["generation"]
            elif "results" in response_body and len(response_body["results"]) > 0:
                output_text = response_body["results"][0].get("outputText", "")
            
            # Rough estimation based on word count
            input_tokens = len(str(response_body).split()) * 1.3
            output_tokens = len(output_text.split()) * 1.3
        
        return input_tokens, output_tokens
    
    def visualize_results(self, results_df, output_file="quota_optimization.svg"):
        """Create visualization of throughput results."""
        if results_df.empty:
            print("No results to visualize")
            return None
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Heatmap of RPM by prompt/response size
        pivot = results_df.pivot_table(
            values="rpm_achieved", 
            index="prompt_size", 
            columns="response_size"
        )
        
        im = axs[0, 0].imshow(pivot, cmap='viridis')
        axs[0, 0].set_title('Requests Per Minute')
        axs[0, 0].set_xlabel('Response Size (tokens)')
        axs[0, 0].set_ylabel('Prompt Size (tokens)')
        axs[0, 0].set_xticks(range(len(pivot.columns)))
        axs[0, 0].set_xticklabels(pivot.columns)
        axs[0, 0].set_yticks(range(len(pivot.index)))
        axs[0, 0].set_yticklabels(pivot.index)
        fig.colorbar(im, ax=axs[0, 0])
        
        # 2. Heatmap of TPM by prompt/response size
        pivot = results_df.pivot_table(
            values="tpm_achieved", 
            index="prompt_size", 
            columns="response_size"
        )
        
        im = axs[0, 1].imshow(pivot, cmap='plasma')
        axs[0, 1].set_title('Tokens Per Minute')
        axs[0, 1].set_xlabel('Response Size (tokens)')
        axs[0, 1].set_ylabel('Prompt Size (tokens)')
        axs[0, 1].set_xticks(range(len(pivot.columns)))
        axs[0, 1].set_xticklabels(pivot.columns)
        axs[0, 1].set_yticks(range(len(pivot.index)))
        axs[0, 1].set_yticklabels(pivot.index)
        fig.colorbar(im, ax=axs[0, 1])
        
        # 3. Scatter plot of total tokens vs RPM
        axs[1, 0].scatter(
            results_df["avg_total_tokens"],
            results_df["rpm_achieved"],
            c=results_df["bottleneck"].map({"RPM": "blue", "TPM": "red"}),
            alpha=0.7
        )
        axs[1, 0].set_title('RPM vs Token Count')
        axs[1, 0].set_xlabel('Total Tokens per Request')
        axs[1, 0].set_ylabel('Requests Per Minute')
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Add bottleneck legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='RPM Limited'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='TPM Limited')
        ]
        axs[1, 0].legend(handles=legend_elements)
        
        # 4. Bar chart of optimal configurations
        # Find the configuration with highest throughput
        best_row = results_df.loc[results_df["rpm_achieved"].idxmax()]
        
        # Create a stacked bar chart for the optimal configuration
        axs[1, 1].bar(
            ["Optimal Configuration"], 
            [best_row["avg_input_tokens"]], 
            label="Input Tokens"
        )
        axs[1, 1].bar(
            ["Optimal Configuration"], 
            [best_row["avg_output_tokens"]], 
            bottom=[best_row["avg_input_tokens"]], 
            label="Output Tokens"
        )
        
        axs[1, 1].set_title(f'Optimal Configuration: {best_row["rpm_achieved"]:.1f} RPM')
        axs[1, 1].set_ylabel('Token Count')
        axs[1, 1].text(
            0, best_row["avg_total_tokens"] + 50,
            f"Prompt: {best_row['prompt_size']} tokens\nResponse: {best_row['response_size']} tokens",
            ha='center'
        )
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_file, **SVG_CONFIG)
        plt.close()
        
        return output_file
```

## Strategies for Maximizing Throughput

### 1. When TPM is the Bottleneck

If your analysis shows you're limited by Tokens Per Minute (TPM):

```python
class TpmOptimizer:
    """Optimize throughput when TPM is the bottleneck."""
    
    def __init__(self, model_id, tpm_limit):
        """Initialize with model ID and TPM limit."""
        self.model_id = model_id
        self.tpm_limit = tpm_limit
        self.bedrock = get_bedrock_client()
    
    def optimize_prompt(self, original_prompt, max_output_tokens):
        """Optimize a prompt to reduce token count while preserving functionality."""
        # Strategies to implement:
        
        # 1. Remove unnecessary context and boilerplate
        prompt = self._remove_boilerplate(original_prompt)
        
        # 2. Use shorthand instructions when possible
        prompt = self._use_shorthand(prompt)
        
        # 3. Truncate examples if present
        prompt = self._truncate_examples(prompt)
        
        # 4. Optimize formatting (newlines, spacing)
        prompt = self._optimize_formatting(prompt)
        
        # Validate the optimized prompt maintains functionality
        # In a real implementation, you might want to test-run the optimized prompt
        
        return prompt
    
    def optimize_output_length(self, original_max_tokens, min_acceptable_tokens):
        """Find optimal output length for maximum throughput."""
        # Start with requested length
        optimized_length = original_max_tokens
        
        # If it's very large, consider reducing
        if original_max_tokens > 1000:
            # Calculate optimal length based on prompt size and TPM limit
            # This would require more complex analysis in a real implementation
            optimized_length = max(min_acceptable_tokens, original_max_tokens // 2)
        
        return optimized_length
    
    def split_request(self, large_prompt, max_tokens_per_chunk=1000):
        """Split a large request into multiple smaller requests."""
        # For very large prompts, split into multiple requests
        if len(large_prompt.split()) > 1500:  # Rough token estimate
            # This is a simplified version - a real implementation would be more sophisticated
            words = large_prompt.split()
            chunks = []
            
            for i in range(0, len(words), 1000):
                chunk = " ".join(words[i:i+1000])
                chunks.append(chunk)
            
            return chunks
        
        return [large_prompt]  # No splitting needed
    
    def _remove_boilerplate(self, prompt):
        """Remove unnecessary instructions and boilerplate."""
        # Example implementation - would be more sophisticated in practice
        boilerplate_phrases = [
            "You are a helpful assistant",
            "Answer the following question",
            "I want you to",
            "Please provide"
        ]
        
        result = prompt
        for phrase in boilerplate_phrases:
            result = result.replace(phrase, "")
        
        return result.strip()
    
    def _use_shorthand(self, prompt):
        """Replace verbose instructions with shorthand."""
        # Example replacements
        replacements = {
            "Please summarize the following text": "Summarize:",
            "Extract the key points from": "Key points:",
            "Answer with a brief explanation": "Brief answer:"
        }
        
        result = prompt
        for verbose, shorthand in replacements.items():
            result = result.replace(verbose, shorthand)
        
        return result
    
    def _truncate_examples(self, prompt):
        """Truncate or reduce examples in the prompt."""
        # This is a simplified implementation
        if "Example 1:" in prompt:
            # Keep only the first example
            parts = prompt.split("Example 2:")
            return parts[0]
        
        return prompt
    
    def _optimize_formatting(self, prompt):
        """Optimize whitespace and formatting."""
        # Remove extra newlines
        result = "\n".join([line for line in prompt.split("\n") if line.strip()])
        
        # Remove extra spaces
        result = " ".join([word for word in result.split() if word])
        
        return result
```

#### TPM Optimization Techniques:

1. **Prompt Compression**
   - Remove unnecessary context and boilerplate text
   - Use shorthand instructions when possible
   - Remove or truncate examples
   - Optimize formatting (newlines, spacing)

2. **Output Length Control**
   - Set the minimum necessary output length
   - Use streaming to get early results before generation completes
   - Implement early stopping when useful information is received

3. **Request Splitting**
   - For large contexts, split into multiple smaller requests
   - Implement a map-reduce pattern for processing large documents
   - Use specialized summarization endpoints for large texts

### 2. When RPM is the Bottleneck

If your analysis shows you're limited by Requests Per Minute (RPM):

```python
class RpmOptimizer:
    """Optimize throughput when RPM is the bottleneck."""
    
    def __init__(self, model_id, rpm_limit):
        """Initialize with model ID and RPM limit."""
        self.model_id = model_id
        self.rpm_limit = rpm_limit
        self.bedrock = get_bedrock_client()
    
    def batch_requests(self, prompts, max_tokens_per_batch=8000):
        """Combine multiple small requests into batches."""
        # This is a simplified implementation
        batched_prompts = []
        current_batch = []
        current_token_estimate = 0
        
        for prompt in prompts:
            # Rough token estimate based on word count
            token_estimate = len(prompt.split()) * 1.3
            
            if current_token_estimate + token_estimate > max_tokens_per_batch:
                # Finalize current batch
                batched_prompts.append(self._create_batch_prompt(current_batch))
                current_batch = [prompt]
                current_token_estimate = token_estimate
            else:
                # Add to current batch
                current_batch.append(prompt)
                current_token_estimate += token_estimate
        
        # Add final batch
        if current_batch:
            batched_prompts.append(self._create_batch_prompt(current_batch))
        
        return batched_prompts
    
    def _create_batch_prompt(self, prompts):
        """Create a batched prompt from multiple individual prompts."""
        if len(prompts) == 1:
            return prompts[0]
        
        batch_prompt = "Process each of the following queries and provide separate answers:\n\n"
        
        for i, prompt in enumerate(prompts):
            batch_prompt += f"Query {i+1}: {prompt}\n\n"
        
        batch_prompt += "Format your response with numbered answers corresponding to each query."
        
        return batch_prompt
    
    def extract_batch_responses(self, batch_response, batch_size):
        """Extract individual responses from a batched response."""
        # Simple extraction based on numbered answers
        # A more robust implementation would use regex patterns
        responses = []
        
        # Split by numbered responses (e.g., "Answer 1:", "Response 1:", etc.)
        parts = []
        for i in range(1, batch_size + 1):
            for pattern in [f"Answer {i}:", f"Response {i}:", f"Query {i} response:"]:
                if pattern in batch_response:
                    parts = batch_response.split(pattern)
                    if len(parts) > i:
                        # Found a response for this query
                        if i < batch_size:
                            # If not the last response, extract until the next pattern
                            next_pattern = None
                            for next_i in range(i+1, batch_size + 1):
                                for p in [f"Answer {next_i}:", f"Response {next_i}:", f"Query {next_i} response:"]:
                                    if p in parts[i]:
                                        next_pattern = p
                                        break
                                if next_pattern:
                                    break
                            
                            if next_pattern:
                                response_parts = parts[i].split(next_pattern)
                                responses.append(response_parts[0].strip())
                            else:
                                responses.append(parts[i].strip())
                        else:
                            # This is the last response
                            responses.append(parts[i].strip())
        
        # If we couldn't extract the expected number of responses,
        # fill with empty strings to match the expected batch size
        while len(responses) < batch_size:
            responses.append("")
        
        return responses
```

#### RPM Optimization Techniques:

1. **Request Batching**
   - Combine multiple small requests into a single larger request
   - Process batch outputs to extract individual responses
   - Balance batch size against token limits

2. **Parallel Processing**
   - Implement asynchronous processing for non-interactive workloads
   - Use threading or async techniques to maximize utilization
   - Implement queue management to smooth out request patterns

3. **Response Caching**
   - Cache common queries and responses
   - Implement similarity matching for near-duplicate queries
   - Use tiered caching strategies (memory, local disk, distributed cache)

### 3. Implementing a Token Bucket Algorithm for Balanced Rate Limiting

To maintain optimal utilization without hitting quota limits:

```python
class TokenBucketRateLimiter:
    """
    Implement a token bucket algorithm for rate limiting Bedrock requests.
    This balances both RPM and TPM constraints.
    """
    
    def __init__(self, rpm_limit, tpm_limit, burst_factor=1.5):
        """
        Initialize the rate limiter.
        
        Args:
            rpm_limit: Requests per minute limit
            tpm_limit: Tokens per minute limit
            burst_factor: Allow bursts up to this factor of the limit
        """
        # Convert to per-second rates
        self.rps_limit = rpm_limit / 60
        self.tps_limit = tpm_limit / 60
        self.burst_factor = burst_factor
        
        # Initialize buckets
        self.request_bucket = self.rps_limit * burst_factor
        self.token_bucket = self.tps_limit * burst_factor
        
        # Track last refill time
        self.last_refill = time.time()
    
    def _refill_buckets(self):
        """Refill the token buckets based on elapsed time."""
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
        self._refill_buckets()
        
        # Check if we have capacity for both one request and the token count
        if self.request_bucket >= 1 and self.token_bucket >= estimated_tokens:
            self.request_bucket -= 1
            self.token_bucket -= estimated_tokens
            return True
        
        return False
    
    def wait_time_for_next_request(self, estimated_tokens):
        """
        Calculate wait time until the next request can be processed.
        
        Args:
            estimated_tokens: Estimated token count for this request
            
        Returns:
            Time in seconds to wait
        """
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
```

## Putting It All Together: A Complete Throughput Optimization System

Now let's implement a complete system that balances all constraints:

```python
class ThroughputOptimizer:
    """
    Complete system for optimizing AWS Bedrock inference throughput 
    under token size and request limits.
    """
    
    def __init__(self, model_id, rpm_limit=None, tpm_limit=None):
        """Initialize the throughput optimizer."""
        self.model_id = model_id
        self.bedrock = get_bedrock_client()
        
        # Detect limits if not provided
        if rpm_limit is None or tpm_limit is None:
            detected_limits = self._detect_quota_limits()
            rpm_limit = rpm_limit or detected_limits.get("rpm", 100)
            tpm_limit = tpm_limit or detected_limits.get("tpm", 10000)
        
        # Initialize rate limiter
        self.rate_limiter = TokenBucketRateLimiter(rpm_limit, tpm_limit)
        
        # Initialize optimizers
        self.tpm_optimizer = TpmOptimizer(model_id, tpm_limit)
        self.rpm_optimizer = RpmOptimizer(model_id, rpm_limit)
        
        # Calculate theoretical max throughput
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        
        # Keep performance metrics
        self.performance_metrics = {
            "requests_processed": 0,
            "requests_throttled": 0,
            "total_tokens_processed": 0,
            "avg_tokens_per_request": 0,
            "avg_wait_time": 0
        }
    
    def _detect_quota_limits(self):
        """Detect quota limits from AWS Service Quotas."""
        # Similar implementation as in QuotaAnalyzer
        try:
            session = boto3.Session(profile_name="aws")
            quotas = session.client('service-quotas')
            
            # Extract model family from ID
            model_family = self.model_id.split('.')[0].lower()
            
            # List all quotas for bedrock
            response = quotas.list_service_quotas(ServiceCode='bedrock')
            
            rpm_limit = None
            tpm_limit = None
            
            # Look for quota codes matching this model
            for quota in response['Quotas']:
                quota_name = quota['QuotaName'].lower()
                if model_family in quota_name:
                    if "requests per minute" in quota_name:
                        rpm_limit = quota['Value']
                    elif "tokens per minute" in quota_name:
                        tpm_limit = quota['Value']
            
            return {
                "rpm": rpm_limit,
                "tpm": tpm_limit
            }
        except Exception:
            # Return default values if Service Quotas API is unavailable
            return {
                "rpm": 100,
                "tpm": 10000
            }
    
    def process_request(self, prompt, max_tokens, optimize=True):
        """
        Process a single request with throughput optimization.
        
        Args:
            prompt: The prompt text
            max_tokens: Maximum tokens to generate
            optimize: Whether to optimize the prompt/response
            
        Returns:
            Response text and metadata
        """
        # Estimate token count
        input_token_estimate = len(prompt.split()) * 1.3
        total_token_estimate = input_token_estimate + max_tokens
        
        # Optimize if requested
        if optimize:
            if self.tpm_limit / self.rpm_limit < total_token_estimate:
                # TPM is likely the bottleneck
                prompt = self.tpm_optimizer.optimize_prompt(prompt, max_tokens)
                max_tokens = self.tpm_optimizer.optimize_output_length(max_tokens, max_tokens // 2)
                # Re-estimate after optimization
                input_token_estimate = len(prompt.split()) * 1.3
                total_token_estimate = input_token_estimate + max_tokens
        
        # Check rate limiter
        if not self.rate_limiter.check_and_consume(total_token_estimate):
            # Calculate wait time
            wait_time = self.rate_limiter.wait_time_for_next_request(total_token_estimate)
            
            # Update metrics
            self.performance_metrics["requests_throttled"] += 1
            self.performance_metrics["avg_wait_time"] = (
                (self.performance_metrics["avg_wait_time"] * self.performance_metrics["requests_processed"]) +
                wait_time
            ) / (self.performance_metrics["requests_processed"] + 1)
            
            # Wait before proceeding
            time.sleep(wait_time)
        
        # Format request
        payload = self._format_request(prompt, max_tokens)
        
        # Process request
        try:
            start_time = time.time()
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract response text and token counts
            output_text, input_tokens, output_tokens = self._process_response(response_body)
            
            # Update metrics
            elapsed = time.time() - start_time
            self.performance_metrics["requests_processed"] += 1
            self.performance_metrics["total_tokens_processed"] += input_tokens + output_tokens
            self.performance_metrics["avg_tokens_per_request"] = (
                self.performance_metrics["total_tokens_processed"] / 
                self.performance_metrics["requests_processed"]
            )
            
            return {
                "text": output_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "elapsed_seconds": elapsed
            }
            
        except Exception as e:
            # Handle error and return error information
            return {
                "error": str(e)
            }
    
    def process_batch(self, prompts, max_tokens_per_response, optimize=True):
        """
        Process multiple prompts optimally.
        
        Args:
            prompts: List of prompts to process
            max_tokens_per_response: Maximum tokens per individual response
            optimize: Whether to optimize processing
            
        Returns:
            List of responses with metadata
        """
        if not prompts:
            return []
        
        if len(prompts) == 1:
            # Single prompt case
            return [self.process_request(prompts[0], max_tokens_per_response, optimize)]
        
        # For multiple prompts
        if optimize:
            # Check if RPM or TPM is the bottleneck
            avg_tokens_per_request = self.performance_metrics.get("avg_tokens_per_request", 1000)
            if self.tpm_limit / self.rpm_limit < avg_tokens_per_request:
                # TPM is the bottleneck, process individually with optimized prompts
                return [self.process_request(p, max_tokens_per_response, True) for p in prompts]
            else:
                # RPM is the bottleneck, batch requests
                batched_prompts = self.rpm_optimizer.batch_requests(
                    prompts, 
                    max_tokens_per_batch=min(8000, self.tpm_limit / 10)
                )
                
                results = []
                for batch_prompt in batched_prompts:
                    # Process the batch
                    batch_size = batch_prompt.count("Query ") if "Query " in batch_prompt else 1
                    batch_response = self.process_request(
                        batch_prompt, 
                        max_tokens_per_response * batch_size,
                        False  # Don't optimize the batch prompt further
                    )
                    
                    if "error" in batch_response:
                        # If batch processing failed, fall back to individual processing
                        return [self.process_request(p, max_tokens_per_response, True) for p in prompts]
                    
                    # Extract individual responses
                    individual_texts = self.rpm_optimizer.extract_batch_responses(
                        batch_response["text"],
                        batch_size
                    )
                    
                    # Create individual response records
                    batch_input_tokens = batch_response["input_tokens"]
                    batch_output_tokens = batch_response["output_tokens"]
                    
                    for text in individual_texts:
                        # Rough distribution of tokens
                        input_tokens = batch_input_tokens / batch_size
                        output_tokens = batch_output_tokens / batch_size
                        
                        results.append({
                            "text": text,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "elapsed_seconds": batch_response["elapsed_seconds"] / batch_size,
                            "batched": True
                        })
                
                return results
        else:
            # No optimization, process individually
            return [self.process_request(p, max_tokens_per_response, False) for p in prompts]
    
    def process_parallel(self, prompts, max_tokens_per_response, max_concurrent=5):
        """
        Process multiple prompts in parallel while respecting rate limits.
        
        Args:
            prompts: List of prompts to process
            max_tokens_per_response: Maximum tokens per response
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of responses
        """
        import threading
        from queue import Queue
        
        if not prompts:
            return []
        
        # Initialize queues
        input_queue = Queue()
        result_queue = Queue()
        
        # Add all prompts to the input queue
        for i, prompt in enumerate(prompts):
            input_queue.put((i, prompt))
        
        # Define worker thread function
        def worker():
            while not input_queue.empty():
                try:
                    # Get next prompt from queue
                    idx, prompt = input_queue.get()
                    
                    # Process it
                    response = self.process_request(prompt, max_tokens_per_response, True)
                    
                    # Add to result queue with original index
                    result_queue.put((idx, response))
                    
                    # Mark task as done
                    input_queue.task_done()
                except Exception as e:
                    # Handle errors
                    result_queue.put((idx, {"error": str(e)}))
                    input_queue.task_done()
        
        # Create worker threads
        threads = []
        for _ in range(min(max_concurrent, len(prompts))):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for all tasks to complete
        input_queue.join()
        
        # Collect results in original order
        results = [None] * len(prompts)
        while not result_queue.empty():
            idx, response = result_queue.get()
            results[idx] = response
        
        return results
    
    def _format_request(self, prompt, max_tokens):
        """Format request based on model type."""
        # Similar to previous implementation
        model_family = self.model_id.split('.')[0].lower()
        
        if "anthropic" in model_family:
            if "claude-3" in self.model_id:
                # Claude 3 models
                return {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            else:
                # Claude 2 and earlier
                return {
                    "prompt": f"Human: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": max_tokens,
                    "temperature": 0.7
                }
        # Other model formats would be included here
        # ...
    
    def _process_response(self, response_body):
        """Extract text and token counts from response."""
        # Similar to previous implementation
        model_family = self.model_id.split('.')[0].lower()
        
        # Extract text based on model type
        if "anthropic" in model_family:
            if "claude-3" in self.model_id:
                output_text = response_body.get('content', [{}])[0].get('text', '')
            else:
                output_text = response_body.get('completion', '')
                
            # Get token counts
            usage = response_body.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
        # Other model response handling would be included here
        # ...
        else:
            # Default fallback
            output_text = str(response_body)
            input_tokens = 0
            output_tokens = 0
        
        return output_text, input_tokens, output_tokens
    
    def get_metrics(self):
        """Get current performance metrics."""
        return self.performance_metrics
    
    def analyze_bottleneck(self):
        """Analyze current bottleneck based on performance data."""
        if self.performance_metrics["requests_processed"] < 10:
            return "Not enough data to determine bottleneck"
        
        avg_tokens = self.performance_metrics["avg_tokens_per_request"]
        
        # Calculate theoretical max RPM based on TPM limit
        max_rpm_under_tpm = self.tpm_limit / avg_tokens if avg_tokens > 0 else float('inf')
        
        if max_rpm_under_tpm < self.rpm_limit:
            return {
                "bottleneck": "TPM",
                "theoretical_max_rpm": max_rpm_under_tpm,
                "avg_tokens_per_request": avg_tokens,
                "recommendation": "Focus on reducing token count per request"
            }
        else:
            return {
                "bottleneck": "RPM",
                "theoretical_max_rpm": self.rpm_limit,
                "avg_tokens_per_request": avg_tokens,
                "recommendation": "Focus on batching requests or using parallelism"
            }
```

## Example Usage

Here's how to use the optimization system:

```python
# Initialize the throughput optimizer
optimizer = ThroughputOptimizer(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

# Simple single request
response = optimizer.process_request(
    "Summarize the benefits of AWS Bedrock in a concise paragraph.",
    max_tokens=100
)
print(f"Response: {response['text']}")
print(f"Tokens: {response['total_tokens']}")

# Batch processing multiple requests
prompts = [
    "What are the key features of AWS Bedrock?",
    "How does AWS Bedrock handle quota limits?",
    "What foundation models are available in AWS Bedrock?",
    "Compare AWS Bedrock to Amazon SageMaker.",
    "How can I optimize throughput in AWS Bedrock?"
]

responses = optimizer.process_batch(prompts, max_tokens_per_response=150)
for i, resp in enumerate(responses):
    print(f"\nQuestion {i+1}: {prompts[i]}")
    print(f"Answer: {resp['text']}")
    print(f"Tokens: {resp['total_tokens']}")

# Check metrics and analyze bottleneck
metrics = optimizer.get_metrics()
print("\nPerformance Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")

bottleneck = optimizer.analyze_bottleneck()
print("\nBottleneck Analysis:")
print(f"  Current Bottleneck: {bottleneck['bottleneck']}")
print(f"  Recommendation: {bottleneck['recommendation']}")
```

## Measuring and Visualizing Optimization Impact

To evaluate the impact of your optimizations:

```python
def measure_optimization_impact(model_id, prompt, max_tokens, repetitions=10):
    """Measure the impact of optimization on throughput."""
    # Create analyzer and optimizer
    analyzer = QuotaAnalyzer(model_id)
    optimizer = ThroughputOptimizer(model_id)
    
    # Measure baseline throughput (no optimization)
    baseline_start = time.time()
    baseline_tokens = 0
    baseline_successes = 0
    
    for _ in range(repetitions):
        response = optimizer.process_request(prompt, max_tokens, optimize=False)
        if "error" not in response:
            baseline_tokens += response["total_tokens"]
            baseline_successes += 1
    
    baseline_time = time.time() - baseline_start
    baseline_rpm = (baseline_successes / baseline_time) * 60
    baseline_tpm = (baseline_tokens / baseline_time) * 60
    
    # Measure optimized throughput
    optimized_start = time.time()
    optimized_tokens = 0
    optimized_successes = 0
    
    for _ in range(repetitions):
        response = optimizer.process_request(prompt, max_tokens, optimize=True)
        if "error" not in response:
            optimized_tokens += response["total_tokens"]
            optimized_successes += 1
    
    optimized_time = time.time() - optimized_start
    optimized_rpm = (optimized_successes / optimized_time) * 60
    optimized_tpm = (optimized_tokens / optimized_time) * 60
    
    # Calculate impact
    rpm_improvement = ((optimized_rpm / baseline_rpm) - 1) * 100 if baseline_rpm > 0 else 0
    tpm_improvement = ((optimized_tpm / baseline_tpm) - 1) * 100 if baseline_tpm > 0 else 0
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart comparing before/after
    x = ['Requests Per Minute', 'Tokens Per Minute']
    baseline = [baseline_rpm, baseline_tpm]
    optimized = [optimized_rpm, optimized_tpm]
    
    x_pos = np.arange(len(x))
    width = 0.35
    
    ax.bar(x_pos - width/2, baseline, width, label='Baseline', color='#1f77b4')
    ax.bar(x_pos + width/2, optimized, width, label='Optimized', color='#2ca02c')
    
    # Add labels and formatting
    ax.set_title('Throughput Optimization Impact', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.set_ylabel('Per Minute', fontsize=12)
    ax.legend()
    
    # Add improvement percentages
    for i, (base, opt, imp) in enumerate(zip(baseline, optimized, [rpm_improvement, tpm_improvement])):
        ax.text(i - width/2, base + 1, f"{base:.1f}", ha='center', va='bottom')
        ax.text(i + width/2, opt + 1, f"{opt:.1f}", ha='center', va='bottom')
        ax.text(i, (base + opt)/2, f"+{imp:.1f}%", ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig("optimization_impact.svg", **SVG_CONFIG)
    plt.close()
    
    return {
        "baseline": {
            "rpm": baseline_rpm,
            "tpm": baseline_tpm,
            "tokens_per_request": baseline_tokens / baseline_successes if baseline_successes > 0 else 0
        },
        "optimized": {
            "rpm": optimized_rpm,
            "tpm": optimized_tpm,
            "tokens_per_request": optimized_tokens / optimized_successes if optimized_successes > 0 else 0
        },
        "improvement": {
            "rpm_percent": rpm_improvement,
            "tpm_percent": tpm_improvement
        }
    }
```

## Conclusion

Optimizing AWS Bedrock inference throughput under token size and request quota limits requires a balanced approach. By identifying your bottleneck (TPM vs. RPM) and implementing the appropriate optimization strategies, you can significantly increase your effective throughput.

Key takeaways:

1. **Analyze Your Workload First**: Determine whether you're token-limited or request-limited
2. **Optimize Accordingly**:
   - If token-limited: Focus on prompt compression and output length control
   - If request-limited: Implement batching and parallel processing
3. **Implement Rate Limiting**: Use a token bucket algorithm to balance both constraints
4. **Adapt Dynamically**: Monitor performance and adjust strategies as workload patterns change

By following these techniques, you can maximize the value from your AWS Bedrock quotas and build more efficient, cost-effective AI applications.