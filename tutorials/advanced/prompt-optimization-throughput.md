# Optimizing Throughput with Model-Specific Prompt Engineering

This tutorial explores how to maximize throughput within AWS Bedrock quota limits by optimizing prompts for specific models.

## Objective

By the end of this tutorial, you'll be able to:
1. Structure prompts efficiently for different model families
2. Measure and optimize token usage to maximize throughput
3. Implement adaptive prompt strategies based on quota consumption
4. Create a system that automatically selects optimal prompt structures

## Prerequisites

- Understanding of AWS Bedrock and its quota system
- Familiarity with different foundation models
- Python 3.8+ with boto3 installed
- AWS account with Bedrock access

## Understanding the Relationship Between Prompts and Throughput

AWS Bedrock imposes quota limits measured in:
- **Tokens Per Minute (TPM)** - Total input+output tokens processed
- **Requests Per Minute (RPM)** - Total API calls

The way you structure prompts directly impacts:
1. **Input token count** - More efficient prompts use fewer tokens
2. **Output token count** - Well-constrained prompts generate smaller responses
3. **Processing speed** - Some prompt structures process faster than others
4. **Error rates** - Poorly structured prompts may cause errors, wasting quota

## Step 1: Measuring Baseline Token Usage

Let's start by creating a function to measure token usage for different prompt structures:

```python
import boto3
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from utils.profile_manager import get_profile
from utils.visualization_config import SVG_CONFIG

def measure_token_usage(model_id, prompt_variants):
    """
    Measure token usage for different prompt structures with the same model.
    
    Args:
        model_id: The model identifier to test
        prompt_variants: Dictionary mapping variant names to prompt payloads
        
    Returns:
        Dictionary with token usage metrics for each variant
    """
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime')
    
    results = {}
    
    for variant_name, payload in prompt_variants.items():
        try:
            # Invoke the model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(payload)
            )
            
            # Parse response and extract token counts
            response_body = json.loads(response['body'].read())
            
            # Extract metrics based on model type
            if "anthropic" in model_id:
                input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
                output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
                output_text = response_body['content'][0]['text']
            elif "meta" in model_id:
                # Llama doesn't provide token counts directly, estimate them
                input_text = payload['prompt']
                output_text = response_body['generation']
                input_tokens = len(input_text.split()) * 1.3  # Rough estimate
                output_tokens = len(output_text.split()) * 1.3  # Rough estimate
            elif "ai21" in model_id:
                input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
                output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
                output_text = response_body.get('completions', [{}])[0].get('data', {}).get('text', '')
            else:
                # For other models, make a rough estimate
                input_text = json.dumps(payload)
                output_text = str(response_body)
                if 'results' in response_body and len(response_body['results']) > 0:
                    output_text = response_body['results'][0].get('outputText', '')
                
                input_tokens = len(input_text.split()) * 1.3  # Rough estimate
                output_tokens = len(output_text.split()) * 1.3  # Rough estimate
            
            # Store the results
            results[variant_name] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "output_sample": output_text[:100] + "..." if len(output_text) > 100 else output_text
            }
            
        except Exception as e:
            results[variant_name] = {
                "error": str(e)
            }
    
    return results

def visualize_token_usage(results, output_file="token_usage.svg"):
    """Create an SVG visualization comparing token usage across prompt variants"""
    variants = list(results.keys())
    input_tokens = [results[v].get("input_tokens", 0) for v in variants]
    output_tokens = [results[v].get("output_tokens", 0) for v in variants]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create stacked bar chart
    bar_width = 0.6
    bars = ax.bar(variants, input_tokens, bar_width, label='Input Tokens', color='#1f77b4')
    bars = ax.bar(variants, output_tokens, bar_width, bottom=input_tokens, 
                 label='Output Tokens', color='#ff7f0e')
    
    # Add labels and styling
    ax.set_title('Token Usage by Prompt Variant', fontsize=16)
    ax.set_ylabel('Token Count', fontsize=12)
    ax.set_xlabel('Prompt Variant', fontsize=12)
    ax.legend()
    
    # Add value labels on bars
    for i, variant in enumerate(variants):
        total = results[variant].get("total_tokens", 0)
        ax.text(i, total + 20, f'{total:.0f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save as SVG
    plt.savefig(output_file, **SVG_CONFIG)
    plt.close()
    
    return output_file
```

## Step 2: Creating Model-Specific Prompt Templates

Now, let's create templates for each model family:

```python
class BedrockPromptTemplates:
    """A class containing model-specific prompt templates"""
    
    @staticmethod
    def format_for_claude(instruction, content=None, system=None):
        """Format a prompt for Claude models"""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        if content:
            messages.append({"role": "user", "content": f"{instruction}\n\n{content}"})
        else:
            messages.append({"role": "user", "content": instruction})
        
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": messages
        }
    
    @staticmethod
    def format_for_llama(instruction, content=None):
        """Format a prompt for Llama models"""
        if content:
            prompt = f"<s>[INST] {instruction}\n\n{content} [/INST]"
        else:
            prompt = f"<s>[INST] {instruction} [/INST]"
        
        return {
            "prompt": prompt,
            "max_gen_len": 1000
        }
    
    @staticmethod
    def format_for_titan(instruction, content=None):
        """Format a prompt for Titan models"""
        if content:
            input_text = f"{instruction}\n\n{content}"
        else:
            input_text = instruction
        
        return {
            "inputText": input_text,
            "textGenerationConfig": {
                "maxTokenCount": 1000
            }
        }
    
    @staticmethod
    def format_for_cohere(instruction, content=None):
        """Format a prompt for Cohere models"""
        if content:
            prompt = f"{instruction}\n\n{content}"
        else:
            prompt = instruction
        
        return {
            "prompt": prompt,
            "max_tokens": 1000
        }
    
    @staticmethod
    def format_for_ai21(instruction, content=None):
        """Format a prompt for AI21 models"""
        if content:
            prompt = f"{instruction}\n\n{content}"
        else:
            prompt = instruction
        
        return {
            "prompt": prompt,
            "maxTokens": 1000
        }
    
    @staticmethod
    def format_for_any_model(model_id, instruction, content=None, system=None):
        """Format a prompt for any supported model"""
        model_family = model_id.split('.')[0].lower() if '.' in model_id else model_id.lower()
        
        if "anthropic" in model_family:
            return BedrockPromptTemplates.format_for_claude(instruction, content, system)
        elif "meta" in model_family or "llama" in model_family:
            return BedrockPromptTemplates.format_for_llama(instruction, content)
        elif "titan" in model_family or "amazon" in model_family:
            return BedrockPromptTemplates.format_for_titan(instruction, content)
        elif "cohere" in model_family:
            return BedrockPromptTemplates.format_for_cohere(instruction, content)
        elif "ai21" in model_family:
            return BedrockPromptTemplates.format_for_ai21(instruction, content)
        else:
            # Default to Titan format
            return BedrockPromptTemplates.format_for_titan(instruction, content)
```

## Step 3: Testing Different Prompt Structures for Throughput

Let's create a test to compare different prompt structures for the same task:

```python
def test_prompt_throughput(model_id, task_description, content=None, repeat_count=5):
    """
    Test how different prompt structures affect throughput for the same task.
    
    Args:
        model_id: The model identifier to test
        task_description: The task to perform
        content: Optional content to include in the prompt
        repeat_count: Number of times to repeat each test
        
    Returns:
        Dictionary with performance metrics
    """
    # Define different prompt variants for the same task
    variants = {
        "detailed": {
            "instruction": f"Please perform the following task: {task_description}. Provide a detailed and thorough response that covers all aspects of the question.",
            "system": "You are a helpful assistant that provides comprehensive responses."
        },
        "concise": {
            "instruction": f"Task: {task_description}\nBe brief and direct.",
            "system": "Provide concise, to-the-point responses."
        },
        "structured": {
            "instruction": f"Follow these steps:\n1. Understand this task: {task_description}\n2. Analyze the key components\n3. Provide a structured response with clear headings",
            "system": "You provide well-structured, organized responses."
        },
        "minimal": {
            "instruction": f"{task_description}",
            "system": None
        }
    }
    
    # Create formatted prompts for each variant
    prompt_variants = {}
    for variant_name, variant_config in variants.items():
        prompt_variants[variant_name] = BedrockPromptTemplates.format_for_any_model(
            model_id, 
            variant_config["instruction"], 
            content, 
            variant_config["system"]
        )
    
    # Measure token usage for each variant
    token_results = measure_token_usage(model_id, prompt_variants)
    
    # Measure throughput (requests per minute) for each variant
    throughput_results = {}
    
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime')
    
    for variant_name, payload in prompt_variants.items():
        start_time = time.time()
        successful = 0
        failed = 0
        total_tokens = 0
        
        for _ in range(repeat_count):
            try:
                response = bedrock_runtime.invoke_model(
                    modelId=model_id,
                    body=json.dumps(payload)
                )
                
                # Extract token usage based on model type
                response_body = json.loads(response['body'].read())
                
                if "anthropic" in model_id:
                    input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
                    output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
                else:
                    # Use the previously measured token counts as estimates
                    input_tokens = token_results[variant_name].get("input_tokens", 0)
                    output_tokens = token_results[variant_name].get("output_tokens", 0)
                
                total_tokens += input_tokens + output_tokens
                successful += 1
                
                # Brief pause to avoid throttling
                time.sleep(0.2)
                
            except Exception as e:
                failed += 1
                print(f"Error with {variant_name}: {str(e)}")
        
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        throughput_results[variant_name] = {
            "successful_requests": successful,
            "failed_requests": failed,
            "total_tokens_processed": total_tokens,
            "elapsed_time_seconds": elapsed_time,
            "requests_per_minute": (successful / elapsed_time) * 60 if elapsed_time > 0 else 0,
            "tokens_per_minute": (total_tokens / elapsed_time) * 60 if elapsed_time > 0 else 0
        }
    
    # Combine results
    combined_results = {
        "model_id": model_id,
        "task": task_description,
        "token_usage": token_results,
        "throughput": throughput_results
    }
    
    return combined_results

def visualize_throughput_comparison(results, output_file="throughput_comparison.svg"):
    """Create an SVG visualization comparing throughput metrics"""
    variants = list(results["throughput"].keys())
    
    # Extract metrics
    rpm_values = [results["throughput"][v]["requests_per_minute"] for v in variants]
    tpm_values = [results["throughput"][v]["tokens_per_minute"] for v in variants]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Requests per minute subplot
    bar_width = 0.6
    bars1 = ax1.bar(variants, rpm_values, bar_width, color='#2ca02c')
    
    ax1.set_title(f'Requests Per Minute by Prompt Variant - {results["model_id"]}', fontsize=16)
    ax1.set_ylabel('Requests Per Minute', fontsize=12)
    
    # Add value labels
    for i, v in enumerate(rpm_values):
        ax1.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)
    
    # Tokens per minute subplot
    bars2 = ax2.bar(variants, tpm_values, bar_width, color='#d62728')
    
    ax2.set_title(f'Tokens Per Minute by Prompt Variant - {results["model_id"]}', fontsize=16)
    ax2.set_ylabel('Tokens Per Minute', fontsize=12)
    ax2.set_xlabel('Prompt Variant', fontsize=12)
    
    # Add value labels
    for i, v in enumerate(tpm_values):
        ax2.text(i, v + 100, f'{v:.0f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save as SVG
    plt.savefig(output_file, **SVG_CONFIG)
    plt.close()
    
    return output_file
```

## Step 4: Implementing a Quota-Aware Prompt Optimizer

Now let's create a system that adapts prompt structures based on quota constraints:

```python
class QuotaAwarePromptOptimizer:
    """
    A class that optimizes prompt structures based on quota constraints.
    """
    
    def __init__(self, model_id, quota_limits=None):
        """
        Initialize the optimizer.
        
        Args:
            model_id: The model identifier to optimize for
            quota_limits: Dictionary with "rpm" and "tpm" limits
        """
        self.model_id = model_id
        self.profile_name = get_profile()
        self.session = boto3.Session(profile_name=self.profile_name)
        self.bedrock_runtime = self.session.client('bedrock-runtime')
        
        # Set quota limits
        if quota_limits:
            self.quota_limits = quota_limits
        else:
            # Try to detect limits from service quotas
            self.quota_limits = self._detect_quota_limits()
        
        # Initialize prompt template storage
        self.prompt_variants = {}
        self.variant_performance = {}
    
    def _detect_quota_limits(self):
        """Try to detect quota limits from AWS Service Quotas"""
        try:
            service_quotas = self.session.client('service-quotas')
            
            # Extract model family from ID
            model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else self.model_id.lower()
            
            # Get all bedrock quotas
            response = service_quotas.list_service_quotas(ServiceCode='bedrock')
            
            rpm_limit = None
            tpm_limit = None
            
            # Look for relevant quota codes
            for quota in response['Quotas']:
                if model_family in quota['QuotaName'].lower():
                    if "requests per minute" in quota['QuotaName'].lower():
                        rpm_limit = quota['Value']
                    elif "tokens per minute" in quota['QuotaName'].lower():
                        tpm_limit = quota['Value']
            
            # If found, return them
            if rpm_limit or tpm_limit:
                return {
                    "rpm": rpm_limit,
                    "tpm": tpm_limit
                }
        
        except Exception as e:
            print(f"Error detecting quota limits: {str(e)}")
        
        # Default conservative limits
        return {
            "rpm": 100,
            "tpm": 10000
        }
    
    def add_prompt_variant(self, variant_name, generator_function):
        """
        Add a prompt variant to the optimizer.
        
        Args:
            variant_name: Name to identify this variant
            generator_function: Function that takes task and content and returns a prompt
        """
        self.prompt_variants[variant_name] = generator_function
    
    def benchmark_variants(self, task, content=None, repeat_count=3):
        """
        Test all prompt variants to measure their performance.
        
        Args:
            task: The task description
            content: Optional content to include
            repeat_count: Number of times to repeat each test
        """
        for variant_name, generator_func in self.prompt_variants.items():
            print(f"Benchmarking variant: {variant_name}")
            
            prompt = generator_func(task, content)
            
            # Measure performance
            successful = 0
            total_tokens = 0
            start_time = time.time()
            
            for _ in range(repeat_count):
                try:
                    response = self.bedrock_runtime.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(prompt)
                    )
                    
                    # Extract token usage based on model type
                    response_body = json.loads(response['body'].read())
                    
                    if "anthropic" in self.model_id:
                        input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
                        output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
                    elif "ai21" in self.model_id:
                        input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
                        output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
                    else:
                        # Rough estimation
                        input_text = json.dumps(prompt)
                        output_text = str(response_body)
                        
                        input_tokens = len(input_text.split()) * 1.3
                        output_tokens = len(output_text.split()) * 1.3
                    
                    total_tokens += input_tokens + output_tokens
                    successful += 1
                    
                except Exception as e:
                    print(f"Error in benchmark: {str(e)}")
                
                # Brief pause
                time.sleep(0.2)
            
            elapsed_time = time.time() - start_time
            
            if successful > 0 and elapsed_time > 0:
                avg_tokens = total_tokens / successful
                tokens_per_second = total_tokens / elapsed_time
                
                self.variant_performance[variant_name] = {
                    "avg_tokens_per_request": avg_tokens,
                    "tokens_per_second": tokens_per_second,
                    "input_example": prompt
                }
            else:
                self.variant_performance[variant_name] = {
                    "error": "Benchmark failed"
                }
    
    def select_optimal_variant(self, optimization_goal="throughput"):
        """
        Select the optimal prompt variant based on the specified goal.
        
        Args:
            optimization_goal: Either "throughput" (max requests) or "efficiency" (min tokens)
            
        Returns:
            Name of the optimal variant
        """
        if not self.variant_performance:
            raise ValueError("Must run benchmark_variants first")
        
        if optimization_goal == "throughput":
            # Find variant with highest tokens_per_second
            best_variant = max(
                self.variant_performance.items(),
                key=lambda x: x[1].get("tokens_per_second", 0) if "error" not in x[1] else 0
            )[0]
        elif optimization_goal == "efficiency":
            # Find variant with lowest avg_tokens_per_request
            best_variant = min(
                self.variant_performance.items(),
                key=lambda x: x[1].get("avg_tokens_per_request", float('inf')) if "error" not in x[1] else float('inf')
            )[0]
        else:
            raise ValueError(f"Unknown optimization goal: {optimization_goal}")
        
        return best_variant
    
    def generate_optimal_prompt(self, task, content=None, optimization_goal="throughput"):
        """
        Generate an optimal prompt for the given task based on quota considerations.
        
        Args:
            task: The task description
            content: Optional content to include
            optimization_goal: What to optimize for
            
        Returns:
            The optimal prompt payload
        """
        # If we haven't benchmarked yet, use a default
        if not self.variant_performance:
            # Use a reasonable default variant
            model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else self.model_id.lower()
            
            if "anthropic" in model_family:
                return BedrockPromptTemplates.format_for_claude(task, content)
            elif "meta" in model_family or "llama" in model_family:
                return BedrockPromptTemplates.format_for_llama(task, content)
            else:
                return BedrockPromptTemplates.format_for_titan(task, content)
        
        # Select the optimal variant
        best_variant = self.select_optimal_variant(optimization_goal)
        
        # Generate prompt using this variant
        return self.prompt_variants[best_variant](task, content)
    
    def visualize_variant_performance(self, output_file="variant_performance.svg"):
        """Create an SVG visualization of variant performance"""
        if not self.variant_performance:
            raise ValueError("Must run benchmark_variants first")
        
        variants = []
        tokens_per_request = []
        tokens_per_second = []
        
        for variant_name, perf in self.variant_performance.items():
            if "error" not in perf:
                variants.append(variant_name)
                tokens_per_request.append(perf.get("avg_tokens_per_request", 0))
                tokens_per_second.append(perf.get("tokens_per_second", 0))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Tokens per request
        ax1.bar(variants, tokens_per_request, color='#1f77b4')
        ax1.set_title('Average Tokens per Request by Variant', fontsize=14)
        ax1.set_ylabel('Tokens', fontsize=12)
        
        # Tokens per second (throughput)
        ax2.bar(variants, tokens_per_second, color='#2ca02c')
        ax2.set_title('Tokens per Second by Variant (Throughput)', fontsize=14)
        ax2.set_ylabel('Tokens/Second', fontsize=12)
        ax2.set_xlabel('Prompt Variant', fontsize=12)
        
        plt.tight_layout()
        
        # Save as SVG
        plt.savefig(output_file, **SVG_CONFIG)
        plt.close()
        
        return output_file
```

## Step 5: Creating a Complete Throughput Optimization Pipeline

Now let's create a complete pipeline that demonstrates how to maximize throughput:

```python
def run_throughput_optimization_demo(model_id, task, content=None):
    """
    Demonstrate the complete throughput optimization pipeline.
    
    Args:
        model_id: The model to optimize for
        task: The task description
        content: Optional content to include
    """
    print(f"Running throughput optimization for {model_id}")
    print(f"Task: {task}")
    
    # Step 1: Create the optimizer
    optimizer = QuotaAwarePromptOptimizer(model_id)
    
    # Step 2: Add prompt variants
    optimizer.add_prompt_variant("detailed", lambda t, c: BedrockPromptTemplates.format_for_any_model(
        model_id,
        f"Please perform the following task: {t}. Provide a detailed and thorough response that covers all aspects of the question.",
        c,
        "You are a helpful assistant that provides comprehensive responses."
    ))
    
    optimizer.add_prompt_variant("concise", lambda t, c: BedrockPromptTemplates.format_for_any_model(
        model_id,
        f"Task: {t}\nBe brief and direct.",
        c,
        "Provide concise, to-the-point responses."
    ))
    
    optimizer.add_prompt_variant("structured", lambda t, c: BedrockPromptTemplates.format_for_any_model(
        model_id,
        f"Follow these steps:\n1. Understand this task: {t}\n2. Analyze the key components\n3. Provide a structured response with clear headings",
        c,
        "You provide well-structured, organized responses."
    ))
    
    optimizer.add_prompt_variant("minimal", lambda t, c: BedrockPromptTemplates.format_for_any_model(
        model_id,
        f"{t}",
        c,
        None
    ))
    
    # Step 3: Benchmark the variants
    print("Benchmarking prompt variants...")
    optimizer.benchmark_variants(task, content)
    
    # Step 4: Visualize the performance
    viz_file = optimizer.visualize_variant_performance(
        f"docs/images/{model_id.replace('.', '_')}_variant_performance.svg"
    )
    print(f"Visualization saved to {viz_file}")
    
    # Step 5: Select optimal variants for different goals
    throughput_variant = optimizer.select_optimal_variant("throughput")
    efficiency_variant = optimizer.select_optimal_variant("efficiency")
    
    print(f"Optimal variant for maximum throughput: {throughput_variant}")
    print(f"Optimal variant for token efficiency: {efficiency_variant}")
    
    # Step 6: Generate optimal prompts
    throughput_prompt = optimizer.generate_optimal_prompt(task, content, "throughput")
    efficiency_prompt = optimizer.generate_optimal_prompt(task, content, "efficiency")
    
    # Step 7: Calculate theoretical maximum throughput
    quota_limits = optimizer.quota_limits
    throughput_perf = optimizer.variant_performance[throughput_variant]
    efficiency_perf = optimizer.variant_performance[efficiency_variant]
    
    if "rpm" in quota_limits and throughput_perf.get("avg_tokens_per_request", 0) > 0:
        max_rpm_throughput = min(
            quota_limits["rpm"],
            quota_limits.get("tpm", float('inf')) / throughput_perf["avg_tokens_per_request"]
        )
        
        max_rpm_efficiency = min(
            quota_limits["rpm"],
            quota_limits.get("tpm", float('inf')) / efficiency_perf["avg_tokens_per_request"]
        )
        
        print("\nTheoretical Maximum Throughput:")
        print(f"Using '{throughput_variant}' variant: {max_rpm_throughput:.1f} requests/minute")
        print(f"Using '{efficiency_variant}' variant: {max_rpm_efficiency:.1f} requests/minute")
        
        if max_rpm_efficiency > max_rpm_throughput:
            print(f"\nIn this case, the more token-efficient '{efficiency_variant}' variant")
            print(f"allows for {max_rpm_efficiency - max_rpm_throughput:.1f} more requests per minute!")
    
    # Return the results
    return {
        "model_id": model_id,
        "task": task,
        "variant_performance": optimizer.variant_performance,
        "throughput_variant": throughput_variant,
        "efficiency_variant": efficiency_variant,
        "visualization_file": viz_file
    }
```

## Using the Optimization Pipeline

Here's how you can use this optimization pipeline:

```python
# Example 1: Optimizing a summarization task for Claude
results = run_throughput_optimization_demo(
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "Summarize the following text in a concise way",
    "AWS Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies through a single API. This service helps you build generative AI applications with security, privacy, and responsible AI. You can choose from a wide range of FMs from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon to find the model best suited for your use case."
)

# Example 2: Optimizing a classification task for Llama
results = run_throughput_optimization_demo(
    "meta.llama2-13b-chat-v1",
    "Classify this text into one of these categories: Technology, Business, Health, Entertainment, or Other",
    "The new quantum computing breakthrough enables calculations previously thought impossible, potentially revolutionizing drug discovery and materials science."
)
```

## Practical Throughput Optimization Strategies

Based on our experiments, here are effective strategies for optimizing throughput:

### 1. Match Prompt Style to Model Strengths

- **Claude Models**: Perform better with structured prompts but can be token-heavy; the "structured" variant often provides the best balance
- **Llama Models**: Respond well to concise prompts with minimal formatting; the "minimal" variant often maximizes throughput
- **Titan Models**: Perform well with clearly labeled tasks; the "concise" variant often provides the best throughput

### 2. Adjust Based on Quota Limiting Factor

If you're limited by:
- **RPM**: Use the most efficient prompt variant to reduce token count per request
- **TPM**: Use the fastest prompt variant to process tokens more quickly

### 3. Implement Adaptive Prompt Selection

Create a system that:
1. Monitors current quota usage
2. Switches to more efficient prompt variants as you approach limits
3. Uses different variants for different priority levels of tasks

### 4. Batch Similar Requests

When possible, combine similar requests to reduce overhead:
1. Use fewer, larger requests instead of many small ones
2. Batch classification or processing tasks
3. Process multiple items in a single request

## Conclusion

By optimizing your prompt structure for specific models, you can significantly increase your effective throughput within AWS Bedrock quota limits. The techniques in this tutorial allow you to:

1. Measure and compare prompt efficiency across variants
2. Select optimal prompt structures based on quota constraints
3. Implement adaptive optimization strategies
4. Maximize the value from your AWS Bedrock quotas

In future tutorials, we'll explore more advanced techniques like dynamic prompt compression, automated prompt optimization, and cross-model load balancing for even greater throughput.