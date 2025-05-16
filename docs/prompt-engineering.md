# Prompt Engineering Across AWS Bedrock Models

This guide explores the different prompt structures, formats, and optimization techniques for the various foundation models available in AWS Bedrock. Proper prompt engineering is essential not only for achieving optimal results but also for maximizing throughput within quota limits.

## Introduction to Model-Specific Prompting

Each foundation model family in AWS Bedrock has its own preferred prompt format and structure. Understanding these differences allows you to:

1. **Optimize token usage** - Reducing unnecessary tokens helps stay within TPM quotas
2. **Improve response quality** - Properly formatted prompts yield better results
3. **Reduce latency** - Efficient prompts can process faster, increasing throughput
4. **Maximize quota utilization** - Different prompt techniques work better for different models

## Model-Specific Prompt Formats

### Anthropic Claude Models

Claude models use a message-based format with roles:

```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 1000,
  "messages": [
    {"role": "user", "content": "Hello, I need help with..."},
    {"role": "assistant", "content": "I'd be happy to help!"},
    {"role": "user", "content": "Can you explain quantum computing?"}
  ]
}
```

**Key considerations:**
- Claude performs best with clear, explicit instructions
- System prompts can provide context and constraints
- Multi-turn conversation history helps with context
- Explicitly define the desired output format

### Meta Llama 2 Models

Llama models use a special tag-based format for instructions:

```json
{
  "prompt": "<s>[INST] Write a story about a robot learning to paint. [/INST]",
  "max_gen_len": 1000,
  "temperature": 0.7,
  "top_p": 0.9
}
```

For multi-turn conversations:

```json
{
  "prompt": "<s>[INST] What is machine learning? [/INST] Machine learning is a subset of artificial intelligence... [INST] Can you give me an example? [/INST]",
  "max_gen_len": 1000
}
```

**Key considerations:**
- Always wrap instructions in [INST] tags
- Include conversation history in a single prompt string
- Alternate [INST] tags with model responses
- Keep instructions concise and clear

### Amazon Titan Models

Titan models use a more straightforward text format:

```json
{
  "inputText": "Write a story about a robot learning to paint.",
  "textGenerationConfig": {
    "maxTokenCount": 1000,
    "temperature": 0.7,
    "topP": 0.9
  }
}
```

**Key considerations:**
- Straightforward, clear instructions work best
- Explicit formatting instructions help control output
- Examples can be helpful for complex tasks
- Less verbose than Claude in many cases

### Cohere Models

Cohere uses a different format with dedicated fields:

```json
{
  "prompt": "Write a story about a robot learning to paint.",
  "max_tokens": 1000,
  "temperature": 0.7,
  "p": 0.9,
  "k": 0,
  "stop_sequences": [],
  "return_likelihoods": "NONE"
}
```

**Key considerations:**
- Clear instructions with examples work well
- Can use ":" to separate instruction from content
- Supports various control parameters
- Often requires less context for good results

### AI21 Jurassic Models

Jurassic models have their own format:

```json
{
  "prompt": "Write a story about a robot learning to paint.",
  "maxTokens": 1000,
  "temperature": 0.7,
  "topP": 0.9,
  "stopSequences": []
}
```

**Key considerations:**
- Benefits from detailed instructions
- Examples help with complex tasks
- Include desired output format in the prompt
- Use colon format for Q&A style interactions

## Comparative Analysis: Same Task, Different Models

Let's explore how the same task might be prompted differently across models:

### Task: Generate a product description for a smart water bottle

#### Claude Approach
```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 250,
  "messages": [
    {
      "role": "user",
      "content": "Write a compelling product description for a smart water bottle with the following features:\n- Tracks water intake\n- Syncs with mobile app\n- Reminds you to drink\n- Temperature monitoring\n- 24oz capacity\n- BPA-free material\n\nThe description should be around 100 words and highlight the health benefits."
    }
  ]
}
```

#### Llama 2 Approach
```json
{
  "prompt": "<s>[INST] Write a compelling product description for a smart water bottle. Include these features: tracks water intake, syncs with mobile app, reminds you to drink, monitors temperature, 24oz capacity, BPA-free material. Keep it around 100 words and emphasize health benefits. [/INST]",
  "max_gen_len": 250
}
```

#### Titan Approach
```json
{
  "inputText": "Product Description Task: Create a compelling description for a smart water bottle. Features: tracks water intake, syncs with mobile app, sends hydration reminders, monitors temperature, 24oz capacity, BPA-free material. Length: approximately 100 words. Focus: highlight health benefits.",
  "textGenerationConfig": {
    "maxTokenCount": 250
  }
}
```

### Analysis of Differences

1. **Verbosity**: Claude prompts tend to be more structured and verbose, while Titan and Llama use more concise formats.
2. **Token Usage**: For this task, the Claude prompt uses more tokens than the Llama or Titan prompts.
3. **Formatting**: Claude uses a clear bullet-point structure, Llama includes the features in a paragraph, and Titan uses a labeled task format.
4. **Instruction Style**: Claude provides more detailed writing instructions, Llama keeps them compact, and Titan uses a task-oriented approach.

## Optimizing Token Usage Across Models

Token optimization strategies vary by model family:

### Claude Token Optimization
- Remove unnecessary pleasantries ("Could you please...")
- Use the system message for instructions instead of the user message
- Consolidate multi-turn context when possible
- Structure instructions as bullet points for clarity without verbosity

### Llama Token Optimization
- Keep instructions concise and direct
- Avoid repetitive information between instruction tags
- Use shorter delimiter formats where possible
- Focus on key constraints rather than long explanations

### Titan Token Optimization
- Use concise, instruction-oriented language
- Label sections clearly but briefly
- Avoid redundant specifications
- Structure complex tasks as numbered steps

## Measuring Token Efficiency

Let's compare the token usage and performance across models for the same task:

```python
import boto3
import json
import time
from utils.profile_manager import get_profile

def measure_token_efficiency(prompt_strategies, task):
    """
    Compare token usage and efficiency across different prompt strategies.
    
    Args:
        prompt_strategies: Dictionary mapping model IDs to their prompt payloads
        task: Description of the task being performed
    
    Returns:
        Dictionary with efficiency metrics
    """
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime')
    
    results = {
        "task": task,
        "timestamp": time.time(),
        "models": {}
    }
    
    for model_id, payload in prompt_strategies.items():
        start_time = time.time()
        
        try:
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(payload)
            )
            
            # Process response based on model type
            if "anthropic" in model_id:
                response_body = json.loads(response['body'].read())
                output = response_body['content'][0]['text']
                input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
                output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
            elif "meta" in model_id:
                response_body = json.loads(response['body'].read())
                output = response_body['generation']
                input_tokens = len(payload['prompt'].split())  # Rough estimation
                output_tokens = len(output.split())  # Rough estimation
            else:  # Default for other models
                response_body = json.loads(response['body'].read())
                # Extract output based on model format
                if 'results' in response_body:
                    output = response_body['results'][0]['outputText']
                elif 'generation' in response_body:
                    output = response_body['generation']
                else:
                    output = str(response_body)
                
                # Rough token estimation
                input_tokens = len(json.dumps(payload).split())
                output_tokens = len(output.split())
            
            elapsed_time = time.time() - start_time
            
            results["models"][model_id] = {
                "success": True,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "response_time_seconds": elapsed_time,
                "tokens_per_second": (input_tokens + output_tokens) / elapsed_time if elapsed_time > 0 else 0,
                "output_sample": output[:100] + "..." if len(output) > 100 else output
            }
            
        except Exception as e:
            results["models"][model_id] = {
                "success": False,
                "error": str(e)
            }
    
    return results
```

## Prompt Templates for Different Tasks

Different tasks benefit from model-specific prompt templates. Here are examples for common tasks:

### Summarization Templates

#### Claude Summarization
```python
def claude_summarize_template(text, max_length=None, focus=None):
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "messages": [
            {
                "role": "user",
                "content": f"Summarize the following text"
                + (f" in {max_length} words or less" if max_length else "")
                + (f", focusing on {focus}" if focus else "")
                + f":\n\n{text}"
            }
        ]
    }
    return prompt
```

#### Llama 2 Summarization
```python
def llama_summarize_template(text, max_length=None, focus=None):
    instruction = f"Summarize this text"
    if max_length:
        instruction += f" in {max_length} words or less"
    if focus:
        instruction += f", focusing on {focus}"
    
    prompt = {
        "prompt": f"<s>[INST] {instruction}: {text} [/INST]",
        "max_gen_len": 300
    }
    return prompt
```

### Classification Templates

#### Claude Classification
```python
def claude_classify_template(text, categories):
    categories_str = ", ".join(categories)
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": f"Classify the following text into one of these categories: {categories_str}.\n\nText: {text}\n\nCategory:"
            }
        ]
    }
    return prompt
```

#### Titan Classification
```python
def titan_classify_template(text, categories):
    categories_str = ", ".join(categories)
    prompt = {
        "inputText": f"Classification task: Assign the following text to exactly one of these categories: {categories_str}.\n\n{text}\n\nSelected category:",
        "textGenerationConfig": {
            "maxTokenCount": 100
        }
    }
    return prompt
```

## Impact on Throughput and Quota Usage

Different prompt structures have measurable impacts on throughput and quota consumption:

1. **Token Efficiency**: More efficient prompts use fewer tokens, allowing more requests within TPM quotas
2. **Processing Speed**: Some models process certain prompt formats faster, affecting throughput
3. **Response Size Control**: Properly constrained prompts produce shorter outputs, saving on output tokens
4. **Error Rates**: Well-structured prompts reduce errors and retries, preserving quota

Here's an example of measuring the quota impact:

```python
def analyze_quota_impact(models, prompt_variants, repeat_count=10):
    """
    Analyze how different prompt structures impact quota usage and throughput.
    
    Args:
        models: List of model IDs to test
        prompt_variants: Dictionary of named variants with prompt templates
        repeat_count: Number of times to repeat each test for reliable data
    
    Returns:
        Dictionary with quota impact analysis
    """
    results = {
        "timestamp": time.time(),
        "models": {},
        "summary": {}
    }
    
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime')
    
    for model_id in models:
        model_results = {}
        
        for variant_name, prompt_template in prompt_variants.items():
            # Apply model-specific formatting
            prompt = format_for_model(model_id, prompt_template)
            
            variant_stats = {
                "total_tokens": 0,
                "total_time": 0,
                "successful_requests": 0,
                "failed_requests": 0
            }
            
            for i in range(repeat_count):
                try:
                    start_time = time.time()
                    
                    response = bedrock_runtime.invoke_model(
                        modelId=model_id,
                        body=json.dumps(prompt)
                    )
                    
                    # Extract token usage from response
                    response_body = json.loads(response['body'].read())
                    token_count = extract_token_count(model_id, response_body)
                    
                    elapsed_time = time.time() - start_time
                    
                    variant_stats["total_tokens"] += token_count
                    variant_stats["total_time"] += elapsed_time
                    variant_stats["successful_requests"] += 1
                    
                except Exception as e:
                    variant_stats["failed_requests"] += 1
                    print(f"Error with {model_id}, variant {variant_name}: {str(e)}")
                
                # Pause briefly between requests
                time.sleep(0.5)
            
            # Calculate averages and rates
            if variant_stats["successful_requests"] > 0:
                variant_stats["avg_tokens_per_request"] = variant_stats["total_tokens"] / variant_stats["successful_requests"]
                variant_stats["avg_time_per_request"] = variant_stats["total_time"] / variant_stats["successful_requests"]
                variant_stats["tokens_per_second"] = variant_stats["total_tokens"] / variant_stats["total_time"] if variant_stats["total_time"] > 0 else 0
                variant_stats["requests_per_minute"] = (variant_stats["successful_requests"] / variant_stats["total_time"]) * 60 if variant_stats["total_time"] > 0 else 0
            
            model_results[variant_name] = variant_stats
        
        results["models"][model_id] = model_results
    
    # Generate summary stats
    for variant_name in prompt_variants.keys():
        variant_summary = {
            "avg_tokens_per_request": sum(results["models"][m][variant_name].get("avg_tokens_per_request", 0) for m in models) / len(models) if models else 0,
            "avg_tokens_per_second": sum(results["models"][m][variant_name].get("tokens_per_second", 0) for m in models) / len(models) if models else 0,
            "success_rate": sum(results["models"][m][variant_name].get("successful_requests", 0) for m in models) / (sum(results["models"][m][variant_name].get("successful_requests", 0) + results["models"][m][variant_name].get("failed_requests", 0)) for m in models) if models else 0
        }
        
        results["summary"][variant_name] = variant_summary
    
    return results
```

## Best Practices for Cross-Model Prompting

When working with multiple models in AWS Bedrock:

1. **Use model-specific adapters** - Create a layer that formats prompts for each model
2. **Focus on the task, not the model** - Design prompts around the task and convert as needed
3. **Measure token usage** - Regularly benchmark to optimize for quota efficiency
4. **Standardize templates** - Create standard templates for common tasks, with model-specific variations
5. **Progressive refinement** - Start with the simplest prompt that works, then optimize

## Automated Prompt Optimization

For advanced use cases, implement automated prompt optimization:

```python
def optimize_prompt_structure(model_id, task_description, base_prompt, optimization_targets):
    """
    Iteratively optimize prompt structure to meet target metrics.
    
    Args:
        model_id: The model to optimize for
        task_description: Description of the task
        base_prompt: Starting prompt template
        optimization_targets: Dict with targets like "max_tokens", "throughput"
    
    Returns:
        Optimized prompt template
    """
    # Implementation would include:
    # 1. Variations of the prompt (more concise, different formatting, etc.)
    # 2. Testing each variation for performance
    # 3. Selecting the best variation based on targets
    # 4. Possibly using the model itself to help optimize further
    pass
```

## Conclusion and Long-Term Roadmap

This document provides a starting point for understanding prompt structure differences across AWS Bedrock models. A comprehensive suite would include:

1. **Expanded model coverage** - Detailed templates for all available models
2. **Task-specific libraries** - Optimized templates for each common task
3. **Automatic formatting layer** - A library to automatically format prompts for any model
4. **Performance benchmarks** - Regular testing of prompt efficiency across models
5. **Token optimization techniques** - Advanced strategies for minimal token usage
6. **Multi-modal prompting** - Specialized techniques for text+image models
7. **Quota simulator** - Tools to predict quota usage based on prompt design

In future iterations, we'll explore each model family in depth, with specific techniques for maximizing throughput while maintaining quality.