---
layout: chapter
title: Understanding Foundation Models in AWS Bedrock
difficulty: beginner
time-estimate: 25 minutes
---

# Understanding Foundation Models in AWS Bedrock

> "You know AWS Bedrock gives you access to multiple AI models, but which one should you choose? Let's cut through the confusion and find the right tool for your job."

## The Problem

---

**Scenario**: You're implementing AI capabilities for your enterprise application using AWS Bedrock. As you start exploring, you realize there are numerous foundation models available from different providers: Anthropic, Amazon, Meta, Cohere, AI21 Labs, and Stability AI.

Each model has different capabilities, pricing structures, token limits, and performance characteristics. You need to select the right model for your specific use cases, but you're overwhelmed by the options.

Your specific challenges include:
1. Understanding the real-world strengths and weaknesses of each model family
2. Determining which models offer the best price-performance ratio for your tasks
3. Identifying which models support advanced features like streaming or structured outputs
4. Finding the right balance between model quality and cost for production use

---

## Key Concepts Explained

Before diving into specific models, let's clarify what makes foundation models different from one another.

### What Differentiates Foundation Models

Think of foundation models like different professional tools – a hammer, screwdriver, and wrench all have their purposes, and while you could use a hammer for everything, some jobs are better suited for other tools.

Foundation models differ in several key dimensions:

1. **Training Data**: The content they were trained on shapes their knowledge and abilities
2. **Architecture**: Different internal structures affect reasoning capabilities and efficiency
3. **Size**: Larger models generally perform better but cost more and run slower
4. **Specialization**: Some models excel at specific tasks like code generation or reasoning
5. **Context Window**: How much text they can process and "remember" at once
6. **Instruction Following**: How well they adhere to specific directions
7. **Output Control**: Ability to generate structured outputs or follow formatting rules

### Understanding Model Versions

Model capabilities evolve over time with new versions. For example, Anthropic's Claude has progressed from Claude 1 to Claude 2 to Claude 3, with each generation offering improved capabilities.

Within a given model family, you'll often see variations like:

- **Opus/Large**: The most capable (and expensive) version
- **Sonnet/Medium**: A balanced option for most use cases
- **Haiku/Small**: Faster and less expensive, but with reduced capabilities

## Model Families in AWS Bedrock

Let's explore the major model families available in AWS Bedrock:

### Anthropic Claude Models

Claude models excel at:
- Following complex instructions precisely
- Nuanced reasoning and analysis
- Safety and reducing harmful outputs
- Long context processing (up to 200K tokens in Claude 3 Opus)

**Best For**:
- Enterprise applications requiring reliability and safety
- Complex reasoning tasks and content generation
- Applications needing long context windows
- Situations requiring nuanced understanding and responses

**Available Models**:
- Claude 3 Opus: The most capable Claude model
- Claude 3 Sonnet: Balanced performance and cost
- Claude 3 Haiku: Fastest and most cost-effective
- Claude 2 and Claude Instant (older generations)

**Code Example**:
```python
import boto3
import json

bedrock = boto3.client('bedrock-runtime')

claude_prompt = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "messages": [
        {"role": "user", "content": "Analyze the following contract clause and explain any potential risks: 'The party shall make best efforts to deliver services in a timely manner.'"}
    ]
}

response = bedrock.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps(claude_prompt)
)

response_body = json.loads(response['body'].read())
print(response_body['content'][0]['text'])
```

### Amazon Titan Models

Titan models offer:
- Tight integration with AWS services
- Good balance of performance and cost
- Text and image generation capabilities
- Embeddings for retrieval and classification

**Best For**:
- Cost-sensitive applications
- Applications requiring deep AWS integration
- Baseline text generation and summarization
- Embedding generation for vector databases

**Available Models**:
- Titan Text (Express and Lite)
- Titan Embeddings
- Titan Image Generator

**Code Example**:
```python
import boto3
import json

bedrock = boto3.client('bedrock-runtime')

titan_prompt = {
    "inputText": "Summarize the key features of AWS Bedrock in 3 bullet points.",
    "textGenerationConfig": {
        "maxTokenCount": 500,
        "temperature": 0.7,
        "topP": 0.9
    }
}

response = bedrock.invoke_model(
    modelId='amazon.titan-text-express-v1',
    body=json.dumps(titan_prompt)
)

response_body = json.loads(response['body'].read())
print(response_body['results'][0]['outputText'])
```

### Meta Llama 2 Models

Llama 2 models provide:
- Strong performance at lower cost than some alternatives
- Open weights (outside of Bedrock) for local deployment
- Good code generation capabilities
- Effective for conversational applications

**Best For**:
- Applications with a good balance of cost and performance
- Code generation and technical content
- Chatbots and conversational agents
- Projects that might later require local deployment

**Available Models**:
- Llama 2 (13B and 70B parameters)
- Llama 2 Chat (tuned for conversations)

**Code Example**:
```python
import boto3
import json

bedrock = boto3.client('bedrock-runtime')

llama_prompt = {
    "prompt": "<s>[INST] Write a Python function to calculate the Fibonacci sequence up to n [/INST]",
    "max_gen_len": 512,
    "temperature": 0.7,
    "top_p": 0.9
}

response = bedrock.invoke_model(
    modelId='meta.llama2-13b-chat-v1',
    body=json.dumps(llama_prompt)
)

response_body = json.loads(response['body'].read())
print(response_body['generation'])
```

### Cohere Command Models

Cohere models excel at:
- Multilingual capabilities across 100+ languages
- Search and retrieval tasks
- Text classification and analysis
- High-quality summarization

**Best For**:
- International applications requiring multiple languages
- Search functionality and semantic matching
- Content summarization and classification
- Applications requiring multilingual capabilities

**Available Models**:
- Command (text generation)
- Command Light (faster, more efficient)
- Command R (enhanced reasoning capabilities)
- Embed (multilingual embeddings)

**Code Example**:
```python
import boto3
import json

bedrock = boto3.client('bedrock-runtime')

cohere_prompt = {
    "prompt": "Translate the following English text to French, Spanish, and German: 'Welcome to our global platform.'",
    "max_tokens": 500,
    "temperature": 0.7
}

response = bedrock.invoke_model(
    modelId='cohere.command-text-v14',
    body=json.dumps(cohere_prompt)
)

response_body = json.loads(response['body'].read())
print(response_body['generations'][0]['text'])
```

### AI21 Jurassic Models

Jurassic models are strong in:
- Structured text generation
- Numerical reasoning and analysis
- Specialized document tasks
- Handling specific formats reliably

**Best For**:
- Financial and data-heavy applications
- Applications requiring structured outputs
- Document summarization and analysis
- Mathematical and numerical reasoning

**Available Models**:
- Jurassic-2 (various sizes)

**Code Example**:
```python
import boto3
import json

bedrock = boto3.client('bedrock-runtime')

jurassic_prompt = {
    "prompt": "Calculate the compound interest on a loan of $10,000 with an annual interest rate of 5% over 3 years.",
    "maxTokens": 500,
    "temperature": 0.7
}

response = bedrock.invoke_model(
    modelId='ai21.j2-mid-v1',
    body=json.dumps(jurassic_prompt)
)

response_body = json.loads(response['body'].read())
print(response_body['completions'][0]['data']['text'])
```

### Stability AI Models

Stability AI provides:
- High-quality image generation from text prompts
- Style control and artistic variations
- Support for various image dimensions
- Fast generation times

**Best For**:
- Creating marketing visuals
- Product design and visualization
- Creative and artistic applications
- Generating custom imagery for content

**Available Models**:
- Stable Diffusion XL
- Stable Diffusion 3

**Code Example**:
```python
import boto3
import json
import base64
from PIL import Image
import io

bedrock = boto3.client('bedrock-runtime')

stability_prompt = {
    "text_prompts": [
        {
            "text": "A futuristic city with flying cars and tall buildings, digital art style",
            "weight": 1.0
        }
    ],
    "cfg_scale": 7,
    "steps": 30,
    "seed": 42,
    "width": 1024,
    "height": 1024
}

response = bedrock.invoke_model(
    modelId='stability.stable-diffusion-xl-v1',
    body=json.dumps(stability_prompt)
)

response_body = json.loads(response['body'].read())
image_bytes = base64.b64decode(response_body['artifacts'][0]['base64'])

# Save or display the image
image = Image.open(io.BytesIO(image_bytes))
image.save("generated_image.png")
```

## Choosing the Right Model for Your Task

Here's a practical framework for selecting the most appropriate model:

### Step 1: Define Your Requirements

Start by clearly defining what you need:
- **Task type**: Text generation, conversation, summarization, code, image generation
- **Quality threshold**: How critical is perfect output quality?
- **Speed requirements**: Is response time critical?
- **Budget constraints**: What are your cost limitations?
- **Context needs**: How much input context do you need to process?

### Step 2: Match Your Task Type to Model Strengths

Here's a quick reference guide:

| Task | Recommended Models | Reasoning |
|------|-------------------|-----------|
| General Content Creation | Claude 3 Sonnet, Titan Text | Good balance of quality and cost |
| Customer Support Chatbot | Claude 3 Haiku, Llama 2 Chat | Fast responses, good conversation handling |
| Legal/Financial Analysis | Claude 3 Opus, Jurassic-2 | Strong reasoning, handles complex documents |
| Code Generation | Llama 2, Claude 3 | Strong technical capabilities |
| Multilingual Applications | Cohere Command | Superior multilingual support |
| Image Generation | Stable Diffusion XL | High-quality image creation |
| Knowledge Base QA | Claude with long context | Can process lengthy documents |

### Step 3: Consider Pricing

Price comparisons (approximate, please check current AWS pricing):

| Model | Input Tokens (per 1M) | Output Tokens (per 1M) | Relative Cost |
|-------|----------------------|------------------------|---------------|
| Claude 3 Opus | $15 | $75 | Highest |
| Claude 3 Sonnet | $3 | $15 | High |
| Claude 3 Haiku | $0.25 | $1.25 | Medium |
| Titan Text Express | $0.20 | $0.30 | Low |
| Llama 2 70B | $0.80 | $1.10 | Medium |
| Cohere Command | $1 | $2 | Medium |
| Jurassic-2 Mid | $0.50 | $1.50 | Medium |

### Step 4: Evaluate and Iterate

Don't rely solely on specifications – test models with your actual use cases:

1. Create a representative set of example inputs
2. Run them through your candidate models
3. Evaluate the outputs against your requirements
4. Consider both quality and cost metrics
5. Iterate based on real-world performance

## Feature Comparison

Here's a feature comparison to help with your decision:

| Feature | Claude 3 | Titan | Llama 2 | Cohere | Jurassic-2 |
|---------|---------|-------|---------|--------|------------|
| Max Context Window | Up to 200K tokens | 8K tokens | 4K tokens | 8K tokens | 8K tokens |
| Streaming Support | Yes | Yes | Yes | Yes | Yes |
| Structured Output | Yes | Limited | Limited | Yes | Yes |
| Multilingual Support | Good | Basic | Basic | Excellent | Good |
| Fine-tuning Support | Yes | Yes | Yes | Yes | Yes |
| Safety Guardrails | Strong | Moderate | Moderate | Moderate | Moderate |
| Code Generation | Good | Basic | Strong | Basic | Basic |
| Image Understanding | Yes (Claude 3) | No | No | No | No |

## Common Pitfalls and Troubleshooting

### Pitfall #1: Model Overspending

**Problem**: Using expensive models for simple tasks that don't require their capabilities.

**Solution**: Start with smaller models and only upgrade when necessary. For example, use Claude 3 Haiku for most tasks and only upgrade to Sonnet or Opus when you need more sophisticated reasoning or larger context windows.

### Pitfall #2: Ignoring Context Windows

**Problem**: Trying to process documents larger than a model's context window.

**Solution**: Implement chunking strategies or use models with larger context windows like Claude 3 Opus. Be aware of token limits when designing your application.

```python
def process_large_document(document, chunk_size=8000, overlap=500):
    """Process a document that exceeds context window by chunking."""
    tokens = tokenize_text(document)  # Implement appropriate tokenization
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(detokenize(chunk))  # Convert back to text
    
    results = []
    for chunk in chunks:
        # Process each chunk with the model
        result = process_with_model(chunk)
        results.append(result)
    
    # Combine results appropriately for your use case
    return combine_results(results)
```

### Pitfall #3: Not Accounting for Token Costs

**Problem**: Unexpected high costs due to not estimating token usage.

**Solution**: Implement token counting and budget management:

```python
def estimate_cost(input_text, expected_output_length, model="claude-3-sonnet"):
    """Estimate cost for a model request."""
    # Rough token estimation (implement proper tokenization for accuracy)
    input_tokens = len(input_text.split()) * 1.3
    output_tokens = expected_output_length
    
    # Example pricing (would need to be updated)
    pricing = {
        "claude-3-opus": {"input": 0.000015, "output": 0.000075},
        "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
        "claude-3-haiku": {"input": 0.00000025, "output": 0.00000125},
        "titan-text-express": {"input": 0.0000002, "output": 0.0000003}
    }
    
    cost = (input_tokens * pricing[model]["input"]) + (output_tokens * pricing[model]["output"])
    return cost
```

## Try It Yourself Challenge

Now it's your turn to gain hands-on experience with different foundation models:

### Challenge: Model Comparison Test

1. Create a simple test to compare 2-3 different models on the same task
2. Implement code that:
   - Sends the same prompt to multiple models
   - Captures response quality, token usage, and latency
   - Provides a simple scoring mechanism for comparison

**Starting Code**:

```python
import boto3
import json
import time
from datetime import datetime

def model_comparison_test(prompt, models):
    """
    Compare different models' performance on the same prompt.
    
    Args:
        prompt: The text prompt to send to each model
        models: List of dictionaries with model information
        
    Returns:
        Dictionary with comparison results
    """
    bedrock = boto3.client('bedrock-runtime')
    results = []
    
    for model in models:
        model_id = model["id"]
        formatted_prompt = format_prompt_for_model(prompt, model_id)
        
        start_time = time.time()
        try:
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(formatted_prompt)
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            response_body = json.loads(response['body'].read())
            output = extract_output(response_body, model_id)
            
            # Add your metrics and evaluation here
            result = {
                "model_id": model_id,
                "latency_seconds": latency,
                "output": output,
                "success": True
            }
            
        except Exception as e:
            result = {
                "model_id": model_id,
                "error": str(e),
                "success": False
            }
        
        results.append(result)
    
    return results

def format_prompt_for_model(prompt, model_id):
    """Format the prompt appropriately for each model."""
    # Implement formatting logic for different model types
    pass

def extract_output(response_body, model_id):
    """Extract the output text from model-specific response format."""
    # Implement extraction logic for different model types
    pass

# TODO: Complete the implementation of the helper functions
# and run the comparison test with your own prompt
```

**Expected Outcome**:
A working script that provides quantitative and qualitative comparison data for different foundation models on the same task.

## Beyond the Basics

Once you've selected your foundation models, consider these advanced strategies:

### 1. Model Ensembles

Combine multiple models to improve reliability and quality:

```python
def ensemble_generation(prompt, models, combination_strategy="voting"):
    """Generate text using an ensemble of models."""
    results = []
    
    # Get responses from all models
    for model in models:
        result = invoke_model(model, prompt)
        results.append(result)
    
    if combination_strategy == "voting":
        # Implement voting mechanism for classification tasks
        return majority_vote(results)
    elif combination_strategy == "confidence":
        # Return result with highest confidence score
        return highest_confidence(results)
    elif combination_strategy == "average":
        # For numeric predictions, average the results
        return average_results(results)
    else:
        # Default: return all results for manual evaluation
        return results
```

### 2. Automatic Model Selection

Dynamically select models based on input characteristics:

```python
def auto_select_model(input_text, max_cost=None):
    """Automatically select the appropriate model based on input."""
    input_length = len(input_text.split())
    contains_code = any(marker in input_text for marker in ["def ", "class ", "function", "```"])
    is_multilingual = detect_non_english(input_text)
    
    if contains_code:
        return "meta.llama2-13b-chat-v1"  # Good for code
    elif is_multilingual:
        return "cohere.command-text-v14"  # Strong multilingual support
    elif input_length > 6000:
        return "anthropic.claude-3-opus-20240229-v1:0"  # Large context window
    elif max_cost and max_cost < 0.001:
        return "amazon.titan-text-express-v1"  # Most affordable
    else:
        return "anthropic.claude-3-haiku-20240307-v1:0"  # Good balance
```

### 3. Hybrid Approaches

Use different models for different parts of your application workflow:

1. **Classifier Model**: Use an efficient model to classify user input
2. **Content Generator**: Use a high-quality model for main content generation
3. **Refinement Model**: Use a specialized model to check and refine outputs

This approach optimizes for both cost and quality throughout the processing pipeline.

## Key Takeaways

- Each foundation model family in AWS Bedrock has distinct strengths and use cases
- Model selection should be based on task requirements, quality needs, and budget constraints
- Test models with representative examples rather than relying solely on specifications
- Consider context window limitations when designing your applications
- Implement token counting and budget management to control costs
- More expensive models aren't always better for every task - match capabilities to requirements
- Advanced strategies like ensembles and hybrid approaches can optimize performance

---

**Next Steps**: Now that you understand the different foundation models, learn about implementing [synchronous inference](/chapters/core-methods/synchronous/) with AWS Bedrock.

---

© 2025 Scott Friedman. Licensed under CC BY-NC-ND 4.0