# Foundation Models in AWS Bedrock

This document provides an overview of the various foundation models available in AWS Bedrock and how to experiment with each.

## Foundation Model Families

AWS Bedrock provides access to multiple foundation model families, each with different capabilities, strengths, and pricing:

### Anthropic Claude

Claude models excel at natural language understanding, conversation, and reasoning tasks.

- **Claude 3 Opus** - Highest capability model for complex tasks
- **Claude 3 Sonnet** - Balanced performance and cost
- **Claude 3 Haiku** - Fastest, most cost-effective version
- **Claude 2** - Previous generation model
- **Claude Instant** - Lightweight model optimized for speed

### Meta Llama 2

Open-source models from Meta that provide strong performance across a range of tasks.

- **Llama 2 Chat 13B** - Optimized for conversational use cases
- **Llama 2 Chat 70B** - Larger model with improved capabilities
- **Llama 2 13B** - Base model for fine-tuning and general use
- **Llama 2 70B** - Larger base model

### Amazon Titan

Amazon's own foundation models designed for enterprise use cases.

- **Titan Text G1 - Express** - Lightweight text generation model
- **Titan Text G1 - Lite** - Compact model for text tasks
- **Titan Embeddings G1** - Specialized for generating embeddings
- **Titan Image Generator** - For image generation tasks
- **Titan Multimodal Embeddings** - For processing text and images

### Stability AI

Models focused on image generation and manipulation.

- **Stable Diffusion XL** - High-quality image generation
- **Stable Diffusion 3** - Latest generation model

### AI21 Labs

Models specializing in language understanding and generation.

- **Jurassic-2 Ultra** - Largest model for complex tasks
- **Jurassic-2 Mid** - Medium-sized model
- **Jurassic-2 Light** - Compact model for simpler tasks

### Cohere

Models focusing on enterprise applications and embeddings.

- **Command** - Text generation model
- **Embed** - Text embedding model

## Modalities

AWS Bedrock supports various input and output modalities:

### Text Generation

All foundation models support text generation with varying capabilities for:
- Completion tasks
- Question answering
- Summarization
- Creative writing
- Code generation

### Image Generation

Models supporting image generation include:
- Stability AI's Stable Diffusion models
- Amazon Titan Image Generator

### Image Understanding

Models supporting image inputs include:
- Claude 3 (all variants)
- Titan Multimodal

### Document Processing

Some models are particularly effective for document processing:
- Claude models with their high context windows
- Amazon Titan Text models

## Experimenting with Different Models

### Using AWS CLI for Model Exploration

List available foundation models:

```bash
aws bedrock list-foundation-models --profile aws --region us-west-2
```

Get detailed information about a specific model:

```bash
aws bedrock get-foundation-model \
  --model-identifier anthropic.claude-3-sonnet-20240229-v1:0 \
  --profile aws \
  --region us-west-2
```

### Python Code for Model Comparison

Here's a simple framework for comparing different models on the same task:

```python
import boto3
import json
from utils.profile_manager import get_profile
from utils.visualization import generate_comparison_chart_svg

# Use the configured profile (defaults to 'aws' for local testing)
profile_name = get_profile()
session = boto3.Session(profile_name=profile_name)
bedrock = session.client('bedrock-runtime')

models = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "meta.llama2-13b-chat-v1",
    "amazon.titan-text-express-v1"
]

prompt = "Explain quantum computing in simple terms."
results = {}

for model in models:
    # Different models require different request formats
    if "anthropic" in model:
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    elif "meta" in model:
        request = {
            "prompt": f"<s>[INST] {prompt} [/INST]",
            "max_gen_len": 1000
        }
    else:  # Amazon Titan and others
        request = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1000,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    
    response = bedrock.invoke_model(
        modelId=model,
        body=json.dumps(request)
    )
    
    response_body = json.loads(response['body'].read())
    
    # Different models return results in different formats
    if "anthropic" in model:
        results[model] = response_body['content'][0]['text']
    elif "meta" in model:
        results[model] = response_body['generation']
    else:
        results[model] = response_body['results'][0]['outputText']

# Generate SVG visualization comparing response lengths, latency, etc.
generate_comparison_chart_svg(results, "model_comparison.svg")
```

## Model Selection Guidelines

When choosing a foundation model for your application, consider:

1. **Task complexity** - More complex tasks may require larger models
2. **Latency requirements** - Smaller models typically have lower latency
3. **Cost considerations** - Pricing varies significantly between models
4. **Context window needs** - Models have different maximum input sizes
5. **Modality requirements** - Only certain models support multimodal inputs
6. **Quota availability** - Higher-capacity models often have stricter quotas

In the next sections, we'll explore detailed examples of working with each model family and compare their performance on specific tasks.