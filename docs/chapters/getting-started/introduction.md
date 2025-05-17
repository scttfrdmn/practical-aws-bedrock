---
layout: chapter
title: Introduction to AWS Bedrock
difficulty: beginner
time-estimate: 20 minutes
---

# Introduction to AWS Bedrock

> "You've heard the buzz about generative AI, but now you need to actually build something with it. Let's cut through the hype and get practical with AWS Bedrock."

## The Problem

---

**Scenario**: You're a developer tasked with adding AI capabilities to your company's applications. After exploring the landscape, you've found that open-source models require significant infrastructure and expertise to deploy properly, while proprietary API services lock you into specific models with limited customization options.

You need a solution that provides:
1. Access to multiple high-quality foundation models
2. Enterprise-grade security and compliance
3. The ability to customize models for your specific needs
4. Integration with your existing AWS infrastructure
5. Predictable pricing and scaling

AWS Bedrock promises to address these needs, but you're wondering: What exactly is Bedrock? How does it work? And most importantly, how do you start building with it?

---

## Key Concepts Explained

AWS Bedrock is a fully managed service that makes foundation models (FMs) from leading AI companies available through a unified API. Think of it as having a collection of the world's most capable AI models at your fingertips, all accessible through a consistent interface.

### Foundation Models vs. Traditional ML Models

Traditional machine learning typically requires:
- Large amounts of labeled training data
- Domain-specific feature engineering
- Model architectures designed for specific tasks

In contrast, foundation models are:
- Pre-trained on vast amounts of general data
- Capable of zero-shot and few-shot learning
- Adaptable to many different tasks without full retraining

Imagine the difference between hiring specialists for each specific task versus having a versatile team that can quickly adapt to various projects with minimal guidance.

### How AWS Bedrock Works

At its core, AWS Bedrock provides three main capabilities:

1. **Model Access**: Use foundation models through a simple API without managing infrastructure
2. **Model Customization**: Adapt models to your specific needs through fine-tuning and RAG
3. **Application Development**: Build AI applications with enterprise features like monitoring and security

The workflow looks like this:

1. You select a foundation model that fits your use case
2. You interact with the model through the Bedrock API
3. The model generates content based on your input
4. You integrate this content into your application

AWS handles all the infrastructure, scaling, and maintenance behind the scenes.

## Available Foundation Models

AWS Bedrock provides access to a variety of models from different AI companies:

| Provider | Models | Strengths |
|----------|--------|-----------|
| Anthropic | Claude 3 (Opus, Sonnet, Haiku) | Strong reasoning, follows instructions precisely, safety |
| Amazon | Titan | Cost-effective, tight AWS integration |
| Meta | Llama 2 | Open weights, good performance-to-cost ratio |
| Cohere | Command | Exceptional multilingual support, search capabilities |
| AI21 Labs | Jurassic | Strong structured outputs, good at numerical reasoning |
| Stability AI | Stable Diffusion | High-quality image generation |

Each model has different capabilities, pricing, and context window sizes, allowing you to choose the right tool for each task.

## Getting Started with AWS Bedrock

Let's walk through the practical steps to start using AWS Bedrock:

### 1. Enable AWS Bedrock in Your Account

Before you can use Bedrock, you need to enable it:

1. Go to the [AWS Bedrock console](https://console.aws.amazon.com/bedrock)
2. Click "Model access" in the left navigation
3. Request access to the models you want to use
4. Wait for approval (typically instant for most models)

### 2. Set Up Your Environment

For Python development:

```bash
# Create a virtual environment
python -m venv bedrock-env
source bedrock-env/bin/activate  # On Windows: bedrock-env\Scripts\activate

# Install required packages
pip install boto3
```

Make sure your AWS credentials are configured:

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, region (e.g., us-west-2), and output format
```

### 3. Basic Inference Example

Here's a simple example of how to invoke a model:

```python
import boto3
import json

# Create a Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'  # Use a region where Bedrock is available
)

# Prepare the prompt
prompt = "Explain AWS Bedrock in one paragraph"

# Create the request payload (format varies by model)
payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 300,
    "messages": [
        {"role": "user", "content": prompt}
    ]
}

# Invoke the model
response = bedrock_runtime.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps(payload)
)

# Parse the response
response_body = json.loads(response['body'].read())
answer = response_body['content'][0]['text']

print(answer)
```

### 4. Using the AWS Management Console

For quick experimentation:

1. Go to the AWS Bedrock console
2. Select "Playgrounds" from the left navigation
3. Choose "Chat" or "Text" depending on your needs
4. Select a model from the dropdown
5. Enter your prompt and click "Run"

This gives you a quick way to test different models and prompts without writing code.

## Core AWS Bedrock APIs

AWS Bedrock provides several key APIs for different use cases:

### InvokeModel (Synchronous)

The most basic API for sending a prompt and receiving a response:

```python
response = bedrock_runtime.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps(payload)
)
```

### InvokeModelWithResponseStream (Streaming)

For receiving responses as they're generated, token by token:

```python
response = bedrock_runtime.invoke_model_with_response_stream(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps(payload)
)

for event in response['body']:
    if 'chunk' in event:
        chunk = json.loads(event['chunk']['bytes'])
        # Process the chunk
        print(chunk['content'][0]['text'], end='', flush=True)
```

### Converse API

For multi-turn conversations with memory:

```python
conversation = []  # Store conversation history

def chat(user_message):
    # Add user message to conversation
    conversation.append({"role": "user", "content": user_message})
    
    response = bedrock_runtime.converse(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        messages=conversation
    )
    
    # Add model response to conversation
    assistant_message = response['messages'][0]['content']
    conversation.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message
```

### Construct API

For generating structured outputs (like JSON):

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age", "email"]
}

response = bedrock_runtime.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": "Extract information about John who is 30 years old with email john@example.com"}
        ],
        "response_format": {
            "type": "json_object",
            "schema": schema
        }
    })
)
```

## Common Use Cases

AWS Bedrock can support a wide range of applications:

1. **Content Generation**: Creating marketing copy, product descriptions, blog posts
2. **Conversational AI**: Building chatbots and virtual assistants
3. **Document Processing**: Summarizing reports, extracting information, answering questions
4. **Code Generation**: Creating code snippets, refactoring, documentation
5. **Image Generation**: Creating visuals for marketing, product design, creative work

## Best Practices for Getting Started

1. **Start Simple**: Begin with basic prompts before adding complexity
2. **Compare Models**: Test the same prompt across different models to see which performs best
3. **Monitor Costs**: Keep track of token usage, especially during development
4. **Iterate Prompts**: Refine your prompts based on the responses you receive
5. **Handle Errors**: Implement proper error handling for production applications

## Pricing Considerations

AWS Bedrock pricing is based on:

1. **Input Tokens**: Text sent to the model
2. **Output Tokens**: Text generated by the model
3. **Model Selected**: More capable models cost more per token

A token is approximately 3-4 characters or 0.75 words in English. Prices vary significantly by model, so choose appropriately based on your requirements.

## Try It Yourself Challenge

Now it's your turn to get hands-on with AWS Bedrock:

1. Enable AWS Bedrock in your account if you haven't already
2. Set up your development environment with the AWS SDK
3. Try running the basic inference example with different prompts
4. Experiment with at least two different models to compare responses
5. Create a simple application that uses AWS Bedrock to solve a real problem

**Expected Outcome**: 
A working code snippet that invokes an AWS Bedrock model and processes the response in a meaningful way.

## Beyond the Basics

Once you're comfortable with the basics, you can explore:

1. **Model Fine-tuning**: Customize models for your specific domain
2. **Retrieval Augmented Generation (RAG)**: Enhance responses with your own data
3. **Multi-modal Processing**: Working with text and images together
4. **Prompt Engineering**: Crafting effective prompts for better results
5. **Advanced Security**: Implementing VPC endpoints and data encryption

## Key Takeaways

- AWS Bedrock provides access to multiple foundation models through a unified API
- No infrastructure management is required - AWS handles scaling and availability
- Models can be customized through fine-tuning and RAG
- Different models have different strengths, costs, and capabilities
- Getting started requires minimal setup - just enable the service and start making API calls

---

**Next Steps**: Now that you understand the basics of AWS Bedrock, learn about the [different foundation models](/chapters/getting-started/foundation-models/) and their capabilities.

---

© 2025 Scott Friedman. Licensed under CC BY-NC-ND 4.0