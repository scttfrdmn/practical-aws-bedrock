---
layout: page
title: AWS CLI Guide for Bedrock
difficulty: basic
time_required: 20 minutes
---

# AWS CLI Guide for Bedrock

This tutorial explains how to use the AWS Command Line Interface (CLI) to work with Amazon Bedrock services.

## Prerequisites

- AWS CLI installed (version 2.x recommended)
- Configured AWS profile with Bedrock permissions
- Basic familiarity with command line interfaces

## Setting Up AWS CLI for Bedrock

### Configure Your Profile

For local testing, we always use the `aws` profile as documented in our CLAUDE.md file:

```bash
aws configure --profile aws
```

Enter your AWS Access Key ID, Secret Access Key, default region (use a region where Bedrock is available, such as `us-west-2`), and preferred output format (json recommended).

### Verify Bedrock Access

Confirm you can access Bedrock services:

```bash
aws bedrock list-foundation-models --profile aws
```

If successful, you'll see a JSON response listing available models.

## Common Bedrock CLI Commands

### List Available Models

```bash
aws bedrock list-foundation-models --profile aws
```

### Get Model Details

```bash
aws bedrock get-foundation-model \
  --model-identifier anthropic.claude-3-sonnet-20240229-v1:0 \
  --profile aws
```

### Check Model Throughput Capacity

```bash
aws bedrock get-foundation-model-throughput-capacity \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --profile aws
```

### List Custom Models

If you have custom models:

```bash
aws bedrock list-custom-models --profile aws
```

### Check Service Quotas

To view your current quotas:

```bash
aws service-quotas list-service-quotas \
  --service-code bedrock \
  --profile aws
```

To request a quota increase:

```bash
aws service-quotas request-service-quota-increase \
  --service-code bedrock \
  --quota-code L-12345678 \
  --desired-value 100 \
  --profile aws
```

### Run Inference (Text Example)

For simple testing, you can invoke a model directly via CLI:

```bash
aws bedrock-runtime invoke-model \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --body '{"anthropic_version":"bedrock-2023-05-31","max_tokens":1000,"messages":[{"role":"user","content":"Explain quantum computing in simple terms."}]}' \
  --profile aws \
  output.json
```

This saves the response to `output.json`.

### Streaming Inference

For streaming responses (requires further processing):

```bash
aws bedrock-runtime invoke-model-with-response-stream \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --body '{"anthropic_version":"bedrock-2023-05-31","max_tokens":1000,"messages":[{"role":"user","content":"Write a short story about robots."}]}' \
  --profile aws \
  output_stream.json
```

## Working with Different Model Formats

Different models require different request formats. Here are examples for major model families:

### Anthropic Claude

```bash
aws bedrock-runtime invoke-model \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --body '{
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "messages": [
      {"role": "user", "content": "Write a poem about machine learning."}
    ]
  }' \
  --profile aws \
  claude_output.json
```

### Meta Llama 2

```bash
aws bedrock-runtime invoke-model \
  --model-id meta.llama2-13b-chat-v1 \
  --body '{
    "prompt": "<s>[INST] Write a poem about machine learning. [/INST]",
    "max_gen_len": 1000,
    "temperature": 0.7,
    "top_p": 0.9
  }' \
  --profile aws \
  llama_output.json
```

### Amazon Titan

```bash
aws bedrock-runtime invoke-model \
  --model-id amazon.titan-text-express-v1 \
  --body '{
    "inputText": "Write a poem about machine learning.",
    "textGenerationConfig": {
      "maxTokenCount": 1000,
      "temperature": 0.7,
      "topP": 0.9
    }
  }' \
  --profile aws \
  titan_output.json
```

## Processing Responses

To extract just the generated text from the response JSON:

```bash
cat claude_output.json | jq -r '.content[0].text'
cat llama_output.json | jq -r '.generation'
cat titan_output.json | jq -r '.results[0].outputText'
```

## Creating Shell Scripts for Common Tasks

Here's an example shell script that tests multiple models with the same prompt:

```bash
#!/bin/bash

PROFILE="aws"
PROMPT="Explain the concept of cloud computing."

# Claude
echo "Testing Claude 3 Sonnet..."
aws bedrock-runtime invoke-model \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --body "{\"anthropic_version\":\"bedrock-2023-05-31\",\"max_tokens\":1000,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" \
  --profile $PROFILE \
  claude_response.json

# Llama 2
echo "Testing Llama 2..."
aws bedrock-runtime invoke-model \
  --model-id meta.llama2-13b-chat-v1 \
  --body "{\"prompt\":\"<s>[INST] $PROMPT [/INST]\",\"max_gen_len\":1000}" \
  --profile $PROFILE \
  llama_response.json

# Titan
echo "Testing Titan..."
aws bedrock-runtime invoke-model \
  --model-id amazon.titan-text-express-v1 \
  --body "{\"inputText\":\"$PROMPT\",\"textGenerationConfig\":{\"maxTokenCount\":1000}}" \
  --profile $PROFILE \
  titan_response.json

echo "All tests complete. Results saved to *_response.json files."
```

Make the script executable with `chmod +x test_models.sh` and run it with `./test_models.sh`.

## Next Steps

Now that you're familiar with the basic AWS CLI commands for Bedrock, you can:

1. Create more advanced scripts for automation
2. Integrate these commands into your development workflow
3. Set up monitoring scripts to track quota usage
4. Explore advanced topics like batch processing