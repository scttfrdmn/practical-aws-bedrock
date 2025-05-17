---
layout: page
title: AWS Bedrock Inference Methods
---

# AWS Bedrock Inference Methods

This document provides a comprehensive overview of the various inference methods available in AWS Bedrock, their use cases, advantages, and implementation details.

## Overview of Inference Methods

AWS Bedrock offers multiple ways to interact with foundation models:

1. **Synchronous Inference** (InvokeModel)
2. **Streaming Inference** (InvokeModelWithResponseStream)
3. **Asynchronous Processing** (CreateModelInvocationJob)
4. **Conversational AI** (Converse API)
5. **Structured Outputs** (Construct API)

Each method has different characteristics and is suited for specific use cases. Understanding these differences is crucial for optimizing throughput, managing quota limits, and providing the best user experience.

## Synchronous Inference (InvokeModel)

### Overview

Synchronous inference is the simplest way to interact with foundation models. It follows a request-response pattern where you send a prompt and wait for the complete response before proceeding.

### When to Use

- Single, independent requests that don't require real-time feedback
- Batch processing where you can wait for the full result
- Simple integrations where streaming adds unnecessary complexity
- Cases where you need the full response before taking any action

### AWS SDK Implementation

```python
import boto3
import json
from utils.profile_manager import get_profile

def invoke_model_sync(model_id, prompt_data):
    """
    Perform synchronous inference using AWS Bedrock.
    
    Args:
        model_id: The model identifier
        prompt_data: Dictionary with the prompt payload
        
    Returns:
        The model's response
    """
    # Use the configured profile (defaults to 'aws' for local testing)
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime')
    
    # Convert the prompt data to JSON string
    body = json.dumps(prompt_data)
    
    # Invoke the model
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=body
    )
    
    # Parse and return the response
    response_body = json.loads(response['body'].read())
    return response_body
```

### AWS CLI Example

```bash
aws bedrock-runtime invoke-model \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --body '{"anthropic_version":"bedrock-2023-05-31","max_tokens":1000,"messages":[{"role":"user","content":"What is quantum computing?"}]}' \
  --profile aws \
  output.json
```

### Quota Considerations

- Subject to both TPM (tokens per minute) and RPM (requests per minute) quotas
- Each request counts as 1 against RPM quota
- Both input and output tokens count against TPM quota
- If a request completes very quickly, you may still be limited by RPM quota

### Best Practices

1. **Batch related requests** - Process multiple items in a single request when possible
2. **Optimize prompt size** - Use concise prompts to reduce token usage
3. **Implement retries with backoff** - Handle throttling errors gracefully
4. **Monitor token usage** - Track both input and output tokens to predict quota consumption

## Streaming Inference (InvokeModelWithResponseStream)

### Overview

Streaming inference allows you to receive the model's response incrementally as it's being generated, rather than waiting for the complete response.

### When to Use

- Interactive applications where showing incremental results improves user experience
- Long-form content generation where you want to display progress
- Applications where perceived latency is more important than total processing time
- Chat interfaces where typing indicators or progressive responses are expected

### AWS SDK Implementation

```python
import boto3
import json
from utils.profile_manager import get_profile

def invoke_model_stream(model_id, prompt_data):
    """
    Perform streaming inference using AWS Bedrock.
    
    Args:
        model_id: The model identifier
        prompt_data: Dictionary with the prompt payload
        
    Returns:
        Generator yielding response chunks as they arrive
    """
    # Use the configured profile (defaults to 'aws' for local testing)
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime')
    
    # Convert the prompt data to JSON string
    body = json.dumps(prompt_data)
    
    # Invoke the model with streaming
    response = bedrock_runtime.invoke_model_with_response_stream(
        modelId=model_id,
        body=body
    )
    
    # Process the streaming response
    stream = response.get('body')
    
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_data = json.loads(chunk.get('bytes').decode())
                yield chunk_data
```

### AWS CLI Example

```bash
aws bedrock-runtime invoke-model-with-response-stream \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --body '{"anthropic_version":"bedrock-2023-05-31","max_tokens":1000,"messages":[{"role":"user","content":"Write a short story about robots."}]}' \
  --profile aws \
  output_stream.json
```

### Quota Considerations

- Subject to the same TPM and RPM quotas as synchronous inference
- Can provide better perceived performance despite the same total processing time
- May improve throughput for large outputs by starting to process results before completion
- Token usage is identical to synchronous requests

### Best Practices

1. **Update UI incrementally** - Display content chunks as they arrive for better UX
2. **Implement robust error handling** - Handle stream interruptions gracefully
3. **Consider connection timeout limits** - For very long responses, be aware of connection limits
4. **Manage incomplete responses** - Design your application to handle partial results if the stream is interrupted

## Asynchronous Processing (CreateModelInvocationJob)

### Overview

Asynchronous processing allows you to submit long-running inference jobs without maintaining an open connection. You submit a job, receive a job ID, and can check for results later.

### When to Use

- Long-running inference tasks that may exceed typical connection timeouts
- Batch processing of multiple requests
- Background processing where you don't need immediate results
- Heavy workloads where you need to manage throughput without overwhelming the system

### AWS SDK Implementation

```python
import boto3
import json
import time
from utils.profile_manager import get_profile

def create_async_inference_job(model_id, prompt_data, output_s3_uri):
    """
    Create an asynchronous inference job.
    
    Args:
        model_id: The model identifier
        prompt_data: Dictionary with the prompt payload
        output_s3_uri: S3 URI where results should be stored
        
    Returns:
        Job ID for tracking the job
    """
    # Use the configured profile (defaults to 'aws' for local testing)
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock = session.client('bedrock')
    
    # Convert the prompt data to JSON string
    input_text = json.dumps(prompt_data)
    
    # Create the job
    response = bedrock.create_model_invocation_job(
        modelId=model_id,
        jobName=f"inference-job-{int(time.time())}",
        inputDataConfig={
            'contentType': 'application/json',
            'text': input_text
        },
        outputDataConfig={
            's3Uri': output_s3_uri
        }
    )
    
    return response['jobArn']

def check_job_status(job_arn):
    """
    Check the status of an asynchronous inference job.
    
    Args:
        job_arn: The ARN of the job to check
        
    Returns:
        Dictionary with job status information
    """
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock = session.client('bedrock')
    
    response = bedrock.get_model_invocation_job(
        jobIdentifier=job_arn
    )
    
    return response
```

### AWS CLI Example

```bash
# Submit an asynchronous job
aws bedrock create-model-invocation-job \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --job-name "my-batch-job-1" \
  --input-data-config \
    contentType="application/json",\
    text="{\"anthropic_version\":\"bedrock-2023-05-31\",\"max_tokens\":1000,\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed analysis of quantum computing.\"}]}" \
  --output-data-config s3Uri="s3://my-bucket/results/" \
  --profile aws

# Check job status
aws bedrock get-model-invocation-job \
  --job-identifier "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/abcde12345" \
  --profile aws
```

### Quota Considerations

- Different quota limits than real-time inference
- Often allows for higher throughput for large batch operations
- May have limits on concurrent jobs rather than RPM
- Consider storage costs for inputs and outputs in S3

### Best Practices

1. **Implement job polling** - Check job status at appropriate intervals
2. **Manage job lifecycles** - Clean up completed jobs and outputs
3. **Use job batching** - Group related requests into a single job when possible
4. **Implement notification mechanisms** - Use SNS or other services to notify when jobs complete

## Conversational AI (Converse API)

### Overview

The Converse API is purpose-built for multi-turn conversations, handling conversation state and memory management automatically.

### When to Use

- Chat applications with conversation history
- Conversational interfaces requiring context management
- Applications where maintaining conversation state is important
- Cases where you need to manage conversation memory efficiently

### AWS SDK Implementation

```python
import boto3
import json
from utils.profile_manager import get_profile

def converse(model_id, messages, system_prompt=None):
    """
    Use the Converse API for a multi-turn conversation.
    
    Args:
        model_id: The model identifier
        messages: List of conversation messages
        system_prompt: Optional system prompt
        
    Returns:
        The model's response
    """
    # Use the configured profile (defaults to 'aws' for local testing)
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock = session.client('bedrock-runtime')
    
    # Prepare request parameters
    request = {
        "messages": messages
    }
    
    if system_prompt:
        request["system"] = system_prompt
    
    # Invoke the Converse API
    response = bedrock.converse(
        modelId=model_id,
        messages=request["messages"],
        **({"system": system_prompt} if system_prompt else {})
    )
    
    return response
```

### Quota Considerations

- May have separate quota limits from standard inference
- Optimization focuses on efficient conversation history management
- Token usage includes conversation history, which grows with conversation length

### Best Practices

1. **Summarize conversation history** - Periodically summarize long conversations
2. **Prune irrelevant messages** - Remove unimportant turns to save tokens
3. **Use system prompts effectively** - Set context with system prompts instead of user messages
4. **Implement conversation memory strategies** - Consider sliding windows or hierarchical summarization

## Structured Outputs (Construct API)

### Overview

The Construct API provides a way to generate structured outputs in specific formats like JSON or XML, with schema validation.

### When to Use

- When you need consistent, structured data from the model
- Applications requiring JSON or XML outputs
- Integration with databases or APIs expecting specific formats
- Cases where output validation is critical

### AWS SDK Implementation

```python
import boto3
import json
from utils.profile_manager import get_profile

def construct_structured_output(model_id, prompt, schema):
    """
    Use the Construct API to generate structured output.
    
    Args:
        model_id: The model identifier
        prompt: The prompt text
        schema: JSON schema defining the expected output structure
        
    Returns:
        Structured output matching the schema
    """
    # Use the configured profile (defaults to 'aws' for local testing)
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock = session.client('bedrock-runtime')
    
    # Invoke the Construct API
    response = bedrock.construct(
        modelId=model_id,
        prompt=prompt,
        schema=schema
    )
    
    return response
```

### Quota Considerations

- May have specific quota limits separate from standard inference
- Optimize by designing minimal schemas that capture only required fields
- Consider response size when designing schemas

### Best Practices

1. **Design precise schemas** - Clearly define expected output format
2. **Include examples in prompts** - Providing examples helps models generate correct formats
3. **Implement validation** - Always validate outputs against schemas
4. **Create fallback mechanisms** - Handle cases where structured generation fails

## Comparative Analysis

### Performance Characteristics

| Inference Method | Latency | Throughput | User Experience | Quota Efficiency |
|------------------|---------|------------|-----------------|------------------|
| Synchronous      | Higher  | Standard   | Wait for full response | Standard |
| Streaming        | Lower perceived | Standard | Progressive display | Standard |
| Asynchronous     | Highest | Highest    | Background processing | Most efficient for batch |
| Converse API     | Similar to sync | Standard | Maintains context | Depends on conversation length |
| Construct API    | Similar to sync | Standard | Structured data | May require more tokens |

### Use Case Matrix

| Use Case | Recommended Method | Why |
|----------|-------------------|-----|
| Chat interfaces | Streaming or Converse | Better user experience, context management |
| Data extraction | Construct | Ensures consistent, validated output |
| Batch document processing | Asynchronous | Handle large volumes efficiently |
| Simple Q&A | Synchronous | Straightforward implementation |
| Long-form content | Streaming | Show progress during generation |

## Implementation Pattern: Multi-Method Inference Manager

A robust application might use different inference methods based on the specific requirements:

```python
class InferenceManager:
    def __init__(self, model_id, profile_name=None):
        self.model_id = model_id
        self.profile_name = profile_name or get_profile()
        self.session = boto3.Session(profile_name=self.profile_name)
        self.bedrock_runtime = self.session.client('bedrock-runtime')
        self.bedrock = self.session.client('bedrock')
    
    def infer(self, prompt_data, method="sync", **kwargs):
        """
        Perform inference using the specified method.
        
        Args:
            prompt_data: The prompt payload
            method: One of "sync", "stream", "async", "converse", "construct"
            **kwargs: Additional method-specific parameters
            
        Returns:
            Inference results in the appropriate format
        """
        if method == "sync":
            return self._sync_inference(prompt_data)
        elif method == "stream":
            return self._streaming_inference(prompt_data)
        elif method == "async":
            return self._async_inference(prompt_data, kwargs.get("output_s3_uri"))
        elif method == "converse":
            return self._converse(prompt_data, kwargs.get("system_prompt"))
        elif method == "construct":
            return self._construct(prompt_data, kwargs.get("schema"))
        else:
            raise ValueError(f"Unknown inference method: {method}")
    
    def _sync_inference(self, prompt_data):
        """Synchronous inference implementation"""
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            body=json.dumps(prompt_data)
        )
        return json.loads(response['body'].read())
    
    def _streaming_inference(self, prompt_data):
        """Streaming inference implementation"""
        response = self.bedrock_runtime.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(prompt_data)
        )
        stream = response.get('body')
        
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    yield json.loads(chunk.get('bytes').decode())
    
    def _async_inference(self, prompt_data, output_s3_uri):
        """Asynchronous inference implementation"""
        # Implementation details...
        pass
    
    def _converse(self, messages, system_prompt=None):
        """Converse API implementation"""
        # Implementation details...
        pass
    
    def _construct(self, prompt, schema):
        """Construct API implementation"""
        # Implementation details...
        pass
```

## Conclusion

Each inference method in AWS Bedrock offers unique advantages for specific use cases. By understanding these differences and implementing the appropriate method for each scenario, you can optimize for throughput, user experience, and quota efficiency.

The next sections will explore each method in greater depth, with specific code examples, optimization techniques, and real-world use cases.