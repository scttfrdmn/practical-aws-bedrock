# Synchronous Inference with AWS Bedrock

Synchronous inference is the most straightforward approach for working with foundation models in AWS Bedrock. This method follows a simple request-response pattern where the client sends a request and waits for the complete response before proceeding.

## What is Synchronous Inference?

In synchronous inference, your application sends a complete prompt to the foundation model and waits until the model generates the entire response before proceeding. This is implemented through AWS Bedrock's `InvokeModel` API.

![Synchronous Inference Flow](../images/synchronous-inference-diagram.svg)

## Key Characteristics

- **Single request-response flow**: Complete prompt in, complete response out
- **Blocking operation**: Your application waits for the full response
- **Simple implementation**: Straightforward to code and understand
- **Subject to timeouts**: Long responses may exceed API timeouts
- **Full response at once**: No partial/incremental updates

## When to Use Synchronous Inference

Synchronous inference is ideal for:

1. **Simple, standalone requests** - When you need a self-contained response to a question or prompt
2. **Backend processing** - Where real-time user experience isn't a factor
3. **Batch operations** - When processing many items sequentially
4. **Short-form outputs** - When responses are typically brief
5. **Simple architectures** - When you want to avoid the complexity of streaming or asynchronous processing

## When Not to Use Synchronous Inference

Consider other methods when:

1. **Generating lengthy content** - Long responses may hit timeouts (Bedrock's `InvokeModel` has a 30-second timeout)
2. **Building interactive experiences** - Users see no output until complete response is available
3. **Working with limited resources** - Your application may block while waiting for responses
4. **Processing very large inputs** - Large context inputs may exceed size limitations

## Quota Considerations

Synchronous inference is subject to two primary quota limits:

1. **Tokens Per Minute (TPM)** - Limits the total input + output tokens processed per minute
2. **Requests Per Minute (RPM)** - Limits the number of API calls regardless of token count

When planning your application, consider how these quotas impact your throughput:

- Each request counts toward your RPM limit, regardless of size
- Both input and output tokens count toward your TPM limit
- TPM is typically the limiting factor for most applications

## Implementation Example

Here's a basic Python example using our library:

```python
from aws_bedrock_inference import BedrockClient

# Create a client for Claude
client = BedrockClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Simple invocation
try:
    response = client.invoke(
        prompt="Explain quantum computing in simple terms.",
        max_tokens=500,
        temperature=0.7
    )
    
    # Process the response
    print(response["output"])
    
    # Check token usage
    print(f"Input tokens: {response.get('input_tokens')}")
    print(f"Output tokens: {response.get('output_tokens')}")
    print(f"Total tokens: {response.get('total_tokens')}")
    
except Exception as e:
    print(f"Error: {str(e)}")
```

## Quota-Aware Implementation

To manage quotas effectively, our library provides a `QuotaAwareBedrockClient` that implements token bucket rate limiting:

```python
from aws_bedrock_inference import QuotaAwareBedrockClient, QuotaExceededException

# Create a quota-aware client
client = QuotaAwareBedrockClient(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    max_rpm=60,  # 60 requests per minute
    max_tpm=100000  # 100K tokens per minute
)

try:
    # Will wait if quota limits are reached
    response = client.invoke(
        prompt="Explain quantum computing in simple terms.",
        max_tokens=500,
        wait_for_quota=True  # Wait for quota if limit reached
    )
    
    print(response["output"])
    
except QuotaExceededException:
    print("Quota limit reached and wait_for_quota=False")
except Exception as e:
    print(f"Error: {str(e)}")
```

## Error Handling

Common errors you may encounter with synchronous inference:

1. **ThrottlingException**: You've exceeded your RPM or TPM quota
2. **ValidationException**: Invalid request parameters or token limits exceeded
3. **ServiceException**: Internal service errors
4. **Timeouts**: Response generation takes too long (>30 seconds)

Best practices for handling these errors:

```python
try:
    response = client.invoke(prompt="Your prompt here")
    # Process success case
except Exception as e:
    if "ThrottlingException" in str(e):
        # Implement exponential backoff and retry
        time.sleep(backoff_time)
        # Retry the request
    elif "ValidationException" in str(e):
        # Check your input parameters
        print("Invalid request parameters")
    elif "ServiceException" in str(e):
        # Temporary service issue
        print("Service error, retry later")
    else:
        # Handle other errors
        print(f"Unexpected error: {str(e)}")
```

## Performance Optimization

To optimize synchronous inference performance:

1. **Minimize prompt size**: Shorter prompts use fewer tokens and process faster
2. **Batch similar requests**: Process in batches when possible
3. **Implement request queuing**: Spread requests out to avoid hitting rate limits
4. **Use quota-aware clients**: Automatically manage TPM/RPM limits
5. **Cache common responses**: Avoid redundant API calls for frequently asked questions

## Model Selection Considerations

Different foundation models have different quota limits, latency profiles, and capabilities:

- **Anthropic Claude models**: Higher token limits, longer context windows
- **Meta Llama 2 models**: Lower latency, more economical token costs
- **Amazon Titan models**: Often higher default quotas
- **Cohere/AI21 models**: Different performance characteristics

Select the most appropriate model based on your use case and quota requirements.

## Comparison with Other Methods

| Aspect | Synchronous | Streaming | Asynchronous |
|--------|-------------|-----------|--------------|
| Implementation | Simple | Moderate | Complex |
| Response Time | Full response at once | Incremental | Delayed |
| User Experience | Wait until complete | Real-time updates | Not for real-time use |
| Timeout Risk | Higher | Lower | Lowest |
| Quota Impact | Single API call | Single API call | Single job |
| Max Content Size | Limited | Limited | Largest |
| Best For | Simple Q&A | Interactive chat | Large batch jobs |

## Next Steps

- Learn about [Streaming Inference](streaming.md) for real-time response generation
- Explore [Asynchronous Processing](asynchronous.md) for handling large requests
- Understand [Quota Management](../quota-management.md) for optimizing throughput