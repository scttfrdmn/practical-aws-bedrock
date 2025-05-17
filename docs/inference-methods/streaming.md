# Streaming Inference with AWS Bedrock

Streaming inference provides a real-time, incremental response experience by returning content chunks as they're generated. This significantly improves perceived latency and enables more interactive applications with AWS Bedrock foundation models.

## What is Streaming Inference?

In streaming inference, your application sends a complete prompt to the foundation model, but instead of waiting for the entire response, the model returns small chunks of the response as they're generated. This is implemented through AWS Bedrock's `InvokeModelWithResponseStream` API.

![Streaming Inference Flow](../images/streaming-inference-diagram.svg)

## Key Characteristics

- **Incremental response delivery**: Content arrives in chunks as it's generated
- **Faster time-to-first-token**: Users see initial content more quickly
- **Interactive experience**: Applications can display content progressively
- **Same token usage**: Uses the same token quota as synchronous requests
- **Same billing model**: No cost difference compared to synchronous inference

## When to Use Streaming Inference

Streaming inference is ideal for:

1. **User-facing applications** - Where user perception of speed is important
2. **Chat interfaces** - To provide a more natural, responsive conversation flow
3. **Content generation** - When generating longer outputs like articles or reports
4. **Real-time collaboration** - Where multiple users might view content generation
5. **Interactive experiences** - Applications where users may want to interrupt generation

## When Not to Use Streaming Inference

Consider other methods when:

1. **Processing in batch** - Where real-time display isn't needed
2. **Needing complete responses** - When your processing requires the full response
3. **Limited bandwidth** - In environments where minimizing network requests is critical
4. **Complex client handling** - If your client application can't easily handle streaming

## Quota Considerations

Streaming inference is subject to the same quota limits as synchronous inference:

1. **Tokens Per Minute (TPM)** - Limits the total input + output tokens processed per minute
2. **Requests Per Minute (RPM)** - Limits the number of API calls regardless of token count

However, streaming has unique quota advantages:

- **Perception optimization**: Users perceive content generation as faster, even under the same quota
- **Early termination**: Users may get what they need before the full response completes
- **Timeout mitigation**: Long responses are less likely to encounter timeout issues

## Implementation Example

Here's a basic Python example using our library:

```python
from aws_bedrock_inference import BedrockStreamingClient

# Create a streaming client for Claude
client = BedrockStreamingClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Basic streaming example
try:
    print("Streaming response:")
    print("-" * 50)
    
    # Use iterator approach
    for chunk in client.invoke_stream(
        prompt="Write a short story about space exploration.",
        max_tokens=500
    ):
        # Process and display each chunk as it arrives
        print(chunk, end="", flush=True)
    
    print("\n" + "-" * 50)
    
except Exception as e:
    print(f"Error: {str(e)}")
```

## Callback-Based Implementation

For more complex handling, you can use callbacks:

```python
from aws_bedrock_inference import BedrockStreamingClient

# Create a streaming client
client = BedrockStreamingClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Define callback functions
def on_content_chunk(chunk):
    """Called for each content chunk."""
    print(chunk, end="", flush=True)

def on_complete(result):
    """Called when streaming completes."""
    print("\nStreaming complete.")
    print(f"Generated approximately {result['estimated_tokens']} tokens.")

def on_error(error):
    """Called when an error occurs."""
    print(f"\nError occurred: {str(error)}")

# Stream with callbacks
client.invoke_stream_with_callbacks(
    prompt="Explain the benefits of streaming responses in AI applications.",
    on_content=on_content_chunk,
    on_complete=on_complete,
    on_error=on_error,
    max_tokens=500
)
```

## Front-End Integration

When integrating streaming responses with front-end applications:

### Web Applications (JavaScript)

```javascript
async function streamResponse() {
    const response = await fetch('/api/stream-completion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: 'Write a story about...' })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const outputElement = document.getElementById('output');
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        // Decode and display the chunk
        const chunk = decoder.decode(value);
        outputElement.textContent += chunk;
    }
}
```

### Server-Side Implementation (Express.js)

```javascript
app.post('/api/stream-completion', async (req, res) => {
    const { prompt } = req.body;
    
    // Set up streaming headers
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Transfer-Encoding', 'chunked');
    
    try {
        // Make request to your Python backend
        const response = await fetch('http://your-python-backend/stream', {
            method: 'POST',
            body: JSON.stringify({ prompt })
        });
        
        // Forward the stream
        const reader = response.body.getReader();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            res.write(value);
        }
        
        res.end();
    } catch (error) {
        res.status(500).send('Error: ' + error.message);
    }
});
```

## Error Handling

Common errors you may encounter with streaming inference:

1. **ThrottlingException**: You've exceeded your RPM or TPM quota
2. **ValidationException**: Invalid request parameters or token limits exceeded
3. **ServiceException**: Internal service errors
4. **StreamInterruptedException**: The stream was interrupted before completion
5. **ConnectionError**: Network issues during streaming

Best practices for handling streaming errors:

```python
try:
    for chunk in client.invoke_stream(prompt="Your prompt here"):
        # Process each chunk
        print(chunk, end="", flush=True)
except Exception as e:
    if "ThrottlingException" in str(e):
        # Implement exponential backoff and retry
        time.sleep(backoff_time)
        # Retry the request
    elif "StreamInterruptedException" in str(e):
        # Handle partial response case
        print("\nStream was interrupted. Partial response:")
        print("".join(received_chunks))
    else:
        # Handle other errors
        print(f"\nError: {str(e)}")
```

## Performance Optimization

To optimize streaming inference performance:

1. **Buffer appropriately**: Decide whether to buffer chunks or display immediately
2. **Implement loading states**: Show typing indicators or placeholders while waiting
3. **Consider chunk processing**: Process chunks for formatting or post-processing
4. **Handle backpressure**: Ensure your application can handle fast-arriving chunks
5. **Implement cancellation**: Allow users to cancel ongoing generations

## Model Selection Considerations

Different foundation models have different streaming characteristics:

- **Anthropic Claude models**: Send larger, more coherent chunks
- **Llama 2 models**: Often provide smaller, more frequent chunks
- **Amazon Titan models**: Varied chunk sizes based on model version
- **Cohere/AI21 models**: Streaming behavior varies by implementation

Test multiple models to find the best streaming experience for your application.

## UX Considerations

When implementing streaming interfaces:

1. **Typing indicators**: Show the AI is "thinking" before content starts streaming
2. **Scrolling behavior**: Auto-scroll to follow new content as it appears
3. **Chunk aggregation**: Smooth out chunk rendering to avoid jarring updates
4. **Error recovery**: Handle interruptions gracefully with retry options
5. **Visual formatting**: Apply formatting as content streams, not just at the end

## Comparison with Other Methods

| Aspect | Streaming | Synchronous | Asynchronous |
|--------|-----------|-------------|--------------|
| Response Time | Immediate, incremental | Delayed, complete | Very delayed |
| User Experience | Interactive, real-time | Wait until complete | Not for real-time use |
| Implementation Complexity | Moderate | Simple | Complex |
| Timeout Risk | Low | Higher | Lowest |
| Quota Impact | Same as synchronous | Same as streaming | Same (but less time pressure) |
| Best For | Interactive, user-facing apps | Simple backend use | Large batch processing |

## Next Steps

- Learn about [Synchronous Inference](synchronous.md) for simpler implementations
- Explore [Asynchronous Processing](asynchronous.md) for handling very large requests
- Understand [Quota Management](../quota-management.md) for optimizing throughput