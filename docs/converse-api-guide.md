# Conversational AI with AWS Bedrock Converse API

The AWS Bedrock Converse API provides a streamlined way to build conversational experiences with foundation models. This specialized API simplifies multi-turn conversations by handling conversation state management, providing a consistent interface across different model families.

## What is the Converse API?

The Converse API is a purpose-built interface for creating conversational applications. It abstracts away model-specific prompt formats and handles conversation memory management, making it easier to create chat-like experiences regardless of which foundation model you're using.

![Converse API Overview](images/converse-api-diagram.svg)

## Key Benefits

- **Unified interface** across different model families (Claude, Llama, Titan, etc.)
- **Simplified conversation management** with built-in memory handling
- **Standardized message format** for human/AI interactions
- **Enhanced conversation control** with system prompts and conversation settings
- **Function calling capabilities** for integrating with external tools and APIs
- **Multimodal support** for conversations with both text and images

## When to Use the Converse API

The Converse API is ideal for:

1. **Chat applications** - Building conversational interfaces and chatbots
2. **Multi-turn interactions** - Any application requiring back-and-forth dialogue
3. **Cross-model compatibility** - Applications that might use different models
4. **Memory-dependent use cases** - When conversation history matters
5. **Mixed media conversations** - Including both text and images in conversations

## Core Concepts

### Conversation Structure

The Converse API structures conversations around these key elements:

- **Messages**: Individual utterances in the conversation
- **Roles**: Identifies who said what (human, assistant, system)
- **System prompt**: Special instructions that guide the model's behavior
- **Conversation ID**: Unique identifier for tracking a conversation
- **Generation parameters**: Control over temperature, token limits, etc.

### Basic Message Flow

```
Human: "Hello, can you help me with AWS Bedrock?"
Assistant: "Hi there! I'd be happy to help you with AWS Bedrock. What would you like to know?"
Human: "What models are available?"
Assistant: "AWS Bedrock offers several foundation models including..."
```

### System Prompts

System prompts define the AI assistant's behavior, capabilities, and limitations:

```
System: "You are a friendly AI assistant specialized in AWS services. Provide concise, 
technically accurate answers. When you don't know something, say so rather than speculating."
```

## Implementation Example

Here's a basic Python example using our library:

```python
from aws_bedrock_inference.converse import ConverseClient

# Create a client for conversational AI
client = ConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Start a new conversation with a system prompt
conversation_id = client.create_conversation(
    system_prompt="You are a helpful assistant specialized in AWS services."
)

# Send the first message
response = client.send_message(
    conversation_id=conversation_id,
    message="What is AWS Bedrock and what can I do with it?"
)

print(f"Assistant: {response}")

# Continue the conversation
follow_up = client.send_message(
    conversation_id=conversation_id,
    message="Which foundation models does it support?"
)

print(f"Assistant: {follow_up}")

# Retrieve conversation history
history = client.get_conversation_history(conversation_id)
print(f"Message count: {len(history['messages'])}")
```

## Memory Management

The Converse API automatically maintains conversation history, but for long conversations, you may need to manage memory:

```python
from aws_bedrock_inference.converse import MemoryAwareConverseClient

# Create a memory-aware client
client = MemoryAwareConverseClient(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    max_history_tokens=8000  # Limit the token count of stored history
)

# Start a conversation
conversation_id = client.create_conversation()

# The client will automatically manage conversation history:
# - Tracking token usage
# - Pruning old messages when needed
# - Keeping recent context
# - Preserving important context

# Check current memory usage
memory_stats = client.get_memory_stats(conversation_id)
print(f"Current token usage: {memory_stats['current_tokens']}")
print(f"Available token space: {memory_stats['available_tokens']}")
```

## Streaming Responses

For real-time interactions, use streaming:

```python
from aws_bedrock_inference.converse import StreamingConverseClient

# Create a streaming client
client = StreamingConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Start a conversation
conversation_id = client.create_conversation()

# Stream a response
print("Assistant: ", end="", flush=True)
for chunk in client.send_message_streaming(
    conversation_id=conversation_id,
    message="Explain how AWS Bedrock handles quota limits"
):
    print(chunk, end="", flush=True)
print()
```

## Function Calling

The Converse API supports function calling, allowing the AI to use external tools:

```python
from aws_bedrock_inference.converse import FunctionCallingConverseClient

# Define functions the AI can call
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or zip code"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }
    }
]

# Create client with function definitions
client = FunctionCallingConverseClient(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    functions=functions
)

# Function implementations
def get_weather(location, unit="celsius"):
    # In a real app, this would call a weather API
    return {
        "location": location,
        "temperature": 72 if unit == "fahrenheit" else 22,
        "condition": "sunny",
        "humidity": "45%"
    }

# Function dispatcher
function_map = {
    "get_weather": get_weather
}

# Start conversation
conversation_id = client.create_conversation()

# Process a message that might trigger function calls
result = client.send_message_with_functions(
    conversation_id=conversation_id,
    message="What's the weather like in Seattle?",
    function_map=function_map
)

print(f"Final response: {result['response']}")
print(f"Function calls made: {len(result['function_calls'])}")
```

## Multimodal Conversations

For conversations that include images:

```python
from aws_bedrock_inference.converse import MultimodalConverseClient

# Create a multimodal client
client = MultimodalConverseClient(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

# Start a conversation
conversation_id = client.create_conversation()

# Send a message with an image
with open("diagram.jpg", "rb") as image_file:
    image_bytes = image_file.read()
    
response = client.send_message_with_image(
    conversation_id=conversation_id,
    message="What does this diagram show?",
    image_bytes=image_bytes,
    image_format="jpeg"
)

print(f"Assistant: {response}")
```

## Error Handling

Implement robust error handling for conversations:

```python
try:
    response = client.send_message(
        conversation_id=conversation_id,
        message="Tell me about AWS services."
    )
    print(f"Assistant: {response}")
    
except Exception as e:
    if "ConversationNotFound" in str(e):
        # Create a new conversation if the old one expired
        conversation_id = client.create_conversation()
        print("Created a new conversation.")
        
    elif "ThrottlingException" in str(e):
        # Implement backoff and retry
        print("Rate limited. Retrying after delay...")
        time.sleep(2)
        
    elif "ModelErrorException" in str(e):
        # Handle content filtering or other model errors
        print("The model encountered an error processing your request.")
        
    else:
        # Handle other errors
        print(f"Error: {str(e)}")
```

## Conversation Persistence

For long-running applications, implement conversation persistence:

```python
import json

def save_conversation(conversation_id, client, filename):
    """Save conversation to a file."""
    history = client.get_conversation_history(conversation_id)
    
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Conversation saved to {filename}")

def load_conversation(client, filename):
    """Load conversation from a file."""
    with open(filename, 'r') as f:
        history = json.load(f)
    
    # Create a new conversation with the loaded history
    conversation_id = client.create_conversation(
        system_prompt=history.get('system_prompt')
    )
    
    # Add messages in order (skipping system prompt)
    for message in history['messages']:
        if message['role'] != 'system':
            client.add_message(
                conversation_id=conversation_id,
                role=message['role'],
                content=message['content']
            )
    
    return conversation_id
```

## Model-Specific Considerations

Different models have unique characteristics when used with the Converse API:

### Anthropic Claude

- Stronger system prompt adherence
- Better at following complex instructions
- Larger context window (up to 100K tokens)
- More detailed function calling capabilities

### Meta Llama 2

- More concise responses
- Different style/tone characteristics
- Smaller context window (typically 4K tokens)
- More economical for high-volume applications

### Amazon Titan

- Often higher default quotas
- Different personality characteristics
- Variation in instruction-following capabilities

## Performance Optimization

To optimize conversations:

1. **Use appropriate system prompts**: Clear, concise instructions in system prompts
2. **Manage conversation length**: Prune history for long conversations
3. **Implement caching**: Cache common responses to reduce API calls
4. **Use streaming**: Improve user experience with streaming responses
5. **Right-size models**: Use smaller models for simpler conversations

## Quota Considerations

Conversations are subject to the same AWS Bedrock quota limits:

1. **TPM (tokens per minute)**: Each message in the conversation contributes to TPM
2. **RPM (requests per minute)**: Each API call counts toward your RPM limit

When planning conversational applications, consider:

- Conversation history increases token usage over time
- Each turn requires at least one API call
- System prompts consume tokens with every request

## Comparison with Direct Model Invocation

| Aspect | Converse API | Direct Model Invocation |
|--------|-------------|------------------------|
| Conversation State | Managed automatically | Must handle manually |
| Cross-Model Compatibility | Unified interface | Model-specific formats |
| Message Structure | Standardized | Varies by model |
| Function Calling | Standardized | Varies by model |
| Implementation Complexity | Lower | Higher |
| Flexibility | Less granular control | More granular control |
| Token Efficiency | May use more tokens | Potentially more efficient |

## Next Steps

- Explore [Converse API Examples](../src/converse) for implementation details
- Learn about [Memory Management](memory-management.md) for long conversations
- See [Function Calling Patterns](function-calling-patterns.md) for tool integration
- Understand [Quota Management](quota-management.md) for scaling conversations