# AWS Bedrock Converse API Examples

This directory contains implementation examples for the AWS Bedrock Converse API, which simplifies building conversational AI applications with AWS Bedrock foundation models.

## Overview

The Converse API provides a standardized interface for multi-turn conversations with foundation models, abstracting away model-specific prompt formats and handling conversation state management automatically.

Key benefits:
- Simplified conversation management
- Model-agnostic interface
- Integrated memory management
- Structured response formats

## Directory Structure

```
converse/
├── basic/               # Basic implementation examples
│   ├── simple_chat.py   # Simple chat implementation
│   ├── memory.py        # Conversation memory management
│   └── parameters.py    # Parameter tuning examples
│
├── advanced/            # Advanced implementation examples
│   ├── streaming.py     # Streaming conversation responses
│   ├── tools.py         # Function calling with Converse API
│   └── multimodal.py    # Handling images in conversations
│
├── utils/               # Shared utilities
│   ├── history.py       # Conversation history management
│   ├── token_limit.py   # Token limit optimization utilities
│   └── templates.py     # System prompt templates
│
└── examples/            # Complete application examples
    ├── chatbot.py       # Complete chatbot implementation
    ├── qa_system.py     # Question answering with context
    └── agent.py         # Tool-using agent implementation
```

## Usage Examples

### Basic Conversation

```python
from aws_bedrock_inference.converse.basic.simple_chat import ConverseClient

# Create a client
client = ConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Start a conversation
conversation_id = client.create_conversation()

# Send messages and get responses
response = client.send_message(conversation_id, "Hello! Can you help me with AWS Bedrock?")
print(response)

# Continue the conversation
response = client.send_message(conversation_id, "What are the different inference methods available?")
print(response)
```

### Conversation with Memory Management

```python
from aws_bedrock_inference.converse.basic.memory import MemoryAwareConverseClient

# Create a memory-aware client that optimizes token usage
client = MemoryAwareConverseClient(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    max_history_tokens=8000
)

# Start a conversation with a system prompt
conversation_id = client.create_conversation(
    system_prompt="You are an AWS expert specializing in foundation models."
)

# Conversation will automatically manage history token usage
for i in range(20):
    user_message = input("> ")
    if user_message.lower() == "exit":
        break
    
    response = client.send_message(conversation_id, user_message)
    print(f"Assistant: {response}")
```

### Streaming Conversation

```python
from aws_bedrock_inference.converse.advanced.streaming import StreamingConverseClient

# Create a streaming client
client = StreamingConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Start a conversation
conversation_id = client.create_conversation()

# Send a message and stream the response
user_message = "Write a short story about space exploration."
for chunk in client.send_message_streaming(conversation_id, user_message):
    print(chunk, end="", flush=True)
```

## Implementation Status

- [  ] Basic Conversation - Not started
- [  ] Memory Management - Not started
- [  ] Streaming Responses - Not started
- [  ] Function Calling - Not started
- [  ] Multimodal Support - Not started

## Next Steps

1. Implement basic conversation handling
2. Add memory management with token optimization
3. Develop streaming response capabilities
4. Create function calling examples
5. Add multimodal conversation support

## Contributing

See the project [CONTRIBUTING](../../CONTRIBUTING.md) guidelines for information on how to contribute to this module.