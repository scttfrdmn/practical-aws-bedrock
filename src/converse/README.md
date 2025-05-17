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
│   ├── memory.py        # Conversation memory management (Planned)
│   └── parameters.py    # Parameter tuning examples (Planned)
│
├── advanced/            # Advanced implementation examples
│   ├── streaming.py     # Streaming conversation responses (Planned)
│   ├── tools.py         # Function calling with Converse API (Planned)
│   └── multimodal.py    # Handling images in conversations (Planned)
│
├── utils/               # Shared utilities
│   ├── history.py       # Conversation history management (Planned)
│   ├── token_limit.py   # Token limit optimization utilities (Planned)
│   └── templates.py     # System prompt templates (Planned)
│
└── examples/            # Complete application examples
    ├── converse_example.py # Basic conversation examples
    ├── chatbot.py        # Complete chatbot implementation (Planned)
    ├── qa_system.py      # Question answering with context (Planned)
    └── agent.py          # Tool-using agent implementation (Planned)
```

## Usage Examples

### Basic Conversation

```python
from src.converse.basic.simple_chat import ConverseClient

# Create a client
client = ConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Start a conversation
conversation_id = client.create_conversation(
    system_prompt="You are a helpful assistant specializing in AWS services."
)

# Send messages and get responses
response = client.send_message(
    conversation_id=conversation_id,
    message="What is AWS Bedrock and what can I do with it?"
)
print(response)

# Continue the conversation
response = client.send_message(
    conversation_id=conversation_id,
    message="Which foundation models does it support?"
)
print(response)
```

### Interactive Chat Example

```python
# Run the interactive chat example
from src.converse.examples.converse_example import interactive_chat_example

# This will start an interactive chat session in the console
interactive_chat_example()
```

## Implementation Status

- [x] Basic Conversation - Implemented in simple_chat.py
- [ ] Memory Management - Not started
- [ ] Streaming Responses - Not started
- [ ] Function Calling - Not started
- [ ] Multimodal Support - Not started

## Next Steps

1. Implement memory management with token optimization
2. Develop streaming response capabilities
3. Create function calling examples
4. Add multimodal conversation support
5. Build advanced examples (chatbot, QA system, agent)

## Contributing

See the project [CONTRIBUTING](../../CONTRIBUTING.md) guidelines for information on how to contribute to this module.

## Documentation

For detailed information about the Converse API implementation, see the [Converse API chapter](../../docs/chapters/apis/converse.md) in the project documentation.

## Last Updated

This module was last updated on May 16, 2024.