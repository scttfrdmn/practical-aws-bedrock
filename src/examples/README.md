# AWS Bedrock Example Applications

This directory contains complete example applications that demonstrate real-world usage patterns for AWS Bedrock.

## Overview

These examples show how to:
- Implement common AWS Bedrock use cases
- Optimize for throughput and cost
- Handle real-world scenarios
- Combine multiple inference methods
- Manage quota limits effectively

## Example Applications

### Text Generation Applications

- **Content Generator** - Generate high-quality content optimized for throughput
- **Text Summarization Service** - Efficiently summarize long documents
- **Batch Processing Pipeline** - Process large document collections efficiently

### Conversational Applications

- **Customer Support Bot** - Multi-turn conversations with context management
- **Knowledge Base Assistant** - Answer questions from a knowledge corpus
- **Interactive Storytelling** - Dynamic story generation with user interactions

### Structured Data Applications

- **Data Extraction Service** - Extract structured data from unstructured text
- **Form Processing Pipeline** - Convert form inputs to structured formats
- **Sentiment Analysis Pipeline** - Extract structured sentiment data

### Multimodal Applications

- **Image Description Service** - Generate detailed descriptions of images
- **Document Analysis Tool** - Extract information from documents with images
- **Visual QA System** - Answer questions about images

## Directory Structure

```
examples/
├── content_generator/     # Content generation application
│   ├── app.py             # Main application code
│   ├── prompt_manager.py  # Prompt management utilities
│   └── README.md          # Application-specific documentation
│
├── chat_assistant/        # Conversational assistant application
│   ├── app.py             # Main application code
│   ├── memory_manager.py  # Conversation memory management
│   └── README.md          # Application-specific documentation
│
├── data_extraction/       # Structured data extraction application
│   ├── app.py             # Main application code
│   ├── schemas/           # JSON schemas for data extraction
│   └── README.md          # Application-specific documentation
│
├── batch_processor/       # Batch processing application
│   ├── app.py             # Main application code
│   ├── job_manager.py     # Asynchronous job management
│   └── README.md          # Application-specific documentation
│
└── quota_manager/         # Quota management utilities
    ├── app.py             # Main application code
    ├── visualizer.py      # Quota usage visualization
    └── README.md          # Application-specific documentation
```

## Usage Example: Content Generator

```python
from aws_bedrock_inference.examples.content_generator.app import ContentGenerator

# Create a content generator
generator = ContentGenerator(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    max_concurrent_requests=5,
    use_streaming=True
)

# Generate content
articles = generator.generate_batch([
    "The Future of Artificial Intelligence",
    "Climate Change Solutions in 2024",
    "Advances in Quantum Computing"
], 
style="informative",
word_count=500)

# Process the generated content
for topic, article in zip(topics, articles):
    print(f"Article on '{topic}':\n{article}\n")
```

## Usage Example: Chat Assistant

```python
from aws_bedrock_inference.examples.chat_assistant.app import ChatAssistant

# Create a chat assistant
assistant = ChatAssistant(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    system_prompt="You are a helpful customer support agent for a cloud computing company."
)

# Start a conversation
conversation_id = assistant.create_conversation()

# Handle conversation turns
while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        break
        
    response = assistant.send_message(conversation_id, user_input)
    print(f"Assistant: {response}")
```

## Usage Example: Batch Processor

```python
from aws_bedrock_inference.examples.batch_processor.app import BatchProcessor

# Create a batch processor
processor = BatchProcessor(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    max_concurrent_jobs=10,
    output_dir="./results"
)

# Process a large batch of files
results = processor.process_directory(
    input_dir="./documents",
    task="summarize",
    max_words=100
)

# Print the results summary
print(f"Processed {len(results)} documents")
print(f"Average processing time: {processor.get_average_processing_time():.2f} seconds")
print(f"Total tokens used: {processor.get_total_tokens_used()}")
```

## Implementation Status

- [  ] Content Generator - Not started
- [  ] Chat Assistant - Not started
- [  ] Data Extraction Service - Not started
- [  ] Batch Processor - Not started
- [  ] Quota Manager - Not started

## Next Steps

1. Implement basic content generation application
2. Add conversation assistant with memory management
3. Create structured data extraction pipeline
4. Develop batch processing for large document sets
5. Add quota management and monitoring utilities

## Contributing

See the project [CONTRIBUTING](../../CONTRIBUTING.md) guidelines for information on how to contribute to this module.