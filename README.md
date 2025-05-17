# Practical AWS Bedrock

A comprehensive, action-oriented guide to building generative AI applications with AWS Bedrock.

## About This Project

Practical AWS Bedrock provides hands-on guidance for developers looking to build production-ready generative AI applications using Amazon Bedrock. This resource focuses on real-world implementation patterns rather than theoretical concepts, with a conversational approach that makes complex topics accessible without oversimplification.

## What You'll Find Here

- **Practical Code Examples**: Production-ready code with proper error handling, quota management, and optimization
- **Step-by-Step Tutorials**: Guided implementations for common use cases
- **Comprehensive Documentation**: Detailed guides for all AWS Bedrock features
- **Performance Optimization**: Techniques for maximizing throughput, minimizing costs, and scaling effectively

## Inference Methods

AWS Bedrock supports multiple inference methods, each with different characteristics:

### Synchronous Inference (InvokeModel)
- Single request/response pattern
- Ideal for real-time applications requiring immediate responses
- Subject to tokens-per-minute (TPM) and requests-per-minute (RPM) quotas

### Streaming Inference (InvokeModelWithResponseStream)
- Returns generated content in chunks as it's created
- Reduces time-to-first-token for improved user experience
- Allows for faster apparent response times while staying within TPM limits

### Batch Processing
- Process multiple requests efficiently
- Better utilization of quota limits for non-realtime applications
- Strategies for batching requests to maximize throughput

### Asynchronous Processing (CreateModelInvocationJob)
- Long-running inference jobs for large inputs
- Higher throughput potential for offline processing
- Working with job queues and monitoring

### Conversational AI (Converse API)
- Handles multi-turn conversations with context management
- Specialized for chat applications and dialogue systems
- Built-in conversation state tracking and memory management
- Optimizing conversation history within token limits

### Structured Outputs (Construct API)
- Specialized for generating structured data (JSON, XML)
- Enforces output schemas for consistent results
- Strategies for reliable parsing and validation
- Optimizing quota usage with structured generation

## Getting Started

Choose a learning path based on your needs:

1. [Introduction to AWS Bedrock](/docs/01-introduction.md)
2. [Foundation Models Guide](/docs/foundation-models.md)
3. [Core Inference Methods](/docs/inference-methods/)
4. [Token Quota Optimization Tutorial](/tutorials/advanced/token-quota-optimization.md)

For a structured approach to learning, follow the [complete learning path](/LEARNING_PATH.md).

## Prerequisites

- AWS Account with Amazon Bedrock access
- Python 3.8+
- Basic understanding of generative AI concepts
- AWS CLI configured with appropriate credentials

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/practical-aws-bedrock.git
cd practical-aws-bedrock

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
practical-aws-bedrock/
├── benchmarks/       # Performance testing and visualization
├── docs/             # Comprehensive documentation for blog posts
├── src/              # Core implementation modules
│   ├── inference/    # Basic inference implementations
│   ├── converse/     # Converse API implementations
│   ├── construct/    # Construct API implementations
│   └── examples/     # Complete example applications
├── tutorials/        # Step-by-step learning materials
│   ├── basic/        # Getting started tutorials
│   ├── intermediate/ # More advanced concepts
│   └── advanced/     # Complex optimization strategies
└── utils/            # Helper functions and tools
```

## Examples

The repository includes examples for different inference scenarios:

- High-volume batch processing
- Real-time chat applications with streaming
- Multi-turn conversations with Converse API
- Structured data generation with Construct API
- Hybrid approaches for mixed workloads
- Handling quota limits gracefully

Each example includes detailed code comments, performance benchmarks, and optimization techniques.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AWS for the Amazon Bedrock service
- The open source community for inspiration and shared knowledge

---

© 2025 Scott Friedman. All rights reserved.

*This resource is not affiliated with or endorsed by Amazon Web Services.*