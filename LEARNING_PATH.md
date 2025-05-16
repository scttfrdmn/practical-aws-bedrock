# AWS Bedrock Inference Learning Path

This document provides a structured learning path to guide you through the AWS Bedrock inference optimization tutorials and documentation.

## Learning Path Overview

This project is organized in a progressive manner, from basic concepts to advanced optimization techniques. Follow this recommended path to build your knowledge systematically:

### 1. Getting Started

Start here to understand the fundamentals:

- [Introduction to AWS Bedrock Inference](/docs/01-introduction.md) - Basic concepts and overview
- [AWS CLI Guide for Bedrock](/tutorials/basic/aws-cli-guide.md) - Basic interaction with Bedrock APIs
- [Foundation Models in AWS Bedrock](/docs/foundation-models.md) - Understanding available models

### 2. Core Inference Methods

Learn about the different methods for model invocation:

- [Synchronous Inference (InvokeModel)](/docs/inference-methods.md) (To be added)
- [Streaming Inference](/tutorials/basic/streaming-inference.md) (To be added)
- [Asynchronous Processing](/tutorials/intermediate/async-inference.md) (To be added)
- [Comparing Inference Methods](/tutorials/intermediate/inference-methods-comparison.md) (To be added)

### 3. Working with Quotas

Understand and optimize within quota constraints:

- [Understanding AWS Bedrock Quotas](/docs/quota-management.md)
- [Discovering and Managing Quotas](/tutorials/intermediate/quota-discovery.md)
- [Error Handling and Retry Strategies](/tutorials/intermediate/error-handling.md) (To be added)

### 4. Optimizing Prompts

Learn how to structure and optimize prompts:

- [Prompt Engineering Across Models](/docs/prompt-engineering.md)
- [Optimizing Throughput with Prompt Engineering](/tutorials/advanced/prompt-optimization-throughput.md)

### 5. Advanced APIs and Features

Explore more advanced Bedrock capabilities:

- [Conversational AI with Converse API](/tutorials/intermediate/converse-api.md) (To be added)
- [Structured Outputs with Construct API](/tutorials/intermediate/construct-api.md) (To be added)
- [Model Fine-tuning](/tutorials/advanced/model-fine-tuning.md) (To be added)

### 6. Putting It All Together

Comprehensive examples that combine multiple techniques:

- [Building a High-Throughput Processing Pipeline](/tutorials/advanced/high-throughput-pipeline.md) (To be added)
- [Multi-Model Inference Orchestration](/tutorials/advanced/multi-model-orchestration.md) (To be added)
- [Production Deployment Patterns](/tutorials/advanced/production-deployment.md) (To be added)

## Quick Reference

### By Topic

- **Quota Management**: [Understanding Quotas](/docs/quota-management.md), [Quota Discovery](/tutorials/intermediate/quota-discovery.md)
- **Prompt Engineering**: [Prompt Engineering Guide](/docs/prompt-engineering.md), [Throughput Optimization](/tutorials/advanced/prompt-optimization-throughput.md)
- **Model Selection**: [Foundation Models Guide](/docs/foundation-models.md)
- **API Usage**: [AWS CLI Guide](/tutorials/basic/aws-cli-guide.md)

### By Difficulty Level

- **Beginner**: [Introduction](/docs/01-introduction.md), [AWS CLI Guide](/tutorials/basic/aws-cli-guide.md)
- **Intermediate**: [Quota Discovery](/tutorials/intermediate/quota-discovery.md)
- **Advanced**: [Prompt Optimization](/tutorials/advanced/prompt-optimization-throughput.md)

## Working Through the Tutorials

1. **Set up your environment** - Follow the README.md instructions to set up your Python environment and AWS credentials
2. **Start with the basics** - Make sure you understand the fundamental concepts before moving to advanced topics
3. **Execute example code** - Run the provided code samples to see concepts in action
4. **Experiment and adapt** - Modify examples to work with your specific use cases
5. **Optimize iteratively** - Apply optimization techniques incrementally, measuring improvements at each step