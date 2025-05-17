# AWS Bedrock Inference Learning Path

This document provides a structured learning path to guide you through the AWS Bedrock inference optimization tutorials and documentation.

## Learning Path Overview

This project is organized in a progressive manner, from basic concepts to advanced optimization techniques. Follow this recommended path to build your knowledge systematically:

### 1. Getting Started

Start here to understand the fundamentals:

- [Introduction to AWS Bedrock Inference](/docs/chapters/getting-started/introduction.html) - Basic concepts and overview
- [AWS CLI Guide for Bedrock](/tutorials/basic/aws-cli-guide.html) - Basic interaction with Bedrock APIs
- [Foundation Models in AWS Bedrock](/docs/chapters/getting-started/foundation-models.html) - Understanding available models

### 2. Core Inference Methods

Learn about the different methods for model invocation:

- [Synchronous Inference (InvokeModel)](/docs/chapters/core-methods/synchronous.html)
- [Streaming Inference](/docs/chapters/core-methods/streaming.html)
- [Asynchronous Processing](/docs/chapters/core-methods/asynchronous.html)
- [Comparing Inference Methods](/docs/inference-methods-comparison.html)

### 3. Working with Quotas

Understand and optimize within quota constraints:

- [Understanding AWS Bedrock Quotas](/docs/quota-management.html)
- [Discovering and Managing Quotas](/tutorials/intermediate/quota-discovery.html)
- [Error Handling and Retry Strategies](/tutorials/intermediate/error-handling.html)

### 4. Optimizing Prompts

Learn how to structure and optimize prompts:

- [Prompt Engineering Across Models](/docs/prompt-engineering.html)
- [Optimizing Throughput with Prompt Engineering](/tutorials/advanced/prompt-optimization-throughput.html)

### 5. Advanced APIs and Features

Explore more advanced Bedrock capabilities:

- [Conversational AI with Converse API](/docs/chapters/apis/converse.html)
- [Structured Outputs with Construct API](/docs/construct-api-guide.html)
- [Model Fine-tuning](/docs/model-fine-tuning.html)

### 6. Putting It All Together

Comprehensive examples that combine multiple techniques:

- [Building a High-Throughput Processing Pipeline](/docs/high-throughput-pipeline.html)
- [Multi-Model Inference Orchestration](/docs/multi-model-orchestration.html)
- [Production Deployment Patterns](/docs/production-deployment-patterns.html)

## Quick Reference

### By Topic

- **Quota Management**: [Understanding Quotas](/docs/quota-management.html), [Quota Discovery](/tutorials/intermediate/quota-discovery.html)
- **Prompt Engineering**: [Prompt Engineering Guide](/docs/prompt-engineering.html), [Throughput Optimization](/tutorials/advanced/prompt-optimization-throughput.html)
- **Model Selection**: [Foundation Models Guide](/docs/chapters/getting-started/foundation-models.html)
- **API Usage**: [AWS CLI Guide](/tutorials/basic/aws-cli-guide.html)

### By Difficulty Level

- **Beginner**: [Introduction](/docs/chapters/getting-started/introduction.html), [AWS CLI Guide](/tutorials/basic/aws-cli-guide.html)
- **Intermediate**: [Quota Discovery](/tutorials/intermediate/quota-discovery.html)
- **Advanced**: [Prompt Optimization](/tutorials/advanced/prompt-optimization-throughput.html)

## Working Through the Tutorials

1. **Set up your environment** - Follow the README.md instructions to set up your Python environment and AWS credentials
2. **Start with the basics** - Make sure you understand the fundamental concepts before moving to advanced topics
3. **Execute example code** - Run the provided code samples to see concepts in action
4. **Experiment and adapt** - Modify examples to work with your specific use cases
5. **Optimize iteratively** - Apply optimization techniques incrementally, measuring improvements at each step