# Claude Configuration and Settings

## AWS CLI Profile

For all AWS CLI commands and Boto3 client sessions, this project uses the following profile configuration:

- **Local testing**: Always use the `aws` profile
- **Production**: Use the profile specified in environment variables or configuration

```python
# Example of how to use the correct profile in code
import boto3

def get_bedrock_client(profile_name="aws"):
    """
    Create a Bedrock client using the specified AWS profile.
    For local testing, this defaults to the 'aws' profile.
    """
    session = boto3.Session(profile_name=profile_name)
    return session.client('bedrock-runtime')
```

## Environment Variables

The project recognizes the following environment variables:

- `AWS_PROFILE`: Override the default profile name
- `AWS_REGION`: Set the AWS region to use for Bedrock services
- `BEDROCK_QUOTA_STRATEGY`: Set the quota management strategy (e.g., "aggressive", "conservative")

## AWS CLI Conventions

When showing AWS CLI examples throughout the documentation, commands should:

1. Always include the profile selection (`--profile aws`)
2. Use the proper region for Bedrock availability
3. Format JSON with double quotes (not single)
4. Include proper error handling examples

Example:
```bash
aws bedrock list-foundation-models --profile aws --region us-west-2
```

## Visualization Standards

All data visualizations in the project should:

1. Be generated in SVG format for optimal clarity and scalability
2. Use a consistent color palette from the `utils/visualization_config.py` file
3. Include proper titles, axis labels, and legends
4. Be accessible (colorblind-friendly)

## Foundation Model Coverage

This project should include examples for all major foundation model families available in AWS Bedrock:

- Anthropic Claude models
- Meta Llama 2 models
- Amazon Titan models
- Stability AI models
- Cohere models
- AI21 Labs models

## Modality Support

Examples should cover all supported modalities:

- Text generation
- Image generation
- Multimodal (text+image)
- Document processing
- Audio processing (if/when available)

## Code Implementation Standards

For all code implementations in this project:

1. **Error Handling**: Implement exponential backoff with jitter for all API calls
2. **Quota Awareness**: All clients should monitor usage and adapt to quota limits
3. **Metrics Collection**: Track and log key performance metrics (latency, token usage, etc.)
4. **Documentation**: Include detailed docstrings for all functions and classes
5. **Type Hints**: Use Python type annotations for better code quality

## File Organization

Follow these naming conventions for all source files:

1. Use snake_case for filenames (e.g., `synchronous_client.py`)
2. Group related functionality in appropriately named modules
3. Place shared utilities in the `utils` directory
4. Keep examples separate from core implementation

## Testing Requirements

All implementations should include:

1. Unit tests for core functionality
2. Integration tests that validate AWS service interaction
3. Performance benchmarking tests
4. Error condition tests

## Development Workflow

When adding new features or examples:

1. First implement core functionality with proper error handling
2. Add comprehensive documentation with usage examples
3. Create benchmark tests to measure performance
4. Add tutorial content explaining when and how to use the feature
5. Verify information is current by checking AWS documentation and announcements

## Documentation Freshness

To maintain high-quality, accurate documentation:

1. Regularly check AWS Bedrock announcements and release notes for updates
2. Verify code examples against the latest AWS SDK version
3. Update model lists and capabilities when new foundation models are added
4. Include "Last Updated" dates in documentation headers
5. Review and update documentation at least quarterly to ensure accuracy
6. Reference AWS official documentation when appropriate, including links to source material