"""
AWS Bedrock inference implementation package.

This package provides robust, quota-aware implementations of all AWS Bedrock
inference methods, including synchronous, streaming, and asynchronous APIs.
"""

# Import key classes to make them available at the package level
from .synchronous.basic_client import BedrockClient
from .synchronous.quota_aware import QuotaAwareBedrockClient, QuotaExceededException
from .streaming.basic_client import BedrockStreamingClient
from .asynchronous.job_client import BedrockJobClient
from .asynchronous.batch_processor import BedrockBatchProcessor

__all__ = [
    'BedrockClient',
    'QuotaAwareBedrockClient',
    'QuotaExceededException',
    'BedrockStreamingClient',
    'BedrockJobClient',
    'BedrockBatchProcessor',
]