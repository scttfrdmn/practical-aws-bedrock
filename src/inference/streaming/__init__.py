"""
Streaming inference methods for AWS Bedrock.

This package contains implementations of streaming inference methods.
"""

from .basic_client import BedrockStreamingClient

__all__ = [
    'BedrockStreamingClient',
]