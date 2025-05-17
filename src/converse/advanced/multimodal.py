"""
Multimodal client for AWS Bedrock Converse API.

This module provides a client for multimodal conversations with AWS Bedrock models
using the Converse API. It enables conversations that include both text and images.
"""

import json
import time
import logging
import uuid
import base64
from typing import Dict, Any, Optional, List, Union, Tuple, BinaryIO

import boto3
from botocore.exceptions import ClientError

from ..basic.simple_chat import ConverseClient


class MultimodalConverseClient(ConverseClient):
    """
    A client for multimodal conversations with AWS Bedrock models.
    
    This client extends the basic ConverseClient to provide support for
    conversations that include both text and images.
    """
    
    def __init__(
        self, 
        model_id: str,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the multimodal Converse client.
        
        Args:
            model_id: The Bedrock model identifier
            profile_name: AWS profile name (defaults to value from get_profile())
            region_name: AWS region name (defaults to value from get_region())
            max_retries: Maximum number of retry attempts for recoverable errors
            base_backoff: Base backoff time (in seconds) for exponential backoff
            logger: Optional logger instance
        """
        super().__init__(
            model_id=model_id,
            profile_name=profile_name,
            region_name=region_name,
            max_retries=max_retries,
            base_backoff=base_backoff,
            logger=logger
        )
        
        # Validate that the model supports multimodal input
        self._validate_multimodal_support()
    
    def _validate_multimodal_support(self) -> None:
        """
        Validate that the selected model supports multimodal input.
        
        Raises:
            ValueError: If the model doesn't support multimodal input
        """
        # Extract model family and version from model ID
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        model_name = self.model_id.split('.')[1].lower() if '.' in self.model_id and len(self.model_id.split('.')) > 1 else ""
        
        # Check if the model supports multimodal input
        supported = False
        
        if "anthropic" in model_family and "claude-3" in model_name:
            # Claude 3 models support multimodal
            supported = "haiku" not in model_name  # haiku doesn't support vision
        elif "amazon" in model_family and "titan-multimodal" in model_name:
            # Amazon Titan Multimodal models
            supported = True
        elif "stability" in model_family:
            # Stability models (though these are typically for image generation)
            supported = True
        
        if not supported:
            self.logger.warning(
                f"Model {self.model_id} may not support multimodal input. "
                f"For best results, use Claude 3 Opus/Sonnet or Titan Multimodal."
            )
    
    def send_message_with_image(
        self, 
        conversation_id: str, 
        message: str,
        image_bytes: Union[bytes, BinaryIO],
        image_format: str = "jpeg",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        other_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a message with an image in a conversation.
        
        Args:
            conversation_id: The conversation ID
            message: The user message
            image_bytes: Binary image data as bytes or file-like object
            image_format: Format of the image ('jpeg', 'png', 'gif', 'webp')
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            other_params: Additional model-specific parameters
            
        Returns:
            The model's response text
            
        Raises:
            ValueError: For invalid input parameters
            KeyError: When conversation_id doesn't exist
            RuntimeError: For unrecoverable errors after retries
        """
        # Validate inputs
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation {conversation_id} not found")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        
        # Validate image format
        supported_formats = ["jpeg", "jpg", "png", "gif", "webp"]
        if image_format.lower() not in supported_formats:
            raise ValueError(f"Unsupported image format. Must be one of: {supported_formats}")
        
        # Prepare image data
        if hasattr(image_bytes, 'read'):
            # If it's a file-like object, read it
            image_data = image_bytes.read()
        else:
            # Otherwise assume it's already bytes
            image_data = image_bytes
        
        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Add multimodal message to conversation history
        self._add_multimodal_message(
            conversation_id=conversation_id,
            text=message,
            base64_image=base64_image,
            image_format=image_format
        )
        
        # Create request body
        request_body = self._create_multimodal_request(
            conversation_id=conversation_id,
            max_tokens=max_tokens,
            temperature=temperature,
            other_params=other_params or {}
        )
        
        # Make the request with retries
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Invoking model {self.model_id} with multimodal content (attempt {attempt + 1})")
                
                start_time = time.time()
                
                response = self.client.converse(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                elapsed_time = time.time() - start_time
                
                # Parse the response
                result = self._parse_converse_response(response)
                
                # Update metrics
                self.request_count += 1
                self._update_token_counts(result)
                
                self.logger.debug(
                    f"Multimodal converse invocation successful in {elapsed_time:.2f}s. "
                    f"Input tokens: {result.get('input_tokens', 'unknown')}, "
                    f"Output tokens: {result.get('output_tokens', 'unknown')}"
                )
                
                # Add assistant response to conversation history
                self.conversations[conversation_id]["messages"].append({
                    "role": "assistant",
                    "content": result["output"]
                })
                self.conversations[conversation_id]["updated_at"] = time.time()
                
                return result["output"]
                
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_message = e.response["Error"]["Message"]
                
                self.logger.warning(
                    f"Error invoking model (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{error_code} - {error_message}"
                )
                
                # Track error
                self.error_count += 1
                
                # Check if the error is recoverable
                if error_code in ["ThrottlingException", "ServiceUnavailableException", "InternalServerException"]:
                    if attempt < self.max_retries:
                        # Calculate backoff time with exponential backoff and jitter
                        backoff_time = self._calculate_backoff(attempt)
                        self.logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                        time.sleep(backoff_time)
                        continue
                
                # If we've exhausted retries or the error is not recoverable, raise
                raise RuntimeError(f"Failed to invoke model after {attempt + 1} attempts: {error_code} - {error_message}")
    
    def _add_multimodal_message(
        self, 
        conversation_id: str,
        text: str,
        base64_image: str,
        image_format: str
    ) -> None:
        """
        Add a multimodal message to the conversation history.
        
        Args:
            conversation_id: The conversation ID
            text: The text content of the message
            base64_image: Base64-encoded image data
            image_format: Format of the image
        """
        # Extract model family from model ID
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        # Create media type
        media_type = f"image/{image_format}"
        
        if "anthropic" in model_family:
            # Claude format for multimodal messages
            content = [
                {"type": "text", "text": text},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image
                    }
                }
            ]
            
            # Add to conversation
            self.conversations[conversation_id]["messages"].append({
                "role": "user",
                "content": content
            })
        else:
            # Generic format for other models
            # Note: This is a simplification; actual format may vary by model
            content = {
                "text": text,
                "image": {
                    "format": image_format,
                    "data": base64_image
                }
            }
            
            # Add to conversation
            self.conversations[conversation_id]["messages"].append({
                "role": "user",
                "content": content
            })
        
        self.conversations[conversation_id]["updated_at"] = time.time()
    
    def _create_multimodal_request(
        self, 
        conversation_id: str,
        max_tokens: int,
        temperature: float,
        other_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a request body for the Converse API with multimodal content.
        
        Args:
            conversation_id: The conversation ID
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            other_params: Additional model-specific parameters
            
        Returns:
            Request body for the Converse API
        """
        # Get conversation history
        messages = self.conversations[conversation_id]["messages"]
        
        # Create base request body
        body = {
            "messages": messages,
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add any additional parameters
        body.update(other_params)
        
        return body


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a multimodal client for Claude
    client = MultimodalConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Create a conversation with a system prompt
    conversation_id = client.create_conversation(
        system_prompt="You are a helpful assistant that can analyze images and provide detailed descriptions."
    )
    
    try:
        # Load a sample image
        with open("sample_image.jpg", "rb") as f:
            image_bytes = f.read()
        
        # Send a message with the image
        print("\nUser: What can you see in this image?")
        
        response = client.send_message_with_image(
            conversation_id=conversation_id,
            message="What can you see in this image?",
            image_bytes=image_bytes,
            image_format="jpeg",
            max_tokens=500,
            temperature=0.7
        )
        
        print("\nAssistant response:")
        print(response)
        
        # Send a follow-up question about the same image
        follow_up = client.send_message(
            conversation_id=conversation_id,
            message="What colors are most prominent in the image?",
            max_tokens=300,
            temperature=0.7
        )
        
        print("\nFollow-up question: What colors are most prominent in the image?")
        print("\nAssistant response:")
        print(follow_up)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get conversation history
    history = client.get_conversation_history(conversation_id)
    print(f"\nConversation has {len(history['messages'])} messages")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")