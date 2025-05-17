"""
Streaming client for AWS Bedrock Converse API.

This module provides a client for streaming responses from AWS Bedrock models
using the Converse API. It enables real-time, token-by-token responses for
conversational applications.
"""

import json
import time
import logging
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple, Iterator, Callable

import boto3
from botocore.exceptions import ClientError

from ..basic.simple_chat import ConverseClient


class StreamingConverseClient(ConverseClient):
    """
    A client for streaming responses from AWS Bedrock models using the Converse API.
    
    This client extends the basic ConverseClient to provide streaming response
    capabilities, delivering content in real-time as it's generated.
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
        Initialize the streaming Converse client.
        
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
    
    def send_message_streaming(
        self, 
        conversation_id: str, 
        message: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        other_params: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """
        Send a message in a conversation and stream the model's response.
        
        Args:
            conversation_id: The conversation ID
            message: The user message
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            other_params: Additional model-specific parameters
            
        Returns:
            An iterator that yields response content chunks
            
        Raises:
            ValueError: For invalid input parameters
            KeyError: When conversation_id doesn't exist
            RuntimeError: For unrecoverable errors after retries
        """
        # Validate inputs
        if not message:
            raise ValueError("Message cannot be empty")
        
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation {conversation_id} not found")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        
        # Add user message to conversation history
        self.conversations[conversation_id]["messages"].append({
            "role": "user",
            "content": message
        })
        self.conversations[conversation_id]["updated_at"] = time.time()
        
        # Create request body
        request_body = self._create_converse_request(
            conversation_id=conversation_id,
            max_tokens=max_tokens,
            temperature=temperature,
            other_params=other_params or {}
        )
        
        # Make the request with retries
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Invoking model {self.model_id} for streaming (attempt {attempt + 1})")
                
                start_time = time.time()
                
                # Use converse_stream API for streaming responses
                response = self.client.converse_stream(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                # Buffer for accumulating the complete response
                complete_response = ""
                chunks_received = 0
                total_bytes = 0
                
                # Process the streaming response
                for event in response.get('stream'):
                    chunk = event.get('chunk')
                    if chunk:
                        if 'bytes' in chunk:
                            # Parse the bytes into a JSON object
                            chunk_data = json.loads(chunk['bytes'].read().decode('utf-8'))
                            
                            # Extract text content based on model family
                            text_chunk = self._extract_streaming_text(chunk_data)
                            
                            if text_chunk:
                                complete_response += text_chunk
                                chunks_received += 1
                                total_bytes += len(text_chunk.encode('utf-8'))
                                
                                # Yield each text chunk
                                yield text_chunk
                
                elapsed_time = time.time() - start_time
                
                # Update metrics
                self.request_count += 1
                # Rough token estimation for metrics
                token_estimate = len(complete_response.split()) * 4 / 3  # ~4/3 tokens per word
                self.token_count += int(token_estimate)
                
                self.logger.debug(
                    f"Streaming complete in {elapsed_time:.2f}s. "
                    f"Received {chunks_received} chunks, {total_bytes} bytes."
                )
                
                # Add the complete response to conversation history
                self.conversations[conversation_id]["messages"].append({
                    "role": "assistant",
                    "content": complete_response
                })
                self.conversations[conversation_id]["updated_at"] = time.time()
                
                # Exit the retry loop after successful completion
                break
                
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_message = e.response["Error"]["Message"]
                
                self.logger.warning(
                    f"Error invoking streaming model (attempt {attempt + 1}/{self.max_retries + 1}): "
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
                raise RuntimeError(f"Failed to invoke streaming model after {attempt + 1} attempts: {error_code} - {error_message}")
    
    def send_message_streaming_with_callbacks(
        self,
        conversation_id: str,
        message: str,
        on_content: Callable[[str], None],
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        other_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a message in a conversation and handle streaming response with callbacks.
        
        Args:
            conversation_id: The conversation ID
            message: The user message
            on_content: Callback function for each content chunk
            on_complete: Optional callback for completion with full response
            on_error: Optional callback for error handling
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            other_params: Additional model-specific parameters
        """
        try:
            # Initialize complete response
            complete_response = ""
            
            # Stream the response
            for chunk in self.send_message_streaming(
                conversation_id=conversation_id,
                message=message,
                max_tokens=max_tokens,
                temperature=temperature,
                other_params=other_params
            ):
                # Accumulate the complete response
                complete_response += chunk
                
                # Call the content callback
                on_content(chunk)
            
            # Call the completion callback if provided
            if on_complete:
                on_complete(complete_response)
                
        except Exception as e:
            # Handle exceptions
            if on_error:
                on_error(e)
            else:
                # Re-raise if no error handler
                raise
    
    def _extract_streaming_text(self, chunk_data: Dict[str, Any]) -> str:
        """
        Extract text content from a streaming chunk based on model family.
        
        Different model families return slightly different JSON structures
        in their streaming responses.
        
        Args:
            chunk_data: The parsed JSON data from a streaming chunk
            
        Returns:
            The extracted text content, or empty string if not found
        """
        # Extract model family from model ID to handle model-specific responses
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        # Extract text content based on model family
        if "anthropic" in model_family:
            # Claude models
            if 'message' in chunk_data and 'content' in chunk_data['message']:
                content_list = chunk_data['message']['content']
                if content_list and len(content_list) > 0 and 'text' in content_list[0]:
                    return content_list[0]['text']
            
            # Alternative format for streaming chunks
            if 'delta' in chunk_data and 'text' in chunk_data['delta']:
                return chunk_data['delta']['text']
                
            # Another alternative format
            if 'contentBlock' in chunk_data and 'text' in chunk_data['contentBlock']:
                return chunk_data['contentBlock']['text']
                
        else:
            # Generic extraction for other models
            # Try common patterns in streaming responses
            if 'delta' in chunk_data and 'text' in chunk_data['delta']:
                return chunk_data['delta']['text']
                
            if 'content' in chunk_data:
                if isinstance(chunk_data['content'], str):
                    return chunk_data['content']
                elif isinstance(chunk_data['content'], list) and len(chunk_data['content']) > 0:
                    if isinstance(chunk_data['content'][0], dict) and 'text' in chunk_data['content'][0]:
                        return chunk_data['content'][0]['text']
            
            if 'text' in chunk_data:
                return chunk_data['text']
                
            if 'completion' in chunk_data:
                return chunk_data['completion']
        
        # If no matching pattern, return empty string
        return ""


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a streaming client for Claude
    client = StreamingConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Create a conversation with a system prompt
    conversation_id = client.create_conversation(
        system_prompt="You are a helpful assistant specializing in AWS services."
    )
    
    try:
        # Example 1: Using the iterator interface
        print("\n--- Example 1: Iterator Interface ---")
        print("User: Explain AWS Bedrock in 3 sentences.")
        
        print("Assistant: ", end="", flush=True)
        for chunk in client.send_message_streaming(
            conversation_id=conversation_id,
            message="Explain AWS Bedrock in 3 sentences.",
            max_tokens=200,
            temperature=0.7
        ):
            print(chunk, end="", flush=True)
        print("\n")
        
        # Example 2: Using the callback interface
        print("\n--- Example 2: Callback Interface ---")
        print("User: What foundation models are available in AWS Bedrock?")
        
        print("Assistant: ", end="", flush=True)
        
        def on_content(chunk):
            print(chunk, end="", flush=True)
            
        def on_complete(complete_response):
            print("\n\n[Streaming complete, received", len(complete_response), "characters in total]")
            
        def on_error(error):
            print(f"\nError: {str(error)}")
        
        client.send_message_streaming_with_callbacks(
            conversation_id=conversation_id,
            message="What foundation models are available in AWS Bedrock?",
            on_content=on_content,
            on_complete=on_complete,
            on_error=on_error,
            max_tokens=300,
            temperature=0.7
        )
        
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