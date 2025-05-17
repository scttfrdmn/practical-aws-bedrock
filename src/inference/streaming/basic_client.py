"""
Basic streaming client for AWS Bedrock inference.

This module provides a simple, reliable client for streaming responses from
AWS Bedrock models, enabling real-time content generation.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple, Iterator, Callable

import boto3
from botocore.exceptions import ClientError

from utils.profile_manager import get_profile, get_region


class BedrockStreamingClient:
    """
    A client for streaming responses from AWS Bedrock models.
    
    This client follows the AWS profile conventions specified in CLAUDE.md and
    provides functionality for consuming streaming responses for improved
    user experience and reduced time-to-first-token.
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
        Initialize the Bedrock streaming client.
        
        Args:
            model_id: The Bedrock model identifier
            profile_name: AWS profile name (defaults to value from get_profile())
            region_name: AWS region name (defaults to value from get_region())
            max_retries: Maximum number of retry attempts for recoverable errors
            base_backoff: Base backoff time (in seconds) for exponential backoff
            logger: Optional logger instance
        """
        self.model_id = model_id
        self.profile_name = profile_name or get_profile()
        self.region_name = region_name or get_region()
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Create AWS session with profile
        self.session = boto3.Session(
            profile_name=self.profile_name, 
            region_name=self.region_name
        )
        
        # Create Bedrock runtime client
        self.client = self.session.client('bedrock-runtime')
        
        # Track metrics
        self.request_count = 0
        self.token_count = 0
        self.error_count = 0
    
    def invoke_stream(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7, 
        system_prompt: Optional[str] = None,
        other_params: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """
        Invoke the model with a prompt and yield streaming responses.
        
        Args:
            prompt: The user prompt or instruction
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            system_prompt: Optional system prompt for models that support it
            other_params: Additional model-specific parameters
            
        Yields:
            Text chunks from the streaming response
            
        Raises:
            ValueError: For invalid input parameters
            RuntimeError: For unrecoverable errors after retries
        """
        # Validate inputs
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        
        # Create model-specific request body
        request_body = self._create_request_body(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            other_params=other_params or {}
        )
        
        # Make the streaming request with retries
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Invoking streaming model {self.model_id} (attempt {attempt + 1})")
                
                start_time = time.time()
                
                stream_response = self.client.invoke_model_with_response_stream(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                # Process the streaming response
                stream = stream_response.get('body')
                content_buffer = []
                token_count = 0
                
                # Track metrics
                self.request_count += 1
                
                # Process each chunk
                for event in stream:
                    # Check for completion events
                    if 'internalServerException' in event:
                        error = event['internalServerException']
                        raise RuntimeError(f"Server error: {error.get('message', 'Unknown error')}")
                    
                    if 'modelStreamErrorException' in event:
                        error = event['modelStreamErrorException']
                        raise RuntimeError(f"Model stream error: {error.get('message', 'Unknown error')}")
                    
                    # Process chunk response based on model
                    chunk = self._process_stream_chunk(event)
                    
                    if chunk:
                        content_buffer.append(chunk)
                        token_count += self._estimate_tokens_in_text(chunk)
                        yield chunk
                
                # Update total token count
                self.token_count += token_count
                
                # Successful completion, break the retry loop
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
    
    def _create_request_body(
        self, 
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        other_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create the appropriate request body for the model family.
        
        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            system_prompt: Optional system prompt
            other_params: Additional model-specific parameters
            
        Returns:
            Model-specific request body
        """
        # Extract model family from model ID
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        if "anthropic" in model_family:
            # Claude models
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
        elif "meta" in model_family or "llama" in model_family:
            # Llama models
            if system_prompt:
                formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            else:
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            body = {
                "prompt": formatted_prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature
            }
            
        elif "cohere" in model_family:
            # Cohere models
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }
            
            if system_prompt:
                body["preamble"] = system_prompt
                
        elif "ai21" in model_family:
            # AI21 models
            body = {
                "prompt": prompt,
                "maxTokens": max_tokens,
                "temperature": temperature
            }
            
        else:
            # Default to Amazon Titan format
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            }
        
        # Add any additional parameters
        body.update(other_params)
        
        return body
    
    def _process_stream_chunk(self, event: Dict[str, Any]) -> Optional[str]:
        """
        Process a chunk from the streaming response.
        
        Args:
            event: Event from the streaming response
            
        Returns:
            Text chunk if available, None otherwise
        """
        # Handle common event formats
        if 'chunk' not in event:
            return None
        
        chunk_data = event['chunk']
        
        # Read bytes data
        if 'bytes' in chunk_data:
            # Parse the bytes data
            chunk_bytes = chunk_data['bytes']
            chunk_json = json.loads(chunk_bytes)
            
            # Extract model family from model ID
            model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
            
            # Extract text based on model family
            if "anthropic" in model_family:
                # Claude responses
                if 'delta' in chunk_json and 'text' in chunk_json['delta']:
                    return chunk_json['delta']['text']
                    
            elif "meta" in model_family or "llama" in model_family:
                # Llama responses
                if 'generation' in chunk_json:
                    return chunk_json['generation']
                    
            elif "cohere" in model_family:
                # Cohere responses
                if 'text' in chunk_json:
                    return chunk_json['text']
                    
            elif "ai21" in model_family:
                # AI21 responses
                if 'completions' in chunk_json and len(chunk_json['completions']) > 0:
                    return chunk_json['completions'][0].get('data', {}).get('text', '')
                    
            else:
                # Default extraction (Amazon Titan)
                if 'outputText' in chunk_json:
                    return chunk_json['outputText']
        
        return None
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff time using exponential backoff with jitter.
        
        Args:
            attempt: The current attempt number (0-indexed)
            
        Returns:
            Backoff time in seconds
        """
        import random
        
        # Calculate exponential backoff: base * 2^attempt
        backoff = self.base_backoff * (2 ** attempt)
        
        # Add jitter (±20%)
        jitter = backoff * 0.2
        backoff = backoff + random.uniform(-jitter, jitter)
        
        return backoff
    
    def _estimate_tokens_in_text(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        This is a rough approximation for tracking purposes.
        
        Args:
            text: The text to estimate
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
            
        # Very simple approximation (not model-specific)
        return len(text.split())
    
    def get_metrics(self) -> Dict[str, int]:
        """
        Get usage metrics for this client instance.
        
        Returns:
            Dictionary with usage metrics
        """
        return {
            "request_count": self.request_count,
            "token_count": self.token_count,
            "error_count": self.error_count
        }
    
    def invoke_stream_with_callbacks(
        self,
        prompt: str,
        on_content: Callable[[str], None],
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        other_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Invoke streaming with callback functions for content handling.
        
        Args:
            prompt: The user prompt or instruction
            on_content: Callback function for each content chunk
            on_complete: Optional callback for when streaming completes
            on_error: Optional callback for error handling
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            system_prompt: Optional system prompt for models that support it
            other_params: Additional model-specific parameters
        """
        try:
            content_buffer = []
            
            # Invoke streaming
            for chunk in self.invoke_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                other_params=other_params
            ):
                # Append to buffer for callback tracking
                content_buffer.append(chunk)
                
                # Call the content callback
                on_content(chunk)
            
            # Streaming completed successfully
            if on_complete:
                result = {
                    "model_id": self.model_id,
                    "full_content": "".join(content_buffer),
                    "total_chunks": len(content_buffer),
                    "estimated_tokens": sum(self._estimate_tokens_in_text(chunk) for chunk in content_buffer)
                }
                on_complete(result)
                
        except Exception as e:
            self.logger.error(f"Error in streaming: {str(e)}")
            
            # Call error callback if provided
            if on_error:
                on_error(e)
            else:
                # Re-raise if no error handler
                raise


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a streaming client for Claude
    client = BedrockStreamingClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Basic streaming example
    try:
        print("\nStreaming response:")
        print("-" * 50)
        
        for chunk in client.invoke_stream(
            prompt="Write a short poem about cloud computing.",
            max_tokens=200
        ):
            print(chunk, end="", flush=True)
        
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Callback-based example
    try:
        print("\nStreaming with callbacks:")
        print("-" * 50)
        
        def on_content_chunk(chunk):
            print(chunk, end="", flush=True)
        
        def on_complete(result):
            print("\n" + "-" * 50)
            print(f"Streaming complete. Generated approximately {result['estimated_tokens']} tokens.")
        
        def on_error(error):
            print(f"\nError occurred: {str(error)}")
        
        client.invoke_stream_with_callbacks(
            prompt="Explain the benefits of streaming responses in AI applications.",
            on_content=on_content_chunk,
            on_complete=on_complete,
            on_error=on_error,
            max_tokens=200
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    print(f"Request count: {metrics['request_count']}")
    print(f"Token count: {metrics['token_count']}")
    print(f"Error count: {metrics['error_count']}")