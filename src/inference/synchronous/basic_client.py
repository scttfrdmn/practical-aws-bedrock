"""
Basic synchronous client for AWS Bedrock inference.

This module provides a simple, reliable client for invoking AWS Bedrock models
synchronously. It includes proper error handling, quota management awareness,
and follows project standards from CLAUDE.md.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple

import boto3
from botocore.exceptions import ClientError

from utils.profile_manager import get_profile, get_region


class BedrockClient:
    """
    A client for making synchronous AWS Bedrock requests with proper error handling.
    
    This client follows the AWS profile conventions specified in CLAUDE.md and
    provides a simple interface for making requests to AWS Bedrock models.
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
        Initialize the Bedrock client.
        
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
    
    def invoke(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7, 
        system_prompt: Optional[str] = None,
        other_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke the model with a prompt and return the response.
        
        Args:
            prompt: The user prompt or instruction
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            system_prompt: Optional system prompt for models that support it
            other_params: Additional model-specific parameters
            
        Returns:
            The parsed response from the model
            
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
        
        # Make the request with retries
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Invoking model {self.model_id} (attempt {attempt + 1})")
                
                start_time = time.time()
                
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                elapsed_time = time.time() - start_time
                
                # Parse the response based on model type
                result = self._parse_response(response)
                
                # Update metrics
                self.request_count += 1
                self._update_token_counts(result)
                
                self.logger.debug(
                    f"Model invocation successful in {elapsed_time:.2f}s. "
                    f"Input tokens: {result.get('input_tokens', 'unknown')}, "
                    f"Output tokens: {result.get('output_tokens', 'unknown')}"
                )
                
                return result
                
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
                "temperature": temperature
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
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the response from the model.
        
        Args:
            response: Raw response from the Bedrock API
            
        Returns:
            Parsed response with standardized fields
        """
        # Read the response body
        response_body = json.loads(response['body'].read())
        
        # Extract model family from model ID
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        # Initialize result with common fields
        result = {
            "model_id": self.model_id,
            "raw_response": response_body
        }
        
        # Parse model-specific response formats
        if "anthropic" in model_family:
            # Claude models
            if 'content' in response_body and len(response_body['content']) > 0:
                result["output"] = response_body['content'][0]['text']
            else:
                result["output"] = ""
                
            if 'usage' in response_body:
                result["input_tokens"] = response_body['usage'].get('input_tokens', 0)
                result["output_tokens"] = response_body['usage'].get('output_tokens', 0)
                result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
            
        elif "meta" in model_family or "llama" in model_family:
            # Llama models
            result["output"] = response_body.get('generation', '')
            
            # Token counts not provided by Llama, make rough estimate
            result["input_tokens"] = len(self._tokenize_rough(result.get("prompt", "")))
            result["output_tokens"] = len(self._tokenize_rough(result["output"]))
            result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
            
        elif "cohere" in model_family:
            # Cohere models
            result["output"] = response_body.get('text', '')
            
            if 'meta' in response_body:
                result["input_tokens"] = response_body['meta'].get('prompt_tokens', 0)
                result["output_tokens"] = response_body['meta'].get('response_tokens', 0)
                result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
            
        elif "ai21" in model_family:
            # AI21 models
            if 'completions' in response_body and len(response_body['completions']) > 0:
                result["output"] = response_body['completions'][0].get('data', {}).get('text', '')
            else:
                result["output"] = ""
                
            if 'usage' in response_body:
                result["input_tokens"] = response_body['usage'].get('input_tokens', 0)
                result["output_tokens"] = response_body['usage'].get('output_tokens', 0)
                result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
            
        else:
            # Default extraction (Amazon Titan)
            if 'results' in response_body and len(response_body['results']) > 0:
                result["output"] = response_body['results'][0].get('outputText', '')
            else:
                result["output"] = ""
                
            # Rough token count estimate for models that don't provide it
            result["input_tokens"] = len(self._tokenize_rough(result.get("prompt", "")))
            result["output_tokens"] = len(self._tokenize_rough(result["output"]))
            result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        
        return result
    
    def _update_token_counts(self, result: Dict[str, Any]) -> None:
        """
        Update token count metrics based on the response.
        
        Args:
            result: The parsed response
        """
        if "total_tokens" in result:
            self.token_count += result["total_tokens"]
    
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
    
    def _tokenize_rough(self, text: str) -> List[str]:
        """
        Perform a rough tokenization for token count estimation.
        This is not an accurate model-specific tokenization but
        provides a reasonable approximation for logging.
        
        Args:
            text: The text to tokenize
            
        Returns:
            A list of approximate tokens
        """
        import re
        
        # Simple tokenization rules (not model-specific)
        if not text:
            return []
            
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
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


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a client for Claude
    client = BedrockClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Basic invocation
    try:
        response = client.invoke(
            prompt="Explain quantum computing in simple terms.",
            max_tokens=500,
            temperature=0.7
        )
        
        print("\nModel Response:")
        print(response["output"])
        
        print("\nToken Usage:")
        print(f"Input tokens: {response.get('input_tokens', 'unknown')}")
        print(f"Output tokens: {response.get('output_tokens', 'unknown')}")
        print(f"Total tokens: {response.get('total_tokens', 'unknown')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    print(f"Request count: {metrics['request_count']}")
    print(f"Token count: {metrics['token_count']}")
    print(f"Error count: {metrics['error_count']}")