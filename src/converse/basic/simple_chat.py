"""
Basic client for AWS Bedrock Converse API.

This module provides a simple, reliable client for creating conversational
experiences with AWS Bedrock models using the Converse API. It follows project
standards from CLAUDE.md.
"""

import json
import time
import logging
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple

import boto3
from botocore.exceptions import ClientError

from utils.profile_manager import get_profile, get_region


class ConverseClient:
    """
    A client for creating conversational experiences with AWS Bedrock models.
    
    This client follows the AWS profile conventions specified in CLAUDE.md and
    provides a simple interface for managing multi-turn conversations using
    the Converse API.
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
        Initialize the Converse client.
        
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
        
        # In-memory conversation storage (in production, use a database)
        self.conversations = {}
    
    def create_conversation(
        self, 
        system_prompt: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Create a new conversation.
        
        Args:
            system_prompt: Optional system prompt to guide the model's behavior
            conversation_id: Optional ID for the conversation (generated if not provided)
            
        Returns:
            The conversation ID
        """
        # Generate a conversation ID if not provided
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        # Initialize conversation structure
        self.conversations[conversation_id] = {
            "messages": [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "model_id": self.model_id
        }
        
        # Add system prompt if provided
        if system_prompt:
            self.conversations[conversation_id]["messages"].append({
                "role": "system",
                "content": system_prompt
            })
            
        self.logger.info(f"Created conversation {conversation_id}")
        return conversation_id
    
    def send_message(
        self, 
        conversation_id: str, 
        message: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        other_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a message in a conversation and get the model's response.
        
        Args:
            conversation_id: The conversation ID
            message: The user message
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
                self.logger.debug(f"Invoking model {self.model_id} (attempt {attempt + 1})")
                
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
                    f"Converse invocation successful in {elapsed_time:.2f}s. "
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
    
    def _create_converse_request(
        self, 
        conversation_id: str,
        max_tokens: int,
        temperature: float,
        other_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a request body for the Converse API.
        
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
    
    def _parse_converse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the response from the Converse API.
        
        Args:
            response: Raw response from the Bedrock Converse API
            
        Returns:
            Parsed response with standardized fields
        """
        # Read the response body
        response_body = json.loads(response['body'].read())
        
        # Initialize result with common fields
        result = {
            "model_id": self.model_id,
            "raw_response": response_body
        }
        
        # Extract model family from model ID to handle model-specific responses
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        # Parse the response
        if "anthropic" in model_family:
            # Claude models
            if 'message' in response_body and 'content' in response_body['message']:
                content_list = response_body['message']['content']
                if content_list and len(content_list) > 0 and 'text' in content_list[0]:
                    result["output"] = content_list[0]['text']
                else:
                    result["output"] = ""
            else:
                result["output"] = ""
                
            if 'usage' in response_body:
                result["input_tokens"] = response_body['usage'].get('input_tokens', 0)
                result["output_tokens"] = response_body['usage'].get('output_tokens', 0)
                result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        else:
            # Generic parsing for other models
            if 'output' in response_body:
                result["output"] = response_body['output']
            elif 'message' in response_body:
                result["output"] = response_body['message'].get('content', '')
            else:
                result["output"] = ""
            
            # Token counts might not be available for all models
            if 'usage' in response_body:
                result["input_tokens"] = response_body['usage'].get('input_tokens', 0)
                result["output_tokens"] = response_body['usage'].get('output_tokens', 0)
                result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
            else:
                # Rough estimation
                result["input_tokens"] = 0
                result["output_tokens"] = 0
                result["total_tokens"] = 0
        
        return result
    
    def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get the conversation history.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            The conversation history
            
        Raises:
            KeyError: When conversation_id doesn't exist
        """
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation {conversation_id} not found")
        
        return self.conversations[conversation_id]
    
    def add_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str
    ) -> None:
        """
        Add a message to a conversation without sending it to the model.
        Useful for adding context or system messages.
        
        Args:
            conversation_id: The conversation ID
            role: Message role ('system', 'user', or 'assistant')
            content: Message content
            
        Raises:
            KeyError: When conversation_id doesn't exist
            ValueError: When role is invalid
        """
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation {conversation_id} not found")
        
        valid_roles = ['system', 'user', 'assistant']
        if role not in valid_roles:
            raise ValueError(f"Invalid role '{role}'. Must be one of: {valid_roles}")
        
        self.conversations[conversation_id]["messages"].append({
            "role": role,
            "content": content
        })
        self.conversations[conversation_id]["updated_at"] = time.time()
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if conversation was deleted, False if it didn't exist
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            self.logger.info(f"Deleted conversation {conversation_id}")
            return True
        return False
    
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
    
    def _update_token_counts(self, result: Dict[str, Any]) -> None:
        """
        Update token count metrics based on the response.
        
        Args:
            result: The parsed response
        """
        if "total_tokens" in result:
            self.token_count += result["total_tokens"]
    
    def get_metrics(self) -> Dict[str, int]:
        """
        Get usage metrics for this client instance.
        
        Returns:
            Dictionary with usage metrics
        """
        return {
            "request_count": self.request_count,
            "token_count": self.token_count,
            "error_count": self.error_count,
            "conversation_count": len(self.conversations)
        }


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a client for Claude
    client = ConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Create a conversation with a system prompt
    conversation_id = client.create_conversation(
        system_prompt="You are a helpful assistant specializing in AWS services."
    )
    
    try:
        # Send the first message
        response = client.send_message(
            conversation_id=conversation_id,
            message="What is AWS Bedrock and what can I do with it?",
            max_tokens=500,
            temperature=0.7
        )
        
        print("\nAssistant response:")
        print(response)
        
        # Send a follow-up message in the same conversation
        follow_up = client.send_message(
            conversation_id=conversation_id,
            message="Which foundation models does it support?",
            max_tokens=500,
            temperature=0.7
        )
        
        print("\nAssistant response:")
        print(follow_up)
        
        # Get conversation history
        history = client.get_conversation_history(conversation_id)
        print(f"\nConversation has {len(history['messages'])} messages")
        
        # Display each message
        for i, msg in enumerate(history['messages']):
            print(f"\nMessage {i+1}:")
            print(f"Role: {msg['role']}")
            print(f"Content: {msg['content'][:100]}..." if len(msg['content']) > 100 else f"Content: {msg['content']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")