"""
Function calling client for AWS Bedrock Converse API.

This module provides a client for implementing function calling capabilities
with AWS Bedrock models using the Converse API. It allows the AI to use
external tools and APIs during conversations.
"""

import json
import time
import logging
import uuid
import inspect
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

import boto3
from botocore.exceptions import ClientError

from ..basic.simple_chat import ConverseClient


class FunctionCallingConverseClient(ConverseClient):
    """
    A client for implementing function calling with AWS Bedrock models.
    
    This client extends the basic ConverseClient to provide function calling
    capabilities, allowing the AI to use external tools and APIs during conversations.
    """
    
    def __init__(
        self, 
        model_id: str,
        functions: List[Dict[str, Any]],
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the function calling Converse client.
        
        Args:
            model_id: The Bedrock model identifier
            functions: List of function definitions
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
        
        # Store function definitions
        self.functions = functions
        
        # Validate function definitions
        self._validate_functions()
    
    def _validate_functions(self) -> None:
        """
        Validate function definitions to ensure they meet required format.
        
        Raises:
            ValueError: If function definitions are invalid
        """
        if not isinstance(self.functions, list):
            raise ValueError("functions must be a list of function definitions")
        
        required_keys = ["name", "description"]
        
        for i, func in enumerate(self.functions):
            if not isinstance(func, dict):
                raise ValueError(f"Function definition at index {i} must be a dictionary")
                
            for key in required_keys:
                if key not in func:
                    raise ValueError(f"Function definition at index {i} missing required key: {key}")
            
            if "parameters" in func and not isinstance(func["parameters"], dict):
                raise ValueError(f"Function parameters at index {i} must be a dictionary")
    
    def send_message_with_functions(
        self, 
        conversation_id: str, 
        message: str,
        function_map: Dict[str, Callable],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        max_function_calls: int = 5,
        other_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a message in a conversation with function calling capabilities.
        
        Args:
            conversation_id: The conversation ID
            message: The user message
            function_map: Dictionary mapping function names to callable implementations
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            max_function_calls: Maximum number of function calls allowed
            other_params: Additional model-specific parameters
            
        Returns:
            Dictionary with the final response and function calls made
            
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
        
        # Validate function map against function definitions
        self._validate_function_map(function_map)
        
        # Add user message to conversation history
        self.conversations[conversation_id]["messages"].append({
            "role": "user",
            "content": message
        })
        self.conversations[conversation_id]["updated_at"] = time.time()
        
        # Initialize function call tracking
        function_calls_made = []
        
        # Process the conversation with function calling
        for call_count in range(max_function_calls + 1):  # +1 for final response
            # Create request body with functions
            request_body = self._create_function_request(
                conversation_id=conversation_id,
                max_tokens=max_tokens,
                temperature=temperature,
                other_params=other_params or {}
            )
            
            # Make the request with retries
            response_result = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    self.logger.debug(f"Invoking model {self.model_id} with functions (attempt {attempt + 1})")
                    
                    start_time = time.time()
                    
                    response = self.client.converse(
                        modelId=self.model_id,
                        body=json.dumps(request_body)
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    # Parse the response
                    response_result = self._parse_converse_response(response)
                    
                    # Update metrics
                    self.request_count += 1
                    self._update_token_counts(response_result)
                    
                    self.logger.debug(
                        f"Converse invocation successful in {elapsed_time:.2f}s. "
                        f"Input tokens: {response_result.get('input_tokens', 'unknown')}, "
                        f"Output tokens: {response_result.get('output_tokens', 'unknown')}"
                    )
                    
                    # Break out of retry loop
                    break
                    
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
            
            # If we couldn't get a response, raise an exception
            if response_result is None:
                raise RuntimeError("Failed to get response from model")
            
            # Check if the model wants to call a function
            function_call = self._extract_function_call(response_result["raw_response"])
            
            if function_call and call_count < max_function_calls:
                # Extract function details
                function_name = function_call.get("name")
                function_args = function_call.get("arguments", {})
                
                # Log the function call
                self.logger.info(f"Model wants to call function: {function_name} with args: {function_args}")
                
                # Track function call
                function_calls_made.append({
                    "name": function_name,
                    "arguments": function_args
                })
                
                # Execute the function
                try:
                    # Convert args from JSON string if needed
                    if isinstance(function_args, str):
                        try:
                            function_args = json.loads(function_args)
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not parse function arguments as JSON: {function_args}")
                            function_args = {"error": "Invalid JSON arguments"}
                    
                    # Call the function and get the result
                    func = function_map[function_name]
                    
                    # Check if function accepts keyword arguments or positional
                    if isinstance(function_args, dict):
                        function_result = func(**function_args)
                    else:
                        function_result = func(function_args)
                    
                    # Add function call result to conversation
                    self._add_function_result(
                        conversation_id=conversation_id,
                        function_name=function_name,
                        function_args=function_args,
                        function_result=function_result
                    )
                    
                except Exception as e:
                    # Log the error
                    self.logger.error(f"Error executing function {function_name}: {str(e)}")
                    
                    # Add function call error to conversation
                    self._add_function_result(
                        conversation_id=conversation_id,
                        function_name=function_name,
                        function_args=function_args,
                        function_result={"error": str(e)}
                    )
            else:
                # No function call or reached max calls, add assistant response
                final_response = response_result["output"]
                
                self.conversations[conversation_id]["messages"].append({
                    "role": "assistant",
                    "content": final_response
                })
                self.conversations[conversation_id]["updated_at"] = time.time()
                
                # Return the final response and function calls
                return {
                    "response": final_response,
                    "function_calls": function_calls_made
                }
        
        # If we reach here, we've exceeded max_function_calls
        self.logger.warning(f"Exceeded maximum function calls ({max_function_calls})")
        return {
            "response": "I apologize, but I've reached the maximum number of function calls allowed. Is there anything else I can help with?",
            "function_calls": function_calls_made
        }
    
    def _validate_function_map(self, function_map: Dict[str, Callable]) -> None:
        """
        Validate that the function map contains implementations for all defined functions.
        
        Args:
            function_map: Dictionary mapping function names to implementations
            
        Raises:
            ValueError: If function map is invalid
        """
        if not isinstance(function_map, dict):
            raise ValueError("function_map must be a dictionary")
        
        # Check that all defined functions have implementations
        for func in self.functions:
            func_name = func["name"]
            if func_name not in function_map:
                raise ValueError(f"No implementation provided for function: {func_name}")
            
            if not callable(function_map[func_name]):
                raise ValueError(f"Implementation for {func_name} is not callable")
    
    def _create_function_request(
        self, 
        conversation_id: str,
        max_tokens: int,
        temperature: float,
        other_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a request body for the Converse API with function definitions.
        
        Args:
            conversation_id: The conversation ID
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            other_params: Additional model-specific parameters
            
        Returns:
            Request body for the Converse API with function definitions
        """
        # Get the basic request body
        body = self._create_converse_request(
            conversation_id=conversation_id,
            max_tokens=max_tokens,
            temperature=temperature,
            other_params=other_params
        )
        
        # Add function definitions based on model family
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        if "anthropic" in model_family:
            # Claude models use tools format
            body["tools"] = [
                {
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func.get("parameters", {})
                }
                for func in self.functions
            ]
        else:
            # Generic format for other models (if supported)
            body["functions"] = self.functions
        
        return body
    
    def _extract_function_call(self, response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract function call information from the model response.
        
        Args:
            response_data: The raw response data from the model
            
        Returns:
            Function call details or None if no function call
        """
        # Extract model family from model ID
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        if "anthropic" in model_family:
            # Claude models
            if 'tool_use' in response_data:
                tool_use = response_data['tool_use']
                return {
                    "name": tool_use.get("name"),
                    "arguments": tool_use.get("input", {})
                }
                
            if 'message' in response_data and 'tool_calls' in response_data['message']:
                tool_calls = response_data['message']['tool_calls']
                if tool_calls and len(tool_calls) > 0:
                    tool_call = tool_calls[0]
                    return {
                        "name": tool_call.get("function", {}).get("name"),
                        "arguments": tool_call.get("function", {}).get("arguments", {})
                    }
        else:
            # Generic format for other models
            if 'function_call' in response_data:
                return response_data['function_call']
                
            if 'message' in response_data and 'function_call' in response_data['message']:
                return response_data['message']['function_call']
        
        # No function call
        return None
    
    def _add_function_result(
        self, 
        conversation_id: str,
        function_name: str,
        function_args: Dict[str, Any],
        function_result: Any
    ) -> None:
        """
        Add function call result to the conversation history.
        
        Args:
            conversation_id: The conversation ID
            function_name: Name of the called function
            function_args: Arguments passed to the function
            function_result: Result returned by the function
        """
        # Format function result for the conversation
        # Extract model family to determine format
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        # Serialize complex objects
        if not isinstance(function_result, (str, int, float, bool, type(None))):
            try:
                function_result = json.dumps(function_result)
            except TypeError:
                function_result = str(function_result)
        
        if "anthropic" in model_family:
            # Claude models use tool_result format
            self.conversations[conversation_id]["messages"].append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_args) if isinstance(function_args, dict) else function_args
                    }
                }]
            })
            
            self.conversations[conversation_id]["messages"].append({
                "role": "tool",
                "content": json.dumps(function_result) if not isinstance(function_result, str) else function_result,
                "tool_call_id": str(uuid.uuid4()),
                "name": function_name
            })
        else:
            # Generic format for other models
            self.conversations[conversation_id]["messages"].append({
                "role": "assistant",
                "content": "",
                "function_call": {
                    "name": function_name,
                    "arguments": json.dumps(function_args) if isinstance(function_args, dict) else function_args
                }
            })
            
            self.conversations[conversation_id]["messages"].append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(function_result) if not isinstance(function_result, str) else function_result
            })
        
        self.conversations[conversation_id]["updated_at"] = time.time()


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define functions the AI can call
    functions = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or zip code"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "find_restaurants",
            "description": "Find restaurants in a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or neighborhood"
                    },
                    "cuisine": {
                        "type": "string",
                        "description": "Type of cuisine (e.g., Italian, Mexican)"
                    },
                    "price_range": {
                        "type": "string",
                        "enum": ["$", "$$", "$$$", "$$$$"],
                        "description": "Price range from $ (cheap) to $$$$ (expensive)"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    # Create function implementations
    def get_weather(location, unit="celsius"):
        """Simulated weather API call"""
        # In a real implementation, this would call a weather API
        weather_data = {
            "seattle": {"temp": 62, "condition": "Cloudy", "humidity": 75},
            "new york": {"temp": 72, "condition": "Sunny", "humidity": 50},
            "san francisco": {"temp": 65, "condition": "Foggy", "humidity": 80},
            "chicago": {"temp": 58, "condition": "Rainy", "humidity": 65},
        }
        
        location = location.lower()
        
        if location in weather_data:
            result = weather_data[location].copy()
            
            # Convert temperature if needed
            if unit == "fahrenheit":
                result["temp"] = int(result["temp"] * 9/5 + 32)
                result["unit"] = "F"
            else:
                result["unit"] = "C"
                
            return result
        else:
            return {"error": f"Weather data not available for {location}"}
    
    def find_restaurants(location, cuisine=None, price_range=None):
        """Simulated restaurant search"""
        # In a real implementation, this would call a restaurant API
        restaurants = [
            {"name": "Bella Italia", "cuisine": "Italian", "price": "$$", "rating": 4.5},
            {"name": "Taco Haven", "cuisine": "Mexican", "price": "$", "rating": 4.2},
            {"name": "Dragon Palace", "cuisine": "Chinese", "price": "$$", "rating": 4.0},
            {"name": "Burger Joint", "cuisine": "American", "price": "$", "rating": 3.8},
            {"name": "Le Gourmet", "cuisine": "French", "price": "$$$$", "rating": 4.8},
            {"name": "Sushi Express", "cuisine": "Japanese", "price": "$$$", "rating": 4.3},
            {"name": "Curry House", "cuisine": "Indian", "price": "$$", "rating": 4.1},
        ]
        
        # Filter by cuisine if specified
        if cuisine:
            restaurants = [r for r in restaurants if r["cuisine"].lower() == cuisine.lower()]
            
        # Filter by price range if specified
        if price_range:
            restaurants = [r for r in restaurants if r["price"] == price_range]
            
        # Return results
        if restaurants:
            return {
                "location": location,
                "restaurants": restaurants[:3]  # Return top 3 restaurants
            }
        else:
            return {"message": "No restaurants found matching your criteria"}
    
    # Map function names to implementations
    function_map = {
        "get_weather": get_weather,
        "find_restaurants": find_restaurants
    }
    
    # Create a function calling client for Claude
    client = FunctionCallingConverseClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        functions=functions
    )
    
    # Create a conversation with a system prompt
    conversation_id = client.create_conversation(
        system_prompt="You are a helpful assistant that can provide information about weather and restaurants."
    )
    
    try:
        # Example 1: Weather query that should trigger a function call
        print("\n--- Example 1: Weather Query ---")
        print("User: What's the weather like in Seattle?")
        
        response = client.send_message_with_functions(
            conversation_id=conversation_id,
            message="What's the weather like in Seattle?",
            function_map=function_map,
            max_tokens=200,
            temperature=0.7
        )
        
        print("\nAssistant response:")
        print(response["response"])
        print(f"\nFunction calls made: {len(response['function_calls'])}")
        for call in response["function_calls"]:
            print(f"- Called {call['name']} with args: {call['arguments']}")
        
        # Example 2: Restaurant query that should trigger a function call
        print("\n--- Example 2: Restaurant Query ---")
        print("User: Can you find me some Italian restaurants?")
        
        response = client.send_message_with_functions(
            conversation_id=conversation_id,
            message="Can you find me some Italian restaurants?",
            function_map=function_map,
            max_tokens=300,
            temperature=0.7
        )
        
        print("\nAssistant response:")
        print(response["response"])
        print(f"\nFunction calls made: {len(response['function_calls'])}")
        for call in response["function_calls"]:
            print(f"- Called {call['name']} with args: {call['arguments']}")
        
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