"""
JSON output client for AWS Bedrock Construct API.

This module provides a simplified client for generating JSON outputs from
AWS Bedrock models without requiring a formal JSON schema.
"""

import json
import time
import logging
import re
from typing import Dict, Any, Optional, List, Union, Tuple

import boto3
from botocore.exceptions import ClientError

from utils.profile_manager import get_profile, get_region
from .schema_client import ConstructClient


class JSONOutputClient:
    """
    A client for generating JSON outputs from AWS Bedrock models.
    
    This client provides a simpler interface for generating JSON outputs
    without requiring formal JSON schema validation. It's useful for cases
    where you want structured JSON but don't need strict schema enforcement.
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
        Initialize the JSON output client.
        
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
    
    def generate_json(
        self, 
        prompt: str,
        output_format: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        other_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a JSON output based on a prompt.
        
        Args:
            prompt: The prompt describing what JSON to generate
            output_format: Optional dictionary describing the expected format
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            system_prompt: Optional system prompt to guide generation
            other_params: Additional model-specific parameters
            
        Returns:
            Generated JSON object
            
        Raises:
            ValueError: For invalid input parameters
            json.JSONDecodeError: When the output is not valid JSON
            RuntimeError: For unrecoverable errors after retries
        """
        # Validate inputs
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        
        # Create the prompt for JSON generation
        json_prompt = self._create_json_prompt(
            prompt=prompt,
            output_format=output_format,
            system_prompt=system_prompt
        )
        
        # Create model-specific request body
        request_body = self._create_request_body(
            prompt=json_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            other_params=other_params or {}
        )
        
        # Make the request with retries
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Invoking model {self.model_id} for JSON generation (attempt {attempt + 1})")
                
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
                
                # Extract the JSON output
                try:
                    json_output = self._extract_json(result["output"])
                    return json_output
                except json.JSONDecodeError as e:
                    # If this is the last retry, propagate the error
                    if attempt == self.max_retries:
                        raise
                    else:
                        # Log the error and retry
                        self.logger.warning(
                            f"JSON parsing failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                            f"{str(e)}"
                        )
                        
                        # Use a more explicit prompt for the next attempt
                        json_prompt = self._create_enhanced_json_prompt(
                            prompt=prompt,
                            output_format=output_format,
                            system_prompt=system_prompt,
                            error_message=str(e),
                            previous_output=result["output"]
                        )
                        
                        # Update request body with enhanced prompt
                        request_body = self._create_request_body(
                            prompt=json_prompt,
                            max_tokens=max_tokens,
                            temperature=max(0.1, temperature - 0.1),  # Slightly reduce temperature
                            other_params=other_params or {}
                        )
                        
                        # Calculate backoff time
                        backoff_time = self._calculate_backoff(attempt)
                        self.logger.info(f"Retrying with enhanced prompt in {backoff_time:.2f} seconds...")
                        time.sleep(backoff_time)
                        continue
                
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
    
    def _create_json_prompt(
        self, 
        prompt: str,
        output_format: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Create a prompt for JSON generation.
        
        Args:
            prompt: The user prompt
            output_format: Optional dictionary describing the expected format
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt for the model
        """
        # Format example if provided
        format_example = ""
        if output_format:
            format_example = f"""
The JSON should follow this format:
{json.dumps(output_format, indent=2)}

Keys should match exactly as shown in the example format.
"""
        
        # Create base prompt
        base_prompt = f"""Generate a valid JSON object based on the following request:

{prompt}

{format_example}

Your response should be ONLY a valid JSON object, with no additional text, explanation, or markdown formatting.
Do not include ```json or ``` markers around the output.
"""

        # If system prompt is provided, incorporate it
        if system_prompt:
            base_prompt = f"{system_prompt}\n\n{base_prompt}"
        
        return base_prompt
    
    def _create_enhanced_json_prompt(
        self, 
        prompt: str,
        output_format: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        error_message: str = "",
        previous_output: str = ""
    ) -> str:
        """
        Create an enhanced prompt when JSON parsing fails.
        
        Args:
            prompt: The user prompt
            output_format: Optional dictionary describing the expected format
            system_prompt: Optional system prompt
            error_message: Error message from previous attempt
            previous_output: Previous failed output
            
        Returns:
            Enhanced prompt for the model
        """
        # Format example if provided
        format_example = ""
        if output_format:
            format_example = f"""
The JSON should follow this format:
{json.dumps(output_format, indent=2)}

Keys should match exactly as shown in the example format.
"""
        
        # Clean previous output for display
        clean_previous = previous_output.strip()
        if len(clean_previous) > 500:
            clean_previous = clean_previous[:500] + "... (truncated)"
        
        # Create enhanced prompt
        enhanced_prompt = f"""Generate a valid JSON object based on the following request:

{prompt}

{format_example}

Your previous response could not be parsed as valid JSON. Error: {error_message}

Previous response:
{clean_previous}

Please provide a response that is ONLY a valid JSON object, with no additional text, explanation, or markdown formatting.
Do not include ```json or ``` markers around the output.

Important JSON requirements:
1. Ensure all keys and values are properly quoted with double quotes
2. Include commas between all key-value pairs except the last one
3. Escape special characters in strings properly
4. Make sure all opened brackets and braces are closed
5. Verify quotes are properly paired
6. Do not include any text before or after the JSON object
"""

        # If system prompt is provided, incorporate it
        if system_prompt:
            enhanced_prompt = f"{system_prompt}\n\n{enhanced_prompt}"
        
        return enhanced_prompt
    
    def _create_request_body(
        self, 
        prompt: str,
        max_tokens: int,
        temperature: float,
        other_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create the appropriate request body for the model family.
        
        Args:
            prompt: The formatted prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            other_params: Additional model-specific parameters
            
        Returns:
            Model-specific request body
        """
        # Extract model family from model ID
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        
        if "anthropic" in model_family:
            # Claude models
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
        elif "meta" in model_family or "llama" in model_family:
            # Llama models
            body = {
                "prompt": f"<s>[INST] {prompt} [/INST]",
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
            result["input_tokens"] = 0  # Can't estimate without the prompt
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
            result["input_tokens"] = 0  # Can't estimate without the prompt
            result["output_tokens"] = len(self._tokenize_rough(result["output"]))
            result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        
        return result
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract a JSON object from text.
        
        Args:
            text: The text containing JSON
            
        Returns:
            Parsed JSON object
            
        Raises:
            json.JSONDecodeError: If valid JSON cannot be extracted
        """
        # Clean up the text to extract just the JSON
        # Remove markdown code block indicators
        text = text.replace("```json", "").replace("```", "")
        
        # Trim whitespace
        text = text.strip()
        
        try:
            # Try to parse the entire text as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON using a more robust approach
            self.logger.debug("Initial JSON parsing failed, trying more robust extraction")
            
            # Look for curly braces that may indicate JSON object
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                # Extract potential JSON substring
                json_text = text[start_idx:end_idx + 1]
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass
            
            # If we still haven't found valid JSON, raise the original error
            raise json.JSONDecodeError(f"Could not extract valid JSON from model output", text, 0)
    
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
    client = JSONOutputClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Define an example output format
    user_format = {
        "name": "John Doe",
        "age": 30,
        "skills": ["Python", "Machine Learning", "AWS"],
        "contact": {
            "email": "john.doe@example.com",
            "phone": "555-123-4567"
        }
    }
    
    # Example prompt
    prompt = "Create a JSON profile for a software developer named Jane Smith who is 28 years old, knows JavaScript, React, and Node.js, and can be contacted at jane.smith@techcompany.com or 555-987-6543."
    
    # Generate JSON data
    try:
        result = client.generate_json(
            prompt=prompt,
            output_format=user_format,
            temperature=0.2
        )
        
        print("\nGenerated JSON:")
        print(json.dumps(result, indent=2))
        
        # Try with a different prompt
        event_format = {
            "title": "Example Event",
            "date": "2023-01-01",
            "location": "Example Venue, City",
            "ticketPrice": 99.99,
            "tags": ["tag1", "tag2"]
        }
        
        event_prompt = "Create a JSON object for a Tech Conference happening on October 10-12, 2023 at the San Francisco Convention Center. Tickets cost $499 and it focuses on AI, Machine Learning, and Cloud Computing."
        
        event_result = client.generate_json(
            prompt=event_prompt,
            output_format=event_format,
            temperature=0.2
        )
        
        print("\nEvent JSON:")
        print(json.dumps(event_result, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    print(f"Request count: {metrics['request_count']}")
    print(f"Token count: {metrics['token_count']}")
    print(f"Error count: {metrics['error_count']}")