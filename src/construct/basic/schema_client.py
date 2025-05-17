"""
Schema-based client for AWS Bedrock Construct API.

This module provides a client for generating structured outputs from AWS Bedrock models
using the Construct API with JSON Schema validation to ensure consistent formats.
"""

import json
import time
import logging
import jsonschema
from typing import Dict, Any, Optional, List, Union, Tuple

import boto3
from botocore.exceptions import ClientError

from utils.profile_manager import get_profile, get_region


class SchemaValidationError(Exception):
    """Exception raised when the generated output fails schema validation."""
    
    def __init__(self, message: str, partial_result: Any = None, validation_errors: List[str] = None):
        """
        Initialize a schema validation error.
        
        Args:
            message: Error message
            partial_result: The output that failed validation
            validation_errors: List of specific validation errors
        """
        super().__init__(message)
        self.partial_result = partial_result
        self.validation_errors = validation_errors or []


class ConstructClient:
    """
    A client for generating structured outputs from AWS Bedrock models.
    
    This client follows the AWS profile conventions specified in CLAUDE.md and
    provides a simple interface for generating schema-validated structured data.
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
        Initialize the Construct client.
        
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
    
    def generate_structured(
        self, 
        input_text: str, 
        schema: Dict[str, Any],
        max_tokens: int = 2000,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        other_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Generate a structured output based on a JSON schema.
        
        Args:
            input_text: The input text to process
            schema: JSON schema that defines the output structure
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            system_prompt: Optional system prompt to guide generation
            other_params: Additional model-specific parameters
            
        Returns:
            Structured data conforming to the specified schema
            
        Raises:
            ValueError: For invalid input parameters
            SchemaValidationError: When the output doesn't validate against the schema
            RuntimeError: For unrecoverable errors after retries
        """
        # Validate inputs
        if not input_text:
            raise ValueError("Input text cannot be empty")
        
        if not schema or not isinstance(schema, dict):
            raise ValueError("Schema must be a valid JSON schema dictionary")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        
        # Validate the schema itself
        try:
            jsonschema.validators.validator_for(schema).check_schema(schema)
        except jsonschema.exceptions.SchemaError as e:
            raise ValueError(f"Invalid JSON schema: {str(e)}")
        
        # Create the prompt for structured generation
        construct_prompt = self._create_construct_prompt(
            input_text=input_text,
            schema=schema,
            system_prompt=system_prompt
        )
        
        # Create model-specific request body
        request_body = self._create_request_body(
            prompt=construct_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            other_params=other_params or {}
        )
        
        # Make the request with retries
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Invoking model {self.model_id} for structured generation (attempt {attempt + 1})")
                
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
                
                # Extract the structured output
                structured_output = self._extract_structured_output(result["output"])
                
                # Validate against schema
                try:
                    self._validate_against_schema(structured_output, schema)
                    return structured_output
                except SchemaValidationError as e:
                    # If this is the last retry, propagate the validation error
                    if attempt == self.max_retries:
                        raise
                    else:
                        # Log the error and retry
                        self.logger.warning(
                            f"Schema validation failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                            f"{str(e)}"
                        )
                        
                        # Use a more explicit prompt for the next attempt
                        construct_prompt = self._create_enhanced_prompt(
                            input_text=input_text,
                            schema=schema,
                            system_prompt=system_prompt,
                            validation_errors=e.validation_errors,
                            partial_result=e.partial_result
                        )
                        
                        # Update request body with enhanced prompt
                        request_body = self._create_request_body(
                            prompt=construct_prompt,
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
    
    def _create_construct_prompt(
        self, 
        input_text: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Create a prompt for structured generation.
        
        Args:
            input_text: The input text to process
            schema: JSON schema defining the output structure
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt for the model
        """
        # Format the schema as a string with pretty printing
        schema_str = json.dumps(schema, indent=2)
        
        # Create base prompt
        base_prompt = f"""Generate a valid JSON object based on the following schema:

{schema_str}

The JSON must strictly follow this schema and be valid. Only include the JSON object in your response.

Here is the input text to extract information from:

{input_text}

Your task is to extract the relevant information from the input text and format it according to the provided JSON schema. 
Return ONLY the JSON object, with no additional text, explanation, or markdown formatting.
"""

        # If system prompt is provided, incorporate it
        if system_prompt:
            base_prompt = f"{system_prompt}\n\n{base_prompt}"
        
        return base_prompt
    
    def _create_enhanced_prompt(
        self, 
        input_text: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        validation_errors: List[str] = None,
        partial_result: Any = None
    ) -> str:
        """
        Create an enhanced prompt when schema validation fails.
        
        Args:
            input_text: The input text to process
            schema: JSON schema defining the output structure
            system_prompt: Optional system prompt
            validation_errors: List of validation errors from previous attempt
            partial_result: The output that failed validation
            
        Returns:
            Enhanced prompt for the model
        """
        # Format the schema as a string with pretty printing
        schema_str = json.dumps(schema, indent=2)
        
        # Format validation errors if available
        validation_info = ""
        if validation_errors and partial_result:
            validation_info = f"""
The previous response failed schema validation with these errors:
{json.dumps(validation_errors, indent=2)}

Here was the invalid response:
{json.dumps(partial_result, indent=2)}

Please fix these errors and ensure your response is valid according to the schema.
"""
        
        # Create enhanced prompt
        enhanced_prompt = f"""Generate a valid JSON object based on the following schema:

{schema_str}

The JSON must strictly follow this schema and be valid. Only include the JSON object in your response.

Here is the input text to extract information from:

{input_text}

{validation_info}

Your task is to extract the relevant information from the input text and format it according to the provided JSON schema.
Return ONLY the JSON object, with no additional text, explanation, or markdown formatting.

Important requirements:
1. Follow the schema exactly, including all required fields
2. Use correct data types for each field
3. Do not add fields not specified in the schema
4. Ensure nested objects and arrays follow their specified schemas
5. Do not include explanatory text, just the JSON object
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
            result["input_tokens"] = len(self._tokenize_rough(prompt)) if 'prompt' in locals() else 0
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
    
    def _extract_structured_output(self, text: str) -> Any:
        """
        Extract structured JSON output from the model's text response.
        
        Args:
            text: The model's text output
            
        Returns:
            Parsed JSON object
            
        Raises:
            ValueError: If valid JSON cannot be extracted
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
            
            # If we still haven't found valid JSON, raise error
            raise ValueError(f"Could not extract valid JSON from model output: {text[:100]}...")
    
    def _validate_against_schema(self, data: Any, schema: Dict[str, Any]) -> None:
        """
        Validate data against a JSON schema.
        
        Args:
            data: The data to validate
            schema: JSON schema to validate against
            
        Raises:
            SchemaValidationError: If validation fails
        """
        validator = jsonschema.Draft7Validator(schema)
        errors = list(validator.iter_errors(data))
        
        if errors:
            # Format validation errors
            error_messages = []
            for error in errors:
                path = ".".join(str(path_part) for path_part in error.path) if error.path else "root"
                error_messages.append(f"At {path}: {error.message}")
            
            raise SchemaValidationError(
                message=f"Schema validation failed with {len(errors)} errors",
                partial_result=data,
                validation_errors=error_messages
            )
    
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
    client = ConstructClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Define a simple schema
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The person's full name"},
            "age": {"type": "integer", "minimum": 0, "description": "The person's age in years"},
            "occupation": {"type": "string", "description": "The person's job or profession"},
            "email": {"type": "string", "format": "email", "description": "The person's email address"}
        },
        "required": ["name", "age"]
    }
    
    # Example text
    sample_text = "John Smith is a 35-year-old software engineer who can be reached at john.smith@example.com"
    
    # Generate structured data
    try:
        result = client.generate_structured(
            input_text=sample_text,
            schema=person_schema,
            temperature=0.2
        )
        
        print("\nGenerated Structured Data:")
        print(json.dumps(result, indent=2))
        
        # Try with a more complex schema
        event_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title or name of the event"},
                "date": {"type": "string", "format": "date", "description": "The date when the event takes place"},
                "location": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the venue"},
                        "address": {"type": "string", "description": "Address of the venue"},
                        "city": {"type": "string", "description": "City where the event takes place"}
                    },
                    "required": ["name", "city"]
                },
                "description": {"type": "string", "description": "A brief description of the event"},
                "ticketPrice": {"type": "number", "minimum": 0, "description": "The price of tickets in USD"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Categories or tags related to the event"
                }
            },
            "required": ["title", "date", "location"]
        }
        
        event_text = """
        Join us for the annual Tech Conference 2023, happening on September 15-17 at the San Francisco Convention Center.
        This year's event features keynote speakers from major tech companies, workshops on AI and machine learning, 
        and networking opportunities. Early bird tickets are available for $299 until August 1st.
        """
        
        event_result = client.generate_structured(
            input_text=event_text,
            schema=event_schema,
            temperature=0.2
        )
        
        print("\nComplex Schema Example:")
        print(json.dumps(event_result, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    print(f"Request count: {metrics['request_count']}")
    print(f"Token count: {metrics['token_count']}")
    print(f"Error count: {metrics['error_count']}")