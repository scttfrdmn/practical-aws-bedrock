"""
Complex schema client for AWS Bedrock Construct API.

This module provides a client for working with complex nested schemas
in structured outputs from AWS Bedrock models.
"""

import json
import time
import logging
import jsonschema
from typing import Dict, Any, Optional, List, Union, Tuple

from ..basic.schema_client import ConstructClient, SchemaValidationError


class ComplexConstructClient(ConstructClient):
    """
    A client for generating structured outputs with complex schemas.
    
    This client extends the basic ConstructClient with additional capabilities
    for handling complex nested schemas, arrays, and conditional logic.
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
        Initialize the complex schema client.
        
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
    
    def generate_structured(
        self, 
        input_text: str, 
        schema: Dict[str, Any],
        max_tokens: int = 4000,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        other_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Generate a structured output based on a complex JSON schema.
        
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
        # Validate schema complexity and adjust parameters if needed
        adjusted_params = self._adjust_for_complexity(schema, max_tokens, temperature)
        
        # For complex schemas, we want to use a slightly different prompt approach
        enhanced_system_prompt = self._enhance_system_prompt(system_prompt, schema)
        
        # Call the parent's generate_structured with adjusted parameters
        return super().generate_structured(
            input_text=input_text,
            schema=schema,
            max_tokens=adjusted_params["max_tokens"],
            temperature=adjusted_params["temperature"],
            system_prompt=enhanced_system_prompt,
            other_params=other_params
        )
    
    def generate_with_schema_decomposition(
        self, 
        input_text: str, 
        schema: Dict[str, Any],
        max_tokens: int = 4000,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        other_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Generate a structured output by decomposing a complex schema.
        
        This method breaks down complex schemas into simpler components,
        generates each component separately, and then combines them.
        
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
        
        # Check if schema is simple enough to process directly
        if not self._is_complex_schema(schema):
            return self.generate_structured(
                input_text=input_text,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                other_params=other_params
            )
        
        # Decompose the schema
        decomposed_schemas = self._decompose_schema(schema)
        
        # Generate each component separately
        component_results = {}
        
        for component_name, component_schema in decomposed_schemas.items():
            component_prompt = self._create_component_prompt(
                input_text=input_text,
                schema=component_schema,
                component_name=component_name
            )
            
            # Generate this component
            try:
                component_result = self.generate_structured(
                    input_text=component_prompt,
                    schema=component_schema,
                    max_tokens=max(1000, max_tokens // 2),  # Adjust tokens for components
                    temperature=temperature,
                    system_prompt=system_prompt,
                    other_params=other_params
                )
                
                component_results[component_name] = component_result
                
            except Exception as e:
                self.logger.warning(f"Error generating component {component_name}: {str(e)}")
                # Continue with other components even if one fails
        
        # Combine the components
        combined_result = self._combine_components(component_results, schema)
        
        # Validate the combined result
        self._validate_against_schema(combined_result, schema)
        
        return combined_result
    
    def _adjust_for_complexity(
        self, 
        schema: Dict[str, Any],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Adjust generation parameters based on schema complexity.
        
        Args:
            schema: The JSON schema
            max_tokens: Original max tokens setting
            temperature: Original temperature setting
            
        Returns:
            Dictionary with adjusted parameters
        """
        complexity_score = self._calculate_schema_complexity(schema)
        
        # Adjust max_tokens based on complexity
        # More complex schemas need more tokens for complete generation
        adjusted_max_tokens = max_tokens
        if complexity_score > 10:
            # For very complex schemas, increase token limit
            adjusted_max_tokens = min(8000, max_tokens * 2)
        
        # Adjust temperature based on complexity
        # More complex schemas benefit from lower temperature for consistency
        adjusted_temperature = temperature
        if complexity_score > 5:
            # For complex schemas, reduce temperature slightly
            adjusted_temperature = max(0.1, temperature * 0.8)
        
        return {
            "max_tokens": adjusted_max_tokens,
            "temperature": adjusted_temperature,
            "complexity_score": complexity_score
        }
    
    def _calculate_schema_complexity(self, schema: Dict[str, Any]) -> int:
        """
        Calculate a complexity score for a JSON schema.
        
        Args:
            schema: The JSON schema
            
        Returns:
            Complexity score (higher means more complex)
        """
        score = 0
        
        # Check for properties (object type)
        if "properties" in schema and isinstance(schema["properties"], dict):
            # Base score from number of properties
            score += len(schema["properties"])
            
            # Add complexity for nested objects and arrays
            for prop_name, prop_schema in schema["properties"].items():
                if "properties" in prop_schema:
                    # Nested object
                    score += self._calculate_schema_complexity(prop_schema)
                elif "items" in prop_schema and "type" in prop_schema and prop_schema["type"] == "array":
                    # Array type
                    if isinstance(prop_schema["items"], dict):
                        # Complex array items
                        score += 1 + self._calculate_schema_complexity(prop_schema["items"])
                    else:
                        # Simple array items
                        score += 1
        
        # Check for required fields
        if "required" in schema and isinstance(schema["required"], list):
            score += len(schema["required"]) * 0.5
        
        # Check for additional constraints
        for constraint in ["minimum", "maximum", "minLength", "maxLength", "pattern", "format"]:
            if constraint in schema:
                score += 0.5
        
        # Check for oneOf, anyOf, allOf (complex logic)
        for complex_key in ["oneOf", "anyOf", "allOf"]:
            if complex_key in schema and isinstance(schema[complex_key], list):
                score += len(schema[complex_key]) * 2
        
        return score
    
    def _enhance_system_prompt(
        self,
        system_prompt: Optional[str],
        schema: Dict[str, Any]
    ) -> str:
        """
        Enhance the system prompt for complex schema generation.
        
        Args:
            system_prompt: Original system prompt
            schema: The JSON schema
            
        Returns:
            Enhanced system prompt
        """
        complexity_score = self._calculate_schema_complexity(schema)
        
        # Create schema-specific enhancements
        if complexity_score <= 3:
            # Simple schema, minimal enhancement
            enhancement = "You are an expert at generating structured data according to provided schemas."
        elif complexity_score <= 8:
            # Moderately complex schema
            enhancement = """
You are an expert at generating structured data based on JSON schemas.
Pay special attention to nested objects and arrays in your output.
Ensure all required fields are included and correctly typed.
"""
        else:
            # Very complex schema
            enhancement = """
You are an expert at generating structured data based on complex JSON schemas.
Approach this systematically by breaking down the schema into components:
1. First identify all required top-level fields
2. Then fill in nested objects and arrays
3. Finally, add optional fields if the information is available
Double-check types, formats, and constraints before finalizing your output.
"""
        
        # Combine with user-provided system prompt if available
        if system_prompt:
            return f"{system_prompt}\n\n{enhancement}"
        else:
            return enhancement
    
    def _is_complex_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Determine if a schema is complex enough to warrant decomposition.
        
        Args:
            schema: The JSON schema
            
        Returns:
            True if the schema is complex, False otherwise
        """
        complexity_score = self._calculate_schema_complexity(schema)
        return complexity_score > 12  # Threshold for decomposition
    
    def _decompose_schema(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Decompose a complex schema into simpler components.
        
        Args:
            schema: The complex JSON schema
            
        Returns:
            Dictionary mapping component names to component schemas
        """
        components = {}
        
        # Only decompose object types with properties
        if "type" in schema and schema["type"] == "object" and "properties" in schema:
            # Group related properties into logical components
            ungrouped_props = {}
            
            # Extract property groups
            for prop_name, prop_schema in schema["properties"].items():
                # Check if this is a complex nested object
                if "type" in prop_schema and prop_schema["type"] == "object" and "properties" in prop_schema:
                    # This is a nested object that can be its own component
                    component_name = prop_name
                    components[component_name] = {
                        "type": "object",
                        "properties": prop_schema["properties"].copy(),
                        "required": prop_schema.get("required", [])
                    }
                else:
                    # Add to ungrouped properties
                    ungrouped_props[prop_name] = prop_schema
            
            # If we have ungrouped properties, create a base component
            if ungrouped_props:
                components["base"] = {
                    "type": "object",
                    "properties": ungrouped_props,
                    "required": [p for p in schema.get("required", []) if p in ungrouped_props]
                }
            
            # If no decomposition was possible, use the original schema
            if not components:
                components["entire"] = schema
        else:
            # Non-object types or schemas without properties can't be decomposed
            components["entire"] = schema
        
        return components
    
    def _create_component_prompt(
        self,
        input_text: str,
        schema: Dict[str, Any],
        component_name: str
    ) -> str:
        """
        Create a prompt focused on a specific schema component.
        
        Args:
            input_text: The original input text
            schema: The component schema
            component_name: Name of the component
            
        Returns:
            Component-specific prompt
        """
        # Format component name for prompt
        formatted_name = component_name.replace("_", " ").title()
        
        # Create focused prompt
        component_prompt = f"""
Extract the {formatted_name} information from the following text.
Focus only on details relevant to the {formatted_name} component.

{input_text}
"""
        
        return component_prompt
    
    def _combine_components(
        self,
        component_results: Dict[str, Any],
        original_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine decomposed component results into a unified result.
        
        Args:
            component_results: Results from individual components
            original_schema: The original schema
            
        Returns:
            Combined result conforming to the original schema
        """
        # For simple object types, just merge the components
        if "type" in original_schema and original_schema["type"] == "object":
            combined = {}
            
            # Merge all component results
            for component_data in component_results.values():
                if isinstance(component_data, dict):
                    combined.update(component_data)
            
            return combined
        else:
            # For non-object types, just return the first component
            # (this should rarely happen with decomposition)
            if component_results:
                return next(iter(component_results.values()))
            else:
                # Fallback to empty object if no components were generated
                return {}


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a client for Claude
    client = ComplexConstructClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Define a complex schema
    product_schema = {
        "type": "object",
        "properties": {
            "product_name": {"type": "string", "description": "The name of the product"},
            "price": {"type": "number", "minimum": 0, "description": "The price of the product"},
            "description": {"type": "string", "description": "A detailed description of the product"},
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of product features"
            },
            "specifications": {
                "type": "object",
                "properties": {
                    "dimensions": {"type": "string", "description": "Product dimensions"},
                    "weight": {"type": "string", "description": "Product weight"},
                    "materials": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Materials used in the product"
                    },
                    "compatibility": {
                        "type": "object",
                        "properties": {
                            "operating_systems": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Compatible operating systems"
                            },
                            "connectivity": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Connectivity options"
                            }
                        }
                    }
                },
                "description": "Technical specifications of the product"
            },
            "in_stock": {"type": "boolean", "description": "Whether the product is in stock"},
            "shipping": {
                "type": "object",
                "properties": {
                    "free_shipping": {"type": "boolean", "description": "Whether shipping is free"},
                    "estimated_delivery": {"type": "string", "description": "Estimated delivery time"},
                    "shipping_options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Name of shipping option"},
                                "price": {"type": "number", "description": "Price of shipping option"},
                                "delivery_time": {"type": "string", "description": "Delivery time for this option"}
                            }
                        },
                        "description": "Available shipping options"
                    }
                },
                "description": "Shipping information for the product"
            },
            "rating": {
                "type": "object",
                "properties": {
                    "average": {"type": "number", "minimum": 0, "maximum": 5, "description": "Average rating"},
                    "count": {"type": "integer", "minimum": 0, "description": "Number of ratings"}
                },
                "description": "Product rating information"
            }
        },
        "required": ["product_name", "price", "description", "in_stock"]
    }
    
    # Example text
    product_text = """
    The new Sony WH-1000XM5 Wireless Noise Cancelling Headphones are the latest in Sony's acclaimed 1000X series.
    Launched in May 2022, these premium headphones retail for $399.99 and come in black and silver color options.
    
    These headphones feature industry-leading noise cancellation with eight microphones and two processors for unprecedented
    noise reduction. The 30mm carbon fiber drivers deliver exceptional sound quality with enhanced clarity in the mid and high frequencies.
    
    Key features include:
    - Up to 30 hours of battery life with quick charging (3 hours from 3 minutes charge)
    - Adaptive Sound Control that adjusts ambient sound settings based on your location
    - Speak-to-Chat technology that automatically reduces volume during conversations
    - Precise Voice Pickup technology for crystal-clear hands-free calls
    - Multipoint connection allowing simultaneous connection to two Bluetooth devices
    
    The headphones weigh 250g and have dimensions of 8.85 x 6.73 x 3.03 inches when folded. They're constructed with
    soft-fit leather and a lightweight design for all-day comfort. The WH-1000XM5 is compatible with iOS and Android devices,
    and supports Bluetooth 5.2 with USB-C connectivity.
    
    These headphones are currently in stock and have a 4.7 out of 5-star rating based on 3,245 customer reviews.
    
    Shipping is free with an estimated delivery of 2-3 business days. Premium next-day delivery is available for $15,
    and standard shipping takes 5-7 days for $5.
    """
    
    # Generate structured data
    try:
        result = client.generate_structured(
            input_text=product_text,
            schema=product_schema,
            temperature=0.3
        )
        
        print("\nGenerated Structured Data:")
        print(json.dumps(result, indent=2))
        
        # Try with schema decomposition for an even more complex schema
        print("\nTrying with schema decomposition:")
        
        decomposed_result = client.generate_with_schema_decomposition(
            input_text=product_text,
            schema=product_schema,
            temperature=0.3
        )
        
        print("Decomposed Result:")
        print(json.dumps(decomposed_result, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    print(f"Request count: {metrics['request_count']}")
    print(f"Token count: {metrics['token_count']}")
    print(f"Error count: {metrics['error_count']}")