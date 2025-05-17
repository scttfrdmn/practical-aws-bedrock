"""
Validation utilities for AWS Bedrock Construct API outputs.

This module provides utilities for validating and testing structured outputs
from AWS Bedrock models against schemas and expected outputs.
"""

import json
import time
import logging
import jsonschema
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

from .schema_client import ConstructClient, SchemaValidationError


class ValidationResult:
    """Class representing the result of a validation operation."""
    
    def __init__(
        self,
        is_valid: bool,
        errors: List[str] = None,
        data: Any = None,
        execution_time: float = 0.0
    ):
        """
        Initialize a validation result.
        
        Args:
            is_valid: Whether the validation passed
            errors: List of validation error messages
            data: The validated data object
            execution_time: Time taken to execute the validation
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.data = data
        self.execution_time = execution_time
    
    def __bool__(self) -> bool:
        """
        Boolean conversion for ValidationResult.
        
        Returns:
            True if validation passed, False otherwise
        """
        return self.is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of the validation result
        """
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "data": self.data,
            "execution_time": self.execution_time
        }
    
    def __str__(self) -> str:
        """
        String representation of the validation result.
        
        Returns:
            String representation
        """
        status = "VALID" if self.is_valid else "INVALID"
        error_count = len(self.errors)
        
        if self.is_valid:
            return f"[{status}] Validation passed in {self.execution_time:.2f}s"
        else:
            return f"[{status}] Validation failed with {error_count} errors in {self.execution_time:.2f}s"


class SchemaValidator:
    """
    Validator for validating JSON objects against schemas.
    
    This class provides utilities for validating structured outputs
    against JSON schemas and applying custom validation rules.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the schema validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_against_schema(
        self,
        data: Any,
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate data against a JSON schema.
        
        Args:
            data: The data to validate
            schema: JSON schema to validate against
            
        Returns:
            ValidationResult with the validation outcome
        """
        start_time = time.time()
        
        try:
            # Validate schema itself
            jsonschema.validators.validator_for(schema).check_schema(schema)
            
            # Validate data against schema
            validator = jsonschema.Draft7Validator(schema)
            errors = list(validator.iter_errors(data))
            
            # Format error messages
            error_messages = []
            for error in errors:
                path = ".".join(str(path_part) for path_part in error.path) if error.path else "root"
                error_messages.append(f"At {path}: {error.message}")
            
            # Create validation result
            is_valid = len(error_messages) == 0
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            return ValidationResult(
                is_valid=is_valid,
                errors=error_messages,
                data=data,
                execution_time=execution_time
            )
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema validation error: {str(e)}"],
                data=data,
                execution_time=execution_time
            )
    
    def validate_with_custom_rules(
        self,
        data: Any,
        rules: List[Callable[[Any], Tuple[bool, Optional[str]]]]
    ) -> ValidationResult:
        """
        Validate data with custom validation rules.
        
        Args:
            data: The data to validate
            rules: List of validation functions that return (is_valid, error_message)
            
        Returns:
            ValidationResult with the validation outcome
        """
        start_time = time.time()
        
        error_messages = []
        
        # Apply each validation rule
        for rule_func in rules:
            try:
                is_valid, error_message = rule_func(data)
                if not is_valid and error_message:
                    error_messages.append(error_message)
            except Exception as e:
                error_messages.append(f"Error in validation rule: {str(e)}")
        
        # Create validation result
        is_valid = len(error_messages) == 0
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return ValidationResult(
            is_valid=is_valid,
            errors=error_messages,
            data=data,
            execution_time=execution_time
        )
    
    def validate_combined(
        self,
        data: Any,
        schema: Optional[Dict[str, Any]] = None,
        rules: Optional[List[Callable[[Any], Tuple[bool, Optional[str]]]]] = None
    ) -> ValidationResult:
        """
        Validate data with both schema and custom rules.
        
        Args:
            data: The data to validate
            schema: Optional JSON schema to validate against
            rules: Optional list of validation functions
            
        Returns:
            ValidationResult with the combined validation outcome
        """
        start_time = time.time()
        errors = []
        
        # Validate against schema if provided
        if schema:
            schema_result = self.validate_against_schema(data, schema)
            if not schema_result.is_valid:
                errors.extend([f"Schema validation: {err}" for err in schema_result.errors])
        
        # Apply custom rules if provided
        if rules:
            rules_result = self.validate_with_custom_rules(data, rules)
            if not rules_result.is_valid:
                errors.extend([f"Custom validation: {err}" for err in rules_result.errors])
        
        # Create validation result
        is_valid = len(errors) == 0
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            data=data,
            execution_time=execution_time
        )


class OutputTester:
    """
    Tester for generating and validating structured outputs.
    
    This class provides utilities for testing structured output generation
    with different inputs, schemas, and validation rules.
    """
    
    def __init__(
        self,
        client: ConstructClient,
        validator: Optional[SchemaValidator] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the output tester.
        
        Args:
            client: ConstructClient instance for generating outputs
            validator: Optional SchemaValidator instance
            logger: Optional logger instance
        """
        self.client = client
        self.validator = validator or SchemaValidator()
        self.logger = logger or logging.getLogger(__name__)
        
        # Track test results
        self.results = []
    
    def test_schema_generation(
        self,
        input_text: str,
        schema: Dict[str, Any],
        expected_fields: Optional[List[str]] = None,
        custom_rules: Optional[List[Callable[[Any], Tuple[bool, Optional[str]]]]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        test_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test structured output generation and validation.
        
        Args:
            input_text: The input text to process
            schema: JSON schema that defines the output structure
            expected_fields: Optional list of field names that must be present
            custom_rules: Optional list of validation functions
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            system_prompt: Optional system prompt
            test_name: Optional name for this test
            
        Returns:
            Dictionary with test results
        """
        # Generate the structured output
        test_start_time = time.time()
        
        try:
            # Generate structured output
            output = self.client.generate_structured(
                input_text=input_text,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )
            
            generation_success = True
            generation_error = None
        except Exception as e:
            output = None
            generation_success = False
            generation_error = str(e)
        
        generation_time = time.time() - test_start_time
        
        # Validate if generation succeeded
        if generation_success:
            # Validate against schema
            schema_result = self.validator.validate_against_schema(output, schema)
            
            # Check for expected fields
            field_errors = []
            if expected_fields:
                missing_fields = [field for field in expected_fields if field not in output]
                if missing_fields:
                    field_errors.append(f"Missing expected fields: {', '.join(missing_fields)}")
            
            # Apply custom validation rules
            custom_result = None
            if custom_rules:
                custom_result = self.validator.validate_with_custom_rules(output, custom_rules)
                
            # Determine overall validation result
            is_valid = schema_result.is_valid and not field_errors
            if custom_result:
                is_valid = is_valid and custom_result.is_valid
            
            # Combine all validation errors
            all_errors = schema_result.errors + field_errors
            if custom_result:
                all_errors.extend(custom_result.errors)
        else:
            # Generation failed
            is_valid = False
            all_errors = [f"Generation failed: {generation_error}"]
            schema_result = None
            custom_result = None
        
        # Record total test time
        test_end_time = time.time()
        test_time = test_end_time - test_start_time
        
        # Create test result
        result = {
            "test_name": test_name or f"Test-{len(self.results) + 1}",
            "generation_success": generation_success,
            "generation_time": generation_time,
            "validation_success": is_valid,
            "validation_errors": all_errors,
            "output": output,
            "total_time": test_time
        }
        
        # Add to results history
        self.results.append(result)
        
        return result
    
    def run_test_suite(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run a suite of generation and validation tests.
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            Dictionary with test suite results
        """
        suite_start_time = time.time()
        
        results = []
        
        # Run each test case
        for i, test_case in enumerate(test_cases):
            test_name = test_case.get("name", f"Test-{i + 1}")
            self.logger.info(f"Running test: {test_name}")
            
            result = self.test_schema_generation(
                input_text=test_case["input_text"],
                schema=test_case["schema"],
                expected_fields=test_case.get("expected_fields"),
                custom_rules=test_case.get("custom_rules"),
                max_tokens=test_case.get("max_tokens", 2000),
                temperature=test_case.get("temperature", 0.2),
                system_prompt=test_case.get("system_prompt"),
                test_name=test_name
            )
            
            results.append(result)
        
        # Calculate summary statistics
        suite_end_time = time.time()
        total_time = suite_end_time - suite_start_time
        
        passed_tests = sum(1 for r in results if r["validation_success"])
        failed_tests = len(results) - passed_tests
        
        summary = {
            "total_tests": len(results),
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": passed_tests / len(results) if results else 0,
            "total_time": total_time,
            "results": results
        }
        
        return summary
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all test results.
        
        Returns:
            List of test result dictionaries
        """
        return self.results


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a client for Claude
    client = ConstructClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Create a validator
    validator = SchemaValidator()
    
    # Test data
    test_data = {
        "name": "John Smith",
        "age": 35,
        "email": "john.smith@example.com",
        "skills": ["Python", "JavaScript", "AWS"]
    }
    
    # Test schema
    test_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string", "format": "email"},
            "skills": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age", "email"]
    }
    
    # Validate against schema
    result = validator.validate_against_schema(test_data, test_schema)
    print(f"Schema validation: {result}")
    
    # Custom validation rule
    def validate_adult(data):
        """Check if person is an adult (18+)"""
        if "age" in data and data["age"] >= 18:
            return True, None
        return False, "Person must be at least 18 years old"
    
    def validate_skill_count(data):
        """Check if person has at least 3 skills"""
        if "skills" in data and len(data["skills"]) >= 3:
            return True, None
        return False, "Person must have at least 3 skills"
    
    # Validate with custom rules
    custom_rules = [validate_adult, validate_skill_count]
    custom_result = validator.validate_with_custom_rules(test_data, custom_rules)
    print(f"Custom validation: {custom_result}")
    
    # Create an output tester
    tester = OutputTester(client=client, validator=validator)
    
    # Define test cases
    test_cases = [
        {
            "name": "Person Info Test",
            "input_text": "John Smith is a 35-year-old software engineer who can be reached at john.smith@example.com",
            "schema": test_schema,
            "expected_fields": ["name", "age", "email"],
            "custom_rules": custom_rules
        },
        {
            "name": "Event Info Test",
            "input_text": """The annual Tech Conference will be held on June 15-17, 2023 at the San Francisco Convention Center.
                           Tickets cost $299 for early bird registration until May 1st.""",
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "location": {"type": "string"},
                    "ticket_price": {"type": "number", "minimum": 0}
                },
                "required": ["title", "date", "location"]
            },
            "expected_fields": ["title", "date", "location", "ticket_price"]
        }
    ]
    
    # Run the test suite
    suite_results = tester.run_test_suite(test_cases)
    
    # Print results summary
    print("\nTest Suite Summary:")
    print(f"Total tests: {suite_results['total_tests']}")
    print(f"Passed tests: {suite_results['passed_tests']}")
    print(f"Failed tests: {suite_results['failed_tests']}")
    print(f"Pass rate: {suite_results['pass_rate'] * 100:.1f}%")
    print(f"Total time: {suite_results['total_time']:.2f}s")
    
    # Print details for each test
    for i, result in enumerate(suite_results['results']):
        print(f"\nTest {i+1}: {result['test_name']}")
        print(f"  Generation success: {result['generation_success']}")
        print(f"  Validation success: {result['validation_success']}")
        
        if not result['validation_success']:
            print(f"  Validation errors:")
            for error in result['validation_errors']:
                print(f"    - {error}")
        
        print(f"  Output: {json.dumps(result['output'], indent=2) if result['output'] else 'None'}")
        print(f"  Total time: {result['total_time']:.2f}s")