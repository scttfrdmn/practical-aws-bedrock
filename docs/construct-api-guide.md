---
layout: page
title: Structured Outputs with AWS Bedrock Construct API
---

# Structured Outputs with AWS Bedrock Construct API

The AWS Bedrock Construct API enables developers to generate structured, schema-conformant data from foundation models. This specialized API solves one of the most challenging problems in working with LLMs: getting consistent, parseable outputs in formats like JSON or XML.

## What is the Construct API?

The Construct API is designed for generating structured outputs from natural language inputs. It allows you to define a schema (like a JSON Schema) that constrains the model's responses to ensure they follow a specific structure, making them reliable for programmatic use.

![Construct API Overview](images/construct-api-diagram.svg)

## Key Benefits

- **Enforced schema validation** ensures outputs match your required structure
- **Improved output reliability** reduces parsing errors and inconsistencies
- **Standardized interface** across different model families
- **Reduced post-processing** eliminates complex output parsing and correction
- **Direct integration** with structured data workflows and databases
- **Reduced hallucinations** in structured fields through constraints

## When to Use the Construct API

The Construct API is ideal for:

1. **Data extraction** - Pulling structured information from unstructured text
2. **Form processing** - Converting form submissions into structured data
3. **API response generation** - Creating consistent API responses
4. **Database population** - Generating structured records for database insertion
5. **Classification tasks** - Organizing content into predefined categories

## Core Concepts

### JSON Schema

At the heart of the Construct API is JSON Schema, a vocabulary that allows you to annotate and validate JSON documents:

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "The person's full name"
    },
    "age": {
      "type": "integer",
      "minimum": 0,
      "description": "The person's age in years"
    },
    "email": {
      "type": "string",
      "format": "email",
      "description": "The person's email address"
    }
  },
  "required": ["name", "age"]
}
```

### Response Structure

The Construct API ensures responses conform to your schema:

```json
{
  "name": "John Smith",
  "age": 35,
  "email": "john.smith@example.com"
}
```

### Schema Enforcement

The API enforces:
- Property types (string, number, boolean, object, array)
- Required fields
- Value constraints (min/max values, patterns, formats)
- Nested objects and arrays with their own constraints

## Implementation Example

Here's a basic Python example using our library:

```python
from aws_bedrock_inference.construct import ConstructClient

# Create a client for structured outputs
client = ConstructClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Define a simple schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

# Generate structured output
result = client.generate_structured(
    "Extract information about John Doe, who is 42 years old and works at acme.com",
    schema=schema
)

print(result)
# Output: {"name": "John Doe", "age": 42, "email": "john.doe@acme.com"}

# Access specific fields
print(f"Name: {result['name']}")
print(f"Age: {result['age']}")
```

## Working with Complex Schemas

For more complex data extraction:

```python
from aws_bedrock_inference.construct import ComplexConstructClient
import json

# Create a client for complex schemas
client = ComplexConstructClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Define a more complex schema
product_schema = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "description": {"type": "string"},
        "features": {
            "type": "array",
            "items": {"type": "string"}
        },
        "specifications": {
            "type": "object",
            "properties": {
                "dimensions": {"type": "string"},
                "weight": {"type": "string"},
                "materials": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "in_stock": {"type": "boolean"},
        "rating": {
            "type": "number",
            "minimum": 0,
            "maximum": 5
        }
    },
    "required": ["product_name", "price", "description"]
}

# Test input
product_text = """
The new Acme Rocket Boots 3000 are the latest in personal propulsion technology.
Launched in 2023, these boots retail for $299.99 and come in red, blue, and silver colors.
Features include: adjustable thrust control, waterproof design, and shock absorption.
The boots weigh 2.5 lbs per pair and measure 12"x5"x8". They're made of carbon fiber, 
titanium alloy, and synthetic rubber. Currently in stock and have a 4.7/5 rating from customers.
"""

# Generate structured output
result = client.generate_structured(product_text, schema=product_schema)

# Pretty print the result
print(json.dumps(result, indent=2))
```

## Validation and Error Handling

Implement robust error handling for schema validation:

```python
try:
    result = client.generate_structured(input_text, schema=schema)
    # Process successful result
    process_data(result)
    
except SchemaValidationError as e:
    # Handle schema validation failures
    print(f"The generated output did not match the schema: {str(e)}")
    print(f"Partial result: {e.partial_result}")
    # Implement fallback or retry logic
    
except ModelErrorException as e:
    # Handle model-specific errors
    print(f"The model encountered an error: {str(e)}")
    
except Exception as e:
    # Handle other errors
    print(f"Unexpected error: {str(e)}")
```

## Data Extraction Pipeline

For processing multiple documents:

```python
from aws_bedrock_inference.construct import DataExtractionPipeline

# Create an extraction pipeline
pipeline = DataExtractionPipeline(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    schema_path="schemas/person_schema.json"
)

# Process a batch of documents
results = pipeline.process_documents([
    "John Smith is a 35-year-old software engineer living in Seattle.",
    "Jane Doe, 28, works as a data scientist in Boston and has a PhD.",
    "Alex Johnson (42) is a marketing director based in Austin, TX."
])

# Process the extracted data
for result in results:
    if result["status"] == "success":
        # Add to database, generate report, etc.
        process_structured_data(result["data"])
    else:
        # Handle extraction failures
        print(f"Failed to extract data: {result['error']}")
```

## Schema Design Best Practices

To get the best results from the Construct API:

1. **Start simple**: Begin with the minimum fields you need
2. **Add clear descriptions**: Include field descriptions to guide the model
3. **Use appropriate types**: Match data types to your expected values
4. **Include examples**: Provide example values when possible
5. **Set reasonable constraints**: Use minimum/maximum values, patterns, etc.
6. **Include fallbacks**: Use nullable fields for optional data
7. **Limit array items**: Avoid unbounded arrays
8. **Test with varied inputs**: Verify schema works across different text formats

## Common Schema Patterns

### Person Information

```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer", "minimum": 0},
    "occupation": {"type": "string"},
    "location": {"type": "string"},
    "contact": {
      "type": "object",
      "properties": {
        "email": {"type": "string", "format": "email"},
        "phone": {"type": "string", "pattern": "^[0-9\\-\\+\\s\\(\\)]+$"}
      }
    }
  },
  "required": ["name"]
}
```

### Article Summary

```json
{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "author": {"type": "string"},
    "publication_date": {"type": "string", "format": "date"},
    "summary": {"type": "string", "maxLength": 500},
    "key_points": {
      "type": "array",
      "items": {"type": "string"},
      "maxItems": 5
    },
    "categories": {
      "type": "array",
      "items": {"type": "string"},
      "maxItems": 3
    }
  },
  "required": ["title", "summary", "key_points"]
}
```

### Product Information

```json
{
  "type": "object",
  "properties": {
    "product_name": {"type": "string"},
    "price": {"type": "number", "minimum": 0},
    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY"]},
    "description": {"type": "string"},
    "features": {
      "type": "array",
      "items": {"type": "string"},
      "maxItems": 10
    },
    "availability": {"type": "boolean"},
    "rating": {"type": "number", "minimum": 0, "maximum": 5}
  },
  "required": ["product_name", "price", "description"]
}
```

## Error Handling Strategies

### Retry with Modified Schema

When schema validation fails, try a simpler schema:

```python
def extract_with_fallback(text, primary_schema, fallback_schema):
    """Try extraction with primary schema, fall back to simpler schema if needed."""
    try:
        # Try with primary (complex) schema
        result = client.generate_structured(text, schema=primary_schema)
        return {"data": result, "schema_used": "primary"}
    except SchemaValidationError:
        try:
            # Fall back to simpler schema
            result = client.generate_structured(text, schema=fallback_schema)
            return {"data": result, "schema_used": "fallback"}
        except Exception as e:
            return {"error": str(e), "schema_used": "none"}
```

### Progressive Schema Enhancement

Build up schema complexity gradually:

```python
def progressive_extraction(text, schemas):
    """
    Try increasingly complex schemas until one succeeds.
    schemas should be ordered from simplest to most complex.
    """
    results = []
    final_result = None
    
    for i, schema in enumerate(schemas):
        try:
            result = client.generate_structured(text, schema=schema)
            results.append({"level": i, "result": result, "success": True})
            final_result = result
        except Exception as e:
            results.append({"level": i, "error": str(e), "success": False})
            # Stop if we can't even match the simplest schema
            if i == 0:
                break
    
    return {
        "final_result": final_result,
        "attempts": results
    }
```

## Model-Specific Considerations

Different models have unique characteristics when used with the Construct API:

### Anthropic Claude

- Better adherence to complex schemas
- Stronger type enforcement
- More detailed error explanations
- Handles more fields and nested structures

### Amazon Titan

- Often faster for simple schemas
- Different error patterns
- May require more explicit schema descriptions

## Performance Optimization

To optimize structured data generation:

1. **Simplify schemas**: Use only the fields you need
2. **Use appropriate context**: Provide relevant context without excess information
3. **Implement caching**: Cache common extraction patterns
4. **Batch similar requests**: Process similar documents together
5. **Consider model selection**: Match model capabilities to schema complexity

## Quota Considerations

The Construct API is subject to the same AWS Bedrock quota limits:

1. **TPM (tokens per minute)**: Complex schemas and inputs consume more tokens
2. **RPM (requests per minute)**: Each API call counts toward your RPM limit

When planning structured data applications, consider:

- Complex schemas increase token usage
- Input text length significantly impacts token consumption
- Failed schema validations still consume quota

## Comparison with Other Methods

| Aspect | Construct API | Custom Parsing | Regular LLM Prompting |
|--------|---------------|---------------|-----------------------|
| Output Consistency | High | Medium | Low |
| Development Time | Low | High | Medium |
| Token Efficiency | Medium | Low | High |
| Error Handling | Built-in validation | Custom validation | Manual parsing |
| Flexibility | Schema-constrained | Custom rules | Unconstrained |
| Complexity Handling | Good with clear schema | Highly customizable | Inconsistent |

## Next Steps

- Explore [Construct API Examples](../src/construct) for implementation details
- Learn about [Schema Design Patterns](schema-design-patterns.md) for common use cases
- See [Data Extraction Techniques](data-extraction-techniques.md) for advanced usage
- Understand [Quota Management](quota-management.md) for scaling structured data generation