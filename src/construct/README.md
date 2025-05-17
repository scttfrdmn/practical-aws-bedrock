# AWS Bedrock Construct API Examples

This directory contains implementation examples for the AWS Bedrock Construct API, which enables generating structured data outputs from foundation models.

## Overview

The Construct API allows you to:
- Generate outputs in specific formats (JSON, XML, etc.)
- Enforce schema constraints on model outputs
- Improve reliability for data extraction tasks
- Standardize parsing and validation

## Directory Structure

```
construct/
├── basic/                 # Basic implementation examples
│   ├── json_output.py     # Generate JSON outputs
│   ├── schema_client.py   # Schema-based generation
│   └── validation.py      # Output validation utilities
│
├── advanced/              # Advanced implementation examples
│   ├── complex_schema.py  # Working with complex nested schemas
│   ├── streaming.py       # Streaming structured outputs
│   └── error_handling.py  # Handling schema validation errors
│
├── schemas/               # Example JSON schemas
│   ├── product.json       # Product information schema
│   ├── person.json        # Person information schema
│   └── event.json         # Event information schema
│
├── utils/                 # Shared utilities
│   ├── schema_loader.py   # Utilities for loading schemas
│   ├── response_parser.py # Parsing and processing responses
│   └── validators.py      # Custom validation functions
│
└── examples/              # Complete application examples
    ├── data_extraction.py # Extract structured data from text
    ├── form_parser.py     # Parse form data into structured format
    └── api_generator.py   # Generate API responses in correct format
```

## Usage Examples

### Basic JSON Generation

```python
from aws_bedrock_inference.construct.basic.json_output import ConstructClient

# Create a client
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
```

### Working with Complex Schemas

```python
from aws_bedrock_inference.construct.advanced.complex_schema import ComplexConstructClient
import json

# Create a client for complex schemas
client = ComplexConstructClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Load a complex schema
with open("aws_bedrock_inference/construct/schemas/product.json", "r") as f:
    product_schema = json.load(f)

# Generate structured output with the complex schema
product_info = client.generate_structured(
    """
    The new Acme Rocket Boots 3000 are the latest in personal propulsion technology.
    Launched in 2023, these boots come in red, blue, and silver colors. 
    They retail for $299.99 and include a 1-year warranty.
    Features include: adjustable thrust control, waterproof design, and shock absorption.
    Customer reviews give them 4.7 out of 5 stars based on 128 reviews.
    """,
    schema=product_schema
)

print(json.dumps(product_info, indent=2))
```

### Data Extraction Pipeline

```python
from aws_bedrock_inference.construct.examples.data_extraction import DataExtractionPipeline

# Create an extraction pipeline
pipeline = DataExtractionPipeline(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    schema_path="aws_bedrock_inference/construct/schemas/event.json"
)

# Process a batch of documents
results = pipeline.process_documents([
    "The annual tech conference will be held on June 15-17, 2023 at the San Jose Convention Center.",
    "Join us for the AI Summit on April 3rd at the Marriott Downtown. Tickets start at $199.",
    "The charity fundraiser is scheduled for December 12, 2023 from 6-10pm at The Grand Hotel."
])

# Print the extracted structured data
for result in results:
    print(json.dumps(result, indent=2))
```

## Implementation Status

- [  ] Basic JSON Generation - Not started
- [  ] Schema-based Generation - Not started
- [  ] Validation Utilities - Not started
- [  ] Complex Schema Handling - Not started
- [  ] Streaming Structured Outputs - Not started

## Next Steps

1. Implement basic JSON output generation
2. Add schema-based generation capabilities
3. Develop validation utilities
4. Create examples with complex schemas
5. Add streaming support for structured outputs

## Contributing

See the project [CONTRIBUTING](../../CONTRIBUTING.md) guidelines for information on how to contribute to this module.