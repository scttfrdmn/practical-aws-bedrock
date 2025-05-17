# Asynchronous Processing with AWS Bedrock

Asynchronous processing provides a powerful approach for handling large-scale, long-running foundation model inference tasks that exceed the limitations of synchronous and streaming methods. This is implemented through AWS Bedrock's `CreateModelInvocationJob` API.

## What is Asynchronous Processing?

In asynchronous processing, your application submits a job to AWS Bedrock, which then processes the request in the background. Instead of waiting for an immediate response, you receive a job ID that can be used to check status and retrieve results when processing completes. Results are stored in an S3 bucket that you specify.

![Asynchronous Processing Flow](../images/asynchronous-processing-diagram.svg)

## Key Characteristics

- **Non-blocking operation**: Submit jobs and continue other tasks
- **No timeout constraints**: Can process very large inputs and generate extensive outputs
- **S3 integration**: Results stored in your S3 bucket
- **Job management**: APIs for tracking status and retrieving results
- **Higher throughput potential**: Process many jobs in parallel
- **Same quota utilization**: Uses the same TPM quota as other methods

## When to Use Asynchronous Processing

Asynchronous processing is ideal for:

1. **Large-scale batch processing** - When processing many documents or requests
2. **Very long inputs** - Content that exceeds the context limits of synchronous APIs
3. **Extended generations** - When generating lengthy outputs like reports or books
4. **Background processing** - Tasks that don't require immediate user feedback
5. **Offline workflows** - Processing pipelines integrated with other AWS services

## When Not to Use Asynchronous Processing

Consider other methods when:

1. **Real-time responses are needed** - When users are waiting for immediate results
2. **Simple, short queries** - Where the overhead of job management isn't justified
3. **Interactive applications** - Where users expect immediate feedback
4. **Limited infrastructure** - When you don't have S3 buckets configured

## Quota Considerations

Asynchronous processing has important quota differences:

1. **TPM still applies**: Jobs consume the same token quota as other methods
2. **RPM is less critical**: Fewer job creation requests needed for large workloads
3. **Concurrent jobs limit**: There's a separate quota for concurrent job execution
4. **No timeout constraints**: Can process much larger inputs and outputs

This method is particularly effective for:
- Spreading throughput across time periods
- Handling very large documents
- Maximizing utilization of your quota allocations

## Implementation Example

Here's a basic Python example using our library:

```python
from aws_bedrock_inference import BedrockJobClient

# Create a job client for Claude
# Note: You must provide a valid S3 bucket with appropriate permissions
client = BedrockJobClient(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    output_s3_uri="s3://your-bucket/bedrock-outputs/"
)

# Create a job
try:
    prompt = "Analyze this lengthy document and summarize the key points..."
    
    job_id = client.create_job(
        prompt=prompt,
        max_tokens=2000,
        job_name="Document-Analysis-Job"
    )
    
    print(f"Created job: {job_id}")
    
    # Wait for job to complete and get result
    print("Waiting for job to complete...")
    result = client.wait_for_job(job_id)
    
    print("\nJob Result:")
    print(result["output"])
    
    # Print token usage if available
    if "total_tokens" in result:
        print(f"\nToken Usage:")
        print(f"Input tokens: {result.get('input_tokens')}")
        print(f"Output tokens: {result.get('output_tokens')}")
        print(f"Total tokens: {result.get('total_tokens')}")
    
except Exception as e:
    print(f"Error: {str(e)}")
```

## Batch Processing Implementation

For processing multiple items, our library provides a `BedrockBatchProcessor`:

```python
from aws_bedrock_inference import BedrockBatchProcessor

# Create a batch processor
processor = BedrockBatchProcessor(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    output_s3_uri="s3://your-bucket/bedrock-outputs/",
    max_concurrent_jobs=5  # Process 5 jobs in parallel
)

# Sample inputs
inputs = [
    {"prompt": "Analyze document A and extract key information..."},
    {"prompt": "Analyze document B and extract key information..."},
    {"prompt": "Analyze document C and extract key information..."},
    {"prompt": "Analyze document D and extract key information..."},
    {"prompt": "Analyze document E and extract key information..."}
]

# Define progress callback
def print_progress(current, total):
    print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

# Process batch
results = processor.process_batch(
    inputs=inputs,
    job_name_prefix="Document-Analysis",
    progress_callback=print_progress
)

# Process the results
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    if result.get("status") == "completed":
        print(f"Output: {result['output'][:100]}...")  # Print first 100 chars
    else:
        print(f"Failed: {result.get('error')}")
```

## Directory Processing

For processing files in a directory:

```python
# Process all text files in a directory
results = processor.process_directory(
    input_dir="./documents",
    output_dir="./results",
    file_pattern="*.txt",
    prompt_template="Analyze the following document and extract key information:\n\n{content}",
    system_prompt="You are an expert document analyzer. Extract all key information in a structured format."
)
```

## Job Status Monitoring

To check job status without waiting for completion:

```python
# Get job status
status = client.get_job_status(job_id)
print(f"Job status: {status['status']}")

# List recent jobs
jobs = client.list_jobs(max_results=10)
for job in jobs:
    print(f"Job ID: {job['job_id']}")
    print(f"Status: {job['status']}")
    print(f"Created: {job['created_at']}")
    
# Cancel a job
if client.cancel_job(job_id):
    print(f"Successfully cancelled job {job_id}")
else:
    print(f"Failed to cancel job {job_id}")
```

## S3 Integration Requirements

To use asynchronous processing, you need:

1. **S3 bucket**: A bucket where AWS Bedrock can store results
2. **IAM permissions**: AWS Bedrock must have permission to write to your bucket
3. **Cross-account access**: If needed, configure bucket policies for access

Example IAM policy snippet for your S3 bucket:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": [
                "s3:PutObject",
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket/bedrock-outputs/*"
        }
    ]
}
```

## Error Handling

Common errors with asynchronous processing:

1. **ValidationException**: Invalid job parameters or token limits
2. **ResourceNotFoundException**: Job ID not found
3. **AccessDeniedException**: S3 permissions issues
4. **ThrottlingException**: Exceeded job creation rate or concurrent job quota

Best practices for handling job errors:

```python
try:
    job_id = client.create_job(prompt="Your prompt here")
    result = client.wait_for_job(job_id)
    # Process success case
except Exception as e:
    if "ThrottlingException" in str(e):
        # Implement exponential backoff and retry
        time.sleep(backoff_time)
        # Retry job creation
    elif "AccessDeniedException" in str(e):
        # S3 permissions issue
        print("Check S3 bucket permissions")
    elif "ValidationException" in str(e):
        # Check job parameters
        print("Invalid job parameters")
    else:
        # Handle other errors
        print(f"Unexpected error: {str(e)}")
```

## Architectural Patterns

Common patterns for asynchronous processing:

### Queue-Based Architecture

```
[Client] → [SQS Queue] → [Lambda Worker] → [Bedrock Jobs] → [S3] → [Process Results]
```

### Event-Driven Processing

```
[S3 Upload] → [S3 Event] → [Lambda] → [Bedrock Job] → [S3 Result] → [SNS Notification]
```

### Batch Processing Pipeline

```
[Data Source] → [Batch Processor] → [Bedrock Jobs] → [S3 Results] → [Data Warehouse]
```

## Performance Optimization

To optimize asynchronous processing:

1. **Parallelization**: Process multiple jobs concurrently  
2. **Job size optimization**: Balance between many small jobs vs. fewer large jobs
3. **Polling strategy**: Implement efficient status checking
4. **Error retries**: Implement robust retry logic with backoff
5. **Monitoring**: Track job status and performance metrics

## Comparison with Other Methods

| Aspect | Asynchronous | Synchronous | Streaming |
|--------|--------------|-------------|-----------|
| Response Time | Minutes/hours | Seconds | Seconds (incremental) |
| Input Size | Largest | Limited | Limited |
| Output Size | Largest | Limited | Limited |
| Timeout Risk | None | High | Medium |
| Implementation | Complex | Simple | Moderate |
| Infrastructure | S3 required | None | None |
| Best For | Batch processing, large inputs/outputs | Simple Q&A | Interactive chat |

## Next Steps

- Learn about [Synchronous Inference](synchronous.md) for simpler implementations
- Explore [Streaming Inference](streaming.md) for real-time response generation
- Understand [Quota Management](../quota-management.md) for optimizing throughput
- See [Batch Processing Patterns](../batch-processing-patterns.md) for advanced scenarios