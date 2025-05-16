# Understanding and Managing AWS Bedrock Quota Limits

AWS Bedrock imposes various quota limits to ensure fair usage across all customers. This guide explores how to discover, monitor, and optimize your usage within these limits.

## Types of Quota Limits in AWS Bedrock

### 1. Tokens Per Minute (TPM)

- Limits the total number of tokens (both input and output) processed per minute
- Set individually for each foundation model
- Usually the most critical limit for high-throughput applications

### 2. Requests Per Minute (RPM)

- Restricts how many API calls you can make to a specific model per minute
- Independent of token count in each request
- Important for applications making many small requests

### 3. Concurrent Requests

- Limits the number of simultaneous requests to a model
- Particularly relevant for streaming and asynchronous operations

### 4. Maximum Input/Output Tokens

- Caps the maximum size of individual requests/responses
- Varies by model (e.g., Claude models have larger context windows than Llama 2)

## Discovering Quota Limits

### Using AWS CLI

To view your current quotas for all Bedrock services:

```bash
aws service-quotas list-service-quotas \
  --service-code bedrock \
  --profile aws
```

To get details about a specific quota:

```bash
aws service-quotas get-service-quota \
  --service-code bedrock \
  --quota-code L-12345678 \
  --profile aws
```

To view the default quotas for services:

```bash
aws service-quotas list-aws-default-service-quotas \
  --service-code bedrock \
  --profile aws
```

### Using AWS SDK (Python)

```python
import boto3
from utils.profile_manager import get_profile

# Use the configured profile (defaults to 'aws' for local testing)
profile_name = get_profile()
session = boto3.Session(profile_name=profile_name)
client = session.client('service-quotas')

# List all quotas for Bedrock
response = client.list_service_quotas(ServiceCode='bedrock')

# Print quota information
for quota in response['Quotas']:
    print(f"Name: {quota['QuotaName']}")
    print(f"Code: {quota['QuotaCode']}")
    print(f"Value: {quota['Value']}")
    print(f"Adjustable: {quota['Adjustable']}")
    print("---")
```

### Checking Model Throughput (TPM Capacity)

To view your provisioned throughput capacity for specific models:

```bash
aws bedrock list-provisioned-model-throughputs --profile aws
```

To check specific model throughput:

```bash
aws bedrock get-foundation-model-throughput-capacity \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --profile aws
```

## Monitoring Quota Usage

### CloudWatch Metrics

AWS Bedrock publishes usage metrics to CloudWatch:

```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/Bedrock \
  --metric-name InvokeModelClientErrors \
  --statistics Sum \
  --period 3600 \
  --start-time 2023-05-01T00:00:00Z \
  --end-time 2023-05-02T00:00:00Z \
  --dimensions Name=ModelId,Value=anthropic.claude-3-sonnet-20240229-v1:0 \
  --profile aws
```

Key metrics to monitor:
- `InvokeModel` (successful invocations)
- `InvokeModelClientErrors` (client errors)
- `InvokeModelUserErrors` (user errors)
- `InvokeModelThrottled` (requests throttled due to quota limits)

### Using Python to Monitor Usage

```python
import boto3
import datetime
from utils.profile_manager import get_profile

def monitor_bedrock_usage(model_id, hours=24):
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    cloudwatch = session.client('cloudwatch')
    
    end_time = datetime.datetime.utcnow()
    start_time = end_time - datetime.timedelta(hours=hours)
    
    # Get invocation counts
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/Bedrock',
        MetricName='InvokeModel',
        Dimensions=[
            {'Name': 'ModelId', 'Value': model_id}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,  # 1 hour periods
        Statistics=['Sum']
    )
    
    # Get throttling events
    throttled = cloudwatch.get_metric_statistics(
        Namespace='AWS/Bedrock',
        MetricName='InvokeModelThrottled',
        Dimensions=[
            {'Name': 'ModelId', 'Value': model_id}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,  # 1 hour periods
        Statistics=['Sum']
    )
    
    print(f"Usage statistics for {model_id} over the past {hours} hours:")
    print("Timestamp\t\tInvocations\tThrottled")
    print("-" * 50)
    
    # Process and display results
    datapoints = {point['Timestamp']: {'invocations': point['Sum']} 
                 for point in response['Datapoints']}
    
    for point in throttled['Datapoints']:
        if point['Timestamp'] in datapoints:
            datapoints[point['Timestamp']]['throttled'] = point['Sum']
        else:
            datapoints[point['Timestamp']] = {'invocations': 0, 'throttled': point['Sum']}
    
    # Sort by timestamp
    for timestamp in sorted(datapoints.keys()):
        data = datapoints[timestamp]
        invocations = data.get('invocations', 0)
        throttled_count = data.get('throttled', 0)
        print(f"{timestamp}\t{invocations:.0f}\t\t{throttled_count:.0f}")

# Example usage
monitor_bedrock_usage('anthropic.claude-3-sonnet-20240229-v1:0')
```

## Requesting Quota Increases

### Using AWS CLI

To request a quota increase:

```bash
aws service-quotas request-service-quota-increase \
  --service-code bedrock \
  --quota-code L-12345678 \
  --desired-value 100 \
  --profile aws
```

### Using AWS Management Console

1. Open the Service Quotas console
2. Search for "Amazon Bedrock"
3. Select the quota you want to increase
4. Click "Request quota increase"
5. Enter the desired value and justification

## Common Quota-Related Error Responses

When you hit a quota limit, you'll receive error responses:

```json
{
  "message": "Rate exceeded",
  "code": "ThrottlingException"
}
```

Or for token limits:

```json
{
  "message": "Token limit exceeded",
  "code": "ValidationException"
}
```

## Model-Specific Quota Considerations

Different foundation models have varying quota limits and requirements:

### Anthropic Claude Models

- Generally have higher token-per-minute quotas
- Support large context windows (up to 100K tokens)
- Quota limits are set per model version (Claude 3 Opus vs. Sonnet vs. Haiku)

### Meta Llama 2 Models

- Typically have lower default quotas than Claude
- Smaller context windows (4K-8K tokens)
- Lower token costs make them economical for certain workloads

### Amazon Titan Models

- Often have higher default quotas since they're Amazon's own models
- Great for users who need higher throughput without quota increase requests

### Image Generation Models

- Have separate quotas from text models
- Often limited by images per minute rather than tokens

## Optimizing Within Quota Limits

### Strategies for Managing TPM Limits

1. **Prompt Engineering**
   - Reduce input token length through concise prompts
   - Use efficient summarization techniques for long inputs

2. **Batching**
   - Combine multiple smaller requests into batches
   - Distribute workloads over time to stay under per-minute limits

3. **Request Scheduling**
   - Implement token bucket algorithms to pace requests
   - Use exponential backoff for retry mechanisms

4. **Model Selection**
   - Choose smaller models for simpler tasks (Claude Haiku vs. Opus)
   - Distribute workloads across multiple model families

5. **Streaming**
   - Use streaming APIs to get results faster while still using the same token quota
   - Improves user experience while working within the same limits

## Visualizing Quota Usage

```python
import matplotlib.pyplot as plt
import pandas as pd
import boto3
import datetime
from utils.profile_manager import get_profile
from utils.visualization_config import SVG_CONFIG

def visualize_quota_usage(model_id, hours=24, output_file="quota_usage.svg"):
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    cloudwatch = session.client('cloudwatch')
    
    end_time = datetime.datetime.utcnow()
    start_time = end_time - datetime.timedelta(hours=hours)
    
    # Get invocation counts
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/Bedrock',
        MetricName='InvokeModel',
        Dimensions=[
            {'Name': 'ModelId', 'Value': model_id}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,  # 1 hour periods
        Statistics=['Sum']
    )
    
    # Get throttling events
    throttled = cloudwatch.get_metric_statistics(
        Namespace='AWS/Bedrock',
        MetricName='InvokeModelThrottled',
        Dimensions=[
            {'Name': 'ModelId', 'Value': model_id}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,  # 1 hour periods
        Statistics=['Sum']
    )
    
    # Process data for visualization
    data = []
    for point in response['Datapoints']:
        data.append({
            'timestamp': point['Timestamp'],
            'invocations': point['Sum'],
            'type': 'Successful'
        })
    
    for point in throttled['Datapoints']:
        data.append({
            'timestamp': point['Timestamp'],
            'invocations': point['Sum'],
            'type': 'Throttled'
        })
    
    df = pd.DataFrame(data)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    if not df.empty:
        # Pivot data for stacked bar chart
        pivot_df = df.pivot_table(
            index='timestamp', 
            columns='type', 
            values='invocations',
            fill_value=0
        )
        
        # Sort by timestamp
        pivot_df = pivot_df.sort_index()
        
        # Create stacked bar chart
        pivot_df.plot(kind='bar', stacked=True, ax=plt.gca(),
                    color=['#2ca02c', '#d62728'])  # Green for success, red for throttled
        
        plt.title(f'API Usage for {model_id}', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Request Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save as SVG
        plt.savefig(output_file, **SVG_CONFIG)
        plt.close()
        
        return output_file
    else:
        print("No data available for the specified time period")
        return None

# Example usage
visualize_quota_usage('anthropic.claude-3-sonnet-20240229-v1:0', 
                    output_file='docs/images/claude_quota_usage.svg')
```

## Next Steps

- Implement a quota monitoring dashboard
- Set up CloudWatch alarms for approaching limits
- Create automated scaling mechanisms to distribute load
- Optimize prompts to reduce token usage
- Explore more advanced parallelization strategies

These techniques will help you maximize the value from AWS Bedrock while working within the established quota limits.