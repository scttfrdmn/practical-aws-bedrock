---
layout: page
title: Discovering and Managing AWS Bedrock Quotas
difficulty: intermediate
time_required: 30 minutes
---

# Discovering and Managing AWS Bedrock Quotas

This tutorial explains how to discover, monitor, and work within AWS Bedrock's quota limits using both AWS CLI and Python.

## Objective

By the end of this tutorial, you'll be able to:
1. Discover available quotas for AWS Bedrock models
2. Monitor your quota usage and detect throttling
3. Implement strategies to optimize within quota constraints
4. Set up visualization for quota monitoring

## Prerequisites

- AWS CLI configured with the 'aws' profile
- Python 3.8+ with boto3 installed
- AWS account with Bedrock access
- Basic understanding of AWS Bedrock services

## Understanding AWS Bedrock Quotas

Before we start, it's important to understand the types of quotas you'll encounter:

1. **Tokens Per Minute (TPM)** - Total input+output tokens processed per minute
2. **Requests Per Minute (RPM)** - Total API calls per minute
3. **Concurrent Requests** - Simultaneous requests in flight
4. **Context Window Limits** - Maximum tokens per individual request

## Step 1: Discovering Available Quotas

### Using AWS CLI

Let's start by listing all available quotas for AWS Bedrock:

```bash
aws service-quotas list-service-quotas \
  --service-code bedrock \
  --profile aws
```

This will return a JSON response with all quotas. Let's filter it to find Claude-specific quotas:

```bash
aws service-quotas list-service-quotas \
  --service-code bedrock \
  --profile aws \
  --query "Quotas[?contains(QuotaName, 'Claude')]"
```

To check for default quotas (service defaults, not your account's specific quotas):

```bash
aws service-quotas list-aws-default-service-quotas \
  --service-code bedrock \
  --profile aws \
  --query "Quotas[?contains(QuotaName, 'Claude')]"
```

### Using Python

Create a Python script to discover and display quotas:

```python
# quota_discovery.py
import boto3
from utils.profile_manager import get_profile

def discover_bedrock_quotas():
    # Use the configured profile (defaults to 'aws' for local testing)
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    client = session.client('service-quotas')
    
    # Get all Bedrock quotas
    response = client.list_service_quotas(ServiceCode='bedrock')
    
    # Group quotas by model family
    model_families = {}
    
    for quota in response['Quotas']:
        # Try to extract model family from quota name
        quota_name = quota['QuotaName']
        
        # Check for known model families
        for family in ['Claude', 'Llama', 'Titan', 'Cohere', 'Stable Diffusion']:
            if family in quota_name:
                if family not in model_families:
                    model_families[family] = []
                model_families[family].append(quota)
                break
        else:
            # General quotas that don't mention a specific model
            if 'General' not in model_families:
                model_families['General'] = []
            model_families['General'].append(quota)
    
    # Print results by model family
    for family, quotas in model_families.items():
        print(f"\n{family} Models:")
        print("-" * 50)
        
        for quota in quotas:
            print(f"Name: {quota['QuotaName']}")
            print(f"Value: {quota['Value']}")
            print(f"Adjustable: {quota['Adjustable']}")
            print(f"Quota Code: {quota['QuotaCode']}")
            print()

if __name__ == "__main__":
    discover_bedrock_quotas()
```

Run this script to get a nicely formatted overview of all quotas grouped by model family.

## Step 2: Checking Model-Specific Throughput Capacity

### Using AWS CLI

For more detailed information about a specific model's capacity:

```bash
aws bedrock get-foundation-model-throughput-capacity \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --profile aws
```

You can also list all provisioned throughputs:

```bash
aws bedrock list-provisioned-model-throughputs --profile aws
```

### Using Python

Let's create a function to check a model's throughput capacity:

```python
def check_model_capacity(model_id):
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock = session.client('bedrock')
    
    try:
        response = bedrock.get_foundation_model_throughput_capacity(
            modelId=model_id
        )
        
        print(f"Throughput capacity for {model_id}:")
        print(f"Status: {response.get('status', 'N/A')}")
        
        if 'capacityUtilization' in response:
            util = response['capacityUtilization']
            print(f"Utilization: {util}")
        
        if 'provisionedCapacity' in response:
            capacity = response['provisionedCapacity']
            print(f"Provisioned Capacity: {capacity}")
            
        return response
    except Exception as e:
        print(f"Error checking capacity for {model_id}: {str(e)}")
        return None
```

## Step 3: Monitoring Quota Usage and Throttling

### Using AWS CloudWatch via CLI

To monitor how often your requests are being throttled:

```bash
# Set the date range for the query
START_TIME=$(date -v-24H -u +"%Y-%m-%dT%H:%M:%SZ")
END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Check throttling events for a model
aws cloudwatch get-metric-statistics \
  --namespace AWS/Bedrock \
  --metric-name InvokeModelThrottled \
  --statistics Sum \
  --period 3600 \
  --start-time $START_TIME \
  --end-time $END_TIME \
  --dimensions Name=ModelId,Value=anthropic.claude-3-sonnet-20240229-v1:0 \
  --profile aws
```

Get successful invocation counts to compare:

```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/Bedrock \
  --metric-name InvokeModel \
  --statistics Sum \
  --period 3600 \
  --start-time $START_TIME \
  --end-time $END_TIME \
  --dimensions Name=ModelId,Value=anthropic.claude-3-sonnet-20240229-v1:0 \
  --profile aws
```

### Using Python with Visualization

Let's create a function that monitors and visualizes quota usage:

```python
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from utils.visualization_config import SVG_CONFIG

def monitor_quota_usage(model_id, hours=24, output_file="quota_usage.svg"):
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    cloudwatch = session.client('cloudwatch')
    
    end_time = datetime.datetime.utcnow()
    start_time = end_time - datetime.timedelta(hours=hours)
    
    # Get metrics for successful and throttled requests
    metrics = {}
    for metric_name in ['InvokeModel', 'InvokeModelThrottled']:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/Bedrock',
            MetricName=metric_name,
            Dimensions=[
                {'Name': 'ModelId', 'Value': model_id}
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour periods
            Statistics=['Sum']
        )
        metrics[metric_name] = response['Datapoints']
    
    # Process data for visualization
    data = []
    
    for point in metrics['InvokeModel']:
        data.append({
            'timestamp': point['Timestamp'],
            'count': point['Sum'],
            'type': 'Successful'
        })
    
    for point in metrics['InvokeModelThrottled']:
        data.append({
            'timestamp': point['Timestamp'],
            'count': point['Sum'],
            'type': 'Throttled'
        })
    
    if not data:
        print("No usage data found for the specified time period")
        return
    
    # Create dataframe and visualize
    df = pd.DataFrame(data)
    
    # Convert to pivot table for stacked bar chart
    pivot_df = df.pivot_table(
        index='timestamp', 
        columns='type', 
        values='count',
        fill_value=0
    ).sort_index()
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    pivot_df.plot(
        kind='bar', 
        stacked=True,
        color=['#2ca02c', '#d62728']  # Green for success, red for throttled
    )
    
    plt.title(f'API Usage for {model_id}', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Request Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Calculate throttle rate
    total_requests = pivot_df.sum().sum()
    throttled = pivot_df['Throttled'].sum() if 'Throttled' in pivot_df.columns else 0
    throttle_rate = (throttled / total_requests * 100) if total_requests > 0 else 0
    
    print(f"Summary for {model_id}:")
    print(f"Total Requests: {total_requests}")
    print(f"Throttled Requests: {throttled}")
    print(f"Throttle Rate: {throttle_rate:.2f}%")
    
    # Save as SVG
    plt.savefig(output_file, **SVG_CONFIG)
    print(f"Visualization saved to {output_file}")
```

## Step 4: Testing Quota Limits

Let's create a script to test the practical limits of a model's quotas:

```python
import boto3
import json
import time
import datetime
from utils.profile_manager import get_profile

def test_quota_limits(model_id, requests_per_batch=10, max_batches=5, delay_seconds=1):
    """
    Test the practical limits of a model's quota by gradually increasing load.
    This will help determine when throttling begins.
    
    WARNING: This may consume a significant portion of your quota!
    """
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    bedrock_runtime = session.client('bedrock-runtime')
    
    # Track results
    results = {
        'model_id': model_id,
        'test_time': datetime.datetime.utcnow().isoformat(),
        'batches': []
    }
    
    prompt = "Summarize this in one sentence: AWS provides cloud computing services."
    
    # Prepare request body based on model type
    if "anthropic" in model_id.lower():
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    elif "llama" in model_id.lower():
        request_body = {
            "prompt": f"<s>[INST] {prompt} [/INST]",
            "max_gen_len": 50
        }
    else:  # Default for other models
        request_body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 50
            }
        }
    
    print(f"Starting quota test for {model_id}")
    print(f"Will attempt {max_batches} batches with increasing load")
    
    for batch_num in range(1, max_batches + 1):
        batch_size = batch_num * requests_per_batch
        print(f"\nBatch {batch_num}: Testing {batch_size} requests...")
        
        batch_results = {
            'batch_number': batch_num,
            'requests_attempted': batch_size,
            'successful': 0,
            'throttled': 0,
            'errors': 0,
            'start_time': datetime.datetime.utcnow().isoformat()
        }
        
        # Execute all requests in the batch
        for i in range(batch_size):
            try:
                start_time = time.time()
                response = bedrock_runtime.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body)
                )
                
                # If we get here, the request was successful
                batch_results['successful'] += 1
                
            except bedrock_runtime.exceptions.ThrottlingException:
                # Request was throttled
                batch_results['throttled'] += 1
                
            except Exception as e:
                # Other error
                batch_results['errors'] += 1
                print(f"Error: {str(e)}")
            
            # Brief pause between individual requests
            time.sleep(0.1)
        
        batch_results['end_time'] = datetime.datetime.utcnow().isoformat()
        batch_results['throttle_rate'] = (batch_results['throttled'] / batch_size) * 100
        
        # Add batch results to overall results
        results['batches'].append(batch_results)
        
        # Print batch summary
        print(f"Batch {batch_num} Results:")
        print(f"  Successful: {batch_results['successful']}")
        print(f"  Throttled: {batch_results['throttled']}")
        print(f"  Errors: {batch_results['errors']}")
        print(f"  Throttle Rate: {batch_results['throttle_rate']:.2f}%")
        
        # If throttling exceeds 50%, we've likely hit the quota limit
        if batch_results['throttle_rate'] > 50:
            print(f"High throttle rate detected. Likely hit quota limit.")
            break
        
        # Pause between batches to avoid excessive throttling
        if batch_num < max_batches:
            print(f"Waiting {delay_seconds} seconds before next batch...")
            time.sleep(delay_seconds)
    
    # Calculate overall statistics
    total_requests = sum(batch['requests_attempted'] for batch in results['batches'])
    total_successful = sum(batch['successful'] for batch in results['batches'])
    total_throttled = sum(batch['throttled'] for batch in results['batches'])
    total_errors = sum(batch['errors'] for batch in results['batches'])
    
    results['summary'] = {
        'total_requests': total_requests,
        'total_successful': total_successful,
        'total_throttled': total_throttled,
        'total_errors': total_errors,
        'overall_throttle_rate': (total_throttled / total_requests) * 100 if total_requests > 0 else 0
    }
    
    print("\nTest Complete")
    print(f"Total Requests: {total_requests}")
    print(f"Total Successful: {total_successful}")
    print(f"Total Throttled: {total_throttled}")
    print(f"Overall Throttle Rate: {results['summary']['overall_throttle_rate']:.2f}%")
    
    # Save results to file
    with open(f"{model_id.replace('.', '_')}_quota_test.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
```

## Step 5: Creating a Quota Usage Dashboard

Let's create a simple SVG dashboard that shows quota usage across multiple models:

```python
def create_quota_dashboard(model_ids, hours=24, output_file="quota_dashboard.svg"):
    """Create a comprehensive dashboard of quota usage across multiple models"""
    profile_name = get_profile()
    session = boto3.Session(profile_name=profile_name)
    cloudwatch = session.client('cloudwatch')
    
    end_time = datetime.datetime.utcnow()
    start_time = end_time - datetime.timedelta(hours=hours)
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(model_ids), 1, figsize=(12, 5 * len(model_ids)))
    
    # Handle case with single model
    if len(model_ids) == 1:
        axes = [axes]
    
    for i, model_id in enumerate(model_ids):
        # Get metrics for successful and throttled requests
        data = []
        
        for metric_name in ['InvokeModel', 'InvokeModelThrottled']:
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/Bedrock',
                MetricName=metric_name,
                Dimensions=[
                    {'Name': 'ModelId', 'Value': model_id}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour periods
                Statistics=['Sum']
            )
            
            type_label = 'Successful' if metric_name == 'InvokeModel' else 'Throttled'
            
            for point in response['Datapoints']:
                data.append({
                    'timestamp': point['Timestamp'],
                    'count': point['Sum'],
                    'type': type_label
                })
        
        if not data:
            axes[i].text(0.5, 0.5, f"No data for {model_id}", 
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=axes[i].transAxes)
            continue
        
        # Create dataframe and plot
        df = pd.DataFrame(data)
        
        # Convert to pivot table for stacked bar chart
        pivot_df = df.pivot_table(
            index='timestamp', 
            columns='type', 
            values='count',
            fill_value=0
        ).sort_index()
        
        # Plot on the appropriate subplot
        pivot_df.plot(
            kind='bar', 
            stacked=True,
            ax=axes[i],
            color=['#2ca02c', '#d62728']  # Green for success, red for throttled
        )
        
        # Add labels and title
        axes[i].set_title(f'Usage for {model_id}', fontsize=14)
        axes[i].set_xlabel('Time', fontsize=10)
        axes[i].set_ylabel('Request Count', fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Calculate throttle rate
        total_requests = pivot_df.sum().sum()
        throttled = pivot_df['Throttled'].sum() if 'Throttled' in pivot_df.columns else 0
        throttle_rate = (throttled / total_requests * 100) if total_requests > 0 else 0
        
        # Add throttle rate annotation
        axes[i].annotate(
            f"Throttle Rate: {throttle_rate:.2f}%",
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    plt.tight_layout()
    
    # Save as SVG
    plt.savefig(output_file, **SVG_CONFIG)
    print(f"Dashboard saved to {output_file}")
    
    return output_file
```

## Step 6: Implementing a Quota-Aware Client

Finally, let's create a client that respects quota limits:

```python
class QuotaAwareBedrockClient:
    """A client wrapper that respects quota limits and handles throttling gracefully"""
    
    def __init__(self, model_id, max_retries=3, base_delay=1.0, profile_name=None):
        self.model_id = model_id
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Use the configured profile (defaults to 'aws' for local testing)
        self.profile_name = profile_name or get_profile()
        self.session = boto3.Session(profile_name=self.profile_name)
        self.bedrock_runtime = self.session.client('bedrock-runtime')
        
        # Token bucket for rate limiting
        self.request_tokens = 60  # Start with full bucket
        self.token_refresh_rate = 1  # Tokens per second
        self.last_refresh_time = time.time()
        
        # Statistics
        self.successful_requests = 0
        self.throttled_requests = 0
        self.retry_count = 0
    
    def _refresh_tokens(self):
        """Refresh tokens based on time passed"""
        current_time = time.time()
        elapsed = current_time - self.last_refresh_time
        
        # Add tokens based on elapsed time
        self.request_tokens = min(60, self.request_tokens + elapsed * self.token_refresh_rate)
        self.last_refresh_time = current_time
    
    def _backoff_time(self, attempt):
        """Calculate exponential backoff with jitter"""
        # 2^attempt * base_delay with 20% jitter
        backoff = (2 ** attempt) * self.base_delay
        jitter = backoff * 0.2 * (2 * np.random.random() - 1)  # ±20% jitter
        return backoff + jitter
    
    def invoke_model(self, request_body):
        """Invoke the model with quota awareness and retries"""
        self._refresh_tokens()
        
        # Check if we have enough tokens for this request
        if self.request_tokens < 1:
            sleep_time = (1 - self.request_tokens) / self.token_refresh_rate
            print(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            self.request_tokens = 1  # Now we have 1 token
        
        # Consume a token
        self.request_tokens -= 1
        
        # Try the request with retries
        for attempt in range(self.max_retries + 1):
            try:
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                # Success - update stats and return
                self.successful_requests += 1
                return response
                
            except self.bedrock_runtime.exceptions.ThrottlingException:
                # Request was throttled
                self.throttled_requests += 1
                
                # If we have retries left, back off and try again
                if attempt < self.max_retries:
                    backoff = self._backoff_time(attempt)
                    print(f"Request throttled. Retrying in {backoff:.2f} seconds (attempt {attempt+1}/{self.max_retries})")
                    self.retry_count += 1
                    time.sleep(backoff)
                else:
                    # No more retries
                    print(f"Request throttled after {self.max_retries} retries")
                    raise
            
            except Exception as e:
                # For other errors, don't retry
                print(f"Error invoking model: {str(e)}")
                raise
    
    def get_stats(self):
        """Return usage statistics"""
        return {
            "model_id": self.model_id,
            "successful_requests": self.successful_requests,
            "throttled_requests": self.throttled_requests,
            "retry_count": self.retry_count,
            "success_rate": (self.successful_requests / (self.successful_requests + self.throttled_requests)) * 100 
                if (self.successful_requests + self.throttled_requests) > 0 else 100
        }
```

## Conclusion

In this tutorial, you've learned how to:

1. Discover AWS Bedrock quotas using both CLI and Python
2. Monitor quota usage and visualize it with SVG charts
3. Test practical quota limits to determine throttling thresholds
4. Implement a quota-aware client that respects limits and handles throttling

These tools and techniques will help you optimize your use of AWS Bedrock while staying within quota constraints.

## Next Steps

- Set up CloudWatch Alarms to notify you when approaching quota limits
- Implement more advanced backoff strategies
- Consider requesting quota increases if needed
- Explore provisioned throughput options for high-volume usage