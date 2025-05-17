# Building High-Throughput Processing Pipelines with AWS Bedrock

This guide covers strategies and architectural patterns for building high-throughput processing pipelines with AWS Bedrock. By combining different inference methods, quota management techniques, and AWS services, you can create scalable pipelines for efficiently processing large volumes of requests.

## Understanding Throughput Constraints

AWS Bedrock imposes service quotas that limit your processing capacity:

1. **Tokens Per Minute (TPM)**: Total input + output tokens processed per minute
2. **Requests Per Minute (RPM)**: Total API calls per minute
3. **Concurrent Requests**: Maximum number of simultaneous jobs (for asynchronous processing)

To build a high-throughput pipeline, you must optimize within these constraints while managing:
- Latency requirements
- Processing order
- Error handling
- Cost efficiency

## Architectural Patterns for High Throughput

### Pattern 1: Queue-Based Processing

![Queue-Based Processing](images/queue-based-pipeline.svg)

```
[Producers] → [SQS Queue] → [Lambda Consumers] → [Bedrock] → [Processing] → [Results Store]
```

**Key components:**
- **SQS Queue**: Buffers incoming requests to avoid throttling
- **Lambda Consumers**: Pull from queue at controlled rates
- **Bedrock Clients**: Manage quota consumption
- **DynamoDB**: Store results and processing state

**Implementation:**

```python
# Lambda consumer function
def lambda_handler(event, context):
    # Get messages from SQS
    sqs_messages = event['Records']
    
    # Initialize quota-aware client
    client = QuotaAwareBedrockClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        max_rpm=60,  # Adjust based on your quota
        max_tpm=100000  # Adjust based on your quota
    )
    
    # Process each message
    for message in sqs_messages:
        try:
            # Parse message
            body = json.loads(message['body'])
            request_id = body['request_id']
            prompt = body['prompt']
            
            # Invoke model (will wait if quota limits reached)
            response = client.invoke(
                prompt=prompt,
                max_tokens=body.get('max_tokens', 500),
                wait_for_quota=True
            )
            
            # Store result
            store_result(request_id, response)
            
        except Exception as e:
            # Handle errors and potentially return message to queue
            handle_error(message, str(e))
```

**Advantages:**
- Decouples producers from consumers
- Automatically handles backpressure
- Can scale horizontally with multiple consumers
- Provides built-in retry mechanisms

### Pattern 2: Asynchronous Job Pipeline

![Asynchronous Job Pipeline](images/async-job-pipeline.svg)

```
[Input Bucket] → [Lambda Trigger] → [Bedrock Jobs] → [Output Bucket] → [Processing Lambda] → [Results]
```

**Key components:**
- **S3 Input Bucket**: Stores input documents/prompts
- **Lambda Trigger**: Creates Bedrock jobs when new files arrive
- **Bedrock Asynchronous Jobs**: Process inputs without timeout constraints
- **S3 Output Bucket**: Collects job results
- **Processing Lambda**: Transforms and routes final results

**Implementation:**

```python
# Lambda trigger for new S3 objects
def lambda_handler(event, context):
    # Get input file details
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Initialize job client
    client = BedrockJobClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        output_s3_uri=f"s3://output-bucket/results/"
    )
    
    # Read input file
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    
    # Create job with metadata
    job_id = client.create_job(
        prompt=f"Process the following document: {content}",
        max_tokens=1000,
        job_name=f"process-{key}",
        tags={
            "source_bucket": bucket,
            "source_key": key
        }
    )
    
    # Store job mapping for tracking
    store_job_tracking(job_id, bucket, key)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'job_id': job_id,
            'source': f"{bucket}/{key}"
        })
    }
```

**Advantages:**
- Handles very large inputs and outputs
- No timeout constraints
- Built for batch processing
- Scales to high throughput with managed concurrency

### Pattern 3: Map-Reduce Processing

![Map-Reduce Processing](images/map-reduce-pipeline.svg)

```
[Input Documents] → [Document Chunker] → [Parallel Inference] → [Result Aggregator] → [Final Output]
```

**Key components:**
- **Document Chunker**: Splits large inputs into processable pieces
- **Parallel Inference**: Processes chunks in parallel
- **Result Aggregator**: Combines chunk results into final output
- **Quota Manager**: Ensures parallel processing stays within limits

**Implementation:**

```python
class MapReduceProcessor:
    """Process large documents by splitting into chunks and aggregating results."""
    
    def __init__(
        self, 
        model_id, 
        max_concurrent=5,
        chunk_size=1000,
        overlap=100
    ):
        self.model_id = model_id
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Create client
        self.client = QuotaAwareBedrockClient(
            model_id=model_id,
            max_rpm=60,
            max_tpm=100000
        )
    
    def process_document(self, document, instruction):
        """Process large document with map-reduce approach."""
        # Split document into chunks
        chunks = self._split_into_chunks(document)
        
        # Process chunks in parallel
        chunk_results = self._process_chunks(chunks, instruction)
        
        # Aggregate results
        final_result = self._aggregate_results(chunk_results, instruction)
        
        return final_result
    
    def _split_into_chunks(self, document):
        """Split document into overlapping chunks."""
        words = document.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i+self.chunk_size])
            chunks.append({
                'id': len(chunks),
                'text': chunk,
                'start_idx': i
            })
        
        return chunks
    
    def _process_chunks(self, chunks, instruction):
        """Process chunks in parallel with throttling."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = []
            
            for chunk in chunks:
                prompt = f"{instruction}\n\nCHUNK {chunk['id']} (of {len(chunks)}):\n{chunk['text']}"
                future = executor.submit(self._process_single_chunk, prompt, chunk['id'])
                futures.append(future)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        # Sort by chunk ID
        results.sort(key=lambda x: x['chunk_id'])
        return results
    
    def _process_single_chunk(self, prompt, chunk_id):
        """Process a single chunk with quota awareness."""
        try:
            response = self.client.invoke(
                prompt=prompt,
                max_tokens=500,
                wait_for_quota=True
            )
            
            return {
                'chunk_id': chunk_id,
                'result': response['output'],
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'chunk_id': chunk_id,
                'error': str(e),
                'status': 'error'
            }
    
    def _aggregate_results(self, chunk_results, instruction):
        """Aggregate chunk results into final output."""
        # Extract successful results
        successful_results = [r['result'] for r in chunk_results if r['status'] == 'success']
        
        # Count errors
        error_count = sum(1 for r in chunk_results if r['status'] == 'error')
        
        # If too many errors, return failure
        if error_count > len(chunk_results) * 0.25:  # More than 25% failed
            return {
                'status': 'failed',
                'reason': f"Too many chunk errors ({error_count}/{len(chunk_results)})",
                'partial_results': successful_results
            }
        
        # Create aggregation prompt
        aggregation_prompt = f"""
        {instruction}
        
        Below are the results from processing the document in chunks. 
        Please combine these into a coherent final result:
        
        {json.dumps(successful_results, indent=2)}
        """
        
        # Invoke model for final aggregation
        try:
            response = self.client.invoke(
                prompt=aggregation_prompt,
                max_tokens=1000,
                wait_for_quota=True
            )
            
            return {
                'status': 'success',
                'result': response['output'],
                'chunk_count': len(chunk_results),
                'error_count': error_count
            }
            
        except Exception as e:
            return {
                'status': 'aggregation_failed',
                'reason': str(e),
                'partial_results': successful_results
            }
```

**Advantages:**
- Handles documents of any size
- Maximizes parallel processing
- Works within token context limits
- Provides fault tolerance

### Pattern 4: Streaming Aggregation

![Streaming Aggregation](images/streaming-aggregation.svg)

```
[Input Stream] → [Processor Lambda] → [Bedrock Streaming] → [Kinesis] → [Aggregator] → [Output]
```

**Key components:**
- **Input Stream**: Continuous feed of items to process
- **Streaming Inference**: Processes items with streaming responses
- **Kinesis Data Stream**: Captures streaming chunks
- **Aggregator**: Reconstructs and processes complete responses

**Implementation:**

```python
# In this pattern, we process a stream of inputs and aggregate the streamed results

# Producer function (simplified)
def stream_processor(stream_name, input_batch):
    """Process a batch of inputs using streaming inference."""
    client = BedrockStreamingClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    kinesis_client = boto3.client('kinesis')
    
    for item in input_batch:
        item_id = item['id']
        prompt = item['prompt']
        
        # Stream the response, sending chunks to Kinesis
        try:
            chunk_count = 0
            for chunk in client.invoke_stream(prompt=prompt, max_tokens=500):
                # Send chunk to Kinesis stream
                kinesis_client.put_record(
                    StreamName=stream_name,
                    Data=json.dumps({
                        'item_id': item_id,
                        'chunk_number': chunk_count,
                        'content': chunk,
                        'is_final': False
                    }),
                    PartitionKey=item_id
                )
                chunk_count += 1
            
            # Mark the end of the stream for this item
            kinesis_client.put_record(
                StreamName=stream_name,
                Data=json.dumps({
                    'item_id': item_id,
                    'chunk_number': chunk_count,
                    'content': '',
                    'is_final': True
                }),
                PartitionKey=item_id
            )
            
        except Exception as e:
            # Send error marker
            kinesis_client.put_record(
                StreamName=stream_name,
                Data=json.dumps({
                    'item_id': item_id,
                    'error': str(e),
                    'is_final': True
                }),
                PartitionKey=item_id
            )

# Consumer function (simplified)
def stream_consumer(event, context):
    """Process Kinesis records to aggregate streaming responses."""
    # Initialize aggregation storage
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('StreamingResponseAggregation')
    
    # Process Kinesis records
    for record in event['Records']:
        # Decode and parse the data
        payload = json.loads(base64.b64decode(record['kinesis']['data']))
        
        item_id = payload['item_id']
        
        if 'error' in payload:
            # Handle error case
            table.update_item(
                Key={'item_id': item_id},
                UpdateExpression="SET processing_status = :status, error_message = :error",
                ExpressionAttributeValues={
                    ':status': 'failed',
                    ':error': payload['error']
                }
            )
            continue
        
        # Regular chunk processing
        is_final = payload.get('is_final', False)
        
        if not is_final:
            # Append chunk to the item's content
            table.update_item(
                Key={'item_id': item_id},
                UpdateExpression="SET content = list_append(if_not_exists(content, :empty_list), :chunk)",
                ExpressionAttributeValues={
                    ':empty_list': [],
                    ':chunk': [payload['content']]
                }
            )
        else:
            # Mark processing as complete
            table.update_item(
                Key={'item_id': item_id},
                UpdateExpression="SET processing_status = :status, completion_time = :time",
                ExpressionAttributeValues={
                    ':status': 'completed',
                    ':time': datetime.datetime.now().isoformat()
                }
            )
```

**Advantages:**
- Enables real-time processing of streaming data
- Provides immediate feedback for long-running operations
- Works well with user-facing applications
- Handles large volumes while maintaining responsiveness

## Quota Management Strategies

Effective quota management is essential for high-throughput pipelines:

### Strategy 1: Token Bucket Rate Limiting

Implement token bucket algorithm to control request rates:

```python
class TokenBucketLimiter:
    """Control request rates using token bucket algorithm."""
    
    def __init__(self, capacity, refill_rate):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens the bucket can hold
            refill_rate: Tokens per second to refill
        """
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate)
        self.last_refill_time = time.time()
        self.lock = threading.RLock()
    
    def consume(self, tokens=1.0, block=True):
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            block: Whether to wait for tokens to be available
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            if not block:
                return False
        
        # If blocking, wait until enough tokens are available
        while True:
            sleep_time = 0.1  # Sleep in small increments
            time.sleep(sleep_time)
            
            with self.lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        
        if elapsed > 0:
            new_tokens = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill_time = now
```

### Strategy 2: Adaptive Throttling

Adjust request rates based on observed throttling:

```python
class AdaptiveThrottling:
    """Dynamically adjust request rates based on throttling feedback."""
    
    def __init__(self, initial_rate, min_rate=0.1, max_rate=10.0):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.backoff_factor = 0.8  # Reduce by 20% on throttling
        self.recovery_factor = 1.1  # Increase by 10% on success
        self.consecutive_successes = 0
        self.consecutive_throttles = 0
        self.success_threshold = 10  # Attempts before increasing rate
        self.lock = threading.Lock()
    
    def on_success(self):
        """Called when request succeeds."""
        with self.lock:
            self.consecutive_throttles = 0
            self.consecutive_successes += 1
            
            # Increase rate after consecutive successes
            if self.consecutive_successes >= self.success_threshold:
                self.current_rate = min(self.max_rate, self.current_rate * self.recovery_factor)
                self.consecutive_successes = 0
    
    def on_throttle(self):
        """Called when request is throttled."""
        with self.lock:
            self.consecutive_successes = 0
            self.consecutive_throttles += 1
            
            # Reduce rate immediately on throttling
            self.current_rate = max(self.min_rate, self.current_rate * self.backoff_factor)
    
    def get_delay(self):
        """Calculate the delay between requests based on current rate."""
        with self.lock:
            return 1.0 / self.current_rate
```

### Strategy 3: Multi-Region Distribution

Distribute requests across regions to increase total quota:

```python
class MultiRegionClient:
    """Distribute requests across multiple regions to increase throughput."""
    
    def __init__(self, model_id, regions=None):
        self.model_id = model_id
        self.regions = regions or ["us-west-2", "us-east-1", "eu-west-1"]
        
        # Create clients for each region
        self.clients = {}
        for region in self.regions:
            self.clients[region] = BedrockClient(
                model_id=model_id,
                region_name=region
            )
        
        # Round-robin counter
        self.next_region_index = 0
        self.lock = threading.RLock()
    
    def invoke(self, prompt, **kwargs):
        """Invoke a model, distributing across regions."""
        with self.lock:
            # Select region using round-robin
            region = self.regions[self.next_region_index]
            
            # Update counter for next call
            self.next_region_index = (self.next_region_index + 1) % len(self.regions)
        
        # Use client for selected region
        client = self.clients[region]
        return client.invoke(prompt=prompt, **kwargs)
```

### Strategy 4: Priority-Based Queuing

Process requests based on priority:

```python
class PriorityQueue:
    """Process requests based on priority."""
    
    def __init__(self, max_concurrent_tasks=5):
        # Create queues for different priority levels
        self.high_priority = queue.Queue()
        self.medium_priority = queue.Queue()
        self.low_priority = queue.Queue()
        
        # Thread pool for processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.active_tasks = 0
        self.lock = threading.RLock()
        
        # Start processing thread
        self.running = True
        self.process_thread = threading.Thread(target=self._process_queues)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def add_task(self, priority, task_function, *args, **kwargs):
        """Add a task to the appropriate priority queue."""
        task = (task_function, args, kwargs)
        
        if priority == "high":
            self.high_priority.put(task)
        elif priority == "medium":
            self.medium_priority.put(task)
        else:
            self.low_priority.put(task)
    
    def _process_queues(self):
        """Process tasks from queues in priority order."""
        while self.running:
            next_task = None
            
            # Check queues in priority order
            if not self.high_priority.empty():
                next_task = self.high_priority.get()
            elif not self.medium_priority.empty():
                next_task = self.medium_priority.get()
            elif not self.low_priority.empty():
                next_task = self.low_priority.get()
            
            if next_task:
                # Wait if we're at max concurrency
                while self.active_tasks >= self.executor.max_workers:
                    time.sleep(0.1)
                
                # Submit task to thread pool
                with self.lock:
                    self.active_tasks += 1
                
                func, args, kwargs = next_task
                self.executor.submit(self._execute_task, func, args, kwargs)
            else:
                # No tasks available, sleep briefly
                time.sleep(0.1)
    
    def _execute_task(self, func, args, kwargs):
        """Execute a task and update active count."""
        try:
            func(*args, **kwargs)
        finally:
            with self.lock:
                self.active_tasks -= 1
    
    def shutdown(self):
        """Stop processing and shutdown."""
        self.running = False
        self.process_thread.join(timeout=1.0)
        self.executor.shutdown()
```

## Monitoring and Visibility

Implement monitoring to track pipeline performance:

```python
class ThroughputMonitor:
    """Monitor throughput and quota usage."""
    
    def __init__(self, model_id, window_size=60):
        self.model_id = model_id
        self.window_size = window_size  # seconds
        
        # Metrics storage
        self.request_timestamps = []
        self.token_counts = []
        self.error_counts = {'throttling': 0, 'validation': 0, 'other': 0}
        
        self.lock = threading.RLock()
    
    def record_request(self, token_count=0, error=None):
        """Record a request and its outcome."""
        now = time.time()
        
        with self.lock:
            # Add to history
            self.request_timestamps.append(now)
            
            if error:
                # Classify error
                if "ThrottlingException" in str(error):
                    self.error_counts['throttling'] += 1
                elif "ValidationException" in str(error):
                    self.error_counts['validation'] += 1
                else:
                    self.error_counts['other'] += 1
            else:
                # Record tokens for successful requests
                self.token_counts.append(token_count)
            
            # Prune old entries
            cutoff = now - self.window_size
            
            while self.request_timestamps and self.request_timestamps[0] < cutoff:
                self.request_timestamps.pop(0)
                if self.token_counts:
                    self.token_counts.pop(0)
    
    def get_metrics(self):
        """Get current throughput metrics."""
        with self.lock:
            # Calculate requests per minute
            current_rpm = len(self.request_timestamps) * (60.0 / self.window_size)
            
            # Calculate tokens per minute
            current_tpm = sum(self.token_counts) * (60.0 / self.window_size)
            
            return {
                "requests_per_minute": current_rpm,
                "tokens_per_minute": current_tpm,
                "error_counts": self.error_counts.copy(),
                "request_count": len(self.request_timestamps),
                "success_rate": (len(self.token_counts) / max(1, len(self.request_timestamps))) * 100
            }
```

## Cost Optimization Strategies

Optimize costs while maintaining throughput:

1. **Right-size models**: Use the smallest model that meets your needs
2. **Prompt engineering**: Reduce token usage through efficient prompts
3. **Caching**: Cache common responses to avoid redundant calls
4. **Batching**: Combine similar requests to reduce overhead
5. **Asynchronous jobs**: Use for larger requests to avoid retries

## Error Handling in High-Throughput Pipelines

Implement robust error handling:

```python
def process_with_retry_and_dlq(item, bedrock_client, dlq_url):
    """Process an item with retry and dead-letter queue for failures."""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Process the item
            result = bedrock_client.invoke(prompt=item['prompt'])
            return result
            
        except Exception as e:
            retry_count += 1
            error_string = str(e)
            
            # Check error type
            if "ThrottlingException" in error_string:
                # Exponential backoff for throttling
                sleep_time = (2 ** retry_count) * 0.5
                time.sleep(sleep_time)
                
            elif "ValidationException" in error_string:
                # Don't retry validation errors
                send_to_dlq(item, error_string, dlq_url)
                return None
                
            elif retry_count >= max_retries:
                # Exhausted retries
                send_to_dlq(item, error_string, dlq_url)
                return None
                
            else:
                # Other errors, simple retry with delay
                time.sleep(1.0)

def send_to_dlq(item, error, dlq_url):
    """Send failed item to dead-letter queue."""
    sqs_client = boto3.client('sqs')
    
    message = {
        'item': item,
        'error': error,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    sqs_client.send_message(
        QueueUrl=dlq_url,
        MessageBody=json.dumps(message)
    )
```

## Putting It All Together: Complete Pipeline Example

Here's a comprehensive example combining multiple strategies:

```python
class BedRockPipeline:
    """High-throughput processing pipeline for AWS Bedrock."""
    
    def __init__(
        self,
        model_id,
        max_concurrency=10,
        input_queue_url=None,
        output_queue_url=None,
        dlq_url=None,
        regions=None
    ):
        self.model_id = model_id
        self.max_concurrency = max_concurrency
        self.input_queue_url = input_queue_url
        self.output_queue_url = output_queue_url
        self.dlq_url = dlq_url
        
        # Initialize multi-region client
        self.client = MultiRegionClient(model_id, regions)
        
        # Initialize rate limiting
        self.rpm_limiter = TokenBucketLimiter(60, 1.0)  # 60 RPM
        self.tpm_limiter = TokenBucketLimiter(100000, 1666.67)  # 100K TPM
        
        # Initialize adaptive throttling
        self.throttler = AdaptiveThrottling(initial_rate=0.5)  # 0.5 RPS = 30 RPM
        
        # Initialize throughput monitor
        self.monitor = ThroughputMonitor(model_id)
        
        # Initialize priority queue
        self.queue = PriorityQueue(max_concurrent_tasks=max_concurrency)
        
        # SQS client
        self.sqs = boto3.client('sqs')
    
    def start(self, run_time=None):
        """
        Start the pipeline.
        
        Args:
            run_time: Optional runtime in seconds, or None to run indefinitely
        """
        self.running = True
        self.start_time = time.time()
        
        try:
            while self.running:
                # Check if runtime exceeded
                if run_time and (time.time() - self.start_time) > run_time:
                    self.running = False
                    break
                
                # Get messages from queue
                if self.input_queue_url:
                    self._process_from_queue()
                
                # Sleep briefly to avoid tight loop
                time.sleep(0.1)
                
                # Log metrics periodically (every 60 seconds)
                if int(time.time()) % 60 == 0:
                    metrics = self.monitor.get_metrics()
                    print(f"Current metrics: RPM={metrics['requests_per_minute']:.1f}, "
                          f"TPM={metrics['tokens_per_minute']:.1f}, "
                          f"Success Rate={metrics['success_rate']:.1f}%")
                
        except KeyboardInterrupt:
            print("Pipeline stopped by user")
        finally:
            self.queue.shutdown()
    
    def _process_from_queue(self):
        """Get and process messages from SQS queue."""
        try:
            # Receive messages
            response = self.sqs.receive_message(
                QueueUrl=self.input_queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5
            )
            
            if 'Messages' not in response:
                return
            
            for message in response['Messages']:
                # Parse message
                try:
                    body = json.loads(message['Body'])
                    priority = body.get('priority', 'medium')
                    
                    # Add to priority queue
                    self.queue.add_task(
                        priority, 
                        self._process_item,
                        body,
                        message['ReceiptHandle']
                    )
                    
                except Exception as e:
                    print(f"Error parsing message: {str(e)}")
                    # Delete malformed message
                    self.sqs.delete_message(
                        QueueUrl=self.input_queue_url,
                        ReceiptHandle=message['ReceiptHandle']
                    )
                
        except Exception as e:
            print(f"Error receiving messages: {str(e)}")
    
    def _process_item(self, item, receipt_handle):
        """Process a single item with rate limiting and error handling."""
        try:
            # Apply rate limiting
            self.rpm_limiter.consume(1.0, block=True)
            
            # Estimate token count
            estimated_tokens = self._estimate_tokens(item.get('prompt', ''))
            self.tpm_limiter.consume(estimated_tokens, block=True)
            
            # Apply adaptive throttling
            time.sleep(self.throttler.get_delay())
            
            # Process the item
            result = self.client.invoke(
                prompt=item.get('prompt', ''),
                max_tokens=item.get('max_tokens', 500),
                temperature=item.get('temperature', 0.7)
            )
            
            # Record success and actual token usage
            actual_tokens = result.get('total_tokens', estimated_tokens)
            self.monitor.record_request(token_count=actual_tokens)
            self.throttler.on_success()
            
            # Send result to output queue if specified
            if self.output_queue_url:
                self._send_result(item, result)
            
            # Delete from input queue
            if self.input_queue_url:
                self.sqs.delete_message(
                    QueueUrl=self.input_queue_url,
                    ReceiptHandle=receipt_handle
                )
            
            return result
            
        except Exception as e:
            error_str = str(e)
            
            # Record error
            self.monitor.record_request(error=e)
            
            # Update throttling strategy if throttled
            if "ThrottlingException" in error_str:
                self.throttler.on_throttle()
            
            # Send to DLQ if specified
            if self.dlq_url:
                self._send_to_dlq(item, error_str)
            
            # Don't delete message, let it return to queue for retry
            return None
    
    def _estimate_tokens(self, text):
        """Estimate token count for a text."""
        # Simple estimation based on word count
        words = text.split()
        return len(words) * 1.3  # Rough estimation
    
    def _send_result(self, item, result):
        """Send result to output queue."""
        message = {
            'request_id': item.get('request_id', str(uuid.uuid4())),
            'result': result,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.sqs.send_message(
            QueueUrl=self.output_queue_url,
            MessageBody=json.dumps(message)
        )
    
    def _send_to_dlq(self, item, error):
        """Send failed item to dead-letter queue."""
        message = {
            'item': item,
            'error': error,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.sqs.send_message(
            QueueUrl=self.dlq_url,
            MessageBody=json.dumps(message)
        )
```

## Benchmarking and Optimization

To optimize your high-throughput pipeline:

1. **Start with benchmarking**: Measure baseline performance
2. **Identify bottlenecks**: Find the limiting factors (RPM, TPM, or processing)
3. **Optimize incrementally**: Make one change at a time and measure improvement
4. **Test at scale**: Verify performance with representative workloads
5. **Monitor continuously**: Track performance metrics in production

## Next Steps

- See [Benchmarking Tools](../benchmarks) for performance testing
- Explore [Quota Management](quota-management.md) for more optimization techniques
- Learn about [AWS Service Integration](aws-service-integration.md) for complete solutions
- Study [Error Handling Strategies](error-handling-strategies.md) for resilient pipelines