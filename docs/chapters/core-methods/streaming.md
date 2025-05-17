---
layout: chapter
title: Streaming Inference with AWS Bedrock
difficulty: intermediate
time-estimate: 35 minutes
---

# Streaming Inference with AWS Bedrock

> "You've built an AI assistant, but users are getting impatient staring at a loading spinner. Let's fix that with streaming responses that appear in real-time."

## The Problem

---

**Scenario**: You've implemented an AI chat application using AWS Bedrock's synchronous API, but your users are complaining about the experience. For longer responses, they're left waiting for 5-10 seconds with no feedback, leading to confusion and frustration. Some users even refresh the page thinking the application has frozen, which wastes their time and your compute resources.

Your product manager has asked you to solve this user experience issue without completely redesigning the application. You need to:

1. Show responses to users as they're being generated
2. Maintain the same quality of responses
3. Implement proper error handling for streaming responses
4. Ensure the solution works across different models in AWS Bedrock
5. Make the backend implementation robust enough for production use

---

## Key Concepts Explained

### Understanding Streaming Inference

Streaming inference provides a way to receive and display model responses incrementally as they're being generated, rather than waiting for the complete response.

Think of the difference like this:
- **Synchronous inference** is like ordering a complete meal and waiting until it's fully prepared before it's brought to your table.
- **Streaming inference** is like a tasting menu where dishes are brought out one by one as soon as each is ready.

The key benefit is reducing perceived latency. The actual total time to generate the complete response is similar, but the user experience is drastically improved because:
1. Users see the first tokens almost immediately
2. They can start reading while the rest is generating
3. The application feels responsive rather than frozen

### How Streaming Works in AWS Bedrock

When you make a streaming request to AWS Bedrock:

1. You send the prompt to the model, similar to synchronous inference
2. The API connection remains open instead of waiting for the full response
3. As the model generates tokens, they're sent back to your application in small chunks
4. Your application processes these chunks and updates the UI in real-time
5. The stream completes when the model finishes generating or hits a stop condition

### The InvokeModelWithResponseStream API

AWS Bedrock provides the `InvokeModelWithResponseStream` API for streaming responses. While the input is nearly identical to the synchronous `InvokeModel` API, the response handling is quite different:

```python
# Synchronous (returns full response at once)
response = bedrock.invoke_model(modelId="...", body="...")
result = json.loads(response['body'].read())

# Streaming (returns chunks as they're generated)
response = bedrock.invoke_model_with_response_stream(modelId="...", body="...")
for event in response['body']:
    if 'chunk' in event:
        chunk_data = json.loads(event['chunk']['bytes'])
        # Process each chunk
```

## Step-by-Step Implementation

Now let's build a robust streaming implementation for AWS Bedrock.

### 1. Setting Up Your Environment

First, ensure you have the necessary AWS SDK installed:

```bash
pip install boto3
```

### 2. Creating a Streaming-Capable Client

Let's build a flexible client that handles both streaming and non-streaming requests:

```python
import boto3
import json
import time
import random
import logging
from typing import Dict, Any, Optional, Union, Generator, Callable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bedrock_client")

class BedrockClient:
    """
    A client for working with AWS Bedrock, supporting both streaming and
    non-streaming inference with proper error handling.
    """
    
    def __init__(
        self, 
        region_name: str = "us-west-2", 
        profile_name: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        Initialize the Bedrock client.
        
        Args:
            region_name: AWS region where Bedrock is available
            profile_name: AWS profile to use (optional)
            max_retries: Maximum number of retries for retriable errors
        """
        # Create session with optional profile
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
        else:
            session = boto3.Session()
        
        # Create Bedrock runtime client
        self.client = session.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )
        
        self.max_retries = max_retries
    
    def generate_text(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        stream: bool = False,
        callback: Optional[Callable] = None
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Generate text using the specified model with support for streaming.
        
        Args:
            model_id: The Bedrock model identifier
            prompt: The text prompt to send to the model
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            callback: Function to call with each chunk when streaming
            
        Returns:
            If stream=False: Complete response as a dictionary
            If stream=True: Generator yielding text chunks
        """
        # Format request payload based on model provider
        payload = self._format_payload(model_id, prompt, temperature, max_tokens)
        
        if stream:
            return self._generate_streaming(model_id, payload, callback)
        else:
            return self._generate_sync(model_id, payload)
    
    def _format_payload(
        self,
        model_id: str,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Format the request payload based on the model provider."""
        model_provider = model_id.split(".")[0].lower()
        
        if "anthropic" in model_provider:
            # Check if it's Claude 3 (newer) or Claude 1/2 (older)
            if "claude-3" in model_id.lower():
                # Claude 3 uses the messages format
                return {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            else:
                # Claude 1/2 uses the prompt format
                return {
                    "prompt": f"Human: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": max_tokens,
                    "temperature": temperature
                }
        elif "amazon" in model_provider or "titan" in model_provider:
            # Amazon Titan models
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            }
        elif "meta" in model_provider or "llama" in model_provider:
            # Meta Llama models
            return {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_gen_len": max_tokens,
                "temperature": temperature
            }
        elif "cohere" in model_provider:
            # Cohere models
            return {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True  # Required for Cohere streaming
            }
        elif "ai21" in model_provider:
            # AI21 Jurassic models
            return {
                "prompt": prompt,
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
    
    def _generate_sync(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text using synchronous inference."""
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                if retries > 0:
                    logger.info(f"Retry {retries}/{self.max_retries} for model {model_id}")
                
                # Invoke the model
                start_time = time.time()
                response = self.client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(payload),
                    contentType="application/json",
                    accept="application/json",
                )
                
                # Calculate latency for logging
                latency = time.time() - start_time
                logger.debug(f"Model {model_id} response received in {latency:.2f}s")
                
                # Parse and return the response
                response_body = json.loads(response["body"].read())
                
                # Extract and format the response text
                result = self._extract_response_text(model_id, response_body)
                
                return {
                    "text": result,
                    "model_id": model_id,
                    "latency": latency,
                    "raw_response": response_body
                }
                
            except self.client.exceptions.ThrottlingException as e:
                retries += 1
                if retries <= self.max_retries:
                    # Exponential backoff with jitter
                    wait_time = min(30, (2 ** retries) + random.uniform(0, 1))
                    logger.warning(f"Request throttled. Retrying in {wait_time:.2f}s")
                    time.sleep(wait_time)
                    last_exception = e
                else:
                    logger.error(f"Max retries exceeded for throttling: {str(e)}")
                    raise RuntimeError(f"Request throttled and max retries exceeded: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error generating text: {str(e)}")
                raise
    
    def _generate_streaming(
        self, 
        model_id: str, 
        payload: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> Generator[str, None, None]:
        """
        Generate text using streaming inference.
        
        Returns a generator that yields text chunks as they arrive.
        """
        try:
            # Invoke model with streaming
            response = self.client.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )
            
            # Process the streaming response
            full_response = ""
            
            # Get the stream
            stream = response.get('body')
            
            # Model-specific tracking variables
            if "anthropic" in model_id:
                is_claude3 = "claude-3" in model_id.lower()
            else:
                is_claude3 = False
                
            # Process each chunk in the stream
            for event in stream:
                # Check if this event contains a chunk
                if 'chunk' in event:
                    try:
                        # Parse the chunk
                        chunk_data = json.loads(event['chunk']['bytes'])
                        
                        # Extract text based on model type
                        chunk_text = self._extract_chunk_text(model_id, chunk_data)
                        
                        if chunk_text:
                            # Append to full response
                            full_response += chunk_text
                            
                            # Call callback if provided
                            if callback:
                                callback(chunk_text, False)  # Not done yet
                            
                            # Yield this chunk
                            yield chunk_text
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error decoding chunk: {str(e)}")
                        continue
            
            # Signal completion via callback if provided
            if callback:
                callback("", True)  # Signal completion
            
            # Return the complete response as the StopIteration value
            return full_response
            
        except self.client.exceptions.ThrottlingException as e:
            logger.error(f"Request throttled: {str(e)}")
            if callback:
                callback(f"Error: Request throttled. Please try again later.", True)
            raise RuntimeError(f"Request throttled: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            if callback:
                callback(f"Error: {str(e)}", True)
            raise
    
    def _extract_response_text(self, model_id: str, response_body: Dict[str, Any]) -> str:
        """Extract the generated text from model-specific response formats."""
        model_provider = model_id.split(".")[0].lower()
        
        try:
            if "anthropic" in model_provider:
                if "claude-3" in model_id.lower():
                    # Claude 3 format
                    return response_body.get("content", [{}])[0].get("text", "")
                else:
                    # Claude 1/2 format
                    return response_body.get("completion", "")
            elif "amazon" in model_provider or "titan" in model_provider:
                # Amazon Titan format
                return response_body.get("results", [{}])[0].get("outputText", "")
            elif "meta" in model_provider or "llama" in model_provider:
                # Meta Llama format
                return response_body.get("generation", "")
            elif "cohere" in model_provider:
                # Cohere format
                return response_body.get("generations", [{}])[0].get("text", "")
            elif "ai21" in model_provider:
                # AI21 format
                return response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
            else:
                # Fallback - return the raw response for unknown models
                return str(response_body)
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing response: {str(e)}")
            return str(response_body)  # Return raw response if parsing fails
    
    def _extract_chunk_text(self, model_id: str, chunk_data: Dict[str, Any]) -> str:
        """Extract text from a streaming chunk based on model type."""
        model_provider = model_id.split(".")[0].lower()
        
        try:
            if "anthropic" in model_provider:
                if "claude-3" in model_id.lower():
                    # Claude 3 streaming format
                    if chunk_data.get("type") == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        return delta.get("text", "")
                    return ""
                else:
                    # Claude 1/2 streaming format
                    return chunk_data.get("completion", "")
            elif "amazon" in model_provider or "titan" in model_provider:
                # Amazon Titan streaming format
                return chunk_data.get("outputText", "")
            elif "meta" in model_provider or "llama" in model_provider:
                # Meta Llama streaming format (simplified)
                return chunk_data.get("generation", "")
            elif "cohere" in model_provider:
                # Cohere streaming format
                return chunk_data.get("text", "")
            elif "ai21" in model_provider:
                # AI21 streaming format
                return chunk_data.get("text", "")
            else:
                # Fallback - return empty string for unknown models
                return ""
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing chunk: {str(e)}")
            return ""  # Return empty string if parsing fails
```

### 3. Using the Streaming Client

Here's how to use our client for streaming responses:

```python
def demo_streaming():
    """Demonstrate streaming text generation with AWS Bedrock."""
    client = BedrockClient()
    
    # Define the prompt
    prompt = "Explain quantum computing in simple terms"
    
    print("Starting streaming response. Text will appear as it's generated:\n")
    
    # Option 1: Use as a generator
    try:
        for chunk in client.generate_text(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            prompt=prompt,
            temperature=0.7,
            max_tokens=500,
            stream=True
        ):
            # Print without newline and flush to show real-time
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\nError during streaming: {str(e)}")
    
    print("\n\nStreaming complete!\n")
    
    # Option 2: Use with a callback function
    def handle_chunk(text, done):
        if not done:
            # Process each chunk as it arrives
            print(text, end="", flush=True)
        else:
            # Final handling when stream is complete
            print("\n[Stream finished]")
    
    print("Starting second streaming response with callback:\n")
    
    try:
        # This won't return anything - chunks are handled by the callback
        client.generate_text(
            model_id="amazon.titan-text-express-v1",
            prompt="Write a short poem about streaming data",
            temperature=0.8,
            max_tokens=200,
            stream=True,
            callback=handle_chunk
        )
    except Exception as e:
        print(f"\nError during streaming: {str(e)}")

if __name__ == "__main__":
    demo_streaming()
```

### 4. Implementing a Web-Based Streaming API

For web applications, you'll need to implement streaming HTTP endpoints. Here's an example using Flask:

```python
from flask import Flask, request, Response, jsonify
import json

app = Flask(__name__)
client = BedrockClient()

@app.route("/api/generate", methods=["POST"])
def generate_text():
    """
    Generate text from AWS Bedrock.
    
    If stream=true in the request, returns a streaming response.
    Otherwise, returns a standard JSON response.
    """
    data = request.json
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    # Extract parameters
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    model_id = data.get("model_id", "anthropic.claude-3-haiku-20240307-v1:0")
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 500))
    stream = bool(data.get("stream", False))
    
    try:
        if stream:
            # Return a streaming response
            def generate():
                try:
                    for chunk in client.generate_text(
                        model_id=model_id,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    ):
                        # Format each chunk as a Server-Sent Event
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                    
                    # Signal the end of the stream
                    yield f"data: {json.dumps({'done': True})}\n\n"
                except Exception as e:
                    # Send error as an event
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            # Return a streaming response with Server-Sent Events
            return Response(
                generate(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"  # Disable Nginx buffering
                }
            )
        else:
            # Return a standard JSON response for non-streaming
            result = client.generate_text(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            return jsonify({
                "text": result["text"],
                "model_id": model_id,
                "latency": result["latency"]
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
```

### 5. Frontend Implementation for Streaming

Here's a simple HTML/JavaScript implementation to consume the streaming API:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Streaming AI Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        #prompt {
            width: 100%;
            height: 100px;
            margin-bottom: 10px;
        }
        #response {
            border: 1px solid #ccc;
            padding: 15px;
            min-height: 200px;
            white-space: pre-wrap;
            background-color: #f9f9f9;
        }
        .controls {
            margin: 15px 0;
            display: flex;
            gap: 10px;
        }
        button {
            padding: 8px 16px;
            cursor: pointer;
        }
        #loading {
            color: #888;
            display: none;
        }
        .cursor {
            display: inline-block;
            width: 10px;
            height: 20px;
            background-color: #333;
            animation: blink 1s infinite;
            vertical-align: middle;
        }
        @keyframes blink {
            50% { opacity: 0; }
        }
    </style>
</head>
<body>
    <h1>AWS Bedrock Streaming Demo</h1>
    
    <div>
        <label for="prompt">Enter your prompt:</label>
        <textarea id="prompt">Explain quantum computing in simple terms.</textarea>
    </div>
    
    <div class="controls">
        <button id="streamBtn">Stream Response</button>
        <button id="syncBtn">Get Complete Response</button>
        <span id="loading">Generating response...</span>
    </div>
    
    <div>
        <h3>Response:</h3>
        <div id="response"></div>
    </div>
    
    <script>
        const promptInput = document.getElementById('prompt');
        const responseDiv = document.getElementById('response');
        const streamBtn = document.getElementById('streamBtn');
        const syncBtn = document.getElementById('syncBtn');
        const loadingSpan = document.getElementById('loading');
        
        // Streaming response handler
        streamBtn.addEventListener('click', async () => {
            const prompt = promptInput.value.trim();
            if (!prompt) return;
            
            // Clear previous response
            responseDiv.innerHTML = '<span class="cursor"></span>';
            loadingSpan.style.display = 'inline';
            
            try {
                // Create event source for streaming
                const eventSource = new EventSource(`/api/generate?stream=true&prompt=${encodeURIComponent(prompt)}`);
                
                // Handle incoming chunks
                eventSource.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    if (data.error) {
                        responseDiv.textContent = `Error: ${data.error}`;
                        eventSource.close();
                        loadingSpan.style.display = 'none';
                    } else if (data.done) {
                        // Stream complete
                        eventSource.close();
                        loadingSpan.style.display = 'none';
                        
                        // Remove the cursor
                        const cursor = responseDiv.querySelector('.cursor');
                        if (cursor) cursor.remove();
                    } else {
                        // Append text before the cursor
                        const cursor = responseDiv.querySelector('.cursor');
                        const textNode = document.createTextNode(data.text);
                        responseDiv.insertBefore(textNode, cursor);
                    }
                };
                
                // Handle errors
                eventSource.onerror = () => {
                    eventSource.close();
                    loadingSpan.style.display = 'none';
                    
                    // Display error message if response is empty
                    if (!responseDiv.textContent) {
                        responseDiv.textContent = 'Error: Connection failed or timed out.';
                    }
                    
                    // Remove the cursor
                    const cursor = responseDiv.querySelector('.cursor');
                    if (cursor) cursor.remove();
                };
                
            } catch (error) {
                responseDiv.textContent = `Error: ${error.message}`;
                loadingSpan.style.display = 'none';
            }
        });
        
        // Synchronous response handler
        syncBtn.addEventListener('click', async () => {
            const prompt = promptInput.value.trim();
            if (!prompt) return;
            
            // Clear previous response
            responseDiv.textContent = '';
            loadingSpan.style.display = 'inline';
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prompt,
                        stream: false
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    responseDiv.textContent = `Error: ${data.error}`;
                } else {
                    responseDiv.textContent = data.text;
                }
                
            } catch (error) {
                responseDiv.textContent = `Error: ${error.message}`;
            } finally {
                loadingSpan.style.display = 'none';
            }
        });
    </script>
</body>
</html>
```

### 6. Handling Stream Interruptions

In real applications, you need to handle cases where streams are interrupted:

```python
def handle_stream_interruptions(
    client, 
    model_id, 
    prompt, 
    max_attempts=3, 
    callback=None
):
    """
    Handle streaming with automatic recovery from interruptions.
    
    Args:
        client: The BedrockClient instance
        model_id: The model to use
        prompt: The text prompt
        max_attempts: Maximum number of retry attempts
        callback: Function to receive chunks
    
    Returns:
        The complete generated text
    """
    attempts = 0
    complete_text = ""
    is_complete = False
    
    # Custom callback to track progress
    def track_progress(chunk, done):
        nonlocal complete_text
        nonlocal is_complete
        
        if not done:
            complete_text += chunk
            
            # Pass chunks to the original callback if provided
            if callback:
                callback(chunk, False)
        else:
            is_complete = True
            if callback:
                callback("", True)
    
    while attempts < max_attempts and not is_complete:
        try:
            if attempts > 0:
                logger.info(f"Retrying stream (attempt {attempts+1}/{max_attempts})")
                
                if callback:
                    callback("\n[Reconnecting to stream...]\n", False)
            
            # Start/resume the stream
            for _ in client.generate_text(
                model_id=model_id,
                prompt=prompt,
                stream=True,
                callback=track_progress
            ):
                # We're just using the callback to process chunks
                pass
                
            # If we get here without exception, we're done
            break
            
        except Exception as e:
            attempts += 1
            logger.warning(f"Stream interrupted: {str(e)}. Attempt {attempts}/{max_attempts}")
            
            if attempts >= max_attempts:
                # If we've exhausted retries, inform the caller
                if callback:
                    callback(f"\n[Error: Stream failed after {max_attempts} attempts: {str(e)}]", True)
                logger.error(f"Stream failed after {max_attempts} attempts: {str(e)}")
                raise RuntimeError(f"Stream failed after {max_attempts} attempts: {str(e)}")
            
            # Wait before retrying
            time.sleep(min(1 * attempts, 5))  # Progressive backoff up to 5 seconds
    
    return complete_text
```

## Common Pitfalls and Troubleshooting

### Pitfall #1: Not Handling Different Model Formats

**Problem**: Each model provider formats streaming chunks differently, leading to broken or missing text.

**Solution**: Implement model-specific parsing:

```python
def parse_streaming_chunk(model_id, chunk_data):
    """Parse streaming chunks based on model provider."""
    # Extract model provider from ID
    provider = model_id.split('.')[0].lower()
    
    # Parse based on provider
    if "anthropic" in provider:
        if "claude-3" in model_id.lower():
            # Claude 3 uses a different format than Claude 2
            if "type" in chunk_data and chunk_data["type"] == "content_block_delta":
                return chunk_data.get("delta", {}).get("text", "")
            return ""
        else:
            # Claude 2 format
            return chunk_data.get("completion", "")
    elif "titan" in provider or "amazon" in provider:
        return chunk_data.get("outputText", "")
    elif "meta" in provider or "llama" in provider:
        return chunk_data.get("generation", "")
    # Add cases for other providers...
    
    # If unknown provider, log a warning and return empty string
    logger.warning(f"Unknown provider format for model {model_id}: {chunk_data}")
    return ""
```

### Pitfall #2: Not Properly Handling Network Interruptions

**Problem**: Streaming connections can be interrupted, leading to incomplete responses.

**Solution**: Implement reconnection logic with state tracking:

```python
class StreamingSession:
    """
    Manages a streaming session with reconnection capabilities.
    Tracks progress to allow resuming from interruptions.
    """
    
    def __init__(self, client, model_id, prompt, callback=None):
        """Initialize the streaming session."""
        self.client = client
        self.model_id = model_id
        self.prompt = prompt
        self.callback = callback
        
        # Tracking state
        self.complete_text = ""
        self.is_complete = False
        self.attempt = 0
        self.max_attempts = 3
        
    def start(self):
        """Start or resume the streaming session."""
        while self.attempt < self.max_attempts and not self.is_complete:
            try:
                # Track progress through custom callback
                for chunk in self.client.generate_text(
                    model_id=self.model_id,
                    prompt=self.prompt,
                    stream=True
                ):
                    # Append to our complete text
                    self.complete_text += chunk
                    
                    # Call the original callback if provided
                    if self.callback:
                        self.callback(chunk, False)  # Not done yet
                
                # Stream completed successfully
                self.is_complete = True
                
                # Final callback
                if self.callback:
                    self.callback("", True)  # Signal completion
                
            except Exception as e:
                self.attempt += 1
                logger.warning(f"Stream interrupted: {str(e)}. Attempt {self.attempt}/{self.max_attempts}")
                
                if self.attempt >= self.max_attempts:
                    if self.callback:
                        self.callback(f"\n[Stream failed after {self.max_attempts} attempts]", True)
                    raise RuntimeError(f"Stream failed after {self.max_attempts} attempts: {str(e)}")
                
                # Wait before retrying
                time.sleep(min(2 ** (self.attempt - 1), 8))  # Exponential backoff
                
                # Inform callback about reconnection
                if self.callback:
                    self.callback("\n[Reconnecting...]\n", False)
        
        return self.complete_text
```

### Pitfall #3: Ignoring Error Handling in Streaming UI

**Problem**: Poor error handling in the frontend creates a confusing user experience when streaming fails.

**Solution**: Implement robust frontend error handling with status indicators:

```javascript
// Improved frontend error handling for streaming
function startStreaming(prompt) {
    // Clear previous response
    responseElement.innerHTML = '';
    
    // Add typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<span></span><span></span><span></span>';
    responseElement.appendChild(typingIndicator);
    
    // Create a status element for connection issues
    const statusElement = document.createElement('div');
    statusElement.className = 'connection-status';
    responseElement.appendChild(statusElement);
    
    // Create EventSource for streaming
    const eventSource = new EventSource(`/api/generate?prompt=${encodeURIComponent(prompt)}&stream=true`);
    
    // Track connection state
    let connectionLost = false;
    let reconnectAttempt = 0;
    const maxReconnectAttempts = 3;
    
    // Handle connection open
    eventSource.onopen = () => {
        if (connectionLost) {
            // Update status for reconnection
            statusElement.textContent = 'Connection restored!';
            statusElement.classList.add('status-success');
            
            // Remove success message after 3 seconds
            setTimeout(() => {
                statusElement.textContent = '';
                statusElement.classList.remove('status-success');
            }, 3000);
            
            connectionLost = false;
        }
    };
    
    // Handle message chunks
    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            // Remove typing indicator when done
            if (data.done) {
                typingIndicator.remove();
                eventSource.close();
                return;
            }
            
            if (data.error) {
                // Show error message
                typingIndicator.remove();
                const errorElement = document.createElement('div');
                errorElement.className = 'error-message';
                errorElement.textContent = data.error;
                responseElement.appendChild(errorElement);
                eventSource.close();
                return;
            }
            
            // Append text chunk
            const textNode = document.createTextNode(data.text);
            responseElement.insertBefore(textNode, typingIndicator);
            
        } catch (error) {
            console.error('Error parsing stream chunk:', error);
        }
    };
    
    // Handle connection errors
    eventSource.onerror = () => {
        connectionLost = true;
        reconnectAttempt++;
        
        // Update status
        statusElement.textContent = `Connection lost. Reconnecting (${reconnectAttempt}/${maxReconnectAttempts})...`;
        statusElement.classList.add('status-error');
        
        if (reconnectAttempt >= maxReconnectAttempts) {
            // Give up after max attempts
            typingIndicator.remove();
            statusElement.textContent = 'Connection failed. Please try again.';
            eventSource.close();
        }
    };
    
    // Return a function to cancel the stream
    return () => {
        eventSource.close();
        typingIndicator.remove();
        statusElement.textContent = 'Generation canceled.';
        setTimeout(() => {
            statusElement.remove();
        }, 3000);
    };
}
```

## Try It Yourself Challenge

Now it's your turn to implement streaming inference with AWS Bedrock.

### Challenge: Build a Streaming Chat Application

Create a simple chat application that uses streaming responses to provide a more interactive experience.

**Starting Code**:

```python
import boto3
import json
import time
import logging
import threading
from typing import Dict, Any, Generator, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat_app")

class ChatMessage:
    """Represents a message in a chat conversation."""
    
    def __init__(self, role: str, content: str):
        """Initialize a chat message."""
        self.role = role  # "user" or "assistant"
        self.content = content
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {
            "role": self.role,
            "content": self.content
        }

class ChatConversation:
    """Manages a conversation with message history."""
    
    def __init__(self, max_messages: int = 10):
        """Initialize a conversation."""
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append(ChatMessage(role, content))
        
        # Trim history if needed
        if len(self.messages) > self.max_messages:
            # Remove oldest messages but keep the first (system prompt)
            self.messages = self.messages[:1] + self.messages[-(self.max_messages-1):]
    
    def get_formatted_messages(self) -> List[Dict[str, str]]:
        """Get messages in a format suitable for model input."""
        return [msg.to_dict() for msg in self.messages]

class StreamingChatApp:
    """
    A chat application with streaming responses from AWS Bedrock.
    """
    
    def __init__(self, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """Initialize the chat application."""
        # TODO: Initialize AWS Bedrock client
        
        # TODO: Initialize conversation with a system prompt
    
    def send_message(self, message: str, stream_handler=None) -> Generator[str, None, None]:
        """
        Send a message and get a streaming response.
        
        Args:
            message: The user message to send
            stream_handler: Optional callback function for handling streamed chunks
            
        Returns:
            Generator that yields response chunks
        """
        # TODO: Implement message sending with streaming response
        pass
    
    def _prepare_chat_payload(self) -> Dict[str, Any]:
        """Prepare the request payload with the conversation history."""
        # TODO: Implement payload preparation based on the model
        pass

# Example usage
if __name__ == "__main__":
    # Create chat application
    chat_app = StreamingChatApp()
    
    # Define a handler for streamed responses
    def print_streaming_response(chunk: str, done: bool) -> None:
        """Print streaming response chunks."""
        if not done:
            print(chunk, end="", flush=True)
        else:
            print("\n--- Response complete ---\n")
    
    # Interactive chat loop
    print("Chat with AI (type 'exit' to quit)")
    print("----------------------------------")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Print assistant response with streaming
        print("\nAI: ", end="", flush=True)
        
        # Send message and handle streaming response
        try:
            for _ in chat_app.send_message(user_input, print_streaming_response):
                # Processing happens in the callback
                pass
        except Exception as e:
            print(f"\nError: {str(e)}")
```

**Expected Outcome**: A working chat application that:
1. Maintains conversation history
2. Shows streaming responses in real-time
3. Handles connection errors gracefully
4. Provides a natural chat experience

## Beyond the Basics

Once you've mastered basic streaming, consider these advanced techniques:

### 1. Server-Sent Events (SSE) with FastAPI

FastAPI provides a more modern web framework with built-in async support:

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import uvicorn

app = FastAPI()
client = BedrockClient()

@app.post("/api/generate/stream")
async def generate_stream(request: Request):
    """Stream responses using Server-Sent Events (SSE)."""
    data = await request.json()
    prompt = data.get("prompt")
    model_id = data.get("model_id", "anthropic.claude-3-haiku-20240307-v1:0")
    
    async def event_generator():
        """Generate Server-Sent Events."""
        try:
            # Use a blocking generator in a ThreadPoolExecutor
            loop = asyncio.get_event_loop()
            
            def generate():
                try:
                    for chunk in client.generate_text(
                        model_id=model_id,
                        prompt=prompt,
                        stream=True
                    ):
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                    
                    # Signal the end
                    yield f"data: {json.dumps({'done': True})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            # Run the blocking generator in a thread pool and stream results
            async for chunk in run_blocking_generator(generate()):
                yield chunk
                
        except Exception as e:
            # Send error as an event
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

async def run_blocking_generator(generator):
    """Run a blocking generator in a thread pool and yield results asynchronously."""
    loop = asyncio.get_event_loop()
    for item in generator:
        yield item
        # Allow other tasks to run
        await asyncio.sleep(0)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Streaming with Progress Estimation

Enhance the user experience by providing progress estimates:

```python
def estimate_completion_progress(
    prompt: str, 
    response_so_far: str, 
    max_tokens: int
) -> float:
    """
    Estimate completion progress percentage based on tokens generated so far.
    
    Args:
        prompt: The input prompt
        response_so_far: Text generated so far
        max_tokens: Maximum tokens to generate
        
    Returns:
        Estimated progress (0.0 to 1.0)
    """
    # Estimate token counts (use proper tokenizers in production)
    prompt_tokens = len(prompt.split()) * 1.3
    response_tokens = len(response_so_far.split()) * 1.3
    
    # Estimate progress based on what we've generated vs max tokens
    progress = min(1.0, response_tokens / max_tokens)
    
    # You could use more sophisticated approaches based on:
    # - Model-specific output patterns
    # - Presence of completion indicators in text
    # - Rate of token generation
    
    return progress

# Example usage in streaming UI
def update_progress_bar(chunk, done, progress_bar):
    """Update progress bar based on streaming progress."""
    global response_text
    
    if not done:
        # Append chunk to accumulated text
        response_text += chunk
        
        # Estimate progress
        progress = estimate_completion_progress(prompt, response_text, max_tokens)
        
        # Update progress bar
        progress_bar.style.width = f"{progress * 100}%"
    else:
        # Set to 100% when done
        progress_bar.style.width = "100%"
```

### 3. Adaptive Token Rate Monitoring

Monitor token generation rates to detect and handle slowdowns:

```python
class TokenRateMonitor:
    """
    Monitors token generation rate during streaming and
    provides adaptive feedback for slow responses.
    """
    
    def __init__(self, expected_tokens_per_second=10.0):
        """Initialize with expected token rate."""
        self.expected_rate = expected_tokens_per_second
        self.start_time = None
        self.tokens_received = 0
        self.last_update_time = None
        self.current_rate = 0.0
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.tokens_received = 0
    
    def update(self, chunk):
        """Update with a new chunk of text."""
        now = time.time()
        
        # Estimate tokens in this chunk
        chunk_tokens = len(chunk.split()) * 1.3
        self.tokens_received += chunk_tokens
        
        # Calculate overall rate
        elapsed = now - self.start_time
        if elapsed > 0:
            self.current_rate = self.tokens_received / elapsed
        
        # Calculate instantaneous rate since last update
        time_since_update = now - self.last_update_time
        instantaneous_rate = chunk_tokens / time_since_update if time_since_update > 0 else 0
        
        self.last_update_time = now
        
        return {
            "overall_rate": self.current_rate,
            "instantaneous_rate": instantaneous_rate,
            "is_slow": self.current_rate < (self.expected_rate * 0.5),
            "total_tokens": self.tokens_received,
            "elapsed_seconds": elapsed
        }
    
    def get_status_message(self):
        """Get user-friendly status message."""
        if not self.start_time:
            return ""
        
        if self.current_rate < (self.expected_rate * 0.3):
            return "Response is generating slowly. This might take a moment..."
        elif self.current_rate < (self.expected_rate * 0.7):
            return "Response is generating at a moderate pace."
        else:
            return ""
```

## Key Takeaways

- Streaming inference provides a better user experience by showing responses as they're generated
- `InvokeModelWithResponseStream` is the key API for streaming in AWS Bedrock
- Different models format their streaming responses differently, requiring model-specific parsing
- Proper error handling and reconnection logic are essential for production applications
- Frontend implementation needs to handle streaming gracefully with appropriate visual indicators
- Streaming doesn't reduce total response time but improves perceived response time dramatically

---

**Next Steps**: Now that you understand streaming inference, learn about [asynchronous processing](/chapters/core-methods/asynchronous/) for handling longer, background tasks with AWS Bedrock.

---

© 2025 Scott Friedman. Licensed under CC BY-NC-ND 4.0