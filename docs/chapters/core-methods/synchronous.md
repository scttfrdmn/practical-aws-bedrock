---
layout: chapter
title: Synchronous Inference with AWS Bedrock
difficulty: beginner
time-estimate: 30 minutes
---

# Synchronous Inference with AWS Bedrock

> "You need to integrate AI into your application right now, with the simplest possible approach. Let's start with the fundamentals of synchronous inference."

## The Problem

---

**Scenario**: You're a developer building a content generation tool that needs to provide AI-powered capabilities such as summarization, rephrasing, and expanding bullet points into paragraphs. Your users expect quick responses, typically within a few seconds.

You've decided to use AWS Bedrock for this task, but you're not sure how to properly implement the API calls, handle errors, and ensure a good user experience. You need a straightforward, reliable implementation that:

1. Connects to AWS Bedrock properly
2. Formats prompts correctly for different models
3. Handles errors gracefully without crashing your application
4. Provides appropriate feedback to users during processing
5. Manages timeouts and retries appropriately

Every failed request means a frustrated user, and you need a production-ready solution rather than just sample code.

---

## Key Concepts Explained

Before diving into code, let's understand the fundamental concepts of synchronous inference with AWS Bedrock.

### What is Synchronous Inference?

Synchronous inference is the simplest form of model invocation - you send a request to the model and wait for the complete response before proceeding. Think of it like making a phone call where you ask a question and stay on the line until you get a full answer.

This approach is called "synchronous" because:
- Your application sends a request
- Your code waits (blocks) while the model processes it
- Processing completes and the full response is returned
- Only then does your application continue execution

### When to Use Synchronous Inference

Synchronous inference is ideal for:
- Simple, standalone requests
- Responses needed within a few seconds
- Situations where you need the complete response before proceeding
- Backend processing without user-facing waiting
- Straightforward implementation without complex handling

### The InvokeModel API

The core of synchronous inference in AWS Bedrock is the `InvokeModel` API. This is the primary entry point for sending prompts to foundation models and receiving responses.

The basic flow looks like:

1. Create a prompt according to the model's expected format
2. Send the prompt using the `InvokeModel` API
3. Wait for processing to complete
4. Receive and parse the full response
5. Handle any errors that occurred during processing

Let's explore how to implement this effectively.

## Step-by-Step Implementation

Now let's build a production-ready implementation of synchronous inference.

### 1. Setting Up Your Environment

First, ensure you have the necessary AWS SDK installed:

```bash
pip install boto3
```

Then, set up your AWS credentials through one of these methods:
- AWS CLI: `aws configure`
- Environment variables
- Credentials file
- IAM roles (for AWS services)

### 2. Creating a Basic Client

Let's create a reusable client class for AWS Bedrock:

```python
import boto3
import json
import time
from typing import Dict, Any, Optional, Union
import logging

class BedrockInferenceClient:
    """
    A client for performing synchronous inference with AWS Bedrock models.
    
    This client handles the details of connecting to AWS Bedrock, formatting
    requests for different model providers, and proper error handling.
    """
    
    def __init__(
        self, 
        region_name: str = "us-west-2", 
        profile_name: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the Bedrock client.
        
        Args:
            region_name: AWS region where Bedrock is available
            profile_name: AWS profile to use (optional)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for retriable errors
        """
        # Set up logging
        self.logger = logging.getLogger("bedrock_client")
        
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
        
        # Store configuration
        self.timeout = timeout
        self.max_retries = max_retries
        
    def invoke_model(
        self, 
        model_id: str, 
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Invoke a model with the provided prompt.
        
        Args:
            model_id: The Bedrock model identifier
            prompt: The text prompt to send to the model
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Parsed response from the model
            
        Raises:
            ValueError: If the prompt or model_id is invalid
            TimeoutError: If the request times out
            RuntimeError: For other invocation errors
        """
        if not prompt or not model_id:
            raise ValueError("Prompt and model_id must be provided")
        
        # Format payload based on model provider
        try:
            payload = self._format_payload(model_id, prompt, temperature, max_tokens)
        except Exception as e:
            self.logger.error(f"Error formatting payload: {str(e)}")
            raise ValueError(f"Failed to format payload for model {model_id}: {str(e)}")
        
        # Invoke model with retries
        return self._invoke_with_retries(model_id, payload)
    
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
                "temperature": temperature
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
    
    def _invoke_with_retries(
        self, 
        model_id: str, 
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke the model with exponential backoff retries."""
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                if retries > 0:
                    # Log retry attempt
                    self.logger.info(f"Retry {retries}/{self.max_retries} for model {model_id}")
                
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
                self.logger.debug(f"Model {model_id} response received in {latency:.2f}s")
                
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
                
            except self.client.exceptions.ModelTimeoutException:
                self.logger.warning(f"Model timeout for {model_id}")
                raise TimeoutError(f"Model {model_id} timed out after {self.timeout} seconds")
                
            except self.client.exceptions.ModelStreamErrorException as e:
                self.logger.error(f"Model stream error: {str(e)}")
                raise RuntimeError(f"Model stream error: {str(e)}")
                
            except self.client.exceptions.ModelNotReadyException:
                self.logger.error(f"Model {model_id} is not ready")
                raise RuntimeError(f"Model {model_id} is not ready for inference")
                
            except self.client.exceptions.ThrottlingException as e:
                retries += 1
                if retries <= self.max_retries:
                    # Exponential backoff with jitter
                    wait_time = min(30, (2 ** retries) + random.uniform(0, 1))
                    self.logger.warning(f"Request throttled. Retrying in {wait_time:.2f}s")
                    time.sleep(wait_time)
                    last_exception = e
                else:
                    self.logger.error(f"Max retries exceeded for throttling: {str(e)}")
                    raise RuntimeError(f"Request throttled and max retries exceeded: {str(e)}")
                    
            except self.client.exceptions.ValidationException as e:
                self.logger.error(f"Validation error: {str(e)}")
                raise ValueError(f"Validation error with request: {str(e)}")
                
            except self.client.exceptions.InternalServerException as e:
                retries += 1
                if retries <= self.max_retries:
                    # Exponential backoff with jitter
                    wait_time = min(30, (2 ** retries) + random.uniform(0, 1))
                    self.logger.warning(f"Server error. Retrying in {wait_time:.2f}s")
                    time.sleep(wait_time)
                    last_exception = e
                else:
                    self.logger.error(f"Max retries exceeded for server error: {str(e)}")
                    raise RuntimeError(f"Server error and max retries exceeded: {str(e)}")
                    
            except Exception as e:
                self.logger.error(f"Unexpected error invoking model: {str(e)}")
                raise RuntimeError(f"Error invoking model {model_id}: {str(e)}")
    
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
            self.logger.error(f"Error parsing response: {str(e)}")
            return str(response_body)  # Return raw response if parsing fails
```

This comprehensive client handles:
- Proper request formatting for different model providers
- Robust error handling with appropriate error types
- Retries with exponential backoff for transient errors
- Response parsing for different model formats
- Logging for debugging and monitoring

### 3. Using the Client

Now let's see how to use our client for common tasks:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create the client
client = BedrockInferenceClient(
    region_name="us-west-2",
    profile_name="default",  # Optional - use a specific AWS profile
    timeout=45,
    max_retries=3
)

def summarize_text(text, max_length=2):
    """Summarize text using AWS Bedrock."""
    prompt = f"Summarize the following text in {max_length} paragraphs:\n\n{text}"
    
    try:
        response = client.invoke_model(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",  # Using Claude 3 Haiku for speed
            prompt=prompt,
            temperature=0.3,  # Lower temperature for more focused summaries
            max_tokens=300
        )
        
        return {
            "summary": response["text"],
            "latency": response["latency"]
        }
    except TimeoutError as e:
        logging.error(f"Summarization timed out: {str(e)}")
        return {"error": "The operation timed out. Please try again with a shorter text."}
    except ValueError as e:
        logging.error(f"Invalid input: {str(e)}")
        return {"error": "There was a problem with your input. Please check and try again."}
    except Exception as e:
        logging.error(f"Summarization failed: {str(e)}")
        return {"error": "An unexpected error occurred. Please try again later."}

def rephrase_content(text, style="professional"):
    """Rephrase content in a specific style."""
    styles = {
        "professional": "in a professional and business-appropriate tone",
        "casual": "in a casual, conversational tone",
        "academic": "in an academic, scholarly tone",
        "simplify": "in simple language a 5th grader could understand"
    }
    
    style_desc = styles.get(style, styles["professional"])
    prompt = f"Rephrase the following text {style_desc}, maintaining all key information:\n\n{text}"
    
    try:
        response = client.invoke_model(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            prompt=prompt,
            temperature=0.7,  # Higher temperature for more creative rephrasing
            max_tokens=500
        )
        
        return {
            "rephrased_text": response["text"],
            "latency": response["latency"]
        }
    except Exception as e:
        logging.error(f"Rephrasing failed: {str(e)}")
        return {"error": f"Could not rephrase text: {str(e)}"}

def expand_bullets(bullet_points):
    """Expand bullet points into full paragraphs."""
    prompt = f"Expand each of the following bullet points into a detailed paragraph:\n\n{bullet_points}"
    
    try:
        response = client.invoke_model(
            model_id="amazon.titan-text-express-v1",  # Using Titan for cost efficiency
            prompt=prompt,
            temperature=0.6,
            max_tokens=800  # Larger output for expanded content
        )
        
        return {
            "expanded_text": response["text"],
            "latency": response["latency"]
        }
    except Exception as e:
        logging.error(f"Bullet expansion failed: {str(e)}")
        return {"error": f"Could not expand bullet points: {str(e)}"}
```

### 4. Handling Large Requests

For requests that might exceed model token limits, implement chunking:

```python
def process_large_document(document, max_chunk_tokens=6000, overlap_tokens=500):
    """
    Process a large document by chunking it into manageable pieces.
    
    Args:
        document: The document text to process
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        
    Returns:
        Combined results from all chunks
    """
    # Estimate token count (very rough approximation)
    # In production, use a proper tokenizer for the model you're using
    estimated_tokens = len(document.split()) * 1.3
    
    if estimated_tokens <= max_chunk_tokens:
        # Document fits in a single chunk
        return summarize_text(document)
    
    # Split into chunks (very simplified - use proper chunking in production)
    words = document.split()
    words_per_chunk = int(max_chunk_tokens / 1.3)
    overlap_words = int(overlap_tokens / 1.3)
    
    chunks = []
    for i in range(0, len(words), words_per_chunk - overlap_words):
        chunk = " ".join(words[i:i + words_per_chunk])
        chunks.append(chunk)
    
    # Process each chunk
    results = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        result = summarize_text(chunk, max_length=1)
        if "summary" in result:
            results.append(result["summary"])
    
    # Combine chunk results
    combined_text = "\n\n".join(results)
    
    # Create a final summary if needed
    if len(results) > 1:
        print("Creating final summary from chunk results...")
        final_summary = summarize_text(combined_text)
        return final_summary
    else:
        return {"summary": combined_text}
```

### 5. Implementing a Web API

Here's how you might integrate this with a Flask web API:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide text to summarize"}), 400
    
    text = data['text']
    max_length = data.get('max_length', 2)
    
    result = summarize_text(text, max_length)
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result)

@app.route('/api/rephrase', methods=['POST'])
def api_rephrase():
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide text to rephrase"}), 400
    
    text = data['text']
    style = data.get('style', 'professional')
    
    result = rephrase_content(text, style)
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result)

@app.route('/api/expand', methods=['POST'])
def api_expand():
    data = request.json
    
    if not data or 'bullets' not in data:
        return jsonify({"error": "Please provide bullet points to expand"}), 400
    
    bullets = data['bullets']
    
    result = expand_bullets(bullets)
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

## Common Pitfalls and Troubleshooting

### Pitfall #1: Ignoring Token Limits

**Problem**: Sending too much text to a model and hitting token limits.

**Solution**: Implement token estimation and chunking:

```python
def estimate_tokens(text):
    """Estimate the number of tokens in a text (rough approximation)."""
    # A better approach would use model-specific tokenizers
    return len(text.split()) * 1.3

def check_token_limits(text, model_id):
    """Check if text is within token limits for the model."""
    estimated_tokens = estimate_tokens(text)
    
    # Model-specific limits
    limits = {
        "anthropic.claude-3-opus-20240229-v1:0": 100000,
        "anthropic.claude-3-sonnet-20240229-v1:0": 100000,
        "anthropic.claude-3-haiku-20240307-v1:0": 100000,
        "anthropic.claude-v2": 100000,
        "meta.llama2-13b-chat-v1": 4096,
        "amazon.titan-text-express-v1": 8000
    }
    
    # Default conservative limit
    default_limit = 4000
    
    # Get limit for this model
    model_limit = limits.get(model_id, default_limit)
    
    return {
        "within_limit": estimated_tokens <= model_limit,
        "estimated_tokens": estimated_tokens,
        "model_limit": model_limit
    }
```

### Pitfall #2: Poor Error Messages to Users

**Problem**: Generic error messages that don't help users resolve issues.

**Solution**: Map specific errors to user-friendly messages:

```python
def user_friendly_error(error):
    """Convert technical errors to user-friendly messages."""
    if isinstance(error, TimeoutError):
        return "Your request is taking longer than expected. Please try a shorter text or try again later."
    elif isinstance(error, ValueError) and "token limit" in str(error).lower():
        return "Your text is too long for our AI to process. Please reduce the length and try again."
    elif "throttling" in str(error).lower():
        return "We're experiencing high demand right now. Please try again in a few moments."
    elif "validation" in str(error).lower():
        return "There was an issue with your input. Please ensure your text doesn't contain any problematic content."
    else:
        return "Something went wrong. Our team has been notified and is working on it. Please try again later."
```

### Pitfall #3: No Request Timeouts

**Problem**: Requests hanging indefinitely, especially for large prompts.

**Solution**: Implement client-side timeouts:

```python
import concurrent.futures
import threading

def invoke_with_timeout(client, model_id, prompt, temperature, max_tokens, timeout_seconds):
    """Invoke model with a client-side timeout using ThreadPoolExecutor."""
    
    def invoke_model():
        return client.invoke_model(
            model_id=model_id,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(invoke_model)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            # Attempt to cancel the task if still running
            print(f"Request timed out after {timeout_seconds} seconds")
            raise TimeoutError(f"Request timed out after {timeout_seconds} seconds")
```

## Try It Yourself Challenge

Now it's your turn to implement synchronous inference with error handling and retries.

### Challenge: Create a Robust Content Generator

Build a tool that can generate different types of content using AWS Bedrock models, handling errors and edge cases gracefully.

**Starting Code**:

```python
import boto3
import json
import time
import random
import logging
from typing import Dict, Any, Optional, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("content_generator")

class ContentGenerator:
    """
    A robust content generator that can create various types of content
    using AWS Bedrock models with proper error handling.
    """
    
    def __init__(self, region_name="us-west-2", profile_name=None):
        """Initialize the content generator."""
        # TODO: Create AWS Bedrock client
        pass
        
    def generate_blog_post(self, topic, length="medium"):
        """
        Generate a blog post on a specific topic.
        
        Args:
            topic: The blog post topic
            length: "short", "medium", or "long"
            
        Returns:
            Generated blog post or error information
        """
        # TODO: Implement blog post generation
        pass
    
    def create_product_description(self, product_details):
        """
        Create a compelling product description.
        
        Args:
            product_details: Dictionary with product information
            
        Returns:
            Generated product description or error information
        """
        # TODO: Implement product description generation
        pass
    
    def generate_social_posts(self, content, platforms=["twitter", "linkedin"]):
        """
        Generate social media posts adapted for different platforms.
        
        Args:
            content: The core content to promote
            platforms: List of social platforms to generate for
            
        Returns:
            Dictionary of platform-specific posts
        """
        # TODO: Implement social post generation
        pass
    
    def _invoke_bedrock(self, model_id, payload):
        """
        Invoke Bedrock model with retries and error handling.
        
        Args:
            model_id: The model to use
            payload: Formatted request payload
            
        Returns:
            Processed model response
        """
        # TODO: Implement robust model invocation
        pass

# Example usage
if __name__ == "__main__":
    generator = ContentGenerator()
    
    # Test blog post generation
    blog = generator.generate_blog_post(
        topic="The impact of AI on content creation",
        length="medium"
    )
    print("BLOG POST:")
    print(blog)
    print("\n---\n")
    
    # Test product description
    product = generator.create_product_description({
        "name": "EcoBoost Water Bottle",
        "features": [
            "Made from recycled materials",
            "Double-walled insulation",
            "Keeps drinks cold for 24 hours",
            "Leak-proof design"
        ],
        "target_audience": "environmentally conscious outdoor enthusiasts"
    })
    print("PRODUCT DESCRIPTION:")
    print(product)
    print("\n---\n")
    
    # Test social posts
    social = generator.generate_social_posts(
        content="We just released our new AI-powered content generation platform that helps marketers create high-quality content in seconds.",
        platforms=["twitter", "linkedin", "facebook"]
    )
    print("SOCIAL POSTS:")
    for platform, post in social.items():
        print(f"{platform.upper()}: {post}")
```

**Expected Outcome**: A fully implemented content generator that:
1. Properly handles AWS Bedrock integration
2. Implements error handling with retries
3. Provides meaningful error messages
4. Formats prompts appropriately for different content types

## Beyond the Basics

Once you've mastered synchronous inference, consider these advanced techniques:

### 1. Model Fallbacks

Implement a fallback mechanism that tries alternative models if the primary one fails:

```python
def invoke_with_fallback(prompt, primary_model, fallback_models):
    """Invoke with automatic fallback to other models."""
    client = BedrockInferenceClient()
    
    # Try primary model first
    try:
        return client.invoke_model(
            model_id=primary_model,
            prompt=prompt
        )
    except Exception as primary_error:
        logger.warning(f"Primary model {primary_model} failed: {str(primary_error)}")
        
        # Try fallback models in sequence
        for fallback_model in fallback_models:
            logger.info(f"Trying fallback model {fallback_model}")
            try:
                return client.invoke_model(
                    model_id=fallback_model,
                    prompt=prompt
                )
            except Exception as fallback_error:
                logger.warning(f"Fallback model {fallback_model} failed: {str(fallback_error)}")
        
        # If we get here, all models failed
        raise RuntimeError(f"All models failed. Primary error: {str(primary_error)}")
```

### 2. Performance Monitoring

Implement a wrapper to track performance metrics for model invocations:

```python
from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class InferenceMetrics:
    """Store metrics for a model invocation."""
    request_id: str
    model_id: str
    prompt_tokens: int
    response_tokens: int
    latency_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = datetime.now()

def track_inference_performance(func):
    """Decorator to track performance metrics for inference calls."""
    def wrapper(self, model_id, prompt, *args, **kwargs):
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Estimate prompt tokens
        prompt_tokens = len(prompt.split()) * 1.3
        
        # Record start time
        start_time = time.time()
        
        try:
            # Call the original function
            result = func(self, model_id, prompt, *args, **kwargs)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimate response tokens
            response_tokens = len(result["text"].split()) * 1.3
            
            # Create metrics
            metrics = InferenceMetrics(
                request_id=request_id,
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                latency_ms=latency_ms,
                success=True
            )
            
            # Log or store metrics (implement as needed)
            self._store_metrics(metrics)
            
            # Return original result with metrics
            result["metrics"] = metrics
            return result
            
        except Exception as e:
            # Calculate latency even for errors
            latency_ms = (time.time() - start_time) * 1000
            
            # Create error metrics
            metrics = InferenceMetrics(
                request_id=request_id,
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                response_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            
            # Log or store metrics (implement as needed)
            self._store_metrics(metrics)
            
            # Re-raise the exception
            raise
    
    return wrapper
```

### 3. Background Processing for Longer Tasks

For tasks that might take longer, implement background processing:

```python
import threading
import queue

class BackgroundProcessor:
    """Process longer inference tasks in the background."""
    
    def __init__(self, num_workers=3):
        """Initialize with a specified number of worker threads."""
        self.task_queue = queue.Queue()
        self.result_map = {}
        self.client = BedrockInferenceClient()
        
        # Start worker threads
        self.workers = []
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_thread, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def submit_task(self, task_id, model_id, prompt):
        """Submit a task for background processing."""
        self.task_queue.put((task_id, model_id, prompt))
        return task_id
    
    def get_result(self, task_id, timeout=None):
        """
        Get the result of a background task.
        
        Returns None if the task is not complete yet.
        """
        if task_id in self.result_map:
            result = self.result_map[task_id]
            # Clean up after retrieving
            del self.result_map[task_id]
            return result
        
        return None
    
    def _worker_thread(self):
        """Worker thread that processes tasks from the queue."""
        while True:
            try:
                # Get task from queue
                task_id, model_id, prompt = self.task_queue.get()
                
                # Process the task
                try:
                    result = self.client.invoke_model(model_id, prompt)
                    self.result_map[task_id] = {
                        "status": "completed",
                        "result": result
                    }
                except Exception as e:
                    self.result_map[task_id] = {
                        "status": "error",
                        "error": str(e)
                    }
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in worker thread: {str(e)}")
```

## Key Takeaways

- Synchronous inference is the simplest way to integrate AI into your applications
- Proper error handling and retries are essential for production use
- Model-specific payload formatting is required for different foundation models
- Token limits must be considered when designing your prompts
- Client-side timeouts prevent requests from hanging indefinitely
- For longer tasks, consider asynchronous approaches or background processing

---

**Next Steps**: Now that you understand synchronous inference, explore [streaming inference](/chapters/core-methods/streaming/) for improved user experience with real-time responses.

---

© 2025 Scott Friedman. Licensed under CC BY-NC-ND 4.0