---
layout: page
title: Multi-Model Inference Orchestration in AWS Bedrock
---

# Multi-Model Inference Orchestration in AWS Bedrock

This guide covers strategies and implementation approaches for orchestrating inference across multiple foundation models in AWS Bedrock. Learn how to build robust systems that leverage the strengths of different models for optimal performance.

## Introduction to Multi-Model Orchestration

Multi-model orchestration involves coordinating and managing inference requests across multiple foundation models to leverage their individual strengths for different parts of a complex workflow. This approach can:

- Optimize for cost, performance, quality, and latency
- Leverage specialized capabilities of different models
- Implement fallback mechanisms for reliability
- Create ensemble systems that combine multiple model outputs

## Orchestration Architectures

### 1. Router-Based Orchestration

In this pattern, a router component analyzes the input and determines which model is best suited to handle it.

```
Input → Router → Model Selection → Model 1/2/3 → Output
```

**Advantages:**
- Optimizes for model strengths
- Can reduce costs by routing to less expensive models when appropriate
- Flexible and easily extensible

**Implementation Example:**

```python
class ModelRouter:
    """Routes requests to the appropriate model based on input analysis."""
    
    def __init__(self, bedrock_client=None):
        """Initialize with a Bedrock client."""
        self.bedrock_client = bedrock_client or get_bedrock_client(profile_name="aws")
        self.models = {
            "text_generation": "anthropic.claude-v2",
            "code_generation": "anthropic.claude-3-sonnet-20240229-v1:0",
            "image_analysis": "anthropic.claude-3-sonnet-20240229-v1:0", 
            "structured_data": "anthropic.claude-3-opus-20240229-v1:0",
            "low_latency": "amazon.titan-text-express-v1"
        }
        
    def analyze_input(self, input_text, input_image=None):
        """Determine the task type based on the input."""
        task_type = "text_generation"  # Default
        
        # Check if input contains code-related keywords
        code_keywords = ["function", "class", "code", "algorithm", "programming"]
        if any(keyword in input_text.lower() for keyword in code_keywords):
            task_type = "code_generation"
            
        # Check if input requests structured data
        structure_keywords = ["json", "table", "csv", "structured", "parse"]
        if any(keyword in input_text.lower() for keyword in structure_keywords):
            task_type = "structured_data"
            
        # Check if there's an image for analysis
        if input_image:
            task_type = "image_analysis"
            
        # Check for urgent/quick response indicators
        quick_keywords = ["urgent", "quick", "fast", "immediate"]
        if any(keyword in input_text.lower() for keyword in quick_keywords):
            task_type = "low_latency"
            
        return task_type
        
    def route_request(self, input_text, input_image=None, override_model=None):
        """Route the request to the appropriate model."""
        if override_model:
            model_id = override_model
        else:
            task_type = self.analyze_input(input_text, input_image)
            model_id = self.models[task_type]
            
        # Log the routing decision
        print(f"Routing request to model: {model_id} for task type: {task_type}")
        
        # Format request based on the selected model
        request_body = self._format_request(model_id, input_text, input_image)
        
        # Invoke the selected model
        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        return json.loads(response["body"].read()), model_id
        
    def _format_request(self, model_id, input_text, input_image=None):
        """Format the request body based on the model requirements."""
        if "anthropic.claude-v2" in model_id:
            return {
                "prompt": f"Human: {input_text}\n\nAssistant:",
                "max_tokens_to_sample": 1000,
                "temperature": 0.7
            }
        elif "anthropic.claude-3" in model_id and input_image:
            # Multimodal request for Claude 3
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": input_text},
                            {"type": "image", "source": {"bytes": input_image}}
                        ]
                    }
                ]
            }
        elif "anthropic.claude-3" in model_id:
            # Text-only request for Claude 3
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": input_text}
                ]
            }
        elif "amazon.titan" in model_id:
            return {
                "inputText": input_text,
                "textGenerationConfig": {
                    "maxTokenCount": 512,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
        # Add more model-specific formats as needed
```

### 2. Pipeline Orchestration

In pipeline orchestration, multiple models work sequentially, with each model performing a specific step in a workflow.

```
Input → Model A → Intermediate Output → Model B → Intermediate Output → Model C → Final Output
```

**Advantages:**
- Leverages specialized capabilities for each step
- Can optimize for cost and quality at each stage
- Enables complex workflows with distinct processing steps

**Implementation Example:**

```python
class ModelPipeline:
    """Orchestrates a pipeline of models for multi-step inference."""
    
    def __init__(self, bedrock_client=None):
        """Initialize with a Bedrock client."""
        self.bedrock_client = bedrock_client or get_bedrock_client(profile_name="aws")
        
    def run_pipeline(self, input_data, pipeline_config):
        """
        Run the model pipeline based on configuration.
        
        Args:
            input_data: The initial input data (text or image)
            pipeline_config: List of pipeline steps, each with model ID and processing function
        
        Returns:
            The final output after all pipeline steps
        """
        current_data = input_data
        results = []
        
        for step in pipeline_config:
            model_id = step["model_id"]
            process_fn = step["process_fn"]
            transform_input_fn = step.get("transform_input_fn")
            transform_output_fn = step.get("transform_output_fn")
            
            # Apply input transformation if provided
            if transform_input_fn:
                current_data = transform_input_fn(current_data)
            
            # Prepare request and invoke model
            request_body = process_fn(current_data)
            
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response["body"].read())
            
            # Extract and transform output if needed
            if transform_output_fn:
                current_data = transform_output_fn(response_body)
            else:
                # Default extraction based on model type
                if "anthropic.claude-v2" in model_id:
                    current_data = response_body.get("completion", "")
                elif "anthropic.claude-3" in model_id:
                    current_data = response_body.get("content", [{}])[0].get("text", "")
                elif "amazon.titan" in model_id:
                    current_data = response_body.get("results", [{}])[0].get("outputText", "")
            
            # Store step results
            results.append({
                "step": len(results) + 1,
                "model_id": model_id,
                "output": current_data
            })
            
        return {
            "final_output": current_data,
            "pipeline_results": results
        }
```

**Example Usage of Pipeline Orchestration:**

```python
# Define pipeline configuration
content_extraction_pipeline = [
    {
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "process_fn": lambda input_image: {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this image."},
                        {"type": "image", "source": {"bytes": input_image}}
                    ]
                }
            ]
        },
        "transform_output_fn": lambda response: response["content"][0]["text"]
    },
    {
        "model_id": "anthropic.claude-v2",
        "process_fn": lambda extracted_text: {
            "prompt": f"Human: Format the following extracted text into a structured JSON document with appropriate fields:\n\n{extracted_text}\n\nAssistant:",
            "max_tokens_to_sample": 2000,
            "temperature": 0.2
        },
        "transform_output_fn": lambda response: json.loads(response["completion"])
    }
]

# Use the pipeline
pipeline = ModelPipeline()
result = pipeline.run_pipeline(image_bytes, content_extraction_pipeline)
structured_data = result["final_output"]
```

### 3. Ensemble Orchestration

Ensemble orchestration involves sending the same input to multiple models and then combining their outputs.

```
         ┌→ Model A →┐
Input ───┼→ Model B →┼→ Output Combiner → Final Output
         └→ Model C →┘
```

**Advantages:**
- Can improve accuracy and reduce uncertainty
- Provides diverse perspectives on the same input
- Robust against individual model weaknesses

**Implementation Example:**

```python
class ModelEnsemble:
    """Combines outputs from multiple models for ensemble inference."""
    
    def __init__(self, bedrock_client=None):
        """Initialize with a Bedrock client."""
        self.bedrock_client = bedrock_client or get_bedrock_client(profile_name="aws")
        
    def run_ensemble(self, input_data, models, combination_strategy="voting"):
        """
        Run ensemble inference across multiple models.
        
        Args:
            input_data: The input prompt or image
            models: List of model IDs to use in the ensemble
            combination_strategy: Strategy for combining results
                                  ("voting", "average", "weighted", "custom")
        
        Returns:
            Combined output based on the specified strategy
        """
        model_outputs = []
        
        # Collect outputs from all models
        for model_id in models:
            # Format request based on model type
            request_body = self._format_request(model_id, input_data)
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response["body"].read())
            
            # Extract output based on model type
            output = self._extract_output(model_id, response_body)
            
            model_outputs.append({
                "model_id": model_id,
                "output": output,
                "raw_response": response_body
            })
        
        # Combine results based on the specified strategy
        if combination_strategy == "voting":
            final_output = self._apply_voting(model_outputs)
        elif combination_strategy == "average":
            final_output = self._apply_averaging(model_outputs)
        elif combination_strategy == "weighted":
            weights = {model: 1/len(models) for model in models}  # Equal weights by default
            final_output = self._apply_weighted_combination(model_outputs, weights)
        elif combination_strategy == "custom" and callable(self.custom_combiner):
            final_output = self.custom_combiner(model_outputs)
        else:
            raise ValueError(f"Unsupported combination strategy: {combination_strategy}")
        
        return {
            "final_output": final_output,
            "model_outputs": model_outputs
        }
    
    def set_custom_combiner(self, combiner_function):
        """Set a custom function for combining model outputs."""
        self.custom_combiner = combiner_function
    
    def _format_request(self, model_id, input_data):
        """Format the request based on model type."""
        # Similar to the format function in ModelRouter
        # ...

    def _extract_output(self, model_id, response):
        """Extract the output text from the model response."""
        # Similar to extraction in ModelPipeline
        # ...
        
    def _apply_voting(self, model_outputs):
        """Apply majority voting to combine categorical outputs."""
        # Implementation for categorical outputs (e.g., classification)
        outputs = [output["output"] for output in model_outputs]
        # Count occurrences of each unique output
        from collections import Counter
        counts = Counter(outputs)
        # Return the most common output
        return counts.most_common(1)[0][0]
    
    def _apply_averaging(self, model_outputs):
        """Apply averaging for numerical outputs."""
        # Implementation for numerical outputs
        try:
            outputs = [float(output["output"]) for output in model_outputs]
            return sum(outputs) / len(outputs)
        except ValueError:
            # Fall back to concatenation for non-numerical outputs
            return " | ".join(output["output"] for output in model_outputs)
    
    def _apply_weighted_combination(self, model_outputs, weights):
        """Apply weighted combination of outputs."""
        # For numerical outputs
        try:
            weighted_sum = 0
            total_weight = 0
            
            for output in model_outputs:
                model_id = output["model_id"]
                value = float(output["output"])
                weight = weights.get(model_id, 1)
                
                weighted_sum += value * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0
        except ValueError:
            # For text outputs, concatenate with weights as prefixes
            weighted_outputs = []
            for output in model_outputs:
                model_id = output["model_id"]
                text = output["output"]
                weight = weights.get(model_id, 1)
                weighted_outputs.append(f"[Weight {weight}] {text}")
            
            return "\n\n".join(weighted_outputs)
```

### 4. Fallback Orchestration

Fallback orchestration implements a reliability pattern where secondary models are used when primary models fail.

```
Input → Primary Model → (Success? → Output) or (Failure? → Fallback Model → Output)
```

**Advantages:**
- Increases system reliability
- Handles quota limits and temporary unavailability
- Can optimize for cost (using less expensive models as primary)

**Implementation Example:**

```python
class FallbackOrchestrator:
    """Implements fallback strategies across multiple models."""
    
    def __init__(self, bedrock_client=None, max_retries=3):
        """Initialize with a Bedrock client."""
        self.bedrock_client = bedrock_client or get_bedrock_client(profile_name="aws")
        self.max_retries = max_retries
        
    def invoke_with_fallback(self, input_data, model_chain, timeout=10):
        """
        Invoke models with fallback chain.
        
        Args:
            input_data: The input text or image
            model_chain: Ordered list of model IDs to try
            timeout: Timeout in seconds for each model attempt
            
        Returns:
            Response from the first successful model invocation
        """
        last_error = None
        model_attempts = []
        
        for model_id in model_chain:
            # Format request based on model type
            request_body = self._format_request(model_id, input_data)
            
            # Try this model with retries
            for attempt in range(self.max_retries):
                try:
                    # Set up timeout for this request
                    response = self.bedrock_client.invoke_model(
                        modelId=model_id,
                        body=json.dumps(request_body)
                    )
                    
                    response_body = json.loads(response["body"].read())
                    
                    # Extract output based on model type
                    output = self._extract_output(model_id, response_body)
                    
                    # Record successful attempt
                    model_attempts.append({
                        "model_id": model_id,
                        "attempt": attempt + 1,
                        "status": "success"
                    })
                    
                    # Return successful result
                    return {
                        "output": output,
                        "raw_response": response_body,
                        "model_id": model_id,
                        "attempts": model_attempts
                    }
                
                except Exception as e:
                    last_error = str(e)
                    
                    # Record failed attempt
                    model_attempts.append({
                        "model_id": model_id,
                        "attempt": attempt + 1,
                        "status": "failed",
                        "error": last_error
                    })
                    
                    # Check if error is non-retriable
                    if "ThrottlingException" in last_error or "ServiceQuotaExceededException" in last_error:
                        # Immediately try next model for quota issues
                        break
                    
                    # Exponential backoff with jitter for retriable errors
                    if attempt < self.max_retries - 1:
                        base_delay = 0.5 * (2 ** attempt)  # Exponential backoff
                        jitter = random.uniform(0, 0.25 * base_delay)  # Add jitter
                        time.sleep(base_delay + jitter)
        
        # If we get here, all models in the chain failed
        raise Exception(f"All models in fallback chain failed. Last error: {last_error}")
    
    def _format_request(self, model_id, input_data):
        """Format the request based on model type."""
        # Similar to previous examples
        # ...

    def _extract_output(self, model_id, response):
        """Extract the output text from the model response."""
        # Similar to previous examples
        # ...
```

## Advanced Orchestration Patterns

### 1. Cost-Optimized Orchestration

Route requests based on cost efficiency while maintaining quality thresholds.

```python
def cost_optimized_invoke(input_text, quality_threshold="medium"):
    """Route to the most cost-effective model that meets the quality threshold."""
    
    # Define model tiers by cost and quality
    model_tiers = {
        "high": {
            "models": ["anthropic.claude-3-opus-20240229-v1:0"],
            "cost_per_1k_tokens": 15.00
        },
        "medium": {
            "models": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            "cost_per_1k_tokens": 3.00
        },
        "low": {
            "models": ["amazon.titan-text-express-v1", "anthropic.claude-instant-v1"],
            "cost_per_1k_tokens": 0.50
        }
    }
    
    # Select appropriate tier based on threshold
    if quality_threshold == "high":
        tier = model_tiers["high"]
    elif quality_threshold == "medium":
        tier = model_tiers["medium"]
    else:
        tier = model_tiers["low"]
    
    # Use the first available model in the tier
    for model_id in tier["models"]:
        try:
            # Invoke model
            return invoke_model(model_id, input_text)
        except Exception:
            continue
    
    # If all models in the tier fail, try lower tier
    if quality_threshold == "high":
        return cost_optimized_invoke(input_text, "medium")
    elif quality_threshold == "medium":
        return cost_optimized_invoke(input_text, "low")
    else:
        raise Exception("All models failed")
```

### 2. Latency-Optimized Orchestration

Prioritize models with lower latency when response time is critical.

```python
def latency_optimized_invoke(input_text, max_latency_ms=500):
    """Route to the fastest model within the latency threshold."""
    
    # Define models by typical latency
    models_by_latency = [
        {"model_id": "amazon.titan-text-express-v1", "avg_latency_ms": 250},
        {"model_id": "anthropic.claude-instant-v1", "avg_latency_ms": 500},
        {"model_id": "anthropic.claude-v2", "avg_latency_ms": 1000},
        {"model_id": "anthropic.claude-3-sonnet-20240229-v1:0", "avg_latency_ms": 1200}
    ]
    
    # Filter models by max latency threshold
    candidates = [m for m in models_by_latency if m["avg_latency_ms"] <= max_latency_ms]
    
    if not candidates:
        # If no models meet the threshold, use the fastest available
        candidates = [min(models_by_latency, key=lambda x: x["avg_latency_ms"])]
    
    # Sort by latency (ascending)
    candidates.sort(key=lambda x: x["avg_latency_ms"])
    
    # Try models in order of increasing latency
    for model in candidates:
        try:
            return invoke_model(model["model_id"], input_text)
        except Exception:
            continue
    
    raise Exception("All latency-optimized models failed")
```

### 3. Adaptive Orchestration

Dynamically learn and adapt routing strategies based on performance feedback.

```python
class AdaptiveOrchestrator:
    """Dynamically adapts routing based on model performance metrics."""
    
    def __init__(self, bedrock_client=None):
        """Initialize with a Bedrock client."""
        self.bedrock_client = bedrock_client or get_bedrock_client(profile_name="aws")
        
        # Initialize performance tracking
        self.model_metrics = {}
        self.learning_rate = 0.1  # For updating metrics
        
    def invoke_model(self, input_text, task_type=None):
        """Invoke the best model for the given input and task."""
        # Determine task type if not provided
        if not task_type:
            task_type = self._classify_task(input_text)
        
        # Get candidate models for this task
        candidates = self._get_candidates(task_type)
        
        # Select best model based on current metrics
        selected_model = self._select_best_model(candidates, task_type)
        
        # Invoke the selected model
        start_time = time.time()
        try:
            # Format request and invoke
            request_body = self._format_request(selected_model, input_text)
            response = self.bedrock_client.invoke_model(
                modelId=selected_model,
                body=json.dumps(request_body)
            )
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            
            # Extract and process response
            response_body = json.loads(response["body"].read())
            output = self._extract_output(selected_model, response_body)
            
            # Update metrics (success case)
            self._update_metrics(selected_model, task_type, {
                "success": True,
                "latency": latency,
                "tokens": self._estimate_tokens(input_text, output)
            })
            
            return {
                "output": output,
                "model_id": selected_model,
                "latency_ms": latency
            }
            
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            # Update metrics (failure case)
            self._update_metrics(selected_model, task_type, {
                "success": False,
                "latency": latency,
                "error": str(e)
            })
            
            # Try fallback model
            fallback_model = self._get_fallback(candidates, selected_model)
            if fallback_model:
                return self.invoke_model(input_text, task_type)
            else:
                raise Exception(f"All models failed for task type: {task_type}")
    
    def _classify_task(self, input_text):
        """Classify the input into a task type."""
        # Simple classification logic
        # Could be enhanced with a dedicated classifier model
        if "code" in input_text.lower() or "function" in input_text.lower():
            return "code_generation"
        elif "summarize" in input_text.lower() or "summary" in input_text.lower():
            return "summarization"
        elif "translate" in input_text.lower():
            return "translation"
        else:
            return "general"
    
    def _get_candidates(self, task_type):
        """Get candidate models for a given task type."""
        task_models = {
            "general": [
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-v2",
                "amazon.titan-text-express-v1"
            ],
            "code_generation": [
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-v2"
            ],
            "summarization": [
                "anthropic.claude-3-haiku-20240307-v1:0",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "amazon.titan-text-express-v1"
            ],
            "translation": [
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "amazon.titan-text-express-v1"
            ]
        }
        
        return task_models.get(task_type, task_models["general"])
    
    def _select_best_model(self, candidates, task_type):
        """Select the best model based on historical performance."""
        # If we have performance data, use it to select
        scored_candidates = []
        
        for model_id in candidates:
            metrics = self.model_metrics.get((model_id, task_type), {
                "success_rate": 0.95,  # Initial optimistic rate
                "avg_latency": 1000,   # Initial conservative latency
                "request_count": 0
            })
            
            # Calculate score (higher is better)
            score = metrics["success_rate"] * 10000 / (metrics["avg_latency"] + 100)
            
            # Exploration factor for less-used models
            exploration_bonus = 0.1 / (metrics["request_count"] + 1)
            
            scored_candidates.append({
                "model_id": model_id,
                "score": score + exploration_bonus
            })
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Return highest scoring model
        return scored_candidates[0]["model_id"]
    
    def _update_metrics(self, model_id, task_type, metrics):
        """Update performance metrics for a model."""
        key = (model_id, task_type)
        
        if key not in self.model_metrics:
            self.model_metrics[key] = {
                "success_rate": 1.0 if metrics["success"] else 0.0,
                "avg_latency": metrics["latency"],
                "request_count": 1
            }
        else:
            current = self.model_metrics[key]
            
            # Update with exponential moving average
            current["success_rate"] = (
                (1 - self.learning_rate) * current["success_rate"] + 
                self.learning_rate * (1.0 if metrics["success"] else 0.0)
            )
            
            if metrics["success"]:
                current["avg_latency"] = (
                    (1 - self.learning_rate) * current["avg_latency"] + 
                    self.learning_rate * metrics["latency"]
                )
            
            current["request_count"] += 1
    
    def _get_fallback(self, candidates, failed_model):
        """Get a fallback model after a failure."""
        if not candidates or len(candidates) <= 1:
            return None
            
        # Return next best model that isn't the failed one
        fallback_candidates = [m for m in candidates if m != failed_model]
        if fallback_candidates:
            return fallback_candidates[0]
        return None
        
    # Additional helper methods (_format_request, _extract_output, _estimate_tokens)
    # ...
```

## Best Practices for Multi-Model Orchestration

### 1. Design Patterns

- **Start Simple**: Begin with basic routing patterns and evolve to more complex orchestration as needed
- **Define Clear Boundaries**: Clearly define which models handle which types of tasks
- **Monitor Performance**: Implement comprehensive metrics collection for all models
- **Implement Circuit Breakers**: Prevent cascading failures by detecting and handling persistent errors
- **Use Asynchronous Processing**: When possible, use asynchronous invocation for better throughput

### 2. Error Handling

- **Implement Exponential Backoff**: Use exponential backoff with jitter for retries
- **Distinguish Error Types**: Handle quota errors, timeout errors, and validation errors differently
- **Log Detailed Errors**: Capture detailed error information for debugging
- **Validate Outputs**: Check model outputs for correctness before proceeding

### 3. Quotas and Cost Management

- **Track Quota Usage**: Monitor RPM and TPM usage to avoid quota limits
- **Implement Token Budgeting**: Set token budgets for different request types
- **Cost-Aware Routing**: Consider model costs when designing routing strategies
- **Cache Common Requests**: Implement caching for repeated or similar requests

### 4. Monitoring and Observability

- **Track Key Metrics**: Monitor latency, throughput, error rates, and cost
- **Implement Distributed Tracing**: Use trace IDs to track requests across models
- **Set Up Alerts**: Create alerts for abnormal performance or error patterns
- **Visualize Performance**: Build dashboards to track orchestration effectiveness

## Implementation Example: A Complete Multi-Model System

Here's an example of a complete multi-model orchestration system that combines multiple patterns:

```python
class BedRockOrchestrator:
    """
    Comprehensive orchestration system for AWS Bedrock models.
    
    Combines multiple orchestration patterns (routing, fallbacks, ensembles)
    with adaptive learning and comprehensive monitoring.
    """
    
    def __init__(self, config=None):
        """Initialize with optional configuration."""
        self.bedrock_client = get_bedrock_client(profile_name="aws")
        self.config = config or self._default_config()
        
        # Initialize component orchestrators
        self.router = ModelRouter(bedrock_client=self.bedrock_client)
        self.pipeline = ModelPipeline(bedrock_client=self.bedrock_client)
        self.ensemble = ModelEnsemble(bedrock_client=self.bedrock_client)
        self.fallback = FallbackOrchestrator(bedrock_client=self.bedrock_client)
        
        # Performance tracking
        self.metrics = {}
        self.request_log = []
    
    def _default_config(self):
        """Create default configuration."""
        return {
            "default_strategy": "router",
            "metrics_enabled": True,
            "max_retries": 3,
            "timeout_seconds": 30,
            "default_models": {
                "primary": "anthropic.claude-3-sonnet-20240229-v1:0",
                "fallback": "amazon.titan-text-express-v1",
                "low_latency": "amazon.titan-text-express-v1"
            }
        }
    
    def process_request(self, input_data, strategy=None, models=None, **kwargs):
        """
        Process a request using the specified orchestration strategy.
        
        Args:
            input_data: The input text or image
            strategy: Orchestration strategy to use ("router", "pipeline", "ensemble", "fallback")
            models: Models to use (overrides defaults)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            The processed output based on the selected strategy
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Use default strategy if not specified
        strategy = strategy or self.config["default_strategy"]
        
        # Track request
        request_info = {
            "request_id": request_id,
            "strategy": strategy,
            "start_time": start_time,
            "input_type": "multimodal" if isinstance(input_data, dict) and "image" in input_data else "text"
        }
        
        try:
            # Process based on strategy
            if strategy == "router":
                result = self._apply_router_strategy(input_data, models, **kwargs)
            elif strategy == "pipeline":
                result = self._apply_pipeline_strategy(input_data, models, **kwargs)
            elif strategy == "ensemble":
                result = self._apply_ensemble_strategy(input_data, models, **kwargs)
            elif strategy == "fallback":
                result = self._apply_fallback_strategy(input_data, models, **kwargs)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Update request info with success
            end_time = time.time()
            request_info.update({
                "status": "success",
                "duration_ms": (end_time - start_time) * 1000,
                "models_used": result.get("models_used", []),
                "output_type": result.get("output_type", "text")
            })
            
            # Add output to result
            result["request_id"] = request_id
            
            return result
            
        except Exception as e:
            # Update request info with failure
            end_time = time.time()
            request_info.update({
                "status": "failure",
                "duration_ms": (end_time - start_time) * 1000,
                "error": str(e)
            })
            
            # Re-raise the exception
            raise
            
        finally:
            # Log request info
            self.request_log.append(request_info)
            
            # Update metrics
            self._update_metrics(request_info)
    
    def _apply_router_strategy(self, input_data, models=None, **kwargs):
        """Apply the router orchestration strategy."""
        override_model = models[0] if models and len(models) > 0 else None
        
        if isinstance(input_data, dict) and "text" in input_data and "image" in input_data:
            # Multimodal input
            result, model_id = self.router.route_request(
                input_data["text"], 
                input_data["image"],
                override_model
            )
        else:
            # Text-only input
            result, model_id = self.router.route_request(
                input_data, 
                None,
                override_model
            )
        
        return {
            "output": result,
            "models_used": [model_id],
            "output_type": "text",
            "strategy": "router"
        }
    
    def _apply_pipeline_strategy(self, input_data, models=None, **kwargs):
        """Apply the pipeline orchestration strategy."""
        pipeline_config = kwargs.get("pipeline_config")
        
        if not pipeline_config:
            # Create default pipeline if not provided
            pipeline_config = self._create_default_pipeline(models)
        
        result = self.pipeline.run_pipeline(input_data, pipeline_config)
        
        return {
            "output": result["final_output"],
            "pipeline_results": result["pipeline_results"],
            "models_used": [step["model_id"] for step in pipeline_config],
            "output_type": "pipeline",
            "strategy": "pipeline"
        }
    
    def _apply_ensemble_strategy(self, input_data, models=None, **kwargs):
        """Apply the ensemble orchestration strategy."""
        # Use provided models or defaults
        ensemble_models = models or [
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-v2",
            "amazon.titan-text-express-v1"
        ]
        
        # Get combination strategy
        combination_strategy = kwargs.get("combination_strategy", "voting")
        
        result = self.ensemble.run_ensemble(
            input_data, 
            ensemble_models,
            combination_strategy
        )
        
        return {
            "output": result["final_output"],
            "model_outputs": result["model_outputs"],
            "models_used": ensemble_models,
            "output_type": "ensemble",
            "strategy": "ensemble",
            "combination_strategy": combination_strategy
        }
    
    def _apply_fallback_strategy(self, input_data, models=None, **kwargs):
        """Apply the fallback orchestration strategy."""
        # Use provided models or defaults
        fallback_chain = models or [
            self.config["default_models"]["primary"],
            self.config["default_models"]["fallback"]
        ]
        
        # Get timeout
        timeout = kwargs.get("timeout", self.config["timeout_seconds"])
        
        result = self.fallback.invoke_with_fallback(
            input_data,
            fallback_chain,
            timeout
        )
        
        return {
            "output": result["output"],
            "model_id": result["model_id"],
            "attempts": result["attempts"],
            "models_used": [a["model_id"] for a in result["attempts"]],
            "output_type": "text",
            "strategy": "fallback"
        }
    
    def _create_default_pipeline(self, models=None):
        """Create a default pipeline configuration."""
        # Use provided models or defaults
        if not models or len(models) < 2:
            models = [
                self.config["default_models"]["primary"],
                self.config["default_models"]["primary"]
            ]
        
        # Create a simple two-step pipeline
        return [
            {
                "model_id": models[0],
                "process_fn": lambda input_data: {
                    "prompt": f"Human: Extract the key information from the following text:\n\n{input_data}\n\nAssistant:",
                    "max_tokens_to_sample": 1000,
                    "temperature": 0.7
                }
            },
            {
                "model_id": models[1],
                "process_fn": lambda extracted_info: {
                    "prompt": f"Human: Analyze the following extracted information and provide insights:\n\n{extracted_info}\n\nAssistant:",
                    "max_tokens_to_sample": 1000,
                    "temperature": 0.7
                }
            }
        ]
    
    def _update_metrics(self, request_info):
        """Update performance metrics based on request info."""
        if not self.config["metrics_enabled"]:
            return
            
        # Update metrics for the strategy
        strategy = request_info["strategy"]
        if strategy not in self.metrics:
            self.metrics[strategy] = {
                "request_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_duration_ms": 0,
                "avg_duration_ms": 0
            }
        
        metrics = self.metrics[strategy]
        metrics["request_count"] += 1
        
        if request_info["status"] == "success":
            metrics["success_count"] += 1
        else:
            metrics["failure_count"] += 1
            
        metrics["total_duration_ms"] += request_info["duration_ms"]
        metrics["avg_duration_ms"] = metrics["total_duration_ms"] / metrics["request_count"]
        
        # Update metrics for each model used
        if "models_used" in request_info:
            for model_id in request_info["models_used"]:
                if model_id not in self.metrics:
                    self.metrics[model_id] = {
                        "request_count": 0,
                        "success_count": 0,
                        "failure_count": 0
                    }
                
                model_metrics = self.metrics[model_id]
                model_metrics["request_count"] += 1
                
                if request_info["status"] == "success":
                    model_metrics["success_count"] += 1
                else:
                    model_metrics["failure_count"] += 1
    
    def get_metrics(self):
        """Get the current metrics."""
        return self.metrics
    
    def get_request_log(self, limit=None):
        """Get the request log with optional limit."""
        if limit:
            return self.request_log[-limit:]
        return self.request_log
```

## Conclusion

Multi-model orchestration in AWS Bedrock offers powerful ways to optimize inference across multiple foundation models. By implementing routing, pipeline, ensemble, and fallback patterns, you can create systems that leverage the unique strengths of different models while managing costs, optimizing performance, and ensuring reliability.

The key to successful orchestration is understanding the capabilities and limitations of each model, implementing robust error handling and monitoring, and designing flexible architectures that can adapt to changing requirements.

As you implement these patterns, focus on:

1. **Clarity**: Design clear, understandable orchestration logic
2. **Reliability**: Implement comprehensive error handling and fallbacks
3. **Adaptability**: Build systems that can evolve as models and requirements change
4. **Efficiency**: Optimize for cost, latency, and throughput based on your specific needs
5. **Observability**: Monitor performance to identify opportunities for improvement

By following these principles and implementing the patterns described in this guide, you can build sophisticated inference systems that maximize the value of AWS Bedrock foundation models for your applications.