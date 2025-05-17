"""
Example demonstrating usage of the synchronous inference client.

This example shows how to use the basic BedrockClient and QuotaAwareBedrockClient
for synchronous inference requests to AWS Bedrock models.
"""

import time
import logging
from src.inference import BedrockClient, QuotaAwareBedrockClient, QuotaExceededException


def basic_client_example():
    """Demonstrate basic client usage."""
    print("\n=== Basic Client Example ===\n")
    
    # Create a client for Claude
    client = BedrockClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Basic invocation
    try:
        print("Sending request to Claude model...")
        response = client.invoke(
            prompt="Explain quantum computing in simple terms.",
            max_tokens=500,
            temperature=0.7
        )
        
        print("\nModel Response:")
        print(response["output"])
        
        print("\nToken Usage:")
        print(f"Input tokens: {response.get('input_tokens', 'unknown')}")
        print(f"Output tokens: {response.get('output_tokens', 'unknown')}")
        print(f"Total tokens: {response.get('total_tokens', 'unknown')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    print(f"Request count: {metrics['request_count']}")
    print(f"Token count: {metrics['token_count']}")
    print(f"Error count: {metrics['error_count']}")


def quota_aware_client_example():
    """Demonstrate quota-aware client usage."""
    print("\n=== Quota-Aware Client Example ===\n")
    
    # Create a quota-aware client for Claude with rate limits
    client = QuotaAwareBedrockClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        max_rpm=60,  # 60 requests per minute
        max_tpm=100000  # 100K tokens per minute
    )
    
    # Process multiple requests to demonstrate quota management
    num_requests = 5
    successful = 0
    throttled = 0
    
    print(f"Sending {num_requests} requests with quota management...")
    
    for i in range(num_requests):
        try:
            print(f"\nRequest {i+1}/{num_requests}:")
            
            # Check quota metrics before request
            metrics = client.get_metrics()
            print(f"RPM remaining: {metrics.get('rpm_remaining', 'N/A')}")
            print(f"TPM remaining: {metrics.get('tpm_remaining', 'N/A')}")
            
            # Invoke model (wait for quota if needed)
            response = client.invoke(
                prompt=f"Write a very short poem about science, topic #{i+1}.",
                max_tokens=200,
                wait_for_quota=True
            )
            
            # Print abbreviated response
            output = response["output"]
            print(f"Response: {output[:100]}..." if len(output) > 100 else output)
            print(f"Tokens: {response.get('total_tokens', 'unknown')}")
            
            successful += 1
            
        except QuotaExceededException:
            print("Quota exceeded, would wait for quota reset")
            throttled += 1
            
        except Exception as e:
            print(f"Error: {str(e)}")
            throttled += 1
        
        # Add a small delay between requests for the example
        time.sleep(0.5)
    
    # Print summary
    print("\nExecution Summary:")
    print(f"Successful requests: {successful}/{num_requests}")
    print(f"Throttled requests: {throttled}/{num_requests}")
    
    # Print final metrics
    metrics = client.get_metrics()
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def multiple_models_example():
    """Demonstrate using multiple models with the same client interface."""
    print("\n=== Multiple Models Example ===\n")
    
    # Test multiple models with the same prompt
    models = [
        "anthropic.claude-3-sonnet-20240229-v1:0",  # Claude
        "meta.llama2-13b-chat-v1",                 # Llama 2
        "amazon.titan-text-express-v1"             # Titan
    ]
    
    prompt = "What are three key benefits of cloud computing?"
    
    for model_id in models:
        print(f"\n--- Model: {model_id} ---\n")
        
        try:
            # Create client for this model
            client = BedrockClient(model_id=model_id)
            
            # Invoke model
            response = client.invoke(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )
            
            # Print abbreviated response
            output = response["output"]
            print(f"Response: {output[:200]}..." if len(output) > 200 else output)
            
            # Print token usage if available
            if "total_tokens" in response:
                print(f"\nToken Usage:")
                print(f"Input tokens: {response.get('input_tokens', 'unknown')}")
                print(f"Output tokens: {response.get('output_tokens', 'unknown')}")
            
        except Exception as e:
            print(f"Error with model {model_id}: {str(e)}")


def system_prompt_example():
    """Demonstrate usage of system prompts."""
    print("\n=== System Prompt Example ===\n")
    
    # Create a client for Claude
    client = BedrockClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Define a system prompt
    system_prompt = """You are a technical expert who provides concise, 
    technical explanations. Keep all responses under 100 words and focus 
    on technical accuracy."""
    
    try:
        # Invoke with system prompt
        response = client.invoke(
            prompt="Explain how HTTPS encryption works.",
            system_prompt=system_prompt,
            max_tokens=500
        )
        
        print("Response with system prompt:")
        print(response["output"])
        print(f"Length in words: {len(response['output'].split())}")
        
        # Compare to response without system prompt
        response_no_system = client.invoke(
            prompt="Explain how HTTPS encryption works.",
            max_tokens=500
        )
        
        print("\nResponse without system prompt:")
        print(response_no_system["output"])
        print(f"Length in words: {len(response_no_system['output'].split())}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    basic_client_example()
    quota_aware_client_example()
    multiple_models_example()
    system_prompt_example()
    
    print("\nAll examples completed.")