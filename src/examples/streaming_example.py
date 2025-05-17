"""
Example demonstrating usage of the streaming inference client.

This example shows how to use the BedrockStreamingClient for streaming
inference requests to AWS Bedrock models.
"""

import time
import logging
from src.inference import BedrockStreamingClient


def basic_streaming_example():
    """Demonstrate basic streaming functionality."""
    print("\n=== Basic Streaming Example ===\n")
    
    # Create a streaming client for Claude
    client = BedrockStreamingClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Basic streaming example
    try:
        print("Streaming response from Claude model:")
        print("-" * 50)
        
        # Use iterator approach
        for chunk in client.invoke_stream(
            prompt="Write a short poem about cloud computing.",
            max_tokens=200
        ):
            print(chunk, end="", flush=True)
            time.sleep(0.01)  # Slow down output for better demo visualization
        
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    print(f"Request count: {metrics['request_count']}")
    print(f"Token count: {metrics['token_count']}")
    print(f"Error count: {metrics['error_count']}")


def callback_streaming_example():
    """Demonstrate callback-based streaming."""
    print("\n=== Callback-Based Streaming Example ===\n")
    
    # Create a streaming client for Claude
    client = BedrockStreamingClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Define callback functions
    def on_content_chunk(chunk):
        """Called for each content chunk."""
        print(chunk, end="", flush=True)
        time.sleep(0.01)  # Slow down output for better demo visualization
    
    def on_complete(result):
        """Called when streaming completes."""
        print("\n" + "-" * 50)
        print(f"Streaming complete. Generated approximately {result['estimated_tokens']} tokens.")
        print(f"Total chunks: {result['total_chunks']}")
    
    def on_error(error):
        """Called when an error occurs."""
        print(f"\nError occurred: {str(error)}")
    
    # Stream with callbacks
    try:
        print("Streaming with callbacks:")
        print("-" * 50)
        
        client.invoke_stream_with_callbacks(
            prompt="Explain the benefits of streaming responses in AI applications.",
            on_content=on_content_chunk,
            on_complete=on_complete,
            on_error=on_error,
            max_tokens=200
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")


def multiple_models_streaming_example():
    """Demonstrate streaming with multiple models."""
    print("\n=== Multiple Models Streaming Example ===\n")
    
    # Test multiple models with the same prompt
    models = [
        "anthropic.claude-3-sonnet-20240229-v1:0",  # Claude
        "meta.llama2-13b-chat-v1",                 # Llama 2
        "amazon.titan-text-express-v1"             # Titan
    ]
    
    prompt = "Write a short story about robots in space."
    
    for model_id in models:
        print(f"\n--- Streaming from Model: {model_id} ---\n")
        
        try:
            # Create streaming client for this model
            client = BedrockStreamingClient(model_id=model_id)
            
            # Stream from model
            print("-" * 50)
            
            content_buffer = []
            
            for chunk in client.invoke_stream(
                prompt=prompt,
                max_tokens=200,
                temperature=0.7
            ):
                content_buffer.append(chunk)
                print(chunk, end="", flush=True)
                time.sleep(0.01)  # Slow down output for better demo visualization
            
            print("\n" + "-" * 50)
            
            # Calculate approximate length
            full_content = "".join(content_buffer)
            print(f"Approximate length: {len(full_content.split())} words")
            
        except Exception as e:
            print(f"Error with model {model_id}: {str(e)}")


def system_prompt_streaming_example():
    """Demonstrate streaming with system prompts."""
    print("\n=== System Prompt Streaming Example ===\n")
    
    # Create a streaming client for Claude
    client = BedrockStreamingClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Define a system prompt
    system_prompt = """You are a creative storyteller who writes engaging, 
    short stories. Keep all responses under 200 words and focus on 
    vivid imagery and compelling characters."""
    
    try:
        # Stream with system prompt
        print("Streaming response with system prompt:")
        print("-" * 50)
        
        for chunk in client.invoke_stream(
            prompt="Write a short adventure story.",
            system_prompt=system_prompt,
            max_tokens=300
        ):
            print(chunk, end="", flush=True)
            time.sleep(0.01)  # Slow down output for better demo visualization
        
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"Error: {str(e)}")


def interactive_streaming_demo():
    """Interactive demo with streaming responses."""
    print("\n=== Interactive Streaming Demo ===\n")
    
    # Create a streaming client for Claude
    client = BedrockStreamingClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Define system prompt for the assistant
    system_prompt = """You are a helpful assistant that provides informative,
    concise responses to user questions."""
    
    print("Interactive chat with streaming responses. Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Exiting interactive demo.")
            break
        
        # Stream the response
        print("\nAssistant: ", end="", flush=True)
        
        try:
            for chunk in client.invoke_stream(
                prompt=user_input,
                system_prompt=system_prompt,
                max_tokens=500
            ):
                print(chunk, end="", flush=True)
                time.sleep(0.01)  # Slight delay for better visual experience
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("-" * 50)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    basic_streaming_example()
    callback_streaming_example()
    multiple_models_streaming_example()
    system_prompt_streaming_example()
    
    # Run interactive demo if desired
    choice = input("\nDo you want to try the interactive demo? (y/n): ")
    if choice.lower() in ['y', 'yes']:
        interactive_streaming_demo()
    
    print("\nAll examples completed.")