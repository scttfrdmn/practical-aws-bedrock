"""
Example demonstrating usage of the AWS Bedrock Converse API client.

This example shows various ways to use the ConverseClient for creating
conversational AI applications with AWS Bedrock models.
"""

import os
import json
import time
import logging
import sys
from typing import Dict, Any, Optional, List, Generator

# Adjust import path based on project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.converse.basic.simple_chat import ConverseClient

# Optional rich library for better console formatting
try:
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Note: Install 'rich' package for better output formatting")


def basic_conversation_example():
    """Demonstrate basic conversation capabilities."""
    print("\n=== Basic Conversation Example ===\n")
    
    # Create a client for Claude
    client = ConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Create a conversation with a system prompt
    system_prompt = "You are a helpful assistant specializing in AWS services."
    conversation_id = client.create_conversation(system_prompt=system_prompt)
    
    print(f"Created conversation: {conversation_id}")
    
    try:
        # Send the first message
        print("\nSending first message...")
        response1 = client.send_message(
            conversation_id=conversation_id,
            message="What is AWS Bedrock and what can I do with it?",
            max_tokens=500
        )
        
        print("\n--- Assistant Response ---")
        if HAS_RICH:
            console.print(Markdown(response1))
        else:
            print(response1)
        
        # Send a follow-up message in the same conversation
        print("\nSending follow-up message...")
        response2 = client.send_message(
            conversation_id=conversation_id,
            message="Which foundation models does it support?",
            max_tokens=500
        )
        
        print("\n--- Assistant Response ---")
        if HAS_RICH:
            console.print(Markdown(response2))
        else:
            print(response2)
        
        # Get and display conversation history
        history = client.get_conversation_history(conversation_id)
        print(f"\nConversation has {len(history['messages'])} messages")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def interactive_chat_example():
    """Run an interactive chat session."""
    print("\n=== Interactive Chat Example ===\n")
    
    # Create a client
    client = ConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # System prompt to guide the model's behavior
    system_prompt = """
    You are a helpful, friendly AI assistant. Your goal is to provide accurate,
    clear, and concise responses to the user's questions. If you don't know
    something, admit it rather than making up information.
    """
    
    # Create a new conversation
    conversation_id = client.create_conversation(system_prompt=system_prompt)
    print(f"Started new conversation: {conversation_id}\n")
    
    # Welcome message
    if HAS_RICH:
        console.print(Markdown("# Interactive Chat with AWS Bedrock"))
        console.print(Markdown("> Using the Converse API with Claude 3 Sonnet"))
        console.print(Markdown("> Type 'exit' to end the conversation\n"))
    else:
        print("# Interactive Chat with AWS Bedrock")
        print("> Using the Converse API with Claude 3 Sonnet")
        print("> Type 'exit' to end the conversation\n")
    
    # Start chat loop
    try:
        while True:
            # Get user input
            user_message = input("\n[You]: ")
            
            # Check for exit command
            if user_message.lower() in ["exit", "quit", "bye"]:
                break
            
            print("\n[Assistant]: ", end="", flush=True)
            
            # Send message and get response
            start_time = time.time()
            response = client.send_message(
                conversation_id=conversation_id,
                message=user_message,
                max_tokens=1000
            )
            end_time = time.time()
            
            # Display response
            if HAS_RICH:
                console.print(Markdown(response))
            else:
                print(response)
                
            # Show timing information
            print(f"[Response time: {end_time - start_time:.2f}s]")
            
    except KeyboardInterrupt:
        print("\nChat session ended by user.")
    
    except Exception as e:
        print(f"Error during chat: {str(e)}")
    
    finally:
        print(f"\nConversation ended. ID: {conversation_id}")
        
        # Display final metrics
        metrics = client.get_metrics()
        print("\nSession Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


def stateful_conversation_example():
    """Demonstrate conversation state management across multiple messages."""
    print("\n=== Stateful Conversation Example ===\n")
    
    # Create a client
    client = ConverseClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Create a new conversation with a task-specific system prompt
    system_prompt = """
    You are an AI assistant specializing in creative writing. Your goal is to help
    the user create a short story through a collaborative process. Each of your
    responses should build upon previous elements of the story while introducing
    new creative elements.
    """
    
    conversation_id = client.create_conversation(system_prompt=system_prompt)
    print(f"Created conversation: {conversation_id}")
    
    try:
        # Initial prompt to start the story
        print("\nStarting the story creation process...")
        prompt1 = "Let's create a short sci-fi story set on a distant planet. Can you start with an opening paragraph?"
        
        response1 = client.send_message(
            conversation_id=conversation_id,
            message=prompt1,
            max_tokens=500
        )
        
        print("\n--- Story Beginning ---")
        if HAS_RICH:
            console.print(Markdown(response1))
        else:
            print(response1)
        
        # Second message that asks to develop a character
        print("\nDeveloping a character...")
        prompt2 = "Great! Now introduce a main character who discovers something unexpected in this setting."
        
        response2 = client.send_message(
            conversation_id=conversation_id,
            message=prompt2,
            max_tokens=500
        )
        
        print("\n--- Character Introduction ---")
        if HAS_RICH:
            console.print(Markdown(response2))
        else:
            print(response2)
        
        # Third message that introduces a conflict
        print("\nIntroducing a conflict...")
        prompt3 = "Now let's introduce a conflict or challenge that the character must face in this world."
        
        response3 = client.send_message(
            conversation_id=conversation_id,
            message=prompt3,
            max_tokens=500
        )
        
        print("\n--- Conflict Introduction ---")
        if HAS_RICH:
            console.print(Markdown(response3))
        else:
            print(response3)
        
        # Final message to wrap up the story
        print("\nWrapping up the story...")
        prompt4 = "Great! Now please provide a resolution and conclusion to our short story."
        
        response4 = client.send_message(
            conversation_id=conversation_id,
            message=prompt4,
            max_tokens=500
        )
        
        print("\n--- Story Conclusion ---")
        if HAS_RICH:
            console.print(Markdown(response4))
        else:
            print(response4)
        
        # Show complete story development process
        print("\n=== Complete Story Development Process ===\n")
        history = client.get_conversation_history(conversation_id)
        
        # Filter out system message
        messages = [msg for msg in history["messages"] if msg["role"] != "system"]
        
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                print(f"\n[User Prompt {i//2 + 1}]:")
                print(content)
            else:
                print(f"\n[AI Response {i//2 + 1}]:")
                if HAS_RICH:
                    console.print(Markdown(content))
                else:
                    print(content)
        
    except Exception as e:
        print(f"Error in story development: {str(e)}")


def different_models_example():
    """Compare responses from different foundation models."""
    print("\n=== Different Models Example ===\n")
    
    # List of models to try
    models = [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-instant-v1",
        "amazon.titan-text-express-v1"
    ]
    
    # The prompt to use for all models
    prompt = "What are the key advantages of using foundation models for business applications?"
    
    for model_id in models:
        print(f"\n--- Testing Model: {model_id} ---\n")
        
        try:
            # Create a client for this model
            client = ConverseClient(model_id=model_id)
            
            # Create a conversation
            conversation_id = client.create_conversation()
            
            # Time the response
            start_time = time.time()
            
            # Send the prompt
            response = client.send_message(
                conversation_id=conversation_id,
                message=prompt,
                max_tokens=500
            )
            
            end_time = time.time()
            
            # Display the response
            print(f"Response (took {end_time - start_time:.2f}s):")
            if HAS_RICH:
                console.print(Markdown(response))
            else:
                print(response)
            
            # Get metrics
            metrics = client.get_metrics()
            print("\nMetrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"Error with model {model_id}: {str(e)}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Choose which example to run
    # basic_conversation_example()
    # stateful_conversation_example()
    # different_models_example()
    interactive_chat_example()  # This one is interactive
    
    print("\nExample completed.")