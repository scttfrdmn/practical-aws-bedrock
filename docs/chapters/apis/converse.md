---
layout: chapter
title: Conversational AI with AWS Bedrock Converse API
difficulty: intermediate
time-estimate: 45 minutes
last-updated: May 16, 2024
---

# Conversational AI with AWS Bedrock Converse API

> "You need to build a robust, production-ready chatbot that handles conversations naturally and scales with your users. Let's solve this with AWS Bedrock's Converse API."

## The Problem

---

**Scenario**: You're leading the development of a customer service automation platform that needs to provide natural, helpful interactions across various channels. After experimenting with basic foundation model prompting, you've encountered several challenges:

1. Managing conversation history and context is complex and error-prone
2. Different models require different prompt formats, making model switching difficult
3. Tracking token usage and optimizing for quotas requires significant custom code
4. Handling streaming responses needs complex implementation
5. Adding function calling capabilities requires model-specific implementations

You need a robust solution that:
- Simplifies conversation management across models
- Handles memory and context window constraints automatically 
- Provides a consistent interface regardless of the underlying model
- Enables streaming for responsive user experiences
- Supports function calling for integrating with your backend systems

---

## Key Concepts Explained

### What is the Converse API?

The AWS Bedrock Converse API is a higher-level interface for building conversational applications with foundation models. It abstracts away the complexities of:

1. **Conversation Management**: Tracking history and maintaining context
2. **Model-Specific Formatting**: Converting conversations into model-specific prompt formats
3. **Memory Optimization**: Managing token usage and context window constraints
4. **Response Streaming**: Enabling real-time, token-by-token responses
5. **Function Calling**: Integrating model capabilities with your application functions

Think of it as a specialized layer built on top of the basic foundation model APIs we've explored so far:

```
┌─────────────────────────────────────────────────┐
│                Your Application                  │
└─────────────────────────────────┬───────────────┘
                                  │
┌─────────────────────────────────▼───────────────┐
│               Converse API                       │
└─┬─────────────────┬───────────────────┬─────────┘
  │                 │                   │
┌─▼────────┐    ┌───▼────┐        ┌────▼─────┐
│InvokeModel│    │Streaming│        │Function  │
│           │    │API      │        │Calling   │
└───────────┘    └─────────┘        └──────────┘
```

### Core Components of the Converse API

The Converse API introduces several key concepts:

1. **Conversations**: Persistent objects that maintain state and history
2. **Messages**: Individual exchanges within a conversation
3. **System Prompts**: Instructions that guide the model's behavior throughout the conversation
4. **Memory Management**: Automatic handling of context window constraints
5. **Tool Definitions**: Functions that the model can call when needed

### How the Converse API Differs from Direct Model Invocation

| Feature | Direct Model Invocation | Converse API |
|---------|------------------------|--------------|
| History Management | Manual (you track messages) | Automatic (API manages state) |
| Prompt Formatting | Model-specific | Abstracted (works across models) |
| Context Window | Manual optimization | Automatic handling |
| Streaming | Model-specific implementation | Standardized interface |
| Function Calling | Model-specific formats | Unified approach |
| State Persistence | Developer responsibility | Handled by the API |

## Step-by-Step Implementation

Let's build a complete conversational AI solution using the AWS Bedrock Converse API.

### 1. Setting Up Your Environment

First, we need to set up our environment with the required dependencies:

```bash
# Install required packages
pip install boto3 rich
```

You'll need these IAM permissions:
- `bedrock:CreateConversation`
- `bedrock:GetConversation`
- `bedrock:ListConversations`
- `bedrock:DeleteConversation`
- `bedrock:ConversationSendMessage`

### 2. Creating a Basic Conversation Client

Let's start by creating a basic client for handling conversations:

```python
import boto3
import json
import logging
from typing import Dict, List, Any, Optional, Generator
from rich.console import Console
from rich.markdown import Markdown

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bedrock_converse")

# Set up rich console for better output formatting
console = Console()

class ConverseClient:
    """
    A client for interacting with AWS Bedrock's Converse API.
    
    This client provides a high-level interface for creating and managing
    conversations with foundation models through the Converse API.
    """
    
    def __init__(
        self,
        model_id: str,
        profile_name: Optional[str] = None,
        region_name: str = "us-west-2"
    ):
        """
        Initialize the Converse client.
        
        Args:
            model_id: The Bedrock model identifier
            profile_name: AWS profile name (optional)
            region_name: AWS region for Bedrock
        """
        # Create session with profile if specified
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
        else:
            # Use default credentials
            session = boto3.Session()
            
        # Create Bedrock client
        self.bedrock = session.client('bedrock', region_name=region_name)
        
        # Store configuration
        self.model_id = model_id
        self.region = region_name
        
        logger.info(f"Initialized Converse client for model: {model_id}")
    
    def create_conversation(
        self,
        system_prompt: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Create a new conversation or retrieve an existing one.
        
        Args:
            system_prompt: Optional system instructions for the model
            conversation_id: Optional existing conversation ID to retrieve
            
        Returns:
            The conversation ID
        """
        try:
            if conversation_id:
                # Retrieve existing conversation
                response = self.bedrock.get_conversation(
                    conversationId=conversation_id
                )
                logger.info(f"Retrieved existing conversation: {conversation_id}")
                return conversation_id
            else:
                # Create new conversation
                args = {
                    "modelId": self.model_id
                }
                
                # Add system prompt if provided
                if system_prompt:
                    args["systemPrompt"] = system_prompt
                
                response = self.bedrock.create_conversation(**args)
                conversation_id = response.get("conversationId")
                
                logger.info(f"Created new conversation: {conversation_id}")
                return conversation_id
                
        except Exception as e:
            logger.error(f"Error creating/retrieving conversation: {str(e)}")
            raise
    
    def send_message(
        self,
        conversation_id: str,
        message: str,
        additional_model_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a message to the conversation and get the response.
        
        Args:
            conversation_id: The conversation identifier
            message: The user message to send
            additional_model_params: Optional model parameters
            
        Returns:
            The model's response text
        """
        try:
            # Prepare request
            request = {
                "conversationId": conversation_id,
                "message": message
            }
            
            # Add additional model parameters if provided
            if additional_model_params:
                request["additionalModelParameters"] = additional_model_params
            
            # Send message
            response = self.bedrock.converse_conversation(**request)
            
            # Extract and return response
            return response.get("output", {}).get("message", "")
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise
    
    def list_conversations(self, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        List existing conversations.
        
        Args:
            max_results: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation information
        """
        try:
            response = self.bedrock.list_conversations(
                maxResults=max_results
            )
            
            conversations = response.get("conversations", [])
            logger.info(f"Retrieved {len(conversations)} conversations")
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
            raise
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: The conversation to delete
            
        Returns:
            True if successful
        """
        try:
            self.bedrock.delete_conversation(
                conversationId=conversation_id
            )
            
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            raise
    
    def get_conversation_details(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a conversation.
        
        Args:
            conversation_id: The conversation identifier
            
        Returns:
            Dictionary with conversation details
        """
        try:
            response = self.bedrock.get_conversation(
                conversationId=conversation_id
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting conversation details: {str(e)}")
            raise
```

### 3. Building an Interactive Chat Application

With our client in place, let's create a simple interactive chat application:

```python
def interactive_chat():
    """Run an interactive chat session with the Converse API."""
    
    # Create client
    client = ConverseClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    
    # System prompt to guide the model's behavior
    system_prompt = """
    You are a helpful, friendly AI assistant. Your goal is to provide accurate,
    clear, and concise responses to the user's questions. If you don't know
    something, admit it rather than making up information.
    """
    
    # Create a new conversation
    conversation_id = client.create_conversation(system_prompt=system_prompt)
    print(f"Started new conversation: {conversation_id}")
    
    # Welcome message
    console.print(Markdown("# Interactive Chat with AWS Bedrock"))
    console.print(Markdown("> Using the Converse API with Claude 3 Sonnet"))
    console.print(Markdown("> Type 'exit' to end the conversation\n"))
    
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
            response = client.send_message(conversation_id, user_message)
            
            # Display response as markdown
            console.print(Markdown(response))
            
    except KeyboardInterrupt:
        print("\nChat session ended by user.")
    
    except Exception as e:
        print(f"Error during chat: {str(e)}")
    
    finally:
        print(f"\nConversation ended. ID: {conversation_id}")
        
        # Note: In a real application, you might want to delete the conversation
        # or store the ID for later continuation

if __name__ == "__main__":
    interactive_chat()
```

### 4. Implementing Streaming for Real-Time Responses

For a more responsive user experience, let's add streaming support:

```python
def send_message_streaming(
    self,
    conversation_id: str,
    message: str,
    additional_model_params: Optional[Dict[str, Any]] = None
) -> Generator[str, None, None]:
    """
    Send a message and stream the response.
    
    Args:
        conversation_id: The conversation ID
        message: The user message
        additional_model_params: Optional model parameters
        
    Yields:
        Text chunks as they are generated
    """
    try:
        # Prepare request
        request = {
            "conversationId": conversation_id,
            "message": message
        }
        
        # Add additional model parameters if provided
        if additional_model_params:
            request["additionalModelParameters"] = additional_model_params
        
        # Send streaming request
        response = self.bedrock.converse_conversation_stream(**request)
        
        # Process streaming response
        for event in response.get("stream", []):
            if "output" in event and "chunk" in event["output"]:
                # Extract and yield text chunk
                chunk = event["output"]["chunk"]
                if chunk:
                    yield chunk
                    
    except Exception as e:
        logger.error(f"Error in streaming: {str(e)}")
        yield f"\nError: {str(e)}"

# Add this to the ConverseClient class

def interactive_streaming_chat():
    """Run an interactive chat session with streaming responses."""
    
    # Create client
    client = ConverseClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    
    # System prompt to guide the model's behavior
    system_prompt = """
    You are a helpful, friendly AI assistant. Your goal is to provide accurate,
    clear, and concise responses to the user's questions. If you don't know
    something, admit it rather than making up information.
    """
    
    # Create a new conversation
    conversation_id = client.create_conversation(system_prompt=system_prompt)
    print(f"Started new conversation: {conversation_id}")
    
    # Welcome message
    console.print(Markdown("# Interactive Chat with AWS Bedrock (Streaming)"))
    console.print(Markdown("> Using the Converse API with Claude 3 Sonnet"))
    console.print(Markdown("> Type 'exit' to end the conversation\n"))
    
    # Start chat loop
    try:
        while True:
            # Get user input
            user_message = input("\n[You]: ")
            
            # Check for exit command
            if user_message.lower() in ["exit", "quit", "bye"]:
                break
            
            print("\n[Assistant]: ", end="", flush=True)
            
            # Send message and stream response
            full_response = ""
            for chunk in client.send_message_streaming(conversation_id, user_message):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            # Store full response for later use if needed
            # (storing not shown in this example)
            
    except KeyboardInterrupt:
        print("\nChat session ended by user.")
    
    except Exception as e:
        print(f"Error during chat: {str(e)}")
    
    finally:
        print(f"\nConversation ended. ID: {conversation_id}")
```

### 5. Adding Function Calling Capabilities

To integrate your AI with external systems, let's implement function calling:

```python
def send_message_with_tools(
    self,
    conversation_id: str,
    message: str,
    tools: List[Dict[str, Any]],
    tool_choice: Optional[str] = None,
    additional_model_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Send a message with function calling capabilities.
    
    Args:
        conversation_id: The conversation ID
        message: The user message
        tools: List of tool definitions (functions)
        tool_choice: Optional specified tool to use
        additional_model_params: Optional model parameters
        
    Returns:
        Response that may include function calls
    """
    try:
        # Prepare request
        request = {
            "conversationId": conversation_id,
            "message": message,
            "tools": tools
        }
        
        # Add tool choice if specified
        if tool_choice:
            request["toolChoice"] = tool_choice
            
        # Add additional model parameters if provided
        if additional_model_params:
            request["additionalModelParameters"] = additional_model_params
        
        # Send message
        response = self.bedrock.converse_conversation(**request)
        
        # Process and return complete response
        return response.get("output", {})
            
    except Exception as e:
        logger.error(f"Error using tools: {str(e)}")
        raise

# Example tool definitions
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Example function to handle tool calls
def handle_tool_calls(tool_calls):
    """Process tool calls from the model."""
    results = []
    
    for call in tool_calls:
        if call["type"] == "function":
            function_name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"])
            
            if function_name == "get_weather":
                # In a real application, this would call a weather API
                location = arguments.get("location", "")
                unit = arguments.get("unit", "celsius")
                
                # Mock response
                weather_data = {
                    "location": location,
                    "temperature": 22 if unit == "celsius" else 72,
                    "unit": unit,
                    "condition": "sunny",
                    "humidity": 45,
                    "wind_speed": 10
                }
                
                results.append({
                    "tool_call_id": call["function"]["tool_call_id"],
                    "function_response": json.dumps(weather_data)
                })
    
    return results

def weather_assistant_demo():
    """Demo a weather assistant using function calling."""
    
    # Create client
    client = ConverseClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    
    # System prompt for a weather assistant
    system_prompt = """
    You are a weather assistant that helps users get current weather information.
    Use the provided weather function to fetch data when users ask about the weather.
    Always confirm the location before making a weather request.
    """
    
    # Create a new conversation
    conversation_id = client.create_conversation(system_prompt=system_prompt)
    print(f"Started new weather assistant conversation: {conversation_id}")
    
    # Welcome message
    console.print(Markdown("# Weather Assistant with Function Calling"))
    console.print(Markdown("> Ask about the weather in any location"))
    console.print(Markdown("> Type 'exit' to end the conversation\n"))
    
    # Start chat loop
    try:
        while True:
            # Get user input
            user_message = input("\n[You]: ")
            
            # Check for exit command
            if user_message.lower() in ["exit", "quit", "bye"]:
                break
            
            print("\n[Assistant]: ", end="")
            
            # Send message with tools
            response = client.send_message_with_tools(
                conversation_id=conversation_id,
                message=user_message,
                tools=WEATHER_TOOLS
            )
            
            # Check if the model wants to call a function
            if "tool_calls" in response and response["tool_calls"]:
                print("Checking weather data...\n")
                
                # Process function calls
                tool_results = handle_tool_calls(response["tool_calls"])
                
                # Send function results back to continue the conversation
                response = client.bedrock.converse_conversation(
                    conversationId=conversation_id,
                    toolResults=tool_results
                )
                
                # Get the final response
                final_message = response.get("output", {}).get("message", "")
                console.print(Markdown(final_message))
            else:
                # Regular response without function call
                message = response.get("message", "")
                console.print(Markdown(message))
            
    except KeyboardInterrupt:
        print("\nWeather assistant ended by user.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        print(f"\nConversation ended. ID: {conversation_id}")
```

### 6. Creating a Memory-Optimized Client

For handling long conversations efficiently, let's implement memory management:

```python
class MemoryManagedConverseClient(ConverseClient):
    """
    A Converse client with automatic memory management for long conversations.
    
    This extension optimizes token usage by:
    1. Tracking approximate token usage
    2. Summarizing conversation history when needed
    3. Maintaining the most relevant context
    """
    
    def __init__(
        self,
        model_id: str,
        max_history_tokens: int = 8000,
        profile_name: Optional[str] = None,
        region_name: str = "us-west-2"
    ):
        """
        Initialize the memory-managed client.
        
        Args:
            model_id: The Bedrock model identifier
            max_history_tokens: Maximum tokens to keep in history
            profile_name: AWS profile name
            region_name: AWS region for Bedrock
        """
        super().__init__(model_id, profile_name, region_name)
        
        # Memory management configuration
        self.max_history_tokens = max_history_tokens
        
        # Token usage tracking
        self.conversation_tokens = {}  # conversation_id -> token count
        
        logger.info(f"Initialized memory-managed client with {max_history_tokens} max tokens")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text.
        
        This is a simple approximation. For production use,
        implement a more accurate tokenizer.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token for English text
        return len(text) // 4
    
    def _update_token_usage(self, conversation_id: str, new_text: str):
        """Update token usage tracking for a conversation."""
        # Initialize if not exists
        if conversation_id not in self.conversation_tokens:
            self.conversation_tokens[conversation_id] = 0
        
        # Add estimated tokens
        tokens = self._estimate_tokens(new_text)
        self.conversation_tokens[conversation_id] += tokens
        
        logger.debug(f"Conversation {conversation_id}: +{tokens} tokens, total: {self.conversation_tokens[conversation_id]}")
    
    def _check_and_optimize_memory(self, conversation_id: str) -> bool:
        """
        Check if memory optimization is needed.
        
        Args:
            conversation_id: The conversation to check
            
        Returns:
            True if optimization was performed
        """
        # Check current token usage
        current_tokens = self.conversation_tokens.get(conversation_id, 0)
        
        # No need to optimize if under threshold
        if current_tokens < self.max_history_tokens:
            return False
        
        # Get conversation details
        details = self.get_conversation_details(conversation_id)
        messages = details.get("messages", [])
        
        if len(messages) < 3:
            # Not enough history to optimize
            return False
        
        # Create a summary of older messages
        oldest_messages = messages[:-4]  # Keep the most recent exchanges intact
        
        if not oldest_messages:
            return False
        
        # Create a summary request
        summary_text = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in oldest_messages
        ])
        
        # Request summary from the model
        summary_prompt = f"""
        Summarize the following conversation history concisely while preserving
        key information and context needed for continuing the conversation:
        
        {summary_text}
        """
        
        try:
            # Create a temporary conversation for summarization
            temp_id = self.create_conversation()
            summary = self.send_message(temp_id, summary_prompt)
            self.delete_conversation(temp_id)
            
            # Reset the conversation with the summary as system prompt
            # This maintains the context while reducing token usage
            current_system = details.get("systemPrompt", "")
            new_system = f"{current_system}\n\nPrevious conversation summary: {summary}"
            
            # Create a new conversation with the updated system prompt
            # (In practice, you would preserve the conversation ID if possible)
            new_id = self.create_conversation(system_prompt=new_system)
            
            # Add the most recent exchanges to maintain continuity
            recent_messages = messages[-4:]
            for msg in recent_messages:
                if msg.get("role") == "user":
                    self.send_message(new_id, msg.get("content", ""))
            
            # Update token tracking
            self.conversation_tokens[new_id] = self._estimate_tokens(new_system) + self._estimate_tokens(
                "\n".join([m.get("content", "") for m in recent_messages])
            )
            
            logger.info(f"Optimized memory for conversation {conversation_id}, new ID: {new_id}")
            logger.info(f"New token count: {self.conversation_tokens[new_id]}")
            
            # Return the new conversation ID (you would handle this in your application)
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing memory: {str(e)}")
            return False
    
    def send_message(
        self,
        conversation_id: str,
        message: str,
        additional_model_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a message with memory management.
        
        Args:
            conversation_id: The conversation ID
            message: The user message
            additional_model_params: Optional model parameters
            
        Returns:
            The model's response
        """
        # Update token tracking for the new message
        self._update_token_usage(conversation_id, message)
        
        # Check if memory optimization is needed
        self._check_and_optimize_memory(conversation_id)
        
        # Send message using parent implementation
        response = super().send_message(
            conversation_id, 
            message,
            additional_model_params
        )
        
        # Update token tracking for the response
        self._update_token_usage(conversation_id, response)
        
        return response
```

## Common Pitfalls and Troubleshooting

### Pitfall #1: Not Tracking API Errors

**Problem**: The Converse API might return specific errors that need special handling.

**Solution**: Implement proper error parsing and handling:

```python
def _handle_bedrock_error(self, error):
    """Parse and handle Bedrock-specific errors."""
    error_code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
    error_message = getattr(error, "response", {}).get("Error", {}).get("Message", str(error))
    
    if error_code == "ValidationException":
        if "content filtering" in error_message.lower():
            return "The request was blocked by content filtering. Please rephrase your message."
        elif "token limit" in error_message.lower():
            return "The conversation exceeded the token limit. Please start a new conversation."
    
    if error_code == "ThrottlingException":
        return "The service is currently experiencing high demand. Please try again in a moment."
    
    # Log detailed error for debugging
    logger.error(f"Bedrock error: {error_code} - {error_message}")
    
    # Return a user-friendly message
    return f"An error occurred: {error_code}. Please try again or contact support."
```

### Pitfall #2: Poor System Prompt Design

**Problem**: System prompts might not effectively guide the model's behavior.

**Solution**: Implement a library of well-tested system prompts:

```python
# Example system prompt templates
SYSTEM_PROMPTS = {
    "customer_service": """
    You are a helpful customer service assistant. Your goal is to provide friendly, 
    accurate information and assistance to customers. Always be polite, concise, and 
    focus on resolving the customer's issue efficiently. If you don't know something, 
    acknowledge that and offer to connect them with a human representative.
    """,
    
    "technical_expert": """
    You are a technical expert specializing in {domain}. Provide detailed, accurate 
    technical information at an advanced level. Use precise terminology, cite 
    standards where relevant, and provide code examples when helpful. Focus on 
    best practices and robust solutions.
    """,
    
    "educational_tutor": """
    You are an educational tutor specializing in {subject}. Your goal is to help 
    students understand concepts thoroughly, not just provide answers. Ask 
    clarifying questions, break down complex topics, and provide examples that 
    reinforce learning. Adjust your explanations based on the student's apparent 
    level of understanding.
    """
}

def create_conversation_with_template(
    self, 
    template_name: str, 
    template_vars: Dict[str, str] = None
) -> str:
    """Create a conversation with a template system prompt."""
    if template_name not in SYSTEM_PROMPTS:
        raise ValueError(f"Template '{template_name}' not found")
    
    template = SYSTEM_PROMPTS[template_name]
    
    # Apply template variables if provided
    if template_vars:
        system_prompt = template.format(**template_vars)
    else:
        system_prompt = template
    
    return self.create_conversation(system_prompt=system_prompt)
```

### Pitfall #3: Not Handling Streaming Errors

**Problem**: Streaming responses can fail midway, leading to partial responses.

**Solution**: Implement robust error handling for streaming:

```python
def send_message_streaming_with_recovery(
    self,
    conversation_id: str,
    message: str,
    max_retries: int = 2,
    additional_model_params: Optional[Dict[str, Any]] = None
) -> Generator[str, None, None]:
    """
    Send a message with streaming and automatic recovery from temporary failures.
    
    Args:
        conversation_id: The conversation ID
        message: The user message
        max_retries: Maximum retry attempts
        additional_model_params: Optional model parameters
        
    Yields:
        Text chunks as they are generated
    """
    retries = 0
    partial_response = ""
    
    while retries <= max_retries:
        try:
            # On first attempt or retry, adjust behavior
            if retries == 0:
                current_message = message
            else:
                # For retries, explain that we're continuing
                recovery_msg = f"Please continue your response from where you left off. The last part I received was: \"{partial_response[-100:]}\""
                current_message = recovery_msg
            
            # Stream response
            for chunk in self.send_message_streaming(conversation_id, current_message, additional_model_params):
                partial_response += chunk
                yield chunk
            
            # If we get here, streaming completed successfully
            break
            
        except Exception as e:
            retries += 1
            logger.warning(f"Streaming error (attempt {retries}/{max_retries+1}): {str(e)}")
            
            if retries <= max_retries:
                # Notify user of the issue
                recovery_message = "\n[Connection interrupted. Recovering...]"
                yield recovery_message
                time.sleep(1)  # Brief pause before retry
            else:
                # Final failure
                failure_message = "\n[Unable to complete response. Please try again.]"
                yield failure_message
                logger.error(f"Failed to recover streaming after {max_retries} attempts")
```

## Try It Yourself Challenge

### Challenge: Build a Customer Support Chatbot

Create a customer support chatbot that can respond to product queries, troubleshoot common issues, and escalate complex problems to human agents.

**Starting Code**:

```python
import boto3
import json
import logging
import time
from typing import Dict, List, Any, Optional, Generator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support_bot")

class CustomerSupportBot:
    """
    A customer support chatbot using the AWS Bedrock Converse API.
    
    This bot can:
    1. Answer product questions
    2. Troubleshoot common issues
    3. Escalate complex issues to human agents
    4. Track conversation satisfaction
    """
    
    def __init__(
        self, 
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        product_catalog: Optional[str] = None,
        faq_database: Optional[str] = None
    ):
        """Initialize the support bot."""
        # TODO: Initialize the Converse client
        # TODO: Load product catalog and FAQs if provided
        # TODO: Set up the bot configuration
        pass
    
    def start_conversation(self) -> str:
        """Start a new support conversation."""
        # TODO: Create a conversation with appropriate system prompt
        # TODO: Send initial greeting
        pass
    
    def process_message(
        self, 
        conversation_id: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Process a customer message and generate a response.
        
        Args:
            conversation_id: The active conversation
            message: The customer's message
            
        Returns:
            Dictionary with response and action information
        """
        # TODO: Implement message processing
        # TODO: Detect intent (product question, troubleshooting, etc.)
        # TODO: Use function calling for lookups when needed
        # TODO: Track satisfaction and escalation needs
        pass
    
    def escalate_to_human(
        self,
        conversation_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """Escalate a conversation to a human agent."""
        # TODO: Implement escalation logic
        # TODO: Generate a summary for the human agent
        # TODO: Update conversation status
        pass
    
    def _search_product_catalog(self, query: str) -> Dict[str, Any]:
        """Search the product catalog for information."""
        # TODO: Implement product search function
        pass
    
    def _lookup_troubleshooting_steps(self, issue: str) -> List[Dict[str, Any]]:
        """Look up troubleshooting steps for common issues."""
        # TODO: Implement troubleshooting lookup
        pass
    
    def get_conversation_metrics(self, conversation_id: str) -> Dict[str, Any]:
        """Get metrics for the conversation."""
        # TODO: Calculate and return conversation metrics
        pass

# Tool definitions for the support bot
SUPPORT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_product_catalog",
            "description": "Search for product information in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The product search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_troubleshooting_steps",
            "description": "Find troubleshooting steps for common issues",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {
                        "type": "string",
                        "description": "The issue description"
                    }
                },
                "required": ["issue"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate this conversation to a human support agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "The reason for escalation"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "The priority level for escalation"
                    }
                },
                "required": ["reason", "priority"]
            }
        }
    }
]

# Example product data
PRODUCT_CATALOG = [
    {
        "id": "p1001",
        "name": "SmartHome Hub Pro",
        "category": "Smart Home",
        "price": 199.99,
        "features": [
            "Voice control",
            "Compatible with 100+ devices",
            "Mobile app integration",
            "Energy usage monitoring"
        ],
        "common_issues": ["connectivity", "device pairing", "app syncing"]
    },
    # Add more products...
]

# Example usage
if __name__ == "__main__":
    # Create support bot
    bot = CustomerSupportBot(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    
    # Start conversation
    conversation_id = bot.start_conversation()
    
    # Interactive loop
    try:
        while True:
            user_input = input("\nCustomer: ")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                break
                
            response = bot.process_message(conversation_id, user_input)
            
            if response.get("escalated"):
                print(f"\nAgent: {response['message']}")
                print(f"[This conversation has been escalated to a human agent.]")
                print(f"[Reason: {response['escalation_reason']}]")
                break
            else:
                print(f"\nAgent: {response['message']}")
                
    except KeyboardInterrupt:
        print("\nSession ended.")
```

**Expected Outcome**: A functional support chatbot that can:
1. Answer product questions using a catalog
2. Provide troubleshooting steps from a knowledge base
3. Detect when a human agent is needed and escalate
4. Maintain a natural conversation flow throughout

## Beyond the Basics

Once you've mastered the Converse API, consider these advanced techniques:

### 1. Implementing Multimodal Conversations

For handling images in conversations:

```python
def send_message_with_image(
    self,
    conversation_id: str,
    message: str,
    image_path: str,
    additional_model_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Send a message with an image attachment.
    
    Args:
        conversation_id: The conversation ID
        message: The user message
        image_path: Path to the image file
        additional_model_params: Optional model parameters
        
    Returns:
        The model's response
    """
    try:
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            
        import base64
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # Format message with image
        message_with_image = {
            "content": [
                {"text": message},
                {
                    "image": encoded_image,
                    "format": "base64"
                }
            ]
        }
        
        # Send via Converse API
        request = {
            "conversationId": conversation_id,
            "message": message_with_image
        }
        
        # Add additional model parameters if provided
        if additional_model_params:
            request["additionalModelParameters"] = additional_model_params
        
        # Send message
        response = self.bedrock.converse_conversation(**request)
        
        # Extract and return response
        return response.get("output", {}).get("message", "")
        
    except Exception as e:
        logger.error(f"Error sending message with image: {str(e)}")
        raise
```

### 2. Building a Knowledge-Grounded Conversation System

For accurate responses based on your own data:

```python
from typing import List, Dict, Any

class KnowledgeBaseClient:
    """Client for accessing a knowledge base."""
    
    def __init__(self, vector_db_endpoint: str):
        """Initialize the knowledge base client."""
        self.endpoint = vector_db_endpoint
        # Setup vector DB connection (using a hypothetical client)
        # self.vector_db = VectorDBClient(endpoint)
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: The search query
            limit: Maximum number of results
            
        Returns:
            List of relevant documents
        """
        # In a real implementation, this would query your vector database
        # For example:
        # results = self.vector_db.search(query, limit=limit)
        
        # Mock implementation for demonstration
        import random
        results = [
            {
                "id": f"doc-{random.randint(1000, 9999)}",
                "title": f"Knowledge Article about {query}",
                "content": f"This is relevant information about {query}...",
                "relevance_score": random.uniform(0.7, 0.99)
            }
            for _ in range(limit)
        ]
        
        return results

class KnowledgeGroundedConverseClient(ConverseClient):
    """
    A Converse client that grounds responses in a knowledge base.
    
    This ensures responses are factually accurate and reflect your
    organization's specific information and policies.
    """
    
    def __init__(
        self,
        model_id: str,
        knowledge_base: KnowledgeBaseClient,
        profile_name: Optional[str] = None,
        region_name: str = "us-west-2"
    ):
        """
        Initialize the knowledge-grounded client.
        
        Args:
            model_id: The Bedrock model identifier
            knowledge_base: Knowledge base client for retrieval
            profile_name: AWS profile name
            region_name: AWS region for Bedrock
        """
        super().__init__(model_id, profile_name, region_name)
        self.knowledge_base = knowledge_base
    
    def send_message(
        self,
        conversation_id: str,
        message: str,
        additional_model_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a message with knowledge retrieval.
        
        Args:
            conversation_id: The conversation ID
            message: The user message
            additional_model_params: Optional model parameters
            
        Returns:
            The model's response
        """
        # Search the knowledge base for relevant information
        search_results = self.knowledge_base.search(message, limit=3)
        
        # Format knowledge context
        if search_results:
            knowledge_context = "\n\n".join([
                f"--- Document: {doc['title']} ---\n{doc['content']}"
                for doc in search_results
            ])
            
            # Create a contextual system prompt
            context_prompt = f"""
            When responding to the user's next message, use the following 
            knowledge base information to ensure accuracy. Only reference this 
            information if relevant to the user's query:
            
            {knowledge_context}
            """
            
            # Update the conversation with this contextual information
            # Note: In a real implementation, you might want to track which
            # knowledge was added to avoid redundancy
            details = self.get_conversation_details(conversation_id)
            current_system = details.get("systemPrompt", "")
            
            # Temporarily update system prompt with knowledge
            # In a production system, you would use a more sophisticated
            # approach to managing conversation context
            self.bedrock.update_conversation(
                conversationId=conversation_id,
                systemPrompt=f"{current_system}\n\n{context_prompt}"
            )
        
        # Send the message using the parent implementation
        return super().send_message(
            conversation_id,
            message,
            additional_model_params
        )
```

### 3. Implementing A/B Testing for System Prompts

To optimize your conversational AI performance:

```python
import random
import time
from collections import defaultdict

class SystemPromptTester:
    """
    Tool for A/B testing different system prompts.
    
    This helps optimize conversation quality and outcomes
    by testing variations systematically.
    """
    
    def __init__(
        self,
        model_id: str,
        prompt_variants: Dict[str, str],
        profile_name: Optional[str] = None,
        region_name: str = "us-west-2"
    ):
        """
        Initialize the tester.
        
        Args:
            model_id: The Bedrock model identifier
            prompt_variants: Dictionary of variant_name -> system_prompt
            profile_name: AWS profile name
            region_name: AWS region for Bedrock
        """
        self.model_id = model_id
        self.profile_name = profile_name
        self.region_name = region_name
        self.prompt_variants = prompt_variants
        
        # Create client
        self.client = ConverseClient(
            model_id=model_id,
            profile_name=profile_name,
            region_name=region_name
        )
        
        # Tracking metrics
        self.variant_metrics = defaultdict(lambda: {
            "conversations": 0,
            "messages": 0,
            "response_times": [],
            "feedback_scores": []
        })
    
    def create_test_conversation(self) -> Dict[str, Any]:
        """
        Create a test conversation with a randomly selected prompt variant.
        
        Returns:
            Dictionary with conversation ID and selected variant
        """
        # Select a random variant
        variant_name = random.choice(list(self.prompt_variants.keys()))
        system_prompt = self.prompt_variants[variant_name]
        
        # Create conversation
        conversation_id = self.client.create_conversation(
            system_prompt=system_prompt
        )
        
        # Update metrics
        self.variant_metrics[variant_name]["conversations"] += 1
        
        return {
            "conversation_id": conversation_id,
            "variant": variant_name
        }
    
    def send_message(
        self,
        conversation_id: str,
        message: str,
        variant: str
    ) -> Dict[str, Any]:
        """
        Send a message in a test conversation.
        
        Args:
            conversation_id: The conversation ID
            message: The user message
            variant: The prompt variant being tested
            
        Returns:
            Dictionary with response and timing information
        """
        # Track message
        self.variant_metrics[variant]["messages"] += 1
        
        # Measure response time
        start_time = time.time()
        
        # Send message
        response = self.client.send_message(conversation_id, message)
        
        # Record response time
        response_time = time.time() - start_time
        self.variant_metrics[variant]["response_times"].append(response_time)
        
        return {
            "response": response,
            "response_time": response_time
        }
    
    def record_feedback(
        self,
        variant: str,
        score: float  # 1.0-5.0 scale
    ):
        """Record feedback for a variant."""
        self.variant_metrics[variant]["feedback_scores"].append(score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current test metrics."""
        results = {}
        
        for variant, metrics in self.variant_metrics.items():
            avg_response_time = 0
            if metrics["response_times"]:
                avg_response_time = sum(metrics["response_times"]) / len(metrics["response_times"])
            
            avg_feedback = 0
            if metrics["feedback_scores"]:
                avg_feedback = sum(metrics["feedback_scores"]) / len(metrics["feedback_scores"])
            
            results[variant] = {
                "conversations": metrics["conversations"],
                "messages": metrics["messages"],
                "avg_response_time": avg_response_time,
                "avg_feedback": avg_feedback
            }
        
        return results
```

## Key Takeaways

- The Converse API simplifies building conversational applications with foundation models
- It handles conversation state, memory management, and model-specific formatting
- Streaming provides a responsive user experience for real-time interactions
- Function calling enables integration with your business systems and data
- Memory optimization is crucial for long-running conversations
- Proper system prompts guide model behavior throughout a conversation

---

**Next Steps**: Now that you understand conversational AI with the Converse API, learn about [Structured Outputs with the Construct API](/chapters/apis/construct/) for generating highly structured data from foundation models.

---

© 2025 Scott Friedman. Licensed under CC BY-NC-ND 4.0