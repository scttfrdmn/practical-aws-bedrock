"""
Complete chatbot example using AWS Bedrock Converse API.

This module provides a complete example of a command-line chatbot application 
built using the AWS Bedrock Converse API with streaming responses and
advanced memory management.
"""

import os
import time
import json
import argparse
import logging
from typing import Dict, Any, Optional, List, Union, Tuple

from ..basic.simple_chat import ConverseClient
from ..basic.memory import MemoryAwareConverseClient
from ..advanced.streaming import StreamingConverseClient
from ..utils.templates import get_template
from ..utils.history import ConversationStore


class BedrockChatbot:
    """
    A complete chatbot application using AWS Bedrock Converse API.
    
    This class combines various features of the Converse API client
    implementations to create a full-featured chatbot application.
    """
    
    def __init__(
        self,
        model_id: str,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        memory_limit: int = 8000,
        use_streaming: bool = True,
        system_template: str = "general",
        storage_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the chatbot.
        
        Args:
            model_id: The Bedrock model identifier
            profile_name: AWS profile name
            region_name: AWS region name
            memory_limit: Token limit for conversation memory
            use_streaming: Whether to use streaming responses
            system_template: Template name or custom system prompt
            storage_path: Path for conversation storage
            logger: Optional logger instance
        """
        self.model_id = model_id
        self.profile_name = profile_name
        self.region_name = region_name
        self.memory_limit = memory_limit
        self.use_streaming = use_streaming
        self.logger = logger or logging.getLogger(__name__)
        
        # Set up storage
        self.storage_path = storage_path
        if storage_path:
            os.makedirs(storage_path, exist_ok=True)
            self.conversation_store = ConversationStore(
                storage_type="json",
                storage_path=storage_path
            )
        else:
            self.conversation_store = None
        
        # Set up system prompt
        try:
            # Try to get template by name
            self.system_prompt = get_template(system_template)
        except KeyError:
            # If not a template name, use as custom prompt
            self.system_prompt = system_template
        
        # Initialize clients
        self._initialize_clients()
        
        # Active conversation tracking
        self.active_conversation_id = None
    
    def _initialize_clients(self) -> None:
        """Initialize the appropriate client implementations."""
        common_args = {
            "model_id": self.model_id,
            "profile_name": self.profile_name,
            "region_name": self.region_name,
            "logger": self.logger
        }
        
        # Create memory-aware client
        self.memory_client = MemoryAwareConverseClient(
            max_history_tokens=self.memory_limit,
            preserve_recent_turns=5,
            **common_args
        )
        
        # Create streaming client if enabled
        if self.use_streaming:
            self.streaming_client = StreamingConverseClient(**common_args)
        else:
            self.streaming_client = None
    
    def start_conversation(
        self,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Start a new conversation or load an existing one.
        
        Args:
            conversation_id: Optional ID to load existing conversation
            system_prompt: Optional system prompt override
            
        Returns:
            The conversation ID
        """
        if conversation_id and self.conversation_store:
            # Try to load existing conversation
            try:
                conversation = self.conversation_store.load_conversation(conversation_id)
                
                # Load conversation into memory client
                self.memory_client.conversations[conversation_id] = conversation
                
                # Load into streaming client if enabled
                if self.streaming_client:
                    self.streaming_client.conversations[conversation_id] = conversation
                
                self.active_conversation_id = conversation_id
                self.logger.info(f"Loaded conversation: {conversation_id}")
                
                return conversation_id
            except FileNotFoundError:
                self.logger.warning(f"Conversation {conversation_id} not found, creating new one")
        
        # Use provided system prompt or default
        prompt = system_prompt or self.system_prompt
        
        # Create new conversation
        new_id = self.memory_client.create_conversation(system_prompt=prompt)
        
        # Also create in streaming client if enabled
        if self.streaming_client:
            self.streaming_client.create_conversation(
                conversation_id=new_id,
                system_prompt=prompt
            )
        
        self.active_conversation_id = new_id
        self.logger.info(f"Created new conversation: {new_id}")
        
        # Save to storage if enabled
        if self.conversation_store:
            conversation = self.memory_client.get_conversation_history(new_id)
            self.conversation_store.save_conversation(conversation)
        
        return new_id
    
    def send_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Send a message and get the response.
        
        Args:
            message: The user message
            conversation_id: Optional conversation ID (uses active if not provided)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            The assistant's response
        """
        # Use provided conversation ID or active one
        convo_id = conversation_id or self.active_conversation_id
        
        if not convo_id:
            # Create a new conversation if none exists
            convo_id = self.start_conversation()
        
        # Send message using appropriate client
        if self.use_streaming:
            # Collect the streamed response
            full_response = ""
            
            # Show "thinking" indicator
            print("Assistant: ", end="", flush=True)
            
            # Stream the response
            for chunk in self.streaming_client.send_message_streaming(
                conversation_id=convo_id,
                message=message,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print()  # Add final newline
            
            # Sync conversation state to memory client
            if convo_id in self.streaming_client.conversations:
                self.memory_client.conversations[convo_id] = self.streaming_client.conversations[convo_id]
        else:
            # Non-streaming response
            print("Assistant: ", end="", flush=True)
            full_response = self.memory_client.send_message(
                conversation_id=convo_id,
                message=message,
                max_tokens=max_tokens,
                temperature=temperature
            )
            print(full_response)
        
        # Save conversation if storage enabled
        if self.conversation_store:
            conversation = self.memory_client.get_conversation_history(convo_id)
            self.conversation_store.save_conversation(conversation)
        
        return full_response
    
    def get_conversation_history(
        self,
        conversation_id: Optional[str] = None,
        format_for_display: bool = False
    ) -> Union[Dict[str, Any], str]:
        """
        Get the conversation history.
        
        Args:
            conversation_id: Optional conversation ID (uses active if not provided)
            format_for_display: Whether to format history for display
            
        Returns:
            Conversation history as dictionary or formatted string
        """
        # Use provided conversation ID or active one
        convo_id = conversation_id or self.active_conversation_id
        
        if not convo_id:
            raise ValueError("No active conversation")
        
        # Get conversation history
        history = self.memory_client.get_conversation_history(convo_id)
        
        if format_for_display:
            # Format for display
            lines = []
            for msg in history["messages"]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                # Format multimodal content
                if isinstance(content, list):
                    # Handle Claude multimodal format
                    text_parts = []
                    has_image = False
                    
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "image":
                                has_image = True
                    
                    content = "\n".join(text_parts)
                    if has_image:
                        content += "\n[Image attached]"
                
                # Add to formatted output
                if role == "system":
                    lines.append(f"[System] {content}")
                elif role == "user":
                    lines.append(f"User: {content}")
                elif role == "assistant":
                    lines.append(f"Assistant: {content}")
                else:
                    lines.append(f"[{role}] {content}")
            
            return "\n\n".join(lines)
        else:
            return history
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List available conversations.
        
        Returns:
            List of conversation metadata
        """
        if not self.conversation_store:
            # If no persistent storage, list in-memory conversations
            return [
                {
                    "id": convo_id,
                    "message_count": len(convo_data["messages"]),
                    "created_at": convo_data.get("created_at", 0),
                    "updated_at": convo_data.get("updated_at", 0)
                }
                for convo_id, convo_data in self.memory_client.conversations.items()
            ]
        else:
            # List from persistent storage
            return self.conversation_store.list_conversations()
    
    def save_conversation(
        self,
        conversation_id: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> str:
        """
        Save a conversation to a JSON file.
        
        Args:
            conversation_id: Optional conversation ID (uses active if not provided)
            file_path: Optional file path (defaults to conversation ID + timestamp)
            
        Returns:
            Path to saved file
        """
        # Use provided conversation ID or active one
        convo_id = conversation_id or self.active_conversation_id
        
        if not convo_id:
            raise ValueError("No active conversation")
        
        # Get conversation history
        history = self.memory_client.get_conversation_history(convo_id)
        
        # Determine file path
        if not file_path:
            timestamp = int(time.time())
            file_path = f"conversation_{convo_id}_{timestamp}.json"
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Saved conversation to {file_path}")
        return file_path
    
    def get_memory_stats(
        self,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Args:
            conversation_id: Optional conversation ID (uses active if not provided)
            
        Returns:
            Dictionary with memory statistics
        """
        # Use provided conversation ID or active one
        convo_id = conversation_id or self.active_conversation_id
        
        if not convo_id:
            raise ValueError("No active conversation")
        
        # Get memory stats
        return self.memory_client.get_memory_stats(convo_id)


def main():
    """Command-line interface for the chatbot."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="AWS Bedrock Chatbot")
    parser.add_argument("--model", "-m", default="anthropic.claude-3-sonnet-20240229-v1:0",
                      help="Bedrock model ID")
    parser.add_argument("--profile", "-p", help="AWS profile name")
    parser.add_argument("--region", "-r", help="AWS region name")
    parser.add_argument("--memory", type=int, default=8000,
                      help="Memory token limit")
    parser.add_argument("--no-streaming", action="store_true",
                      help="Disable streaming responses")
    parser.add_argument("--template", "-t", default="general",
                      help="System prompt template name")
    parser.add_argument("--storage", "-s", help="Storage path for conversations")
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Enable verbose logging")
    parser.add_argument("--load", "-l", help="Load existing conversation by ID")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("bedrock-chatbot")
    
    # Create chatbot
    chatbot = BedrockChatbot(
        model_id=args.model,
        profile_name=args.profile,
        region_name=args.region,
        memory_limit=args.memory,
        use_streaming=not args.no_streaming,
        system_template=args.template,
        storage_path=args.storage,
        logger=logger
    )
    
    # Start or load conversation
    conversation_id = chatbot.start_conversation(conversation_id=args.load)
    
    logger.info(f"Active conversation: {conversation_id}")
    
    # Print welcome message
    print("\n=== AWS Bedrock Chatbot ===")
    print(f"Model: {args.model}")
    print("Type 'exit' to quit, 'help' for commands")
    print("===========================\n")
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("User: ")
            
            # Check for special commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "help":
                print("\nCommands:")
                print("  exit - Exit the chatbot")
                print("  help - Show this help message")
                print("  history - Show conversation history")
                print("  save - Save conversation to file")
                print("  stats - Show memory usage statistics")
                print("  new - Start a new conversation")
                print("  list - List available conversations")
                print("  load <id> - Load conversation by ID")
                continue
            elif user_input.lower() == "history":
                formatted_history = chatbot.get_conversation_history(format_for_display=True)
                print("\n--- Conversation History ---")
                print(formatted_history)
                print("----------------------------\n")
                continue
            elif user_input.lower() == "save":
                file_path = chatbot.save_conversation()
                print(f"Conversation saved to {file_path}")
                continue
            elif user_input.lower() == "stats":
                stats = chatbot.get_memory_stats()
                print("\n--- Memory Stats ---")
                print(f"Token usage: {stats['current_tokens']}/{stats['max_tokens']} ({stats['utilization_percentage']:.1f}%)")
                print(f"Message count: {stats['message_count']} total, {stats['user_message_count']} user, {stats['assistant_message_count']} assistant")
                print("--------------------\n")
                continue
            elif user_input.lower() == "new":
                conversation_id = chatbot.start_conversation()
                print(f"Started new conversation: {conversation_id}")
                continue
            elif user_input.lower() == "list":
                conversations = chatbot.list_conversations()
                print("\n--- Available Conversations ---")
                for i, convo in enumerate(conversations):
                    updated = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(convo.get("updated_at", 0)))
                    print(f"{i+1}. ID: {convo['id']}")
                    print(f"   Messages: {convo.get('message_count', 'unknown')}")
                    print(f"   Last updated: {updated}")
                print("------------------------------\n")
                continue
            elif user_input.lower().startswith("load "):
                # Extract conversation ID
                load_id = user_input[5:].strip()
                try:
                    conversation_id = chatbot.start_conversation(conversation_id=load_id)
                    print(f"Loaded conversation: {conversation_id}")
                except Exception as e:
                    print(f"Error loading conversation: {str(e)}")
                continue
            
            # Process regular message
            chatbot.send_message(user_input)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()