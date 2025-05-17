"""
Memory-aware client for AWS Bedrock Converse API.

This module provides a conversation client that efficiently manages memory
and token usage for long conversations with AWS Bedrock models.
It extends the basic ConverseClient with memory management capabilities.
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Union

from .simple_chat import ConverseClient


class MemoryAwareConverseClient(ConverseClient):
    """
    A client for AWS Bedrock Converse API that manages conversation memory.
    
    This client extends the basic ConverseClient with optimized memory management,
    tracking token usage and pruning conversation history to stay within token limits.
    """
    
    def __init__(
        self, 
        model_id: str,
        max_history_tokens: int = 8000,
        include_system_in_limit: bool = False,
        preserve_system_prompt: bool = True,
        preserve_recent_turns: int = 5,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the memory-aware Converse client.
        
        Args:
            model_id: The Bedrock model identifier
            max_history_tokens: Maximum tokens to keep in conversation history
            include_system_in_limit: Whether to include system prompt in the token limit
            preserve_system_prompt: Whether to always keep system prompts
            preserve_recent_turns: Minimum number of recent turns to preserve
            profile_name: AWS profile name (defaults to value from get_profile())
            region_name: AWS region name (defaults to value from get_region())
            max_retries: Maximum number of retry attempts for recoverable errors
            base_backoff: Base backoff time (in seconds) for exponential backoff
            logger: Optional logger instance
        """
        super().__init__(
            model_id=model_id,
            profile_name=profile_name,
            region_name=region_name,
            max_retries=max_retries,
            base_backoff=base_backoff,
            logger=logger
        )
        
        self.max_history_tokens = max_history_tokens
        self.include_system_in_limit = include_system_in_limit
        self.preserve_system_prompt = preserve_system_prompt
        self.preserve_recent_turns = preserve_recent_turns
        
        # Token tracking per conversation
        self.conversation_tokens = {}
    
    def create_conversation(
        self, 
        system_prompt: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Create a new conversation with memory tracking.
        
        Args:
            system_prompt: Optional system prompt to guide the model's behavior
            conversation_id: Optional ID for the conversation (generated if not provided)
            
        Returns:
            The conversation ID
        """
        # Create the conversation using the parent method
        conversation_id = super().create_conversation(
            system_prompt=system_prompt,
            conversation_id=conversation_id
        )
        
        # Initialize token tracking for this conversation
        self.conversation_tokens[conversation_id] = {
            "message_tokens": {},  # Maps message index to token count
            "total_tokens": 0
        }
        
        # If we have a system prompt, estimate its token count
        if system_prompt:
            token_estimate = self._estimate_tokens(system_prompt)
            self.conversation_tokens[conversation_id]["message_tokens"][0] = token_estimate
            self.conversation_tokens[conversation_id]["total_tokens"] = token_estimate
        
        return conversation_id
    
    def send_message(
        self, 
        conversation_id: str, 
        message: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        other_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a message in a conversation with memory management.
        
        This method adds token tracking and memory pruning to the basic send_message.
        
        Args:
            conversation_id: The conversation ID
            message: The user message
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            other_params: Additional model-specific parameters
            
        Returns:
            The model's response text
        """
        # Check if memory needs to be pruned before adding the new message
        self._prune_conversation_history(conversation_id)
        
        # Estimate token count for the new message
        message_token_count = self._estimate_tokens(message)
        
        # Add token count to tracking
        messages_count = len(self.conversations[conversation_id]["messages"])
        self.conversation_tokens[conversation_id]["message_tokens"][messages_count] = message_token_count
        self.conversation_tokens[conversation_id]["total_tokens"] += message_token_count
        
        # Send the message using the parent method
        response = super().send_message(
            conversation_id=conversation_id,
            message=message,
            max_tokens=max_tokens,
            temperature=temperature,
            other_params=other_params
        )
        
        # Update token tracking for the response
        response_token_count = self._estimate_tokens(response)
        messages_count = len(self.conversations[conversation_id]["messages"])
        self.conversation_tokens[conversation_id]["message_tokens"][messages_count - 1] = response_token_count
        self.conversation_tokens[conversation_id]["total_tokens"] += response_token_count
        
        return response
    
    def add_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str
    ) -> None:
        """
        Add a message to a conversation with token tracking.
        
        Args:
            conversation_id: The conversation ID
            role: Message role ('system', 'user', or 'assistant')
            content: Message content
        """
        # Call parent method to add the message
        super().add_message(conversation_id, role, content)
        
        # Update token tracking
        token_count = self._estimate_tokens(content)
        messages_count = len(self.conversations[conversation_id]["messages"])
        self.conversation_tokens[conversation_id]["message_tokens"][messages_count - 1] = token_count
        self.conversation_tokens[conversation_id]["total_tokens"] += token_count
    
    def _prune_conversation_history(self, conversation_id: str) -> None:
        """
        Prune conversation history to stay within token limits.
        
        This method removes older messages while preserving system prompts
        and recent conversation turns.
        
        Args:
            conversation_id: The conversation ID
        """
        if conversation_id not in self.conversations:
            return
        
        messages = self.conversations[conversation_id]["messages"]
        token_tracking = self.conversation_tokens[conversation_id]
        
        # Check if we're under the limit
        if token_tracking["total_tokens"] <= self.max_history_tokens:
            return
        
        # Determine which messages can be removed
        removable_messages = []
        preserved_indices = set()
        
        # Always preserve system prompts if configured
        if self.preserve_system_prompt:
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    preserved_indices.add(i)
        
        # Always preserve the most recent turns
        total_turns = len(messages) // 2  # Approximate turn count (user + assistant pairs)
        preserved_turns = min(self.preserve_recent_turns, total_turns)
        preserved_messages = preserved_turns * 2
        
        for i in range(max(0, len(messages) - preserved_messages), len(messages)):
            preserved_indices.add(i)
        
        # Identify messages that can be removed
        for i in range(len(messages)):
            if i not in preserved_indices:
                removable_messages.append(i)
        
        # If no messages can be removed, we can't prune
        if not removable_messages:
            self.logger.warning(
                f"Cannot prune conversation {conversation_id}: all messages are preserved. "
                f"Consider increasing max_history_tokens or reducing preserve_recent_turns."
            )
            return
        
        # Remove messages starting from the oldest until we're under the limit
        removed_tokens = 0
        removed_indices = []
        
        # Sort removable messages by age (oldest first)
        removable_messages.sort()
        
        for i in removable_messages:
            # Skip if we've already removed enough
            if token_tracking["total_tokens"] - removed_tokens <= self.max_history_tokens:
                break
                
            # Track the removed tokens and indices
            removed_tokens += token_tracking["message_tokens"].get(i, 0)
            removed_indices.append(i)
        
        # If no messages would be removed, exit
        if not removed_indices:
            return
            
        # Create new message list and token tracking
        new_messages = []
        new_token_tracking = {
            "message_tokens": {},
            "total_tokens": token_tracking["total_tokens"] - removed_tokens
        }
        
        # Copy remaining messages and update token tracking
        new_idx = 0
        for i, msg in enumerate(messages):
            if i not in removed_indices:
                new_messages.append(msg)
                new_token_tracking["message_tokens"][new_idx] = token_tracking["message_tokens"].get(i, 0)
                new_idx += 1
        
        # Update conversation and token tracking
        self.conversations[conversation_id]["messages"] = new_messages
        self.conversation_tokens[conversation_id] = new_token_tracking
        
        self.logger.info(
            f"Pruned conversation {conversation_id}: removed {len(removed_indices)} messages, "
            f"freed {removed_tokens} tokens. New total: {new_token_tracking['total_tokens']} tokens."
        )
    
    def get_memory_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get memory usage statistics for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Dictionary with memory usage statistics
            
        Raises:
            KeyError: When conversation_id doesn't exist
        """
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation {conversation_id} not found")
        
        if conversation_id not in self.conversation_tokens:
            raise KeyError(f"Token tracking not found for conversation {conversation_id}")
        
        token_tracking = self.conversation_tokens[conversation_id]
        
        stats = {
            "current_tokens": token_tracking["total_tokens"],
            "max_tokens": self.max_history_tokens,
            "available_tokens": max(0, self.max_history_tokens - token_tracking["total_tokens"]),
            "utilization_percentage": min(100, (token_tracking["total_tokens"] / self.max_history_tokens) * 100),
            "message_count": len(self.conversations[conversation_id]["messages"]),
            "system_message_count": sum(1 for msg in self.conversations[conversation_id]["messages"] if msg["role"] == "system"),
            "user_message_count": sum(1 for msg in self.conversations[conversation_id]["messages"] if msg["role"] == "user"),
            "assistant_message_count": sum(1 for msg in self.conversations[conversation_id]["messages"] if msg["role"] == "assistant")
        }
        
        return stats
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and its token tracking.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if conversation was deleted, False if it didn't exist
        """
        result = super().delete_conversation(conversation_id)
        
        if result and conversation_id in self.conversation_tokens:
            del self.conversation_tokens[conversation_id]
            
        return result
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        This uses a simple approximation method. For production use,
        consider using a model-specific tokenizer for more accurate counts.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token for English text
        # This is a rough approximation; model-specific tokenizers would be more accurate
        if not text:
            return 0
            
        # Count words (rough approximation)
        words = text.split()
        word_count = len(words)
        
        # Estimate tokens (typically 0.75 tokens per word for English)
        token_estimate = max(1, int(word_count * 0.75))
        
        return token_estimate


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a memory-aware client for Claude
    client = MemoryAwareConverseClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        max_history_tokens=8000,
        preserve_recent_turns=3
    )
    
    # Create a conversation with a system prompt
    conversation_id = client.create_conversation(
        system_prompt="You are a helpful assistant specializing in AWS services."
    )
    
    try:
        # Send several messages to demonstrate memory management
        user_messages = [
            "What is AWS Bedrock?",
            "Which foundation models does it support?",
            "How do I optimize for throughput when using Bedrock?",
            "What's the difference between synchronous and asynchronous inference?",
            "Can you explain how token quotas work in Bedrock?"
        ]
        
        for i, message in enumerate(user_messages):
            print(f"\n--- Turn {i+1} ---")
            
            # Check memory stats before sending
            stats = client.get_memory_stats(conversation_id)
            print(f"Memory before: {stats['current_tokens']}/{stats['max_tokens']} tokens ({stats['utilization_percentage']:.1f}%)")
            
            # Send message
            print(f"\nUser: {message}")
            response = client.send_message(
                conversation_id=conversation_id,
                message=message,
                max_tokens=300  # Short responses for demo
            )
            print(f"\nAssistant: {response}")
            
            # Check memory stats after
            stats = client.get_memory_stats(conversation_id)
            print(f"\nMemory after: {stats['current_tokens']}/{stats['max_tokens']} tokens ({stats['utilization_percentage']:.1f}%)")
            print(f"Messages: {stats['message_count']} total, {stats['user_message_count']} user, {stats['assistant_message_count']} assistant")
            
            # Forcing a memory prune by simulating a very long conversation
            if i == 2:  # After the third message
                print("\n--- Simulating memory pressure ---")
                # Add some fake token count to force pruning
                client.conversation_tokens[conversation_id]["total_tokens"] = 7500
                
                # Send next message (will trigger pruning)
                print(f"\nUser: {user_messages[i+1]}")
                response = client.send_message(
                    conversation_id=conversation_id,
                    message=user_messages[i+1],
                    max_tokens=300
                )
                print(f"\nAssistant: {response}")
                
                # Check memory after pruning
                stats = client.get_memory_stats(conversation_id)
                print(f"\nMemory after pruning: {stats['current_tokens']}/{stats['max_tokens']} tokens ({stats['utilization_percentage']:.1f}%)")
                print(f"Messages remaining: {stats['message_count']} total")
                
                # Skip to avoid sending this message twice
                i += 1
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get conversation history
    history = client.get_conversation_history(conversation_id)
    print(f"\nFinal conversation has {len(history['messages'])} messages")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")