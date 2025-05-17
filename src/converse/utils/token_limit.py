"""
Token limit utilities for AWS Bedrock Converse API.

This module provides tools for estimating, tracking, and optimizing token usage
in conversations with AWS Bedrock models.
"""

import re
import json
import tiktoken
from typing import Dict, Any, Optional, List, Union, Tuple


class TokenCounter:
    """
    A utility class for counting tokens in text strings and conversation messages.
    
    This class provides methods for accurately estimating token counts for different
    model families using appropriate tokenizers when available.
    """
    
    def __init__(self, model_id: str = None):
        """
        Initialize the token counter.
        
        Args:
            model_id: Optional model identifier (for model-specific tokenization)
        """
        self.model_id = model_id
        self.encoder = None
        
        # Initialize appropriate tokenizer if possible
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self) -> None:
        """Initialize the appropriate tokenizer based on model_id."""
        if not self.model_id:
            return
        
        # Extract model family from model ID
        model_family = self.model_id.split('.')[0].lower() if '.' in self.model_id else ""
        model_name = self.model_id.split('.')[1].lower() if '.' in self.model_id and len(self.model_id.split('.')) > 1 else ""
        
        # Try to initialize tokenizer based on model family
        try:
            # Different model families need different tokenizers
            if "anthropic" in model_family and "claude" in model_name:
                # Claude models (approximate with cl100k_base which Claude uses)
                self.encoder = tiktoken.get_encoding("cl100k_base")
            elif "meta" in model_family or "llama" in model_name:
                # Meta's Llama models
                self.encoder = tiktoken.encoding_for_model("gpt-4")  # Approximation
            elif "cohere" in model_family:
                # Cohere models
                self.encoder = tiktoken.encoding_for_model("gpt-4")  # Approximation
            elif "amazon" in model_family and "titan" in model_name:
                # Amazon Titan models
                self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Approximation
        except Exception as e:
            # Fall back to approximate tokenization if tiktoken not available
            # or if the model family isn't recognized
            self.encoder = None
    
    def count_tokens(self, text: Union[str, List, Dict]) -> int:
        """
        Count tokens in text or structured content.
        
        Args:
            text: Text string, list of strings, or dictionary with text content
            
        Returns:
            Estimated token count
        """
        if text is None:
            return 0
            
        # Handle different input types
        if isinstance(text, str):
            return self._count_string_tokens(text)
        elif isinstance(text, list):
            return sum(self.count_tokens(item) for item in text)
        elif isinstance(text, dict):
            return self._count_dict_tokens(text)
        else:
            # Convert to string for other types
            return self._count_string_tokens(str(text))
    
    def _count_string_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
            
        if self.encoder:
            # Use the initialized tokenizer
            return len(self.encoder.encode(text))
        else:
            # Fallback approximation method
            return self._approximate_tokens(text)
    
    def _count_dict_tokens(self, obj: Dict) -> int:
        """Count tokens in a dictionary structure."""
        if not obj:
            return 0
            
        # For dictionaries like message objects, handle specially
        if all(k in obj for k in ["role", "content"]):
            # This looks like a message object
            role_tokens = self._count_string_tokens(obj["role"])
            
            # Content could be string or structured (for multimodal)
            content = obj["content"]
            if isinstance(content, str):
                content_tokens = self._count_string_tokens(content)
            elif isinstance(content, list):
                # Handle multimodal content array (Claude format)
                content_tokens = 0
                for item in content:
                    if isinstance(item, dict) and "type" in item:
                        if item["type"] == "text":
                            content_tokens += self._count_string_tokens(item.get("text", ""))
                        elif item["type"] == "image":
                            # Claude charges a base cost for images plus size-based cost
                            # We use a constant approximation since we can't analyze the image
                            content_tokens += 1000  # Approximate cost for a medium image
            else:
                # Generic handling
                content_tokens = self.count_tokens(content)
                
            # Message formatting overhead varies by model
            overhead = 4  # Approximation for message formatting
            
            return role_tokens + content_tokens + overhead
        
        # For other dictionaries, count all keys and values
        total = 0
        for key, value in obj.items():
            total += self._count_string_tokens(str(key))
            total += self.count_tokens(value)
        
        return total
    
    def _approximate_tokens(self, text: str) -> int:
        """
        Approximate token count using simple heuristics.
        
        This is a fallback when a proper tokenizer isn't available.
        
        Args:
            text: Text string
            
        Returns:
            Approximate token count
        """
        if not text:
            return 0
            
        # Approximation based on words and punctuation
        # This is crude but gives a ballpark estimate
        
        # Count words (roughly 0.75 tokens per word for English)
        words = len(re.findall(r'\w+', text))
        word_tokens = int(words * 0.75)
        
        # Count punctuation and special characters (roughly 1 token each)
        punctuation = len(re.findall(r'[^\w\s]', text))
        
        # Count whitespace (roughly 0.25 tokens per whitespace)
        whitespace = len(re.findall(r'\s+', text))
        whitespace_tokens = int(whitespace * 0.25)
        
        # Sum the components with a small overhead
        return word_tokens + punctuation + whitespace_tokens + 2  # +2 for overhead
    
    def count_conversation_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in a full conversation.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Total token count for the conversation
        """
        if not messages:
            return 0
            
        total = 0
        for message in messages:
            total += self.count_tokens(message)
            
        # Add conversation formatting overhead
        overhead = 3  # Approximation for conversation formatting
        
        return total + overhead


class TokenTrimmer:
    """
    A utility class for trimming conversations to fit within token limits.
    
    This class provides methods for intelligently pruning conversation history
    while preserving important context.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        max_tokens: int = 8000,
        preserve_system_prompt: bool = True,
        preserve_recent_turns: int = 3,
        token_counter: Optional[TokenCounter] = None
    ):
        """
        Initialize the token trimmer.
        
        Args:
            model_id: Model identifier
            max_tokens: Maximum allowed tokens
            preserve_system_prompt: Whether to always preserve system prompts
            preserve_recent_turns: Minimum number of recent turns to preserve
            token_counter: Optional TokenCounter instance to use
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.preserve_system_prompt = preserve_system_prompt
        self.preserve_recent_turns = preserve_recent_turns
        
        # Initialize token counter
        self.token_counter = token_counter or TokenCounter(model_id)
    
    def trim_conversation(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Trim a conversation to fit within token limits.
        
        Args:
            messages: List of message dictionaries
            current_tokens: Optional pre-calculated token count
            
        Returns:
            Tuple of (trimmed messages list, new token count)
        """
        if not messages:
            return messages, 0
            
        # Count tokens if not provided
        if current_tokens is None:
            current_tokens = self.token_counter.count_conversation_tokens(messages)
            
        # Check if trimming is needed
        if current_tokens <= self.max_tokens:
            return messages, current_tokens
            
        # Identify messages to preserve
        preserved_indices = set()
        
        # Preserve system prompts if configured
        if self.preserve_system_prompt:
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    preserved_indices.add(i)
        
        # Always preserve the most recent turns
        total_turns = len(messages) // 2  # Approximate turn count (user + assistant pairs)
        preserved_turns = min(self.preserve_recent_turns, total_turns)
        preserved_messages = preserved_turns * 2
        
        for i in range(max(0, len(messages) - preserved_messages), len(messages)):
            preserved_indices.add(i)
        
        # Create candidate messages list for removal (oldest first)
        candidates = [i for i in range(len(messages)) if i not in preserved_indices]
        candidates.sort()  # Sort by index (oldest first)
        
        # If no candidates, we can't trim
        if not candidates:
            return messages, current_tokens
        
        # Trim messages until under the limit
        removed_indices = set()
        token_count = current_tokens
        
        for idx in candidates:
            # Check if we're under the limit
            if token_count <= self.max_tokens:
                break
                
            # Calculate tokens in this message
            msg_tokens = self.token_counter.count_tokens(messages[idx])
            
            # Remove message
            token_count -= msg_tokens
            removed_indices.add(idx)
        
        # Create new message list without removed messages
        trimmed_messages = [msg for i, msg in enumerate(messages) if i not in removed_indices]
        
        return trimmed_messages, token_count


class TokenOptimizer:
    """
    A utility class for optimizing token usage in conversations.
    
    This class provides methods for summarizing conversation history,
    combining related messages, and other token-saving strategies.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        max_tokens: int = 8000,
        summary_model_id: Optional[str] = None,
        token_counter: Optional[TokenCounter] = None
    ):
        """
        Initialize the token optimizer.
        
        Args:
            model_id: Model identifier for the main conversation
            max_tokens: Maximum allowed tokens
            summary_model_id: Model ID to use for summarization (defaults to model_id)
            token_counter: Optional TokenCounter instance to use
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.summary_model_id = summary_model_id or model_id
        
        # Initialize token counter
        self.token_counter = token_counter or TokenCounter(model_id)
    
    def summarize_conversation(
        self,
        messages: List[Dict[str, Any]],
        target_token_count: int
    ) -> Dict[str, Any]:
        """
        Create a summarized version of conversation history.
        
        Args:
            messages: List of message dictionaries
            target_token_count: Target token count for the summary
            
        Returns:
            Summary message dictionary
        """
        # This would typically call a model to generate a summary
        # Here we just create a placeholder that would be replaced
        # with actual implementation using a Bedrock client
        
        # For demonstration purposes:
        summary_text = (
            "This is a conversation summary placeholder. In a real implementation, "
            "this would call a foundation model to generate a concise summary of "
            "the conversation history, focusing on key points and important context."
        )
        
        # Create summary message
        summary = {
            "role": "system",
            "content": summary_text,
            "summary": True,
            "original_message_count": len(messages)
        }
        
        return summary
    
    def optimize_conversation(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        summarize_threshold: int = 6000
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Optimize a conversation to reduce token usage.
        
        This method applies several optimization strategies:
        1. Remove redundant whitespace
        2. Combine consecutive system messages
        3. Summarize old conversation history if over threshold
        
        Args:
            messages: List of message dictionaries
            current_tokens: Optional pre-calculated token count
            summarize_threshold: Token threshold for summarization
            
        Returns:
            Tuple of (optimized messages list, new token count)
        """
        if not messages:
            return messages, 0
            
        # Count tokens if not provided
        if current_tokens is None:
            current_tokens = self.token_counter.count_conversation_tokens(messages)
            
        # Check if optimization is needed
        if current_tokens <= self.max_tokens:
            return messages, current_tokens
            
        # Step 1: Optimize whitespace in text content
        optimized_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), str):
                # Remove redundant whitespace
                content = msg["content"]
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Create new message with optimized content
                new_msg = msg.copy()
                new_msg["content"] = content
                optimized_messages.append(new_msg)
            else:
                # Keep message as is if content isn't a string
                optimized_messages.append(msg)
        
        # Recount tokens
        optimized_tokens = self.token_counter.count_conversation_tokens(optimized_messages)
        
        # Step 2: Combine consecutive system messages
        if optimized_tokens > self.max_tokens:
            combined_messages = []
            system_buffer = []
            
            for msg in optimized_messages:
                if msg.get("role") == "system":
                    # Accumulate system messages
                    system_buffer.append(msg.get("content", ""))
                else:
                    # Flush system buffer if needed
                    if system_buffer:
                        combined_content = "\n\n".join(system_buffer)
                        combined_messages.append({
                            "role": "system",
                            "content": combined_content
                        })
                        system_buffer = []
                    
                    # Add non-system message
                    combined_messages.append(msg)
            
            # Add any remaining system messages
            if system_buffer:
                combined_content = "\n\n".join(system_buffer)
                combined_messages.append({
                    "role": "system",
                    "content": combined_content
                })
            
            optimized_messages = combined_messages
            optimized_tokens = self.token_counter.count_conversation_tokens(optimized_messages)
        
        # Step 3: Summarize old history if still over threshold
        if optimized_tokens > summarize_threshold:
            # Find the point to split history
            preserved_count = 8  # Preserve at least 4 turns (user+assistant pairs)
            
            if len(optimized_messages) > preserved_count:
                # Split messages into old and recent
                old_messages = optimized_messages[:-preserved_count]
                recent_messages = optimized_messages[-preserved_count:]
                
                # Extract system messages to keep separately
                system_messages = [msg for msg in old_messages if msg.get("role") == "system"]
                
                # Summarize non-system old messages
                old_non_system = [msg for msg in old_messages if msg.get("role") != "system"]
                if old_non_system:
                    summary = self.summarize_conversation(old_non_system, target_token_count=500)
                    
                    # Combine system messages with summary and recent messages
                    final_messages = system_messages + [summary] + recent_messages
                    
                    optimized_messages = final_messages
                    optimized_tokens = self.token_counter.count_conversation_tokens(optimized_messages)
        
        return optimized_messages, optimized_tokens


# Example usage
if __name__ == "__main__":
    # Example conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant specializing in AWS services."},
        {"role": "user", "content": "What is AWS Bedrock?"},
        {"role": "assistant", "content": "AWS Bedrock is a fully managed service that provides access to foundation models (FMs) from leading AI companies through a unified API. It allows you to build generative AI applications without having to manage the underlying infrastructure or train your own models."},
        {"role": "user", "content": "Which foundation models does it support?"},
        {"role": "assistant", "content": "AWS Bedrock supports a variety of foundation models from different providers, including:\n\n1. **Anthropic Claude models** - Claude, Claude Instant, and Claude 2\n2. **Amazon Titan models** - Amazon's own foundation models\n3. **AI21 Labs Jurassic models** - Including Jurassic-2\n4. **Stability AI models** - For image generation\n5. **Cohere models** - For text generation and embeddings\n\nEach model has different capabilities, context lengths, and pricing. You can choose the model that best fits your specific use case and requirements."}
    ]
    
    # Initialize tools
    counter = TokenCounter(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    trimmer = TokenTrimmer(model_id="anthropic.claude-3-sonnet-20240229-v1:0", max_tokens=4000)
    optimizer = TokenOptimizer(model_id="anthropic.claude-3-sonnet-20240229-v1:0", max_tokens=4000)
    
    # Count tokens
    token_count = counter.count_conversation_tokens(conversation)
    print(f"Original conversation: {token_count} tokens, {len(conversation)} messages")
    
    # Create a larger conversation for demonstration
    large_conversation = conversation.copy()
    
    # Add more messages
    for i in range(10):
        large_conversation.extend([
            {"role": "user", "content": f"Tell me about use case {i+1} for AWS Bedrock."},
            {"role": "assistant", "content": f"Use case {i+1} for AWS Bedrock involves building a sophisticated application that requires natural language processing. This could include customer service chatbots, content generation systems, or document analysis tools. The key advantage is that you can leverage pre-trained foundation models without needing to build and train your own models from scratch, significantly reducing time to market and development costs."}
        ])
    
    # Count tokens in large conversation
    large_token_count = counter.count_conversation_tokens(large_conversation)
    print(f"Large conversation: {large_token_count} tokens, {len(large_conversation)} messages")
    
    # Trim conversation
    trimmed_messages, trimmed_count = trimmer.trim_conversation(large_conversation)
    print(f"Trimmed conversation: {trimmed_count} tokens, {len(trimmed_messages)} messages")
    
    # Optimize conversation
    optimized_messages, optimized_count = optimizer.optimize_conversation(large_conversation)
    print(f"Optimized conversation: {optimized_count} tokens, {len(optimized_messages)} messages")