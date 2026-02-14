# app/memory.py - Short-Term Memory Implementation for Agent Learning
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class ShortTermMemory:
    """
    Short-term memory implementation for AI agents.

    This class demonstrates how agents can maintain context within a session
    using conversation history and sliding window techniques.

    Key Learning Concepts:
    - Context window management
    - Memory persistence within sessions
    - Conversation history tracking
    - Memory eviction strategies
    """

    def __init__(self, max_items: int = 10, max_tokens: int = 2000):
        """
        Initialize short-term memory.

        Args:
            max_items: Maximum number of conversation items to store
            max_tokens: Maximum total tokens across all stored items
        """
        self.max_items = max_items
        self.max_tokens = max_tokens
        self.memory_items: List[Dict[str, Any]] = []
        self.session_id = self._generate_session_id()
        self.created_at = datetime.now()
        self.total_tokens = 0

    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        import uuid
        return f"session_{uuid.uuid4().hex[:8]}"

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Simple estimation: ~4 characters per token for English text.
        """
        return len(text) // 4

    def add_conversation(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a conversation item to memory.

        Args:
            role: 'user', 'assistant', or 'system'
            content: The conversation content
            metadata: Optional additional data (timestamps, tool calls, etc.)
        """
        # Estimate tokens for this content
        tokens = self._estimate_tokens(content)

        # Create item dict with role, content, timestamp, tokens, metadata
        item = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'tokens': tokens,
            'metadata': metadata or {}
        }

        # Append to memory_items and update total_tokens
        self.memory_items.append(item)
        self.total_tokens += tokens

        # Call _evict_if_needed() to enforce limits
        self._evict_if_needed()

    def add_tool_call(self, tool_name: str, input_data: Dict[str, Any],
                     output_data: Dict[str, Any], success: bool = True) -> None:
        """
        Add a tool call to memory.

        Args:
            tool_name: Name of the tool called
            input_data: Input parameters to the tool
            output_data: Output from the tool
            success: Whether the tool call was successful
        """
        content = f"Tool call: {tool_name}"
        metadata = {
            'type': 'tool_call',
            'tool_name': tool_name,
            'input': input_data,
            'output': output_data,
            'success': success
        }

        self.add_conversation('assistant', content, metadata)

    def add_system_event(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a system event to memory.

        Args:
            event: Description of the system event
            data: Optional additional event data
        """
        metadata = {
            'type': 'system_event',
            'event': event,
            'data': data or {}
        }

        self.add_conversation('system', event, metadata)

    def _evict_if_needed(self) -> None:
        """
        Apply eviction strategy when memory limits are exceeded.

        Uses sliding window approach: removes oldest items first when limits are exceeded.
        """
        # Remove oldest items if we exceed max_items
        while len(self.memory_items) > self.max_items:
            removed_item = self.memory_items.pop(0)
            self.total_tokens -= removed_item['tokens']

        # Remove oldest items if we exceed max_tokens
        while self.total_tokens > self.max_tokens and self.memory_items:
            removed_item = self.memory_items.pop(0)
            self.total_tokens -= removed_item['tokens']

        # Ensure total_tokens doesn't go negative (safety check)
        if self.total_tokens < 0:
            self.total_tokens = 0

    def get_conversation_history(self, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.

        Args:
            include_metadata: Whether to include metadata in the response

        Returns:
            List of conversation items in chronological order
        """
        # Return memories in chronological order (already stored in order)
        if include_metadata:
            # Return full items including all metadata
            return list(self.memory_items)
        else:
            # Return only role, content, timestamp (exclude metadata and tokens)
            return [
                {
                    'role': item['role'],
                    'content': item['content'],
                    'timestamp': item['timestamp']
                }
                for item in self.memory_items
            ]

    def get_recent_conversation(self, last_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent N conversation items.

        Args:
            last_n: Number of recent items to return

        Returns:
            List of recent conversation items
        """
        return self.memory_items[-last_n:] if last_n > 0 else []

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current memory state.

        Returns:
            Dictionary containing memory statistics
        """
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'total_items': len(self.memory_items),
            'total_tokens': self.total_tokens,
            'max_items': self.max_items,
            'max_tokens': self.max_tokens,
            'memory_usage_percent': (len(self.memory_items) / self.max_items) * 100,
            'token_usage_percent': (self.total_tokens / self.max_tokens) * 100,
            'oldest_item': self.memory_items[0]['timestamp'] if self.memory_items else None,
            'newest_item': self.memory_items[-1]['timestamp'] if self.memory_items else None
        }

    def search_memory(self, query: str, role_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search through memory for items containing the query.

        Args:
            query: Search query
            role_filter: Optional role to filter by ('user', 'assistant', 'system')

        Returns:
            List of matching memory items
        """
        results = []
        query_lower = query.lower()

        for item in self.memory_items:
            # Apply role filter if specified
            if role_filter and item['role'] != role_filter:
                continue

            # Search in content
            if query_lower in item['content'].lower():
                results.append(item)
                continue

            # Search in metadata
            if item.get('metadata', {}).get('tool_name', '').lower().find(query_lower) != -1:
                results.append(item)

        return results

    def clear_memory(self) -> None:
        """Clear all memory items"""
        self.memory_items.clear()
        self.total_tokens = 0

    def export_memory(self, filepath: str) -> None:
        """
        Export memory to a JSON file.

        Args:
            filepath: Path to save the memory data
        """
        export_data = {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'memory_items': self.memory_items,
            'total_tokens': self.total_tokens
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def import_memory(self, filepath: str) -> None:
        """
        Import memory from a JSON file.

        Args:
            filepath: Path to load the memory data from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.session_id = data.get('session_id', self.session_id)
        self.memory_items = data.get('memory_items', [])
        self.total_tokens = data.get('total_tokens', 0)

    def get_context_window(self, max_tokens: Optional[int] = None) -> str:
        """
        Get a formatted context window for use in prompts.

        Args:
            max_tokens: Maximum tokens for the context window

        Returns:
            Formatted string of recent conversation
        """
        if not self.memory_items:
            return "No conversation history."

        # Use provided max_tokens or default to 80% of our limit
        token_limit = max_tokens or int(self.max_tokens * 0.8)

        context_parts = []
        current_tokens = 0

        # Start from most recent and work backwards
        for item in reversed(self.memory_items):
            item_tokens = item['tokens']
            if current_tokens + item_tokens > token_limit:
                break

            # Format the conversation item
            role = item['role'].upper()
            content = item['content']
            context_parts.insert(0, f"{role}: {content}")
            current_tokens += item_tokens

        return "\n".join(context_parts)

    def __str__(self) -> str:
        """String representation of the memory state"""
        summary = self.get_memory_summary()
        return f"ShortTermMemory(session={self.session_id}, items={summary['total_items']}, tokens={summary['total_tokens']})"

    def __repr__(self) -> str:
        """Detailed representation of the memory state"""
        return f"ShortTermMemory(session_id='{self.session_id}', items={len(self.memory_items)}, tokens={self.total_tokens})"


# Example usage and demonstration functions
def demonstrate_memory_usage():
    """
    Demonstrate the ShortTermMemory class with a three-line conversation.
    This is perfect for student learning!
    """
    print("🧠 Short-Term Memory Demonstration")
    print("=" * 50)

    # Create memory with small limits for demonstration
    memory = ShortTermMemory(max_items=5, max_tokens=500)

    print(f"Created memory: {memory}")
    print()

    # Simulate a three-line conversation
    print("📝 Adding conversation items...")

    # User asks a question
    memory.add_conversation("user", "What's the weather like in Paris?")
    print(f"After user question: {memory.get_memory_summary()}")

    # Assistant responds and calls a tool
    memory.add_conversation("assistant", "Let me check the weather for you.")
    memory.add_tool_call("weather", {"location": "Paris"}, {"temperature": 22, "condition": "sunny"})
    print(f"After assistant response + tool call: {memory.get_memory_summary()}")

    # User asks follow-up
    memory.add_conversation("user", "What about restaurants in Paris?")
    print(f"After follow-up question: {memory.get_memory_summary()}")

    print("\n📊 Memory Analysis:")
    print("-" * 30)

    # Show what's stored
    print("Current conversation history:")
    for i, item in enumerate(memory.get_conversation_history(), 1):
        print(f"  {i}. [{item['role']}] {item['content']} ({item['tokens']} tokens)")

    print(f"\nMemory summary: {memory.get_memory_summary()}")

    # Show context window
    print(f"\nContext window for prompts:")
    print(memory.get_context_window())

    # Demonstrate search
    print(f"\nSearching for 'weather':")
    weather_items = memory.search_memory("weather")
    for item in weather_items:
        print(f"  Found: [{item['role']}] {item['content']}")

    return memory


if __name__ == "__main__":
    # Run the demonstration
    demo_memory = demonstrate_memory_usage()
