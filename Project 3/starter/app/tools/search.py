# tools/search.py (unified version)
import os
import json
import asyncio
from semantic_kernel import Kernel
from typing import List, Dict, Any, Optional
from semantic_kernel.functions import kernel_function


class SearchTools:
    def __init__(self):
        self.project_endpoint = os.getenv("PROJECT_ENDPOINT")
        self.agent_id = os.getenv("AGENT_ID")
        self.connection_id = os.getenv("BING_CONNECTION_ID")
        self.cred = None

    @kernel_function(
        name="web_search",
        description="General-purpose web search via Azure AI Agent with Bing grounding"
    )
    def web_search(self, query: str, max_results: int = 5, filter_keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run a web search using Bing grounding via Azure AI Agent.
        Args:
            query: the raw user query (e.g., 'luxury hotels in Dubai under $300/night')
            max_results: max number of results to return
            filter_keywords: optional list of keywords to enforce in results
        Returns:
            List of dicts with {title, url, snippet}
        """
        if not all([self.project_endpoint, self.agent_id, self.connection_id]):
            return [{
                "title": "Missing configuration",
                "url": "https://bing.com",
                "snippet": "Missing PROJECT_ENDPOINT, AGENT_ID, or BING_CONNECTION_ID"
            }]

        try:
            from azure.ai.projects import AIProjectClient
            from azure.identity import DefaultAzureCredential
 
            if self.cred is None:
                self.cred = DefaultAzureCredential()
            
            client = AIProjectClient(endpoint=self.project_endpoint, credential=self.cred)
            thread = client.agents.threads.create()

            # Create message with Bing search query
            # Instruct agent to return ONLY a JSON array with title, url, snippet fields
            search_prompt = f"""Search for: {query}

Return ONLY a JSON array with search results. Each result must have these fields:
- title: the page title
- url: the page URL
- snippet: a brief description

Example format:
[
  {{"title": "Example Page", "url": "https://example.com", "snippet": "Description here"}},
  {{"title": "Another Page", "url": "https://another.com", "snippet": "Another description"}}
]

Return ONLY the JSON array, no other text."""

            client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=search_prompt
            )

            # Run the agent to process the message
            client.agents.runs.create_and_process(
                thread_id=thread.id,
                agent_id=self.agent_id
            )

            messages = list(client.agents.messages.list(thread_id=thread.id))
            assistant_msgs = [m for m in messages if m.role == "assistant"]
            content_blocks = assistant_msgs[-1].content if assistant_msgs else []

            raw_json = None
            # Extract raw_json from content_blocks
            # Check if first content block is type "text", then get the value from text field
            if content_blocks and len(content_blocks) > 0:
                first_block = content_blocks[0]
                if hasattr(first_block, 'type') and first_block.type == "text":
                    raw_json = first_block.text.value if hasattr(first_block.text, 'value') else str(first_block.text)
                elif hasattr(first_block, 'text'):
                    raw_json = first_block.text.value if hasattr(first_block.text, 'value') else str(first_block.text)

            client.agents.threads.delete(thread_id=thread.id)

            if not raw_json:
                return [{
                    "title": "No results",
                    "url": "",
                    "snippet": "No JSON returned by Bing grounding"
                }]

            print("\n🧪 RAW Bing Assistant Response (first 1000 chars):\n", raw_json[:1000])
            print(f"\n📊 Total response length: {len(raw_json)} chars")
            raw_json = raw_json.strip()

            if raw_json.startswith("```"):
                lines = raw_json.split('\n')
                if lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw_json = '\n'.join(lines).strip()

            if not raw_json.startswith('['):
                start_idx = raw_json.find('[')
                end_idx = raw_json.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    raw_json = raw_json[start_idx:end_idx+1]
                else:
                    raise ValueError(f"No JSON array found in response. Got: {raw_json[:200]}")

            # Parse raw_json into Python list
            results = json.loads(raw_json)

            # Normalize
            normalized = [{
                "title": r.get("title", "Unknown"),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", "")
            } for r in results]

            # Optional post-filter
            if filter_keywords:
                filtered = [
                    r for r in normalized
                    if any(kw.lower() in (r["title"] + r["snippet"]).lower() for kw in filter_keywords)
                ]
                return filtered[:max_results] or normalized[:max_results]

            return normalized[:max_results]

        except Exception as e:
            import traceback
            print(f"\n❌ Search tool error: {e}")
            print(f"❌ Traceback: {traceback.format_exc()}")
            return [{
                "title": "Search error",
                "url": "",
                "snippet": f"Failed to retrieve search results: {e}"
            }]
