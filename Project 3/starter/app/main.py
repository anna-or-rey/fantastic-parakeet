# app/main.py - Travel Concierge Agent with Semantic Kernel
import os
import json
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding, OpenAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory
from app.state import AgentState
from app.memory import ShortTermMemory
from app.long_term_memory.core import LongTermMemory
from app.utils.config import validate_all_config
from app.utils.logger import setup_logger
from app.tools.weather import WeatherTools
from app.tools.fx import FxTools
from app.tools.search import SearchTools
from app.tools.card import CardTools
from app.tools.knowledge import KnowledgeTools
from app.models import TripPlan
import json as json_module

logger = setup_logger("travel_agent")





# ------------------------------
# KERNEL CREATION
# ------------------------------
def create_kernel():
    """
    Create and configure the Semantic Kernel instance.

    Sets up:
    1. Azure OpenAI services (AzureChatCompletion, AzureTextEmbedding)
    2. Tool plugins (WeatherTools, FxTools, SearchTools, CardTools, KnowledgeTools)
    """
    kernel = Kernel()

    # Add Azure OpenAI Chat Completion service
    chat_deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
    kernel.add_service(
        AzureChatCompletion(
            service_id=chat_deployment,
            deployment_name=chat_deployment,
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
    )
    logger.info(f"✅ Added AzureChatCompletion service: {chat_deployment}")

    # Add Azure OpenAI Text Embedding service
    embed_deployment = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")
    kernel.add_service(
        AzureTextEmbedding(
            service_id=embed_deployment,
            deployment_name=embed_deployment,
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
    )
    logger.info(f"✅ Added AzureTextEmbedding service: {embed_deployment}")

    # Register tool plugins
    kernel.add_plugin(WeatherTools(), plugin_name="weather")
    kernel.add_plugin(FxTools(), plugin_name="fx")
    kernel.add_plugin(SearchTools(), plugin_name="search")
    kernel.add_plugin(CardTools(), plugin_name="card")
    kernel.add_plugin(KnowledgeTools(), plugin_name="knowledge")
    logger.info("✅ Registered 5 tool plugins: weather, fx, search, card, knowledge")

    return kernel

def LongTermMemory():
    raise NotImplementedError


# -------------------------------
# MAIN REQUEST PIPELINE (AGENTIC LOOP)
# -------------------------------
async def run_request(user_input: str, memory: ShortTermMemory = None, long_term_memory: LongTermMemory = None) -> str:
    try:
        validate_all_config()
        kernel = create_kernel()

        # Initialize state machine
        state = AgentState()
        logger.info(f"📍 State: {state.phase}")

        # Initialize or use existing short-term memory
        if memory is None:
            memory = ShortTermMemory(max_items=10, max_tokens=4000)
        memory.add_conversation("user", user_input)

        # Store user input in long-term memory (Cosmos DB)
        if long_term_memory is not None:
            try:
                await long_term_memory.add_memory(
                    session_id=memory.session_id,
                    content=user_input,
                    memory_type="conversation",
                    importance_score=0.5,
                    tags=["user_input"],
                )
            except Exception as e:
                logger.warning(f"Failed to store user input in long-term memory: {e}")

        state.advance()
        logger.info(f"📍 State: {state.phase}")
        logger.info(f"Request: {user_input}")

        # Get chat service
        chat_service = kernel.get_service(type=ChatCompletionClientBase)

        # Create chat history
        chat_history = ChatHistory()

        # ------------------------------
        # SYSTEM MESSAGE PROMPT
        # ------------------------------
        system_message = """You are a professional AI travel concierge agent with access to real-time tools.
Your job is to help users plan trips by gathering weather data, finding restaurants/hotels/attractions,
converting currencies, and recommending the best credit cards for their travel spending.

## AVAILABLE TOOLS

1. **weather.get_weather(city="...")** - Get 7-day weather forecast for a destination city
   - Use when: User mentions a destination and wants weather info
   - Returns: Temperature, conditions, and recommendations

2. **search.web_search(query="...", max_results=5)** - Search the web via Bing for travel info
   - Use when: User asks about restaurants, hotels, attractions, events, or local tips
   - Returns: List of results with title, url, snippet

3. **fx.convert_fx(amount=100, base="USD", target="EUR")** - Convert currency
   - Use when: User asks about costs, currency exchange, or mentions spending abroad
   - Returns: Conversion rate and converted amount

4. **card.recommend_card(category="...", country="...", amount=100)** - Get credit card recommendation
   - Categories: dining, travel, hotels, shopping, general
   - Use when: User mentions a credit card or asks which card to use
   - Returns: Best card with benefits and FX fee info

5. **knowledge.search_knowledge(query="...", card_name="...", category="...")** - Search credit card knowledge base
   - Use when: User asks about specific card benefits, lounge access, or perks
   - Returns: Relevant knowledge base entries about card benefits

## TOOL USAGE GUIDELINES

For a typical trip planning query, you should:
1. Call get_weather() for the destination city
2. Call web_search() for restaurants, hotels, or attractions as requested
3. Call convert_fx() to show local currency costs
4. Call recommend_card() if user mentions a card or asks for spending advice
5. Call search_knowledge() for detailed card benefit information

## OUTPUT FORMAT

You MUST respond with a valid JSON object matching this exact structure:

{
  "destination": "City, Country",
  "travel_dates": "Start date to End date",
  "weather": {
    "temperature_c": 22.5,
    "conditions": "partly cloudy",
    "recommendation": "Pack light layers"
  },
  "results": [
    {
      "title": "Restaurant/Hotel Name",
      "snippet": "Brief description",
      "url": "https://...",
      "price_range": "$$",
      "rating": 4.5,
      "category": "restaurant"
    }
  ],
  "card_recommendation": {
    "card": "BankGold",
    "benefit": "4x points on dining",
    "fx_fee": "None",
    "source": "knowledge_base"
  },
  "currency_info": {
    "usd_to_eur": 0.92,
    "sample_meal_usd": 50,
    "sample_meal_eur": 46,
    "points_earned": 200
  },
  "citations": ["https://source1.com", "https://source2.com"],
  "next_steps": ["Book restaurant reservations", "Notify bank of travel"]
}

## ANTI-HALLUCINATION RULES (CRITICAL)

1. ONLY include data that was actually returned by the tools you called
2. Use null for optional fields if no data was obtained:
   - "weather": null (if get_weather was not called or failed)
   - "card_recommendation": null (if no card info requested)
   - "currency_info": null (if no currency conversion done)
   - "results": null or [] (if no search was performed)
3. Use "N/A" for destination/travel_dates if the query is not about trip planning
4. NEVER fabricate URLs, ratings, prices, or any factual data
5. If a tool call fails, acknowledge the limitation - do not make up data
6. Include actual URLs from search results in the citations array

Always be helpful and professional while strictly adhering to these rules."""

        chat_history.add_system_message(system_message)

        # Add conversation history from memory for context
        for item in memory.get_conversation_history():
            if item.get("role") == "user":
                chat_history.add_user_message(item.get("content", ""))
            elif item.get("role") == "assistant":
                chat_history.add_assistant_message(item.get("content", ""))

        chat_history.add_user_message(user_input)

        # Enable automatic function calling
        execution_settings = OpenAIChatPromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
            temperature=0.7,
            max_tokens=2000
        )

        state.advance()
        logger.info(f"📍 State: {state.phase}")
        logger.info("🤖 LLM will automatically call tools as needed...")

        # Let LLM automatically call tools
        response = await chat_service.get_chat_message_contents(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        )

        state.advance()
        logger.info(f"📍 State: {state.phase}")

        agent_response = response[0].content
        logger.info(f"✅ Agent response received: {len(agent_response)} chars")

        # Save assistant response to memory
        memory.add_conversation("assistant", agent_response[:500])
        
        # Store assistant response in long-term memory (Cosmos DB)
        if long_term_memory is not None:
            try:
                await long_term_memory.add_memory(
                    session_id=memory.session_id,
                    content=agent_response[:500],
                    memory_type="conversation",
                    importance_score=0.6,
                    tags=["assistant_response"],
                )
            except Exception as e:
                logger.warning(f"Failed to store assistant response in long-term memory: {e}")
            
        # Parse and validate response with Pydantic (Lesson 2 pattern)
        try:
            # Extract JSON from response (handle cases where LLM includes extra text)
            json_start = agent_response.find('{')
            json_end = agent_response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = agent_response[json_start:json_end]
            response_data = json_module.loads(json_str)

            logger.info("✅ JSON parsed successfully")

            # Validate with TripPlan Pydantic model
            trip_plan = TripPlan(**response_data)
            logger.info(f"✅ Pydantic validation passed: {trip_plan.destination}")

            # Auto-populate citations from search results if empty
            if (not trip_plan.citations or trip_plan.citations == []) and trip_plan.results:
                trip_plan.citations = [r.url for r in trip_plan.results if r.url]
                logger.info(f"✅ Auto-populated {len(trip_plan.citations)} citations from results")

            state.advance()
            logger.info(f"📍 State: {state.phase}")

            # Return validated Pydantic model as JSON
            result = {
                "trip_plan": trip_plan.model_dump(),
                "metadata": {
                    "session_id": state.session_id,
                    "tools_called": ["automatic_via_llm"],
                    "data_quality": "validated_with_pydantic",
                    "memory_items": len(memory.get_conversation_history())
                }
            }

            return json.dumps(result, indent=2, default=str)

        except (json_module.JSONDecodeError, ValueError) as e:
            logger.warning(f"⚠️ JSON parsing failed: {e}")
            logger.warning("Falling back to raw agent response")

            # Fallback: return raw response if JSON parsing fails
            result = {
                "raw_response": agent_response,
                "metadata": {
                    "session_id": state.session_id,
                    "tools_called": ["automatic_via_llm"],
                    "data_quality": "unvalidated",
                    "parse_error": str(e)
                }
            }

            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            logger.error(f"❌ Pydantic validation failed: {e}")
            logger.warning("Falling back to raw agent response")

            # Fallback: return raw response if validation fails
            result = {
                "raw_response": agent_response,
                "metadata": {
                    "session_id": state.session_id,
                    "tools_called": ["automatic_via_llm"],
                    "data_quality": "validation_failed",
                    "validation_error": str(e)
                }
            }

            return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(f"Request failed: {e}")
        return json.dumps({"error": str(e), "status": "failed"}, indent=2)



# -------------------------------
# CLI ENTRY POINT
# -------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Travel Concierge Agent")
    parser.add_argument("--input", help="User input for the agent")
    args = parser.parse_args()

    if args.input:
        result = asyncio.run(run_request(args.input))
        print(result)
    else:
        print("Travel Concierge Agent (type 'quit' to exit)")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                result = asyncio.run(run_request(user_input))
                try:
                    data = json.loads(result)
                    if "plan" in data:
                        from app.utils.pretty_print import print_plan
                        print_plan(data["plan"])
                    else:
                        print(result)
                except Exception:
                    print(result)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()