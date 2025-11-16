from google import genai
from app.core.config import GEMINI_API_KEY, GEMINI_MODEL
from agno.agent import Agent
from agno.models.google import Gemini
from app.models.query_model import QueryModel
from agno.agent import RunOutput
from app.services.vector_database_service import knowledge

agent = Agent(
    model=Gemini(
        id="gemini-2.5-flash-lite",
        api_key=GEMINI_API_KEY,
    ),
    knowledge=knowledge,
    system_message="You are a helpful assistant that provides concise and accurate answers."
)


class LLMService:
    def __init__(self):
        pass

    def generate(self, query: QueryModel) -> RunOutput:
        prompt = f"{query.query}"
        output = agent.run(prompt)
        return output