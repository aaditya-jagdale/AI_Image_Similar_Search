from agno.agent import Agent
from agno.models.google import Gemini
from app.core.config import GEMINI_API_KEY
#RunOutput
from agno.agent import RunOutput

agent = Agent(
    model=Gemini(
        id="gemini-2.5-flash-lite",
        api_key=GEMINI_API_KEY,
    ),
    system_message="You are a helpful assistant that provides concise and accurate answers."
)

res: RunOutput =  agent.run("Who was the greatest ruler of all time?")
print(res.content)