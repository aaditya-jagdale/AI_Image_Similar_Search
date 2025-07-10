from fastapi import APIRouter, Depends, HTTPException
from app.models.request_models import TextChatRequest, ProductModels
from app.utils.auth import get_current_user
from agno.agent import Agent
from agno.team import Team
# from agno.models.google import Gemini
from agno.models.ollama import Ollama
import logging
from dotenv import load_dotenv
import os
from textwrap import dedent
from datetime import datetime
from typing import List
load_dotenv()
from pydantic import BaseModel, Field
import asyncio

router = APIRouter()
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class SqlOutput(BaseModel):
    sql_query: str = Field(description="The SQL query to fetch the data")


class TextChatResponse(BaseModel):
    message: str = Field(description="This is the response to the user's query")
    items: list[ProductModels] = Field(description="This is the list of products that match the query")


async def perform_text_chat(request: TextChatRequest, company_data: dict, user_id: str = Depends(get_current_user)):    
    # Convert incoming messages to generic chat history format expected by the agent
    message_history = [{"role": m.role, "content": m.message} for m in request.messages]

    # Grab the latest user message
    latest_user_message = None
    for m in reversed(request.messages):
        if m.role.lower() == "user":
            latest_user_message = m.message
            break
    if latest_user_message is None and request.messages:
        latest_user_message = request.messages[-1].message
    if latest_user_message is None:
        raise HTTPException(status_code=400, detail="No user message provided")
    
    support_agent = Agent(
        model=Ollama(id="llama3.2:3b"),
        name="Support Agent",
        description="A helpful assistant that can answer questions about the company and the products",
        context={
            "company_data": company_data,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "user_id": user_id,
            "current_time": datetime.now().strftime("%H:%M:%S"),
            "current_day": datetime.now().strftime("%A"),
            "chat_history": message_history,
        },
        add_context=True,
        num_history_responses=20,
        read_tool_call_history=True,
        read_chat_history=True,
        response_model=TextChatResponse,
        markdown=True,
        system_message=dedent("""
        <goal>
        - You are a helpful assistant that can answer questions about the company and the products.
        - Your PRIMARY goal is to understand user's query, deeply analyze it and then answer the question.
        - Your Secondary goal is to subtly push user to give email address and name (You must lean on the side of making sure you get the email but not force it)
        - The user may not ask a direct question because he is not sure what he wants.
        - Its your job to clarify the user's query and then answer the question.
        </goal>
        <instructions>
        - You will answer questions about the company and the products.
        - The company is a textile export and manufacturing company.
        - The company manufactures and exports a wide range of products and a buyer is looking for a specific product.
        - The buyers will ask information about the product and the company, and you must answer them professionally.
        - You will use the information provided to you to answer the question.
        - You will not make up information.
        - You will not answer questions that are not related to the company and the products.
        - You cannot send direct email right now. So only ask email for "We will contact you soon" purposes
        </instructions>
        <persona>
        - You are a helpful assistant
        - You must keep the conversation bubbly and engaging.
        - You should refrain from using "I" in your responses. You must use "We" while referring to company
        - Keep the tone business professional.
        - If the user seems a lot interested in products you can ask to contact the company.
        - But dont ask to contact the company in every message.
        - You are having a conversation with a potential buyer.
        - Because this is a conversation do not use too fancy words. Have a human like normal conversation.
        - A normal reply would be just between 1-4 sentences maximum. Dont go beyond that.
        </persona>
        """),

    )

    sql_agent = Agent(
        name="SQL Agent",
        # model=Gemini(id="gemini-2.5-flash-preview-05-20", api_key=GEMINI_API_KEY),
        model=Ollama(id="llama3.2:3b"),
        response_model=SqlOutput,
        structured_outputs=True,
        system_message="""
    You are an expert Postgres SQL generator. Your job is to convert natural language queries into accurate, safe SELECT-only SQL queries for the following schema.

    Never return INSERT, UPDATE, DELETE, DROP, or any write operation. Use SELECT only.

    Table: textile_products
    Columns:
    - product_id (text)
    - name (text)
    - material (text)
    - pattern (text)
    - type (text)
    - gsm (integer)
    - price_per_meter (numeric in USD)

    Examples:
    - "cotton materials under 200 GSM" → `SELECT * FROM textile_products WHERE material ILIKE '%cotton%' AND gsm < 200`
    - "shirts with stripe pattern" → `SELECT * FROM textile_products WHERE type ILIKE '%shirt%' AND pattern ILIKE '%stripe%'`

    Always include % in the ILIKE queries.
    """
    )

    customer_support_team = Team(
        # model=Gemini(id="gemini-2.5-flash-preview-05-20", api_key=GEMINI_API_KEY),
        model=Ollama(id="llama3.2:3b"),
        members=[support_agent, sql_agent],
        mode="collaborate",
        system_message=dedent("""
        You are a customer support team that can answer questions about the company and the products.
        
        When to use each agent:
        
        1. Support Agent (support_agent):
        - Use for general company inquiries and product discussions
        - Handle customer service interactions and recommendations
        - Provide friendly, conversational responses
        - When user asks about company policies, shipping, or general questions
        
        2. SQL Agent (sql_agent):
        - Use when specific product searches or filtering is needed
        - When user asks to see products with specific criteria (material, GSM, type, etc.)
        
        Important:
        - Always route product search queries to the SQL agent
        - Route general inquiries and follow-up discussions to the support agent
        - Maintain conversation context when switching between agents
        """),
    )
    
    # Get the response from the agent
    response = customer_support_team.print_response(latest_user_message)
    
    # Return the structured response
    return ''

if __name__ == "__main__":
    items = asyncio.run(perform_text_chat(
        request=TextChatRequest(
            messages=[
                {"role": "user", "message": "Show me cotton materials above 100 GSM"}
            ]
        ),
        company_data={},
        user_id="0b54dd49-abcc-41b5-90e4-618cd5e519f0",
    ))
    # print(items.items)