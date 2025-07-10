import psycopg2
from dotenv import load_dotenv
import os
import re
from agno.agent import Agent
from agno.models.google import Gemini
import asyncio
from pydantic import BaseModel, Field
from agno.tools import tool
load_dotenv()
from app.models.request_models import ProductModels

USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class SqlOutput(BaseModel):
    sql_query: str = Field(description="The SQL query to fetch the data")

def clean_sql(sql: str) -> str:
    sql = sql.replace(";", "")
    sql = sql.replace("```sql", " ")
    sql = sql.replace("\n", " ")
    sql = sql.replace("`", "")
    return sql

# Detect destructive SQL
def is_safe_sql(sql: str) -> bool:
    forbidden = [
        "delete", "update", "insert", "drop", "truncate",
        "alter", "create", "grant", "revoke"
    ]
    return not any(re.search(rf"\b{kw}\b", sql, re.IGNORECASE) for kw in forbidden)

# Inject RLS
def inject_user_filter(sql: str, user_id: str) -> str:
    sql = sql.strip().rstrip(";")
    if "where" in sql.lower():
        return f"{sql} AND user_id = '{user_id}'"
    else:
        return f"{sql} WHERE user_id = '{user_id}'"

# Enforce max result limit
def enforce_limit(sql: str, limit: int = 10) -> str:
    lowered = sql.lower()
    if "limit" in lowered:
        sql = sql[:lowered.rfind("limit")].strip()
    return f"{sql} LIMIT {limit};"

def fetch_textile_items(user_id: str, user_message: str) -> dict:
    # STEP 1: Convert NL to SQL (LLM mock)

    sql_agent = Agent(
        name="SQL Agent",
        model=Gemini(id="gemini-2.5-flash-preview-05-20", api_key=GEMINI_API_KEY),
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
    # Run the agent asynchronously and obtain the SQL string
    raw_response = sql_agent.run(user_message)

    # If the agent returned a coroutine, execute it; otherwise, use the value directly.
    if asyncio.iscoroutine(raw_response):
        agent_response = asyncio.run(raw_response).content
    else:
        agent_response = raw_response

    if isinstance(agent_response, str):
        sql_str = agent_response
    elif hasattr(agent_response, "sql_query"):
        sql_str = agent_response.sql_query
    elif hasattr(agent_response, "content"):
        # Fallback: some agent implementations expose the raw content
        sql_str = agent_response.content
    else:
        # Unable to extract a SQL string – return early with error state.
        return {
            "success": False,
            "reason": "Unable to parse SQL from LLM response",
            "data": []
        }

    # Ensure we have a string before cleaning up the SQL string (remove code-block markers etc.)
    if not isinstance(sql_str, str):
        sql_str = getattr(sql_str, "sql_query", str(sql_str))

    sql_str = clean_sql(sql_str)

    # STEP 2: Validate SQL
    if not is_safe_sql(sql_str):
        return {
            "success": False,
            "reason": "Unsafe SQL detected",
            "data": []
        }

    # STEP 3: Secure & limit
    secure_sql = enforce_limit(inject_user_filter(sql_str, user_id))

    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        cursor = connection.cursor()
        cursor.execute(secure_sql)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        products : list[ProductModels] = []
        for result in results:
            products.append(
                ProductModels(
                product_id=result["product_id"] if result["product_id"] else None,
                name=result["name"],
                material=result["material"] if result["material"] else None,
                pattern=result["pattern"] if result["pattern"] else None,
                type=result["type"] if result["type"] else None,
                gsm=int(result["gsm"]) if result["gsm"] else None,
                price_per_meter=result["price_per_meter"] if result["price_per_meter"] else None,
                image_url=result["image_url"] if result["image_url"] else None,
                price_currency=result["price_currency"] if result["price_currency"] else None,
                price_local=result["price_local"] if result["price_local"] else None,
                user_id=result["user_id"] if result["user_id"] else None,
                id=result["id"] if result["id"] else None,
                
            ))
        return {
            "products": products
        }

    except Exception as e:
        print(e)
        return {
            "products": []
        }

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


if __name__ == "__main__":
    items = fetch_textile_items(
        user_id="0b54dd49-abcc-41b5-90e4-618cd5e519f0",
        user_message="Show me cotton materials above 100 GSM"
    )

    print(items)
    for item in items["products"]:
        print(item.name)