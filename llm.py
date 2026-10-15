import ollama
import json

from tools import TOOLS


MODEL = "gemma3:4b"


TOOL_DESCRIPTIONS = """
You are an AI assistant with access to four tools.

==================================================

1. calculator

Use this for mathematical calculations.

Example:

User:
What is 25 * 40?

Tool call:

TOOL_CALL
{
    "tool": "calculator",
    "arguments": {
        "expression": "25 * 40"
    }
}

==================================================

2. get_current_time

Use this when the user asks for the current
time or date.

Example:

User:
What time is it?

Tool call:

TOOL_CALL
{
    "tool": "get_current_time",
    "arguments": {}
}

==================================================

3. web_search

Use this when the user asks for information
that requires an internet search, especially
current or recent information.

Example:

User:
What are the latest AI news?

Tool call:

TOOL_CALL
{
    "tool": "web_search",
    "arguments": {
        "query": "latest AI news"
    }
}

==================================================

4. get_weather

Use this when the user asks about weather.

Example:

User:
What is the weather in Delhi?

Tool call:

TOOL_CALL
{
    "tool": "get_weather",
    "arguments": {
        "city": "Delhi"
    }
}

==================================================

IMPORTANT:

If a tool is required, respond ONLY with:

TOOL_CALL
{JSON}

Do not write anything else.

If no tool is required, answer normally.

Keep normal answers concise and suitable for speaking aloud.
"""


def ask_llm(user_message):

    prompt = f"""
{TOOL_DESCRIPTIONS}

User message:

{user_message}

Decide whether you need a tool.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"num_predict": 160},
    )

    return response["message"]["content"]


def execute_tool(tool_response):

    try:

        data = tool_response.replace(
            "TOOL_CALL",
            ""
        ).strip()

        data = json.loads(data)

        tool_name = data["tool"]

        arguments = data.get(
            "arguments",
            {}
        )

        if tool_name not in TOOLS:

            return "Unknown tool."

        tool_function = TOOLS[tool_name]

        result = tool_function(**arguments)

        return result

    except Exception as e:

        return f"Tool execution error: {e}"


def final_answer(user_message, tool_result):

    response = ollama.chat(

        model=MODEL,

        messages=[
            {
                "role": "user",

                "content": f"""
You are an AI voice assistant.

User:
{user_message}

Tool result:
{tool_result}

Give the user a short, natural answer.

Do not mention internal tool calling.
"""
            }
        ],
        options={"num_predict": 160},
    )

    return response["message"]["content"]