from datetime import datetime
import ast
import operator
import requests
from duckduckgo_search import DDGS

1. CALCULATOR TOOL


def calculator(expression: str):

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def evaluate(node):

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid number")

        if isinstance(node, ast.BinOp):

            left = evaluate(node.left)
            right = evaluate(node.right)

            op = allowed_operators.get(type(node.op))

            if op is None:
                raise ValueError("Operator not allowed")

            return op(left, right)

        if isinstance(node, ast.UnaryOp):

            operand = evaluate(node.operand)

            op = allowed_operators.get(type(node.op))

            if op is None:
                raise ValueError("Operator not allowed")

            return op(operand)

        raise ValueError("Invalid expression")

    try:

        tree = ast.parse(expression, mode="eval")

        result = evaluate(tree.body)

        return str(result)

    except Exception as e:

        return f"Calculation error: {e}"



# 2. TIME TOOL


def get_current_time():

    now = datetime.now()

    return now.strftime(
        "%A, %d %B %Y, %I:%M %p"
    )


# =====================================================
# 3. WEB SEARCH TOOL
# =====================================================

def web_search(query: str):

    try:

        results = []

        with DDGS() as ddgs:

            search_results = ddgs.text(
                query,
                max_results=5
            )

            for result in search_results:

                results.append({
                    "title": result.get("title"),
                    "body": result.get("body"),
                    "url": result.get("href")
                })

        if not results:
            return "No search results found."

        return str(results)

    except Exception as e:

        return f"Web search error: {e}"



# 4. WEATHER TOOL

def get_weather(city: str):

    try:

        # First find city coordinates
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params
        )

        geo_data = geo_response.json()

        if "results" not in geo_data:
            return f"Could not find city: {city}"

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        city_name = location["name"]

        # Get weather
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "wind_speed_10m"
            )
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params
        )

        weather_data = weather_response.json()

        current = weather_data["current"]

        return {
            "city": city_name,
            "temperature": current["temperature_2m"],
            "feels_like": current["apparent_temperature"],
            "humidity": current["relative_humidity_2m"],
            "wind_speed": current["wind_speed_10m"]
        }

    except Exception as e:

        return f"Weather error: {e}"



# ALL AVAILABLE TOOLS

TOOLS = {

    "calculator": calculator,

    "get_current_time": get_current_time,

    "web_search": web_search,

    "get_weather": get_weather

}
