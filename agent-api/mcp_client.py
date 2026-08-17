import json

from fastmcp import Client


MCP_URL = "http://127.0.0.1:8002/mcp"

client = Client(MCP_URL)


def parse_tool_content(result):
    if result.data is not None:
        return result.data

    if result.structured_content is not None:
        return result.structured_content

    parsed = []

    for item in result.content:
        if not hasattr(item, "text"):
            continue

        text = item.text

        try:
            parsed.append(json.loads(text))
        except json.JSONDecodeError:
            parsed.append(text)

    return parsed


async def search_material_master(
    q: str | None = None,
    plant: str | None = None,
    material_type: str | None = None,
    material_group: str | None = None,
):
    arguments = {}

    if q:
        arguments["q"] = q

    if plant:
        arguments["plant"] = plant

    if material_type:
        arguments["material_type"] = material_type

    if material_group:
        arguments["material_group"] = material_group

    async with client:
        result = await client.call_tool(
            "search_material_master",
            arguments,
        )

        return parse_tool_content(result)


async def query_material_master(query_plan: dict):
    async with client:
        result = await client.call_tool(
            "query_material_master",
            {"query_plan": query_plan},
        )
        parsed = parse_tool_content(result)
        if isinstance(parsed, list) and len(parsed) == 1:
            return parsed[0]
        return parsed


async def get_material_master(material_number: str):
    async with client:
        result = await client.call_tool(
            "get_material_master",
            {
                "material_number": material_number
            },
        )

        parsed = parse_tool_content(result)

        if isinstance(parsed, list) and len(parsed) == 1:
            return parsed[0]

        return parsed

async def explain_material_master(material_number: str):
    async with client:
        result = await client.call_tool(
            "explain_material_master",
            {
                "material_number": material_number
            }
        )

        parsed = parse_tool_content(result)

        if isinstance(parsed, list) and len(parsed) == 1:
            return parsed[0]

        return parsed
    
async def create_material_master(
    material_description: str,
    material_type: str,
    material_group: str,
    base_unit: str,
    plant: str,
    standard_price: float | None = None,
    currency: str = "USD",
):
    arguments = {
        "material_description": material_description,
        "material_type": material_type,
        "material_group": material_group,
        "base_unit": base_unit,
        "plant": plant,
        "currency": currency,
    }

    if standard_price is not None:
        arguments["standard_price"] = standard_price

    async with client:
        result = await client.call_tool(
            "create_material_master",
            arguments,
        )

        parsed = parse_tool_content(result)

        if isinstance(parsed, list) and len(parsed) == 1:
            return parsed[0]

        return parsed
