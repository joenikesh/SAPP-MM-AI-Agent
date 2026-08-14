from mcp.server.fastmcp import FastMCP
from rules.sap_codes import explain_material_codes
from material_api_client import (
    search_materials,
    get_material,
)

mcp = FastMCP("SAP MM MCP Server")
from rules.validation import validate_material_request
from rules.defaults import DEFAULTS_BY_MATERIAL_TYPE

from material_api_client import (
    search_materials,
    get_material,
    create_material,
)

@mcp.tool()
async def search_material_master(
    q: str | None = None,
    plant: str | None = None,
    material_type: str | None = None,
    material_group: str | None = None,
):
    """
    Search SAP material master records.

    Use this tool when looking for materials by description,
    plant, material type, or material group.
    """

    return await search_materials(
        q=q,
        plant=plant,
        material_type=material_type,
        material_group=material_group,
    )


@mcp.tool()
async def get_material_master(material_number: str):
    """
    Get the full material master record for one material number.
    """

    return await get_material(material_number)

@mcp.tool()
async def explain_material_master(material_number: str):
    """
    Get a material master record together with
    authoritative SAP MM meanings for coded fields.
    """

    material = await get_material(material_number)

    code_meanings = explain_material_codes(material)

    return {
        "material": material,
        "code_meanings": code_meanings,
    
    }
@mcp.tool()
async def create_material_master(
    material_description: str,
    material_type: str,
    material_group: str,
    base_unit: str,
    plant: str,
    standard_price: float | None = None,
    currency: str = "USD",
):
    """
    Create a new SAP material master record.

    Hidden SAP MM defaults and validation rules
    are applied before creation.
    """

    material_data = {
        "material_description": material_description,
        "material_type": material_type,
        "material_group": material_group,
        "base_unit": base_unit,
        "plant": plant,
        "standard_price": standard_price,
        "currency": currency,
    }

    defaults = DEFAULTS_BY_MATERIAL_TYPE.get(
        material_type,
        {}
    )

    material_data.update(defaults)

    errors = validate_material_request(
        material_data
    )

    if errors:
        return {
            "status": "VALIDATION_FAILED",
            "errors": errors,
            "material": material_data,
        }

    result = await create_material(
        material_data
    )

    return {
        "status": "CREATED",
        "material": result,
    }

if __name__ == "__main__":
    mcp.run()