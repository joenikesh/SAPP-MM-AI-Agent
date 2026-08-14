import httpx

MATERIAL_API_BASE_URL = "http://127.0.0.1:8001"


async def search_materials(
    q: str | None = None,
    plant: str | None = None,
    material_type: str | None = None,
    material_group: str | None = None,
):
    params = {}

    if q:
        params["q"] = q

    if plant:
        params["plant"] = plant

    if material_type:
        params["material_type"] = material_type

    if material_group:
        params["material_group"] = material_group

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{MATERIAL_API_BASE_URL}/materials/search",
            params=params,
        )

        response.raise_for_status()
        return response.json()


async def get_material(material_number: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{MATERIAL_API_BASE_URL}/materials/{material_number}"
        )

        response.raise_for_status()
        return response.json()

    import httpx

MATERIAL_API_BASE_URL = "http://127.0.0.1:8001"


async def create_material(material_data: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MATERIAL_API_BASE_URL}/materials",
            json=material_data,
        )

        response.raise_for_status()

        return response.json()