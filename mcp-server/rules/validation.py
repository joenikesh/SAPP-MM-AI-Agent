VALID_PLANTS = {"1000", "1100", "1200"}
VALID_BASE_UNITS = {"EA", "KG", "L", "M", "CS"}
VALID_MATERIAL_TYPES = {"FERT", "HALB", "ROH", "HAWA"}


def validate_material_request(data: dict) -> list[str]:
    errors = []

    required_fields = [
        "material_description",
        "material_type",
        "material_group",
        "base_unit",
        "plant",
    ]

    for field in required_fields:
        if not data.get(field):
            errors.append(f"{field} is required")

    if data.get("plant") not in VALID_PLANTS:
        errors.append("Invalid plant")

    if data.get("base_unit") not in VALID_BASE_UNITS:
        errors.append("Invalid base unit")

    if data.get("material_type") not in VALID_MATERIAL_TYPES:
        errors.append("Invalid material type")

    if data.get("standard_price") is not None:
        if data["standard_price"] < 0:
            errors.append("Standard price cannot be negative")

    return errors