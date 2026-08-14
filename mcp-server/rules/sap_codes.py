MATERIAL_TYPES = {
    "FERT": "Finished Product",
    "HALB": "Semi-Finished Product",
    "ROH": "Raw Material",
    "HAWA": "Trading Goods",
}

PROCUREMENT_TYPES = {
    "E": "In-house production",
    "F": "External procurement",
}

MRP_TYPES = {
    "PD": "MRP planning",
    "VB": "Manual reorder point planning",
}

PRICE_CONTROL = {
    "S": "Standard Price",
    "V": "Moving Average Price",
}


def explain_material_codes(material: dict) -> dict:
    return {
        "material_type": {
            "code": material.get("material_type"),
            "meaning": MATERIAL_TYPES.get(
                material.get("material_type"),
                "Unknown"
            ),
        },
        "procurement_type": {
            "code": material.get("procurement_type"),
            "meaning": PROCUREMENT_TYPES.get(
                material.get("procurement_type"),
                "Unknown"
            ),
        },
        "mrp_type": {
            "code": material.get("mrp_type"),
            "meaning": MRP_TYPES.get(
                material.get("mrp_type"),
                "Unknown"
            ),
        },
        "price_control": {
            "code": material.get("price_control"),
            "meaning": PRICE_CONTROL.get(
                material.get("price_control"),
                "Unknown"
            ),
        },
    }