import json
import re

from transformers import pipeline

from prompts import SYSTEM_PROMPT
from mcp_client import (
    search_material_master,
    explain_material_master,
    create_material_master,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
)


ALLOWED_ACTIONS = {
    "search_material",
    "get_material",
    "create_material",
    "need_information",
    "answer_sap_mm",
    "reject",
}


OUT_OF_SCOPE_MESSAGE = (
    "I can only assist with SAP MM Material Master tasks, including "
    "material search, material details, SAP MM field explanations, "
    "validation, and material creation."
)


async def run_agent(
    message: str,
    session_id: str,
    pending_creations: dict,
) -> str:

    # -------------------------------------------------
    # HANDLE PENDING CREATION CONFIRMATION
    # -------------------------------------------------

    normalized = message.strip().lower()

    if session_id in pending_creations:
        pending = pending_creations[session_id]

        if normalized in {
            "yes",
            "y",
            "confirm",
            "create",
            "proceed",
        }:
            print("CONFIRMATION RECEIVED")

            result = await create_material_master(
                material_description=pending["material_description"],
                material_type=pending["material_type"],
                material_group=pending["material_group"],
                base_unit=pending["base_unit"],
                plant=pending["plant"],
                standard_price=pending.get("standard_price"),
                currency=pending.get("currency", "USD"),
            )

            del pending_creations[session_id]

            return format_create_result(result)

        if normalized in {
            "no",
            "n",
            "cancel",
            "stop",
        }:
            del pending_creations[session_id]

            return "Material creation cancelled."

    # -------------------------------------------------
    # RESTRICTED PLANNER
    # -------------------------------------------------

    planner_prompt = f"""
You are a STRICTLY RESTRICTED SAP MM Material Master agent.

You are only allowed to help with SAP MM Material Master topics.

ALLOWED SCOPE:
- search material master records
- list materials
- retrieve material details
- explain SAP MM material master fields
- explain material type codes
- explain procurement type codes
- explain MRP type codes
- explain price control codes
- create material master records
- validate material creation requests
- ask for missing material creation information

OUT OF SCOPE:
- weather
- news
- politics
- sports
- entertainment
- general programming
- coding help
- personal advice
- medical questions
- legal questions
- financial advice
- general knowledge
- internet research
- web browsing
- unrelated SAP modules
- anything not directly related to SAP MM Material Master

If the user asks anything outside the allowed scope,
you MUST return:

{{
  "action": "reject"
}}

Never answer an out-of-scope question.

Available actions:

1. search_material
Use when the user wants to:
- find materials
- search materials
- list materials
- search by description
- search by plant
- search by material type
- search by material group

2. get_material
Use when the user asks for one specific material
and provides a material number such as:
SYN-FG-000001

3. create_material
Use when the user wants to create a new material.

Required creation fields:
- material_description
- material_type
- material_group
- base_unit
- plant

Optional:
- standard_price
- currency

4. need_information
Use when the user wants to create a material
but required information is missing.

5. answer_sap_mm
Use only for SAP MM Material Master questions
that do not require live database access.

6. reject
Use for anything outside SAP MM Material Master.

IMPORTANT:
Do NOT invent missing creation values.
Do NOT browse the internet.
Do NOT claim to search the web.
Do NOT answer unrelated questions.
Do NOT create new actions.

Return ONLY valid JSON.
Do not include explanations.
Do not include markdown.
Do not include code fences.

Examples:

User:
Find pump materials in plant 1000

Response:
{{
  "action": "search_material",
  "q": "Pump",
  "plant": "1000"
}}

User:
Show material SYN-FG-000001

Response:
{{
  "action": "get_material",
  "material_number": "SYN-FG-000001"
}}

User:
Create a finished pump material called AI Pump 010
in plant 1000, material group FG010,
base unit EA, price 250 USD

Response:
{{
  "action": "create_material",
  "material_description": "AI Pump 010",
  "material_type": "FERT",
  "material_group": "FG010",
  "base_unit": "EA",
  "plant": "1000",
  "standard_price": 250,
  "currency": "USD"
}}

User:
Create a finished material

Response:
{{
  "action": "need_information",
  "missing_fields": [
    "material_description",
    "material_group",
    "base_unit",
    "plant"
  ]
}}

User:
What does FERT mean in SAP MM?

Response:
{{
  "action": "answer_sap_mm"
}}

User:
What is the weather today?

Response:
{{
  "action": "reject"
}}

User:
Write Python code for me

Response:
{{
  "action": "reject"
}}

User:
Who won the election?

Response:
{{
  "action": "reject"
}}

User request:
{message}

Response:
"""

    result = generator(
        planner_prompt,
        max_new_tokens=180,
        do_sample=False,
    )

    generated = result[0]["generated_text"]
    generated = generated[len(planner_prompt):].strip()

    print("\n===== QWEN PLANNER OUTPUT =====")
    print(generated)
    print("===============================\n")

    # -------------------------------------------------
    # PARSE FIRST JSON OBJECT
    # -------------------------------------------------

    try:
        match = re.search(
            r"\{.*?\}",
            generated,
            re.DOTALL,
        )

        if not match:
            raise ValueError("No JSON object found")

        json_text = match.group(0)

        print("EXTRACTED JSON:")
        print(json_text)

        decision = json.loads(json_text)

        print("PLANNER DECISION:", decision)

    except (json.JSONDecodeError, ValueError) as e:
        print(
            "ERROR: Could not parse planner JSON:",
            e,
        )

        return OUT_OF_SCOPE_MESSAGE

    action = decision.get("action")

    # -------------------------------------------------
    # PYTHON-SIDE ACTION ALLOWLIST
    # -------------------------------------------------

    if action not in ALLOWED_ACTIONS:
        print("BLOCKED UNKNOWN ACTION:", action)
        return OUT_OF_SCOPE_MESSAGE

    # -------------------------------------------------
    # REJECT OUT-OF-SCOPE
    # -------------------------------------------------

    if action == "reject":
        print("ACTION: Rejected out-of-scope request")
        return OUT_OF_SCOPE_MESSAGE

    # -------------------------------------------------
    # SEARCH MATERIAL
    # -------------------------------------------------

    if action == "search_material":
        print("ACTION: Calling MCP search_material_master")

        materials = await search_material_master(
            q=decision.get("q"),
            plant=decision.get("plant"),
            material_type=decision.get("material_type"),
            material_group=decision.get("material_group"),
        )

        return format_search_results(
            message=message,
            materials=materials,
        )

    # -------------------------------------------------
    # GET SPECIFIC MATERIAL
    # -------------------------------------------------

    if action == "get_material":
        material_number = decision.get(
            "material_number"
        )

        if not material_number:
            return (
                "Please provide the material number "
                "you want me to retrieve."
            )

        print(
            "ACTION: Calling MCP explain_material_master:",
            material_number,
        )

        material = await explain_material_master(
            material_number
        )

        return format_material(
            message=message,
            material=material,
        )

    # -------------------------------------------------
    # CREATE MATERIAL - PREVIEW ONLY
    # -------------------------------------------------

    if action == "create_material":
        required_fields = [
            "material_description",
            "material_type",
            "material_group",
            "base_unit",
            "plant",
        ]

        missing = [
            field
            for field in required_fields
            if not decision.get(field)
        ]

        if missing:
            return (
                "I need more information before creating "
                "the material. Missing: "
                + ", ".join(missing)
            )

        proposal = {
            "material_description": decision.get(
                "material_description"
            ),
            "material_type": decision.get(
                "material_type"
            ),
            "material_group": decision.get(
                "material_group"
            ),
            "base_unit": decision.get(
                "base_unit"
            ),
            "plant": decision.get(
                "plant"
            ),
            "standard_price": decision.get(
                "standard_price"
            ),
            "currency": decision.get(
                "currency",
                "USD",
            ),
        }

        pending_creations[session_id] = proposal

        return (
            "I'm ready to create this material:\n\n"
            f"Description: {proposal['material_description']}\n"
            f"Material Type: {proposal['material_type']}\n"
            f"Material Group: {proposal['material_group']}\n"
            f"Base Unit: {proposal['base_unit']}\n"
            f"Plant: {proposal['plant']}\n"
            f"Standard Price: {proposal.get('standard_price')}\n"
            f"Currency: {proposal.get('currency')}\n\n"
            "Reply 'yes' to confirm or 'cancel' to stop."
        )

    # -------------------------------------------------
    # NEED MORE INFORMATION
    # -------------------------------------------------

    if action == "need_information":
        missing_fields = decision.get(
            "missing_fields",
            [],
        )

        if missing_fields:
            return (
                "I need more information before creating "
                "the material. Please provide: "
                + ", ".join(missing_fields)
            )

        return (
            "I need more information before creating "
            "the material."
        )

    # -------------------------------------------------
    # SAP MM GENERAL ANSWER ONLY
    # -------------------------------------------------

    if action == "answer_sap_mm":
        print("ACTION: SAP MM informational answer")

        return generate_sap_mm_answer(
            message
        )

    # -------------------------------------------------
    # DEFAULT DENY
    # -------------------------------------------------

    return OUT_OF_SCOPE_MESSAGE


def generate_sap_mm_answer(
    message: str,
) -> str:

    prompt = f"""
{SYSTEM_PROMPT}

You are restricted to SAP MM Material Master.

The user asked:

{message}

Answer ONLY if the question is directly related
to SAP MM Material Master.

Do not answer unrelated questions.

Do not browse the internet.
Do not claim to have searched the internet.

Keep the answer concise.

Assistant:
"""

    result = generator(
        prompt,
        max_new_tokens=250,
        do_sample=False,
    )

    generated = result[0]["generated_text"]

    return generated[len(prompt):].strip()


def format_search_results(
    message: str,
    materials,
) -> str:

    if not materials:
        return "I couldn't find any matching materials."

    prompt = f"""
You are an SAP MM Material Master assistant.

The user asked:

{message}

The MCP server returned these REAL material records:

{json.dumps(materials, indent=2)}

Use ONLY these records.

Do not invent data.
Do not browse the internet.
Do not tell the user how to manually search SAP.

For each material include:
- Material number
- Description
- Material type
- Plant
- Standard price
- Currency

Keep the answer concise.

Assistant:
"""

    result = generator(
        prompt,
        max_new_tokens=500,
        do_sample=False,
    )

    generated = result[0]["generated_text"]

    return generated[len(prompt):].strip()


def format_material(
    message: str,
    material,
) -> str:

    if not material:
        return "I couldn't find that material."

    prompt = f"""
You are an SAP MM Material Master assistant.

The user asked:

{message}

The MCP server returned this REAL material record
with authoritative SAP code meanings:

{json.dumps(material, indent=2)}

Use ONLY the supplied data.

Do not browse the internet.
Do not invent meanings.
Do not infer meanings for unexplained codes.

Use code_meanings as the authoritative source
for SAP code definitions.

Prioritize:
- Material number
- Description
- Material type and meaning
- Material group
- Plant
- Procurement type and meaning
- MRP type and meaning
- Valuation class
- Price control and meaning
- Standard price
- Currency

Assistant:
"""

    result = generator(
        prompt,
        max_new_tokens=400,
        do_sample=False,
    )

    generated = result[0]["generated_text"]

    return generated[len(prompt):].strip()


def format_create_result(
    result,
) -> str:

    if not result:
        return (
            "The material creation request returned "
            "no result."
        )

    status = result.get("status")

    if status == "VALIDATION_FAILED":
        errors = result.get(
            "errors",
            [],
        )

        if not errors:
            return (
                "Material creation failed validation."
            )

        return (
            "Material creation failed validation:\n\n"
            + "\n".join(
                f"- {error}"
                for error in errors
            )
        )

    if status == "CREATED":
        payload = result.get(
            "material",
            {},
        )

        if isinstance(payload, dict):
            material = payload.get(
                "material",
                payload,
            )
        else:
            material = {}

        return (
            "Material created successfully.\n\n"
            f"Material Number: {material.get('material')}\n"
            f"Description: {material.get('material_description')}\n"
            f"Material Type: {material.get('material_type')}\n"
            f"Plant: {material.get('plant')}\n"
            f"Procurement Type: {material.get('procurement_type')}\n"
            f"MRP Type: {material.get('mrp_type')}\n"
            f"Valuation Class: {material.get('valuation_class')}\n"
            f"Price Control: {material.get('price_control')}\n"
            f"Standard Price: {material.get('standard_price')}\n"
            f"Currency: {material.get('currency')}"
        )

    return (
        "The material creation request completed, "
        "but the returned status was not recognized."
    )