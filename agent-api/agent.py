import asyncio
import json
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List

from transformers import pipeline

try:
    import torch
except ImportError:  # pragma: no cover - torch should always be present
    torch = None

from prompts import SYSTEM_PROMPT
from mcp_client import (
    search_material_master,
    explain_material_master,
    create_material_master,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Similarity controls.
# Tune these later as you test with real material data.
SIMILARITY_THRESHOLD = 0.68
MAX_SIMILAR_RESULTS = 3

# Conversation workflow stages.
STAGE_SIMILARITY_CONFIRMATION = "similarity_confirmation"
STAGE_CREATION_CONFIRMATION = "creation_confirmation"

# How long an abandoned material-creation flow is kept before it's
# swept away. Prevents pending_creations from growing forever on a
# long-running server.
PENDING_CREATION_TTL_SECONDS = 30 * 60


# Build the pipeline with an explicit device/dtype instead of relying
# on defaults, which silently fall back to fp32 on CPU and are much
# slower than necessary when a GPU is available.
_pipeline_kwargs: Dict[str, Any] = {"model": MODEL_NAME}

if torch is not None and torch.cuda.is_available():
    _pipeline_kwargs["device_map"] = "auto"
    _pipeline_kwargs["torch_dtype"] = torch.bfloat16

generator = pipeline(
    "text-generation",
    **_pipeline_kwargs,
)


async def generate_async(prompt: str, **kwargs) -> str:
    """
    Run the (blocking, synchronous) text-generation pipeline off the
    event loop so a single slow generation doesn't stall every other
    concurrent request being served by this process.
    """

    result = await asyncio.to_thread(
        generator,
        prompt,
        **kwargs,
    )

    generated = result[0]["generated_text"]
    return generated[len(prompt):].strip()


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


YES_RESPONSES = {
    "yes",
    "y",
    "yeah",
    "yep",
    "correct",
    "confirm",
    "that's it",
    "thats it",
    "that is it",
    "use it",
    "use existing",
    "use existing material",
}


NO_RESPONSES = {
    "no",
    "n",
    "nope",
    "not it",
    "that's not it",
    "thats not it",
    "continue",
    "continue creation",
    "create new",
    "create new material",
}


CANCEL_RESPONSES = {
    "cancel",
    "stop",
    "abort",
}


# ============================================================
# SCOPE / SAFETY FAST PATHS
#
# These run in plain Python, before any model call. They exist to
# (a) avoid spending an LLM call on obviously in/out-of-scope
# messages and (b) act as a deterministic backstop so a small model's
# occasional misclassification can't let an off-topic answer or a
# fabricated "I searched the web" claim reach the user.
# ============================================================

# Keywords that are unambiguously outside SAP MM Material Master.
# This is intentionally conservative: it only short-circuits requests
# that are *obviously* off-topic. Anything ambiguous still goes to
# the LLM planner, which has its own scope instructions.
OBVIOUS_OFF_TOPIC_TERMS = {
    "weather", "forecast", "temperature outside",
    "election", "president", "prime minister",
    "stock price", "crypto", "bitcoin",
    "recipe", "cook", "restaurant",
    "joke", "riddle", "poem", "song lyrics",
    "sports score", "football score", "basketball score",
    "movie", "tv show", "celebrity",
    "write me code", "write python", "write javascript",
}

# SAP MM material numbers in this system look like SYN-FG-000001.
# When a message is *just* a lookup of one of these, we can route
# straight to get_material without spending a planner call.
MATERIAL_NUMBER_PATTERN = re.compile(r"\b[A-Z]{2,4}-[A-Z]{2,4}-\d{4,8}\b")

# Phrases that indicate the model has claimed to browse the internet
# or cite an external source it does not have access to. Prompt-level
# instructions ("do not claim to browse the internet") are a soft
# constraint a small model can still violate, so this is a hard
# post-generation filter applied on top of them.
BROWSING_CLAIM_PATTERN = re.compile(
    r"(searched (the )?(internet|web|google)"
    r"|according to (google|the internet|wikipedia)"
    r"|i (found|looked)(\s+\w+){0,2}\s+online"
    r"|browsing the (web|internet)"
    r"|as an ai(,| )i (can|will) search)",
    re.IGNORECASE,
)


def is_obviously_off_topic(message: str) -> bool:
    normalized = normalize_text(message)

    if "material" in normalized or "sap" in normalized:
        # Mentions the domain directly — let the LLM planner decide,
        # rather than risk a false-positive reject here.
        return False

    return any(term in normalized for term in OBVIOUS_OFF_TOPIC_TERMS)


def extract_material_number(message: str):
    match = MATERIAL_NUMBER_PATTERN.search(message.upper())
    return match.group(0) if match else None


def extract_json_object(text: str) -> dict:
    """
    Parse the first balanced JSON object in `text`.

    Using json.JSONDecoder().raw_decode instead of a regex means
    nested objects/arrays in the model's output don't silently
    truncate the match the way `re.search(r"\\{.*?\\}")` can.
    """

    decoder = json.JSONDecoder()
    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object found in model output")

    obj, _ = decoder.raw_decode(text, start)

    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON value was not an object")

    return obj


def strip_browsing_claims(text: str) -> str:
    """
    Hard backstop: if the model's free-text answer claims to have
    browsed the internet or cited an external source, replace it
    with an explicit, honest statement of what this assistant can
    actually do, instead of forwarding the claim to the user.
    """

    if BROWSING_CLAIM_PATTERN.search(text):
        return (
            "I can only answer using SAP MM Material Master data and "
            "definitions available to me directly — I don't browse "
            "the internet or cite outside sources."
        )

    return text


def sweep_expired_pending_creations(pending_creations: dict) -> None:
    """
    Remove abandoned material-creation flows so pending_creations
    doesn't grow without bound on a long-running server.
    """

    now = time.time()

    expired = [
        session_id
        for session_id, entry in pending_creations.items()
        if now - entry.get("_created_at", now) > PENDING_CREATION_TTL_SECONDS
    ]

    for session_id in expired:
        del pending_creations[session_id]


# A small, fixed glossary of SAP MM code meanings. These are stable,
# SAP-defined enumerations, not org-specific data, so they're looked
# up deterministically instead of asking the model to recall them
# from parametric memory. Extend this with the codes your org
# actually uses; the LLM is only a fallback for phrasing questions
# this glossary doesn't cover.
SAP_CODE_GLOSSARY = {
    "FERT": "Finished product — manufactured in-house and ready for sale.",
    "HALB": "Semi-finished product — partially processed, used in further production.",
    "ROH": "Raw material — procured externally and consumed in production.",
    "HAWA": "Trading good — purchased and resold without further processing.",
    "DIEN": "Service — a non-physical material type used for service procurement.",
    "VERP": "Packaging material.",
    "PD": "MRP type: MRP-controlled planning (demand-driven).",
    "VB": "MRP type: Reorder point planning.",
    "ND": "MRP type: No planning — not relevant for MRP.",
    "F": "Procurement type: External procurement (purchased).",
    "E": "Procurement type: In-house production.",
    "X": "Procurement type: Both external and in-house allowed.",
    "S": "Price control: Standard price — valuated at a fixed price.",
    "V": "Price control: Moving average price — valuated at a recalculated average.",
}


# ============================================================
# TEXT / RESPONSE HELPERS
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Normalize text for matching user responses and
    material descriptions.
    """
    if value is None:
        return ""

    text = str(value).lower().strip()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def is_yes(message: str) -> bool:
    normalized = normalize_text(message)

    return (
        normalized in YES_RESPONSES
        or normalized.startswith("yes ")
    )


def is_no(message: str) -> bool:
    normalized = normalize_text(message)

    return (
        normalized in NO_RESPONSES
        or normalized.startswith("no ")
    )


def is_cancel(message: str) -> bool:
    normalized = normalize_text(message)

    return normalized in CANCEL_RESPONSES


# ============================================================
# MATERIAL FIELD HELPERS
# ============================================================

def get_material_value(
    material: Dict[str, Any],
    *keys: str,
) -> Any:
    """
    Read a material field while supporting different naming
    styles returned by the MCP server or CSV.
    """

    for key in keys:
        if (
            key in material
            and material[key] not in (None, "")
        ):
            return material[key]

    return None


# ============================================================
# SIMILARITY ENGINE
# ============================================================

def description_similarity(
    requested: str,
    existing: str,
) -> float:
    """
    Return a description similarity score from 0.0 to 1.0.
    """

    requested_norm = normalize_text(requested)
    existing_norm = normalize_text(existing)

    if not requested_norm or not existing_norm:
        return 0.0

    # General text sequence similarity.
    sequence_score = SequenceMatcher(
        None,
        requested_norm,
        existing_norm,
    ).ratio()

    # Token-based word overlap.
    requested_tokens = set(
        requested_norm.split()
    )

    existing_tokens = set(
        existing_norm.split()
    )

    if requested_tokens and existing_tokens:

        intersection = len(
            requested_tokens
            & existing_tokens
        )

        union = len(
            requested_tokens
            | existing_tokens
        )

        token_score = (
            intersection / union
            if union
            else 0.0
        )

    else:
        token_score = 0.0

    # Description matching is based on both word overlap
    # and general text similarity.
    score = (
        sequence_score * 0.55
        + token_score * 0.45
    )

    return min(
        score,
        1.0,
    )


def score_material_similarity(
    proposal: Dict[str, Any],
    material: Dict[str, Any],
) -> float:
    """
    Score a REAL existing material against a proposed
    new material.

    The LLM does NOT determine this score.
    """

    existing_description = get_material_value(
        material,
        "material_description",
        "MaterialDescription",
        "description",
        "Description",
    )

    base_description_score = description_similarity(
        proposal.get(
            "material_description",
            "",
        ),
        str(
            existing_description
            or ""
        ),
    )

    # Description is the primary duplicate indicator.
    score = (
        base_description_score
        * 0.82
    )

    # Attribute matches boost confidence.
    comparisons = [
        (
            proposal.get(
                "material_type"
            ),
            get_material_value(
                material,
                "material_type",
                "MaterialType",
            ),
            0.07,
        ),
        (
            proposal.get(
                "material_group"
            ),
            get_material_value(
                material,
                "material_group",
                "MaterialGroup",
            ),
            0.05,
        ),
        (
            proposal.get(
                "base_unit"
            ),
            get_material_value(
                material,
                "base_unit",
                "BaseUnit",
            ),
            0.03,
        ),
        (
            proposal.get(
                "plant"
            ),
            get_material_value(
                material,
                "plant",
                "Plant",
            ),
            0.03,
        ),
    ]

    for (
        proposed_value,
        existing_value,
        boost,
    ) in comparisons:

        if (
            proposed_value is not None
            and existing_value is not None
            and normalize_text(
                proposed_value
            )
            == normalize_text(
                existing_value
            )
        ):
            score += boost

    return min(
        score,
        1.0,
    )


def normalize_material_for_display(
    material: Dict[str, Any],
    similarity_score: float,
) -> Dict[str, Any]:
    """
    Normalize a real MCP material result into a predictable
    structure for display.
    """

    return {
        "material_number": get_material_value(
            material,
            "material",
            "Material",
            "material_number",
            "MaterialNumber",
        ),

        "material_description": get_material_value(
            material,
            "material_description",
            "MaterialDescription",
            "description",
            "Description",
        ),

        "material_type": get_material_value(
            material,
            "material_type",
            "MaterialType",
        ),

        "material_group": get_material_value(
            material,
            "material_group",
            "MaterialGroup",
        ),

        "base_unit": get_material_value(
            material,
            "base_unit",
            "BaseUnit",
        ),

        "plant": get_material_value(
            material,
            "plant",
            "Plant",
        ),

        "procurement_type": get_material_value(
            material,
            "procurement_type",
            "ProcurementType",
        ),

        "mrp_type": get_material_value(
            material,
            "mrp_type",
            "MRPType",
        ),

        "valuation_class": get_material_value(
            material,
            "valuation_class",
            "ValuationClass",
        ),

        "price_control": get_material_value(
            material,
            "price_control",
            "PriceControl",
        ),

        "standard_price": get_material_value(
            material,
            "standard_price",
            "StandardPrice",
        ),

        "currency": get_material_value(
            material,
            "currency",
            "Currency",
        ),

        "similarity_score": round(
            similarity_score,
            3,
        ),
    }


async def find_similar_materials(
    proposal: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Mandatory duplicate check.

    This function searches REAL material records and performs
    similarity scoring in Python.

    Material creation cannot bypass this function.
    """

    print(
        "ACTION: Mandatory duplicate/similarity check"
    )

    try:

        # Search materials from the requested plant.
        #
        # We intentionally do not supply q here because an
        # exact text search could miss similar wording.
        candidates = await search_material_master(
            q=None,
            plant=proposal.get(
                "plant"
            ),
            material_type=None,
            material_group=None,
        )

    except Exception as exc:

        print(
            "ERROR: Similarity search failed:",
            exc,
        )

        raise

    if not candidates:
        return []

    scored_matches: List[
        Dict[str, Any]
    ] = []

    for material in candidates:

        if not isinstance(
            material,
            dict,
        ):
            continue

        score = score_material_similarity(
            proposal,
            material,
        )

        if (
            score
            >= SIMILARITY_THRESHOLD
        ):

            scored_matches.append(
                normalize_material_for_display(
                    material,
                    score,
                )
            )

    scored_matches.sort(
        key=lambda item: item[
            "similarity_score"
        ],
        reverse=True,
    )

    return scored_matches[
        :MAX_SIMILAR_RESULTS
    ]


# ============================================================
# MATERIAL CREATION DISPLAY
# ============================================================

def format_creation_preview(
    proposal: Dict[str, Any],
) -> str:
    """
    Show the proposed material before final creation.
    """

    return (
        "I'm ready to create this material:\n\n"
        f"Description: "
        f"{proposal['material_description']}\n"
        f"Material Type: "
        f"{proposal['material_type']}\n"
        f"Material Group: "
        f"{proposal['material_group']}\n"
        f"Base Unit: "
        f"{proposal['base_unit']}\n"
        f"Plant: "
        f"{proposal['plant']}\n"
        f"Standard Price: "
        f"{proposal.get('standard_price')}\n"
        f"Currency: "
        f"{proposal.get('currency')}\n\n"
        "Reply 'yes' to confirm creation or "
        "'cancel' to stop."
    )


def format_similar_material_prompt(
    matches: List[Dict[str, Any]],
) -> str:
    """
    Display the closest existing material before allowing
    creation to continue.
    """

    best_match = matches[0]

    percent = round(
        best_match[
            "similarity_score"
        ]
        * 100
    )

    response = (
        "I found an existing material that looks similar "
        "to what you're trying to create.\n\n"

        f"Material Number: "
        f"{best_match.get('material_number')}\n"

        f"Description: "
        f"{best_match.get('material_description')}\n"

        f"Material Type: "
        f"{best_match.get('material_type')}\n"

        f"Material Group: "
        f"{best_match.get('material_group')}\n"

        f"Base Unit: "
        f"{best_match.get('base_unit')}\n"

        f"Plant: "
        f"{best_match.get('plant')}\n"

        f"Procurement Type: "
        f"{best_match.get('procurement_type')}\n"

        f"MRP Type: "
        f"{best_match.get('mrp_type')}\n"

        f"Valuation Class: "
        f"{best_match.get('valuation_class')}\n"

        f"Price Control: "
        f"{best_match.get('price_control')}\n"

        f"Standard Price: "
        f"{best_match.get('standard_price')}\n"

        f"Currency: "
        f"{best_match.get('currency')}\n"

        f"Similarity: "
        f"{percent}%\n\n"

        "This looks similar to the material you're requesting. "
        "Is this what you're looking for?\n\n"

        "Reply 'yes' to use the existing material or "
        "'no' to continue creating a new material."
    )

    if len(matches) > 1:

        response += (
            f"\n\nI found "
            f"{len(matches) - 1} "
            "additional potentially similar material"
        )

        if len(matches) > 2:
            response += "s"

        response += "."

    return response


def format_existing_material_selected(
    material: Dict[str, Any],
) -> str:
    """
    Response when the user confirms that the existing
    material is the one they need.
    """

    return (
        "Understood. I will not create a new material.\n\n"
        "The existing material is:\n\n"

        f"Material Number: "
        f"{material.get('material_number')}\n"

        f"Description: "
        f"{material.get('material_description')}\n"

        f"Material Type: "
        f"{material.get('material_type')}\n"

        f"Material Group: "
        f"{material.get('material_group')}\n"

        f"Base Unit: "
        f"{material.get('base_unit')}\n"

        f"Plant: "
        f"{material.get('plant')}"
    )


# ============================================================
# MAIN AGENT
# ============================================================

async def run_agent(
    message: str,
    session_id: str,
    pending_creations: dict,
) -> str:

    # Drop any abandoned material-creation flows before doing
    # anything else, so pending_creations doesn't grow unbounded.
    sweep_expired_pending_creations(pending_creations)

    # ========================================================
    # HANDLE EXISTING MATERIAL CREATION WORKFLOW
    # ========================================================

    normalized = normalize_text(
        message
    )

    if (
        session_id
        in pending_creations
    ):

        pending = pending_creations[
            session_id
        ]

        stage = pending.get(
            "_stage",
            STAGE_CREATION_CONFIRMATION,
        )

        # ----------------------------------------------------
        # SIMILAR MATERIAL CONFIRMATION
        # ----------------------------------------------------

        if (
            stage
            == STAGE_SIMILARITY_CONFIRMATION
        ):

            if is_cancel(message):

                del pending_creations[
                    session_id
                ]

                return (
                    "Material creation cancelled."
                )

            # YES means:
            # Existing material is what the user wants.
            if is_yes(message):

                best_match = pending.get(
                    "best_match",
                    {},
                )

                del pending_creations[
                    session_id
                ]

                return (
                    format_existing_material_selected(
                        best_match
                    )
                )

            # NO means:
            # User wants to continue creating a NEW material.
            if is_no(message):

                proposal = pending[
                    "proposal"
                ]

                # Move to separate creation confirmation.
                pending_creations[
                    session_id
                ] = {
                    "_stage":
                        STAGE_CREATION_CONFIRMATION,
                    "_created_at":
                        time.time(),
                    **proposal,
                }

                return (
                    format_creation_preview(
                        proposal
                    )
                )

            return (
                "Please reply 'yes' if the existing material "
                "is what you're looking for, 'no' to continue "
                "creating a new material, or 'cancel' to stop."
            )

        # ----------------------------------------------------
        # FINAL NEW MATERIAL CONFIRMATION
        # ----------------------------------------------------

        if (
            stage
            == STAGE_CREATION_CONFIRMATION
        ):

            if (
                is_yes(message)
                or normalized
                in {
                    "create",
                    "proceed",
                }
            ):

                print(
                    "FINAL CREATION CONFIRMATION RECEIVED"
                )

                # This stage can only be reached after:
                #
                # 1. No similar material was found
                #
                # OR
                #
                # 2. The user explicitly rejected the
                #    existing material.
                result = (
                    await create_material_master(
                        material_description=
                            pending[
                                "material_description"
                            ],

                        material_type=
                            pending[
                                "material_type"
                            ],

                        material_group=
                            pending[
                                "material_group"
                            ],

                        base_unit=
                            pending[
                                "base_unit"
                            ],

                        plant=
                            pending[
                                "plant"
                            ],

                        standard_price=
                            pending.get(
                                "standard_price"
                            ),

                        currency=
                            pending.get(
                                "currency",
                                "USD",
                            ),
                    )
                )

                del pending_creations[
                    session_id
                ]

                return format_create_result(
                    result
                )

            if (
                is_no(message)
                or is_cancel(message)
            ):

                del pending_creations[
                    session_id
                ]

                return (
                    "Material creation cancelled."
                )

            return (
                "A material is waiting for creation confirmation. "
                "Reply 'yes' to create it or 'cancel' to stop."
            )

        # ----------------------------------------------------
        # UNKNOWN WORKFLOW STATE
        # ----------------------------------------------------

        print(
            "BLOCKED UNKNOWN PENDING STAGE:",
            stage,
        )

        del pending_creations[
            session_id
        ]

        return (
            "The material workflow state was not recognized, "
            "so I did not create anything. "
            "Please start the material request again."
        )

    # ========================================================
    # DETERMINISTIC FAST PATHS (no model call)
    #
    # These handle the unambiguous cases directly in Python. This
    # both saves an LLM round trip and gives a hard guarantee — not
    # dependent on the model's classification — that obviously
    # off-topic messages never reach a generation step, and that an
    # explicit material-number lookup is never misrouted.
    # ========================================================

    if is_obviously_off_topic(message):

        print("ACTION: Rejected via deterministic off-topic filter")

        return OUT_OF_SCOPE_MESSAGE

    fast_path_material_number = extract_material_number(message)

    if fast_path_material_number and len(message.split()) <= 6:

        print(
            "ACTION: Fast-path get_material:",
            fast_path_material_number,
        )

        material = await explain_material_master(fast_path_material_number)

        return format_material(
            message=message,
            material=material,
        )

    # ========================================================
    # RESTRICTED LLM PLANNER
    # ========================================================

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


IMPORTANT FOR CREATE MATERIAL:

The backend performs a mandatory duplicate/similarity
check against REAL material master records before
creation confirmation.

You must NOT invent existing materials.

You must NOT invent similarity results.

You must NOT skip or override the backend duplicate check.

You must NOT tell the user that no duplicate exists.
The backend makes that decision.


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

Do NOT claim that a material was created unless
the backend creation function returned a successful
CREATED status.

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

    generated = await generate_async(
        planner_prompt,
        max_new_tokens=180,
        do_sample=False,
    )

    print(
        "\n===== QWEN PLANNER OUTPUT ====="
    )

    print(
        generated
    )

    print(
        "===============================\n"
    )

    # ========================================================
    # PARSE FIRST JSON OBJECT
    #
    # extract_json_object uses json.JSONDecoder().raw_decode, which
    # correctly handles a balanced (possibly nested) object, unlike
    # a non-greedy regex which truncates on the first "}" it sees.
    # ========================================================

    try:

        decision = extract_json_object(
            generated
        )

        print(
            "PLANNER DECISION:",
            decision,
        )

    except (
        json.JSONDecodeError,
        ValueError,
    ) as e:

        print(
            "ERROR: Could not parse planner JSON:",
            e,
        )

        return (
            OUT_OF_SCOPE_MESSAGE
        )

    action = decision.get(
        "action"
    )

    # ========================================================
    # PYTHON-SIDE ACTION ALLOWLIST
    # ========================================================

    if (
        action
        not in ALLOWED_ACTIONS
    ):

        print(
            "BLOCKED UNKNOWN ACTION:",
            action,
        )

        return (
            OUT_OF_SCOPE_MESSAGE
        )

    # ========================================================
    # REJECT OUT-OF-SCOPE
    # ========================================================

    if action == "reject":

        print(
            "ACTION: Rejected out-of-scope request"
        )

        return (
            OUT_OF_SCOPE_MESSAGE
        )

    # ========================================================
    # SEARCH MATERIAL
    # ========================================================

    if (
        action
        == "search_material"
    ):

        print(
            "ACTION: Calling MCP search_material_master"
        )

        materials = (
            await search_material_master(
                q=decision.get(
                    "q"
                ),

                plant=decision.get(
                    "plant"
                ),

                material_type=
                    decision.get(
                        "material_type"
                    ),

                material_group=
                    decision.get(
                        "material_group"
                    ),
            )
        )

        return format_search_results(
            message=message,
            materials=materials,
        )

    # ========================================================
    # GET SPECIFIC MATERIAL
    # ========================================================

    if (
        action
        == "get_material"
    ):

        material_number = (
            decision.get(
                "material_number"
            )
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

        material = (
            await explain_material_master(
                material_number
            )
        )

        return format_material(
            message=message,
            material=material,
        )

    # ========================================================
    # CREATE MATERIAL
    # ========================================================

    if (
        action
        == "create_material"
    ):

        required_fields = [
            "material_description",
            "material_type",
            "material_group",
            "base_unit",
            "plant",
        ]

        missing = [
            field
            for field
            in required_fields
            if not decision.get(
                field
            )
        ]

        if missing:

            return (
                "I need more information before creating "
                "the material. Missing: "
                + ", ".join(
                    missing
                )
            )

        proposal = {

            "material_description":
                decision.get(
                    "material_description"
                ),

            "material_type":
                decision.get(
                    "material_type"
                ),

            "material_group":
                decision.get(
                    "material_group"
                ),

            "base_unit":
                decision.get(
                    "base_unit"
                ),

            "plant":
                decision.get(
                    "plant"
                ),

            "standard_price":
                decision.get(
                    "standard_price"
                ),

            "currency":
                decision.get(
                    "currency",
                    "USD",
                ),
        }

        # ----------------------------------------------------
        # MANDATORY SIMILAR-MATERIAL CHECK
        # ----------------------------------------------------

        try:

            matches = (
                await find_similar_materials(
                    proposal
                )
            )

        except Exception:

            # Fail closed.
            #
            # If the duplicate search fails,
            # creation is blocked.
            return (
                "I couldn't complete the required "
                "duplicate-material check, so I did not "
                "proceed with material creation. "
                "Please try again after the material "
                "master connection is available."
            )

        # ----------------------------------------------------
        # POSSIBLE DUPLICATE FOUND
        # ----------------------------------------------------

        if matches:

            pending_creations[
                session_id
            ] = {

                "_stage":
                    STAGE_SIMILARITY_CONFIRMATION,

                "_created_at":
                    time.time(),

                "proposal":
                    proposal,

                "matches":
                    matches,

                "best_match":
                    matches[0],
            }

            return (
                format_similar_material_prompt(
                    matches
                )
            )

        # ----------------------------------------------------
        # NO DUPLICATE FOUND
        # ----------------------------------------------------

        # Only now can we move to final creation confirmation.
        pending_creations[
            session_id
        ] = {

            "_stage":
                STAGE_CREATION_CONFIRMATION,

            "_created_at":
                time.time(),

            **proposal,
        }

        return (
            format_creation_preview(
                proposal
            )
        )

    # ========================================================
    # NEED INFORMATION
    # ========================================================

    if (
        action
        == "need_information"
    ):

        missing_fields = (
            decision.get(
                "missing_fields",
                [],
            )
        )

        if missing_fields:

            return (
                "I need more information before creating "
                "the material. Please provide: "
                + ", ".join(
                    missing_fields
                )
            )

        return (
            "I need more information before creating "
            "the material."
        )

    # ========================================================
    # GENERAL SAP MM MATERIAL MASTER ANSWER
    # ========================================================

    if (
        action
        == "answer_sap_mm"
    ):

        print(
            "ACTION: SAP MM informational answer"
        )

        return await generate_sap_mm_answer(
            message
        )

    # ========================================================
    # DEFAULT DENY
    # ========================================================

    return (
        OUT_OF_SCOPE_MESSAGE
    )


# ============================================================
# GENERAL SAP MM ANSWER
# ============================================================

async def generate_sap_mm_answer(
    message: str,
) -> str:

    # Deterministic lookup first. Most "what does X mean" questions
    # hit a fixed SAP-defined code, so answer those without any model
    # call — zero hallucination risk and effectively instant.
    normalized = normalize_text(message)

    for code, meaning in SAP_CODE_GLOSSARY.items():
        # Match the code as a whole token so "F" doesn't match inside
        # an unrelated word.
        if re.search(rf"\b{re.escape(code.lower())}\b", normalized):
            return f"{code}: {meaning}"

    # Fall back to the model only for phrasing/explanation questions
    # the glossary doesn't cover.
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

    answer = await generate_async(
        prompt,
        max_new_tokens=250,
        do_sample=False,
        repetition_penalty=1.1,
    )

    return strip_browsing_claims(answer)


# ============================================================
# FORMAT SEARCH RESULTS
# ============================================================

def format_search_results(
    message: str,
    materials,
) -> str:
    """
    Render real MCP search results as plain text.

    This is deliberately plain string formatting rather than an LLM
    call: the data is already structured and correct, so handing it
    to the model to "rewrite" only adds latency and a chance of it
    dropping, altering, or inventing a field.
    """

    if not materials:

        return (
            "I couldn't find any matching materials."
        )

    lines = [f"Found {len(materials)} matching material(s):\n"]

    for material in materials:

        if not isinstance(material, dict):
            continue

        number = get_material_value(
            material, "material", "Material", "material_number", "MaterialNumber"
        )
        description = get_material_value(
            material, "material_description", "MaterialDescription",
            "description", "Description",
        )
        material_type = get_material_value(material, "material_type", "MaterialType")
        plant = get_material_value(material, "plant", "Plant")
        price = get_material_value(material, "standard_price", "StandardPrice")
        currency = get_material_value(material, "currency", "Currency")

        lines.append(
            f"• {number} — {description}\n"
            f"  Type: {material_type} | Plant: {plant} | "
            f"Price: {price} {currency or ''}".rstrip()
        )

    return "\n".join(lines)


# ============================================================
# FORMAT SINGLE MATERIAL
# ============================================================

def format_material(
    message: str,
    material,
) -> str:
    """
    Render a single real MCP material record as plain text.

    Code meanings come, in priority order, from:
    1. The MCP server's own `code_meanings` (authoritative — it knows
       this org's actual configuration), if present.
    2. The static SAP_CODE_GLOSSARY fallback.
    3. The raw code, unexplained, if neither has it — we never let
       the model guess at a meaning it wasn't given.

    No model call: same reasoning as format_search_results.
    """

    if not material:

        return (
            "I couldn't find that material."
        )

    code_meanings = material.get("code_meanings", {}) if isinstance(material, dict) else {}

    def explain(code):
        if not code:
            return code
        if code in code_meanings:
            return f"{code} ({code_meanings[code]})"
        if code in SAP_CODE_GLOSSARY:
            return f"{code} ({SAP_CODE_GLOSSARY[code]})"
        return code

    material_type = get_material_value(material, "material_type", "MaterialType")
    procurement_type = get_material_value(material, "procurement_type", "ProcurementType")
    mrp_type = get_material_value(material, "mrp_type", "MRPType")
    price_control = get_material_value(material, "price_control", "PriceControl")

    lines = [
        f"Material Number: {get_material_value(material, 'material', 'Material', 'material_number', 'MaterialNumber')}",
        f"Description: {get_material_value(material, 'material_description', 'MaterialDescription', 'description', 'Description')}",
        f"Material Type: {explain(material_type)}",
        f"Material Group: {get_material_value(material, 'material_group', 'MaterialGroup')}",
        f"Plant: {get_material_value(material, 'plant', 'Plant')}",
        f"Procurement Type: {explain(procurement_type)}",
        f"MRP Type: {explain(mrp_type)}",
        f"Valuation Class: {get_material_value(material, 'valuation_class', 'ValuationClass')}",
        f"Price Control: {explain(price_control)}",
        f"Standard Price: {get_material_value(material, 'standard_price', 'StandardPrice')}",
        f"Currency: {get_material_value(material, 'currency', 'Currency')}",
    ]

    return "\n".join(lines)


# ============================================================
# FORMAT CREATE RESULT
# ============================================================

def format_create_result(
    result,
) -> str:

    if not result:

        return (
            "The material creation request returned "
            "no result."
        )

    status = result.get(
        "status"
    )

    # --------------------------------------------------------
    # VALIDATION FAILURE
    # --------------------------------------------------------

    if (
        status
        == "VALIDATION_FAILED"
    ):

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
                for error
                in errors
            )
        )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if (
        status
        == "CREATED"
    ):

        payload = result.get(
            "material",
            {},
        )

        if isinstance(
            payload,
            dict,
        ):

            material = payload.get(
                "material",
                payload,
            )

        else:

            material = {}

        return (
            "Material created successfully.\n\n"

            f"Material Number: "
            f"{material.get('material')}\n"

            f"Description: "
            f"{material.get('material_description')}\n"

            f"Material Type: "
            f"{material.get('material_type')}\n"

            f"Plant: "
            f"{material.get('plant')}\n"

            f"Procurement Type: "
            f"{material.get('procurement_type')}\n"

            f"MRP Type: "
            f"{material.get('mrp_type')}\n"

            f"Valuation Class: "
            f"{material.get('valuation_class')}\n"

            f"Price Control: "
            f"{material.get('price_control')}\n"

            f"Standard Price: "
            f"{material.get('standard_price')}\n"

            f"Currency: "
            f"{material.get('currency')}"
        )

    # --------------------------------------------------------
    # UNKNOWN STATUS
    # --------------------------------------------------------

    return (
        "The material creation request completed, "
        "but the returned status was not recognized."
    )