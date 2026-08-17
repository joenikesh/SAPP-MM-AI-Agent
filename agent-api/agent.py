import asyncio
import json
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List

from transformers import pipeline

try:
    import torch
except ImportError:
    torch = None

from prompts import SYSTEM_PROMPT
from query_plan import validate_query_plan

from mcp_client import (
    search_material_master,
    explain_material_master,
    create_material_master,
    query_material_master,
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


_pipeline_kwargs: Dict[str, Any] = {
    "model": MODEL_NAME,
}

# NOTE: torch.cuda.is_available() is always False on macOS -- there is
# no CUDA on Mac. Without an explicit Apple Silicon (MPS) check, this
# was silently falling back to plain CPU inference on Mac even though
# an MPS-capable GPU was available, which is a large chunk of why
# generation was slow enough to hit the frontend's response timeout.
if torch is not None and torch.cuda.is_available():
    _pipeline_kwargs["device_map"] = "auto"
    _pipeline_kwargs["torch_dtype"] = torch.bfloat16
elif (
    torch is not None
    and getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
):
    _pipeline_kwargs["device"] = "mps"
    _pipeline_kwargs["torch_dtype"] = torch.float16


generator = pipeline(
    "text-generation",
    **_pipeline_kwargs,
)


async def generate_async(
    prompt: str,
    **kwargs,
) -> str:
    """
    Run Hugging Face generation outside the FastAPI event loop.
    """

    result = await asyncio.to_thread(
        generator,
        prompt,
        **kwargs,
    )

    generated = result[0]["generated_text"]

    if generated.startswith(prompt):
        generated = generated[len(prompt):]

    return generated.strip()


# ============================================================
# AGENT CONFIGURATION
# ============================================================

ALLOWED_ACTIONS = {
    "search_material",
    "query_materials",
    "get_material",
    "create_material",
    "need_information",
    "answer_sap_mm",
    "reject",
}


OUT_OF_SCOPE_MESSAGE = (
    "I can only assist with SAP MM Material Master tasks, "
    "including material search, material details, SAP MM field "
    "explanations, validation, and material creation."
)


# Similar-material controls.
# Lowered from 0.68 alongside the description-similarity rebalance
# above so that short/generic requests (e.g. "Pump") can actually
# cross the bar against verbose real descriptions that contain them.
SIMILARITY_THRESHOLD = 0.62
MAX_SIMILAR_RESULTS = 5


# Material creation workflow stages.
STAGE_SIMILARITY_CONFIRMATION = "similarity_confirmation"
STAGE_CREATION_CONFIRMATION = "creation_confirmation"


# Abandoned creation flows expire after 30 minutes.
PENDING_CREATION_TTL_SECONDS = 30 * 60


# ============================================================
# USER CONFIRMATION RESPONSES
# ============================================================

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
# HARD SCOPE FILTERS
# ============================================================

OBVIOUS_OFF_TOPIC_TERMS = {
    "weather",
    "forecast",
    "temperature outside",
    "election",
    "president",
    "prime minister",
    "stock price",
    "crypto",
    "bitcoin",
    "recipe",
    "restaurant",
    "joke",
    "riddle",
    "poem",
    "song lyrics",
    "sports score",
    "football score",
    "basketball score",
    "movie",
    "tv show",
    "celebrity",
    "write me code",
    "write python",
    "write javascript",
}


MATERIAL_NUMBER_PATTERN = re.compile(
    r"\b[A-Z]{2,4}-[A-Z]{2,5}-\d{4,8}\b"
)


BROWSING_CLAIM_PATTERN = re.compile(
    r"(searched (the )?(internet|web|google)"
    r"|according to (google|the internet|wikipedia)"
    r"|i (found|looked)(\s+\w+){0,2}\s+online"
    r"|browsing the (web|internet)"
    r"|as an ai(,| )i (can|will) search)",
    re.IGNORECASE,
)


# ============================================================
# SAP CODE GLOSSARY
# ============================================================

SAP_CODE_GLOSSARY = {
    "FERT": (
        "Finished product — manufactured in-house and "
        "ready for sale."
    ),

    "HALB": (
        "Semi-finished product — partially processed and "
        "used in further production."
    ),

    "ROH": (
        "Raw material — normally procured externally and "
        "consumed in production."
    ),

    "HAWA": (
        "Trading good — purchased and resold without "
        "further processing."
    ),

    "DIEN": (
        "Service — a non-physical material type used "
        "for service procurement."
    ),

    "VERP": "Packaging material.",

    "PD": (
        "MRP type: MRP-controlled planning."
    ),

    "VB": (
        "MRP type: Reorder point planning."
    ),

    "ND": (
        "MRP type: No planning."
    ),

    "F": (
        "Procurement type: External procurement."
    ),

    "E": (
        "Procurement type: In-house production."
    ),

    "X": (
        "Procurement type: Both external and in-house "
        "procurement allowed."
    ),

    "S": (
        "Price control: Standard price."
    ),

    "V": (
        "Price control: Moving average price."
    ),
}


# ============================================================
# GROUNDING CHECK FOR create_material FIELDS
# ============================================================
#
# The planner LLM is told never to invent material_type,
# material_group, or base_unit. Small models don't reliably follow
# that instruction on their own (e.g. guessing material_type "FERT"
# for "create a pump in plant 1000" purely from domain priors, with
# nothing in the message actually saying so). This section is a
# deterministic backstop: before a create_material decision is
# trusted, each of these fields must be traceable to something the
# user actually typed, or it's dropped and treated as missing.

MATERIAL_TYPE_KEYWORDS: Dict[str, set[str]] = {
    "FERT": {
        "finished",
        "finished good",
        "finished goods",
        "finished product",
        "finished material",
    },
    "HALB": {
        "semi finished",
        "semifinished",
        "half finished",
    },
    "ROH": {
        "raw material",
        "raw materials",
    },
    "HAWA": {
        "trading good",
        "trading goods",
        "trading material",
        "resale",
        "resell",
    },
    "DIEN": {
        "service",
        "services",
    },
    "VERP": {
        "packaging",
        "packaging material",
    },
}


def message_supports_material_type(
    message: str,
    type_code: Any,
) -> bool:
    """
    True only if the material type code itself, or one of its
    recognized synonym phrases, actually appears in the user's
    message -- never true purely because the model inferred it.
    """

    normalized_message = normalize_text(message)
    code = str(type_code or "").strip().upper()

    if not code:
        return False

    if re.search(
        rf"\b{re.escape(code.lower())}\b",
        normalized_message,
    ):
        return True

    for keyword in MATERIAL_TYPE_KEYWORDS.get(code, ()):
        if normalize_text(keyword) in normalized_message:
            return True

    return False


def message_supports_literal_value(
    message: str,
    value: Any,
) -> bool:
    """
    True only if the exact value appears as a whole word/phrase in
    the user's message. Used for fields with no fixed vocabulary
    to map synonyms from (material_group, base_unit).
    """

    if value is None:
        return False

    normalized_message = normalize_text(message)
    normalized_value = normalize_text(value)

    if not normalized_value:
        return False

    return bool(
        re.search(
            rf"\b{re.escape(normalized_value)}\b",
            normalized_message,
        )
    )


def strip_ungrounded_create_fields(
    decision: Dict[str, Any],
    message: str,
) -> Dict[str, Any]:
    """
    Drop any material_type / material_group / base_unit the planner
    returned but that isn't actually grounded in the user's message,
    so the missing-field flow triggers instead of silently creating
    a material with guessed attributes.

    material_description and plant are intentionally excluded here:
    description is expected to be the user's own free text (not a
    fixed-vocabulary code), and plant is already validated numerically
    by normalize_plant.
    """

    cleaned = dict(decision)

    material_type = cleaned.get("material_type")
    if material_type and not message_supports_material_type(
        message,
        material_type,
    ):
        print("BLOCKED UNGROUNDED material_type GUESS:", material_type)
        cleaned["material_type"] = None

    material_group = cleaned.get("material_group")
    if material_group and not message_supports_literal_value(
        message,
        material_group,
    ):
        print("BLOCKED UNGROUNDED material_group GUESS:", material_group)
        cleaned["material_group"] = None

    base_unit = cleaned.get("base_unit")
    if base_unit and not message_supports_literal_value(
        message,
        base_unit,
    ):
        print("BLOCKED UNGROUNDED base_unit GUESS:", base_unit)
        cleaned["base_unit"] = None

    return cleaned


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(
    value: Any,
) -> str:
    """
    Normalize text for comparisons.
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
    )

    return text.strip()


def is_yes(
    message: str,
) -> bool:

    normalized = normalize_text(message)

    return (
        normalized in YES_RESPONSES
        or normalized.startswith("yes ")
    )


def is_no(
    message: str,
) -> bool:

    normalized = normalize_text(message)

    return (
        normalized in NO_RESPONSES
        or normalized.startswith("no ")
    )


def is_cancel(
    message: str,
) -> bool:

    normalized = normalize_text(message)

    return normalized in CANCEL_RESPONSES


# ============================================================
# SEARCH PARAMETER SANITIZATION
# ============================================================

def clean_optional_value(
    value: Any,
    invalid_values: set[str],
) -> str | None:
    """
    Remove empty and hallucinated placeholder values produced
    by the small planner model.
    """

    if value is None:
        return None

    cleaned = str(value).strip()

    if not cleaned:
        return None

    invalid_normalized = {
        normalize_text(item)
        for item in invalid_values
    }

    if normalize_text(cleaned) in invalid_normalized:
        return None

    return cleaned


def singularize_search_word(
    word: str,
) -> str:
    """
    Very conservative search singularization.

    This solves searches such as:
        pumps   -> pump
        valves  -> valve
        sensors -> sensor

    We intentionally avoid trying to implement a complete
    English stemming algorithm.
    """

    lower = word.lower()

    # ies -> y
    # batteries -> battery
    if (
        len(word) > 4
        and lower.endswith("ies")
    ):
        return word[:-3] + "y"

    # ses / xes / zes / ches / shes
    # boxes -> box
    if (
        lower.endswith("ses")
        or lower.endswith("xes")
        or lower.endswith("zes")
        or lower.endswith("ches")
        or lower.endswith("shes")
    ):
        return word[:-2]

    # ordinary plural
    if (
        len(word) > 3
        and lower.endswith("s")
        and not lower.endswith("ss")
    ):
        return word[:-1]

    return word


def normalize_search_query(
    value: Any,
) -> str | None:
    """
    Convert planner q output into a safe database search term.

    Examples:

    pumps
        -> pump

    Material Description
        -> None

    database
        -> None
    """

    q = clean_optional_value(
        value,
        {
            "material",
            "materials",
            "material description",
            "description",
            "database",
            "the database",
            "material database",
            "material master",
            "material master database",
            "all materials",
            "all material",
            "search term",
            "search query",
            "query",
            "none",
            "null",
            "n/a",
        },
    )

    if not q:
        return None

    words = q.split()

    # Only singularize single-word searches.
    if len(words) == 1:
        q = singularize_search_word(q)

    return q


def normalize_plant(
    value: Any,
) -> str | None:
    """
    Validate plant codes generated by the planner.

    Prototype plant values are numeric SAP plant codes.
    """

    plant = clean_optional_value(
        value,
        {
            "plant",
            "plant number",
            "plant code",
            "plant id",
            "number",
            "none",
            "null",
            "n/a",
        },
    )

    if not plant:
        return None

    # Don't allow Qwen text such as "Plant Number"
    # to reach the database.
    if not plant.isdigit():
        print(
            "BLOCKED INVALID PLANT VALUE:",
            plant,
        )

        return None

    return plant


def normalize_material_type(
    value: Any,
) -> str | None:

    material_type = clean_optional_value(
        value,
        {
            "material type",
            "type",
            "none",
            "null",
            "n/a",
        },
    )

    if not material_type:
        return None

    return material_type.upper()


def normalize_material_group(
    value: Any,
) -> str | None:

    material_group = clean_optional_value(
        value,
        {
            "material group",
            "group",
            "none",
            "null",
            "n/a",
        },
    )

    if not material_group:
        return None

    return material_group.upper()


def normalize_search_decision(
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Sanitize every parameter before an MCP search.

    The LLM decides intent.

    Python decides what values are safe enough to send
    to the database.
    """

    normalized = {
        "q": normalize_search_query(
            decision.get("q")
        ),

        "plant": normalize_plant(
            decision.get("plant")
        ),

        "material_type": normalize_material_type(
            decision.get("material_type")
        ),

        "material_group": normalize_material_group(
            decision.get("material_group")
        ),
    }

    print(
        "NORMALIZED SEARCH:",
        normalized,
    )

    return normalized


# ============================================================
# BASIC SAFETY HELPERS
# ============================================================

def is_obviously_off_topic(
    message: str,
) -> bool:

    normalized = normalize_text(message)

    if (
        "material" in normalized
        or "sap" in normalized
    ):
        return False

    return any(
        term in normalized
        for term in OBVIOUS_OFF_TOPIC_TERMS
    )


def extract_material_number(
    message: str,
):

    match = MATERIAL_NUMBER_PATTERN.search(
        message.upper()
    )

    return (
        match.group(0)
        if match
        else None
    )


def extract_json_object(
    text: str,
) -> dict:
    """
    Read the first valid JSON object returned by Qwen.
    """

    decoder = json.JSONDecoder()

    start = text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found in planner output."
        )

    obj, _ = decoder.raw_decode(
        text,
        start,
    )

    if not isinstance(
        obj,
        dict,
    ):
        raise ValueError(
            "Planner output was not a JSON object."
        )

    return obj


def strip_browsing_claims(
    text: str,
) -> str:

    if BROWSING_CLAIM_PATTERN.search(
        text
    ):

        return (
            "I can only answer using SAP MM Material Master "
            "data and definitions available to me directly. "
            "I don't browse the internet or use external sources."
        )

    return text


def sweep_expired_pending_creations(
    pending_creations: dict,
) -> None:

    now = time.time()

    expired = [
        session_id
        for session_id, entry
        in pending_creations.items()
        if (
            now
            - entry.get(
                "_created_at",
                now,
            )
            > PENDING_CREATION_TTL_SECONDS
        )
    ]

    for session_id in expired:

        del pending_creations[
            session_id
        ]


def extract_excluded_currency(message: str) -> str | None:
    """Recognize a high-confidence 'currency other than X' comparison."""
    normalized = normalize_text(message)
    if "currency" not in normalized or not any(
        word in normalized.split() for word in {"other", "different", "non"}
    ):
        return None
    match = re.search(r"\b(?:than|from)\s+([a-z]{3})\b", normalized)
    if match:
        return match.group(1).upper()
    codes = re.findall(r"\b[A-Z]{3}\b", message)
    return codes[-1] if codes else None


def is_previous_search_follow_up(message: str) -> bool:
    normalized = normalize_text(message)
    phrases = {
        "searched before", "previous search", "last search",
        "previous material", "last material", "earlier material",
    }
    return any(phrase in normalized for phrase in phrases)


def remember_query_context(
    session_contexts: dict,
    session_id: str,
    request: str,
    reply: str,
    found_results: bool,
) -> None:
    context = session_contexts.setdefault(session_id, {})
    context["last_query"] = {"request": request, "reply": reply}
    if found_results:
        context["last_successful_query"] = {"request": request, "reply": reply}


# Phrases that signal "continue from what we were just discussing"
# rather than "start a brand new, unrelated search."
CONTINUITY_CUES = {
    "relevant", "similar", "those", "these", "them", "related",
    "same", "matching", "match", "ones", "one",
}


def message_has_continuity_cue(message: str) -> bool:
    normalized = set(normalize_text(message).split())
    return bool(normalized & CONTINUITY_CUES)


def remember_search_entities(
    session_contexts: dict,
    session_id: str,
    entities: Dict[str, Any],
) -> None:
    """
    Slot-filling working memory for the last search's entities
    (q/search_text, plant, material_type, material_group).

    Only overwrites a slot when the new turn actually supplies a
    value, so a follow-up turn that omits a field (e.g. restates
    only the plant) doesn't erase what was learned earlier in the
    session.
    """

    context = session_contexts.setdefault(session_id, {})
    stored = context.setdefault("last_search_entities", {})

    for key, value in entities.items():
        if value:
            stored[key] = value


def get_last_search_entities(
    session_contexts: dict,
    session_id: str,
) -> Dict[str, Any]:
    return session_contexts.get(session_id, {}).get(
        "last_search_entities",
        {},
    )


# ============================================================
# MATERIAL VALUE HELPERS
# ============================================================

def unwrap_material(
    material: Any,
) -> Any:
    """
    explain_material_master may return:

    {
        "material": {...},
        "code_meanings": {...}
    }

    This helper returns the inner material when necessary.
    """

    if not isinstance(
        material,
        dict,
    ):
        return material

    inner = material.get(
        "material"
    )

    if (
        isinstance(inner, dict)
        and (
            "material" in inner
            or "material_description" in inner
        )
    ):
        return inner

    return material


def get_material_value(
    material: Dict[str, Any],
    *keys: str,
) -> Any:

    if not isinstance(
        material,
        dict,
    ):
        return None

    for key in keys:

        if (
            key in material
            and material[key]
            not in (
                None,
                "",
            )
        ):
            return material[key]

    return None


# ============================================================
# MATERIAL SIMILARITY
# ============================================================

def description_similarity(
    requested: str,
    existing: str,
) -> float:

    requested_norm = normalize_text(
        requested
    )

    existing_norm = normalize_text(
        existing
    )

    if (
        not requested_norm
        or not existing_norm
    ):
        return 0.0

    sequence_score = SequenceMatcher(
        None,
        requested_norm,
        existing_norm,
    ).ratio()

    requested_tokens = set(
        requested_norm.split()
    )

    existing_tokens = set(
        existing_norm.split()
    )

    if (
        requested_tokens
        and existing_tokens
    ):

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

        # How much of the REQUESTED description is fully covered by
        # the existing one. Short, generic requests like "Pump" should
        # score highly against "Precision Pump Assembly 001" even
        # though SequenceMatcher/Jaccard penalize them for the length
        # mismatch. Without this, brief descriptions can never trigger
        # the duplicate check no matter how many real duplicates exist.
        containment_score = (
            intersection / len(requested_tokens)
        )

    else:

        token_score = 0.0
        containment_score = 0.0

    score = (
        sequence_score * 0.20
        + token_score * 0.30
        + containment_score * 0.50
    )

    return min(
        score,
        1.0,
    )


def score_material_similarity(
    proposal: Dict[str, Any],
    material: Dict[str, Any],
) -> float:

    existing_description = get_material_value(
        material,
        "material_description",
        "MaterialDescription",
        "description",
        "Description",
    )

    base_score = description_similarity(
        proposal.get(
            "material_description",
            "",
        ),
        str(
            existing_description
            or ""
        ),
    )

    score = (
        base_score
        * 0.82
    )

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

    material = unwrap_material(
        material
    )

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

    print(
        "ACTION: Mandatory duplicate/similarity check"
    )

    try:

        candidates = (
            await search_material_master(
                q=None,

                plant=normalize_plant(
                    proposal.get(
                        "plant"
                    )
                ),

                material_type=None,
                material_group=None,
            )
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
# FORMAT SEARCH RESULTS
# ============================================================

def format_search_results(
    message: str,
    materials,
) -> str:

    if not materials:

        return (
            "I couldn't find any matching materials."
        )

    # Some MCP responses may contain one nested list.
    if (
        isinstance(materials, list)
        and len(materials) == 1
        and isinstance(
            materials[0],
            list,
        )
    ):
        materials = materials[0]

    if isinstance(
        materials,
        dict,
    ):

        materials = [
            materials
        ]

    lines = [
        f"I found {len(materials)} matching material"
        + (
            ""
            if len(materials) == 1
            else "s"
        )
        + ":",
        "",
    ]

    valid_count = 0

    for material in materials:

        if not isinstance(
            material,
            dict,
        ):
            continue

        material = unwrap_material(
            material
        )

        number = get_material_value(
            material,
            "material",
            "Material",
            "material_number",
            "MaterialNumber",
        )

        description = get_material_value(
            material,
            "material_description",
            "MaterialDescription",
            "description",
            "Description",
        )

        material_type = get_material_value(
            material,
            "material_type",
            "MaterialType",
        )

        plant = get_material_value(
            material,
            "plant",
            "Plant",
        )

        price = get_material_value(
            material,
            "standard_price",
            "StandardPrice",
        )

        currency = get_material_value(
            material,
            "currency",
            "Currency",
        )

        valid_count += 1

        lines.append(
            f"• {number} — {description}\n"
            f"  Type: {material_type} | "
            f"Plant: {plant} | "
            f"Price: {price} {currency or ''}".rstrip()
        )

    if valid_count == 0:

        return (
            "The material search completed, but I couldn't "
            "read the returned material records."
        )

    return "\n".join(
        lines
    )


def format_query_results(result: Any) -> str:
    """Format deterministic query output without asking the LLM to reinterpret it."""
    if not isinstance(result, dict):
        return "The material query completed, but returned an unreadable response."
    if result.get("kind") == "aggregate":
        aggregate = result.get("aggregate", {})
        label = aggregate.get("function", "result")
        field = aggregate.get("field")
        rows = result.get("rows")
        if isinstance(rows, list):
            if not rows:
                return "I couldn't find any matching materials."
            group_by = aggregate.get("group_by")
            lines = [f"{label.title()} {field or 'materials'} by {group_by}:", ""]
            lines.extend(f"• {row.get(group_by)}: {row.get('value')}" for row in rows)
            return "\n".join(lines)
        return f"{label.title()} {field or 'materials'}: {result.get('value')}"
    return format_search_results("", result.get("items", []))


# ============================================================
# FORMAT ONE MATERIAL
# ============================================================

def format_material(
    message: str,
    material,
) -> str:

    if not material:

        return (
            "I couldn't find that material."
        )

    wrapper = (
        material
        if isinstance(
            material,
            dict,
        )
        else {}
    )

    code_meanings = wrapper.get(
        "code_meanings",
        {},
    )

    material = unwrap_material(
        material
    )

    if not isinstance(
        material,
        dict,
    ):

        return (
            "I couldn't read the returned material record."
        )

    def explain(
        code,
    ):

        if not code:
            return code

        # MCP may supply meanings in different structures.
        if isinstance(
            code_meanings,
            dict,
        ):

            if code in code_meanings:
                meaning = code_meanings[
                    code
                ]

                if isinstance(
                    meaning,
                    str,
                ):
                    return (
                        f"{code} ({meaning})"
                    )

        if code in SAP_CODE_GLOSSARY:

            return (
                f"{code} "
                f"({SAP_CODE_GLOSSARY[code]})"
            )

        return code

    material_type = get_material_value(
        material,
        "material_type",
        "MaterialType",
    )

    procurement_type = get_material_value(
        material,
        "procurement_type",
        "ProcurementType",
    )

    mrp_type = get_material_value(
        material,
        "mrp_type",
        "MRPType",
    )

    price_control = get_material_value(
        material,
        "price_control",
        "PriceControl",
    )

    lines = [
        (
            "Material Number: "
            + str(
                get_material_value(
                    material,
                    "material",
                    "Material",
                    "material_number",
                    "MaterialNumber",
                )
            )
        ),

        (
            "Description: "
            + str(
                get_material_value(
                    material,
                    "material_description",
                    "MaterialDescription",
                    "description",
                    "Description",
                )
            )
        ),

        (
            "Material Type: "
            + str(
                explain(
                    material_type
                )
            )
        ),

        (
            "Material Group: "
            + str(
                get_material_value(
                    material,
                    "material_group",
                    "MaterialGroup",
                )
            )
        ),

        (
            "Base Unit: "
            + str(
                get_material_value(
                    material,
                    "base_unit",
                    "BaseUnit",
                )
            )
        ),

        (
            "Plant: "
            + str(
                get_material_value(
                    material,
                    "plant",
                    "Plant",
                )
            )
        ),

        (
            "Procurement Type: "
            + str(
                explain(
                    procurement_type
                )
            )
        ),

        (
            "MRP Type: "
            + str(
                explain(
                    mrp_type
                )
            )
        ),

        (
            "Valuation Class: "
            + str(
                get_material_value(
                    material,
                    "valuation_class",
                    "ValuationClass",
                )
            )
        ),

        (
            "Price Control: "
            + str(
                explain(
                    price_control
                )
            )
        ),

        (
            "Standard Price: "
            + str(
                get_material_value(
                    material,
                    "standard_price",
                    "StandardPrice",
                )
            )
        ),

        (
            "Currency: "
            + str(
                get_material_value(
                    material,
                    "currency",
                    "Currency",
                )
            )
        ),
    ]

    return "\n".join(
        lines
    )


# ============================================================
# CREATE MATERIAL DISPLAY
# ============================================================

def format_creation_preview(
    proposal: Dict[str, Any],
) -> str:

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

    if len(
        matches
    ) > 1:

        additional = (
            len(matches)
            - 1
        )

        response += (
            f"\n\nI found {additional} additional "
            "potentially similar material"
        )

        if additional > 1:
            response += "s"

        response += "."

    return response


def format_existing_material_selected(
    material: Dict[str, Any],
) -> str:

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


def format_create_result(
    result,
) -> str:

    if not result:

        return (
            "The material creation request returned no result."
        )

    if not isinstance(
        result,
        dict,
    ):

        return (
            "The material creation request returned an "
            "unexpected response."
        )

    status = result.get(
        "status"
    )

    # MCP wrapper may contain another material API result.
    if (
        not status
        and isinstance(
            result.get("material"),
            dict,
        )
    ):

        nested = result.get(
            "material"
        )

        if "status" in nested:

            result = nested

            status = result.get(
                "status"
            )

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

    if (
        status
        == "CREATED"
    ):

        payload = result.get(
            "material",
            {},
        )

        # Material API currently returns:
        #
        # {
        #   "status": "CREATED",
        #   "material": {...}
        # }
        #
        # and MCP can wrap that again.
        if (
            isinstance(
                payload,
                dict,
            )
            and payload.get(
                "status"
            )
            == "CREATED"
        ):

            payload = payload.get(
                "material",
                {},
            )

        if (
            isinstance(
                payload,
                dict,
            )
            and isinstance(
                payload.get("material"),
                dict,
            )
        ):

            payload = payload.get(
                "material"
            )

        material = (
            payload
            if isinstance(
                payload,
                dict,
            )
            else {}
        )

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

    return (
        "The material creation request completed, "
        "but the returned status was not recognized."
    )


# ============================================================
# GENERIC SAP MM ANSWERS
# ============================================================

async def generate_sap_mm_answer(
    message: str,
) -> str:

    normalized = normalize_text(
        message
    )

    # Try deterministic glossary first.
    for (
        code,
        meaning,
    ) in SAP_CODE_GLOSSARY.items():

        if re.search(
            rf"\b{re.escape(code.lower())}\b",
            normalized,
        ):

            return (
                f"{code}: {meaning}"
            )

    prompt = f"""
{SYSTEM_PROMPT}

You are strictly restricted to SAP MM Material Master.

User question:

{message}

Rules:

- Answer only SAP MM Material Master questions.
- Do not browse the internet.
- Do not claim that you searched the internet.
- Do not invent company-specific configuration.
- If you do not know something, say so.
- Keep the answer concise and factual.

Assistant:
"""

    answer = await generate_async(
        prompt,
        max_new_tokens=250,
        do_sample=False,
        repetition_penalty=1.1,
    )

    return strip_browsing_claims(
        answer
    )


# ============================================================
# MAIN AGENT
# ============================================================

async def run_agent(
    message: str,
    session_id: str,
    pending_creations: dict,
    session_contexts: dict | None = None,
) -> str:

    if session_contexts is None:
        session_contexts = {}

    sweep_expired_pending_creations(
        pending_creations
    )

    normalized = normalize_text(
        message
    )

    # ========================================================
    # EXISTING MATERIAL CREATION WORKFLOW
    # ========================================================

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
        # SIMILAR MATERIAL FOUND
        # ----------------------------------------------------

        if (
            stage
            == STAGE_SIMILARITY_CONFIRMATION
        ):

            if is_cancel(
                message
            ):

                del pending_creations[
                    session_id
                ]

                return (
                    "Material creation cancelled."
                )

            if is_yes(
                message
            ):

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

            if is_no(
                message
            ):

                proposal = pending[
                    "proposal"
                ]

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
        # FINAL CREATION CONFIRMATION
        # ----------------------------------------------------

        if (
            stage
            == STAGE_CREATION_CONFIRMATION
        ):

            if (
                is_yes(
                    message
                )
                or normalized
                in {
                    "create",
                    "proceed",
                }
            ):

                print(
                    "FINAL CREATION CONFIRMATION RECEIVED"
                )

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

                return (
                    format_create_result(
                        result
                    )
                )

            if (
                is_no(
                    message
                )
                or is_cancel(
                    message
                )
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
    # DETERMINISTIC FAST PATHS
    # ========================================================

    if is_previous_search_follow_up(message):
        context = session_contexts.get(session_id, {})
        last_query = context.get("last_query")
        last_success = context.get("last_successful_query")
        if not last_query:
            return "There is no previous material search in this session."
        if last_success and last_success != last_query:
            return (
                f"Your most recent query was: “{last_query['request']}”\n"
                f"{last_query['reply']}\n\n"
                f"The most recent successful search was: “{last_success['request']}”\n"
                f"{last_success['reply']}"
            )
        return (
            f"Your most recent material search was: “{last_query['request']}”\n"
            f"{last_query['reply']}"
        )

    excluded_currency = extract_excluded_currency(message)
    if excluded_currency:
        query_plan = validate_query_plan({
            "filters": [{
                "field": "currency",
                "operator": "ne",
                "value": excluded_currency,
            }],
            "sort": [{"field": "currency", "direction": "asc"}],
            "limit": 20,
        })
        try:
            result = await query_material_master(query_plan)
        except Exception as exc:
            print("ERROR: MCP currency query failed:", exc)
            return "I couldn't query the material master because the material service is unavailable."
        reply = format_query_results(result)
        found = bool(result.get("items")) if isinstance(result, dict) else False
        remember_query_context(session_contexts, session_id, message, reply, found)
        return reply

    if is_obviously_off_topic(
        message
    ):

        print(
            "ACTION: Rejected via deterministic off-topic filter"
        )

        return OUT_OF_SCOPE_MESSAGE

    material_number = extract_material_number(
        message
    )

    if (
        material_number
        and len(
            message.split()
        )
        <= 8
    ):

        print(
            "ACTION: Fast-path material lookup:",
            material_number,
        )

        try:

            material = (
                await explain_material_master(
                    material_number
                )
            )

        except Exception as exc:

            print(
                "ERROR: Material lookup failed:",
                exc,
            )

            return (
                "I couldn't retrieve that material from "
                "the material master."
            )

        return (
            format_material(
                message=message,
                material=material,
            )
        )

    # ========================================================
    # QWEN PLANNER
    # ========================================================

    last_entities = get_last_search_entities(
        session_contexts,
        session_id,
    )

    if last_entities:
        recent_context_block = (
            "RECENT CONVERSATION CONTEXT:\n\n"
            "The user's previous search in this session used: "
            + json.dumps(last_entities)
            + "\n\nIf the new request refers back to that topic without "
            "repeating it explicitly (e.g. it says 'those materials', "
            "'the relevant materials', or otherwise omits a keyword "
            "shortly after discussing a specific material), reuse the "
            "prior keyword/material_type/material_group. If the new "
            "request clearly starts a different topic, ignore this "
            "context.\n"
        )
    else:
        recent_context_block = ""

    planner_prompt = f"""
You are a STRICTLY RESTRICTED SAP MM Material Master routing agent.

Your ONLY job is to convert the user's request into one JSON action.

{recent_context_block}
ALLOWED DOMAIN:

- SAP MM Material Master
- material searches
- material details
- material types
- material groups
- plants
- base units
- procurement types
- MRP types
- valuation classes
- price controls
- material creation

OUT OF SCOPE:

- weather
- news
- politics
- elections
- sports
- entertainment
- programming
- coding
- personal advice
- medical questions
- legal questions
- financial advice
- general internet research
- unrelated SAP modules

If a request is outside SAP MM Material Master:

{{
  "action": "reject"
}}


AVAILABLE ACTIONS


1. search_material

Use for:
- find materials
- list materials
- search descriptions
- search by plant
- search by type
- search by group

Possible fields:

q
plant
material_type
material_group

IMPORTANT SEARCH RULES:

The q field must contain ONLY the useful material search word.

Examples:

"find pumps"
q must be "pump", NOT "pumps".

"show valves"
q must be "valve".

"give me a material from the database"
q must be null.

"show materials"
q must be null.

Never put these in q:

"Material Description"
"Description"
"Database"
"Material"
"Materials"
"Search Query"

If the user did not specify a plant, omit plant or use null.

Never invent a plant.

Never return placeholder values such as:

"Plant Number"
"Plant Code"
"Material Description"


2. query_materials

Use for analytical or ranked requests: price/stock/weight comparisons,
cheapest/highest/top/bottom, thresholds, ranges, sorting, counts, averages,
totals, minimums, maximums, or grouping.

Return a plan with this exact shape:

{{
  "action": "query_materials",
  "query_plan": {{
    "search_text": "pump",
    "filters": [{{"field": "plant", "operator": "eq", "value": "1000"}}],
    "sort": [{{"field": "standard_price", "direction": "asc"}}],
    "limit": 1
  }}
}}

Allowed fields:
material, material_description, material_type, material_group, base_unit,
old_material_number, division, plant, storage_location, purchasing_group,
procurement_type, mrp_type, mrp_controller, lot_size, valuation_class,
price_control, currency, weight_unit, volume_unit, profit_center,
tax_classification, sales_status, created_on, batch_management,
serial_number_profile, country_of_origin, standard_price,
moving_average_price, price_unit, safety_stock, reorder_point,
planned_delivery_time_days, gr_processing_time_days, gross_weight,
net_weight, volume

Allowed operators: eq, ne, gt, gte, lt, lte, contains, in.
Allowed sort directions: asc, desc. Limit must be 1 through 100.
Allowed aggregates: count, min, max, avg, sum. Optional group_by must be a
text field. For prices, use standard_price unless the user explicitly asks
for moving average price. Add a currency filter only if the user names one;
never pretend to convert currencies.

"expensive material" means sort standard_price descending and limit 1.
"cheapest pump in plant 1000" means search_text pump, plant eq 1000,
sort standard_price ascending, limit 1.
"materials over $500" means standard_price gt 500 and currency eq USD.
"highest standard price" means sort standard_price descending and limit 1.
"currency other than USD" means currency ne USD. Do not put the words
"currency", "other", or "USD" into search_text.

Examples:

User: materials over $500
Response:
{{
  "action": "query_materials",
  "query_plan": {{
    "filters": [
      {{"field": "standard_price", "operator": "gt", "value": 500}},
      {{"field": "currency", "operator": "eq", "value": "USD"}}
    ],
    "limit": 20
  }}
}}

User: highest standard price
Response:
{{
  "action": "query_materials",
  "query_plan": {{
    "sort": [{{"field": "standard_price", "direction": "desc"}}],
    "limit": 1
  }}
}}


3. get_material

Use when the user provides one specific material number.

Example:

{{
  "action": "get_material",
  "material_number": "SYN-FG-000001"
}}


4. create_material

Required fields:

material_description
material_type
material_group
base_unit
plant

Optional fields:

standard_price
currency

Do not invent required values.

The backend performs the mandatory duplicate check.


5. need_information

Use when the user wants to create a material but required
creation information is missing.


6. answer_sap_mm

Use for SAP MM Material Master questions that do not require
live material-master database access.


7. reject

Use for anything outside SAP MM Material Master.


STRICT RULES:

Return exactly ONE JSON object.

Do not write any text after the JSON.

Do not repeat the JSON.

Do not use markdown.

Do not use code fences.

Do not browse the internet.

Do not invent database records.

Do not invent missing values. For create_material, material_type,
material_group, and base_unit must come only from words the user
actually typed -- never guessed from real-world domain knowledge
(e.g. assuming a pump is FERT). If not stated, treat as missing.


EXAMPLES


User:
What are the pumps in plant 1000?

Response:
{{
  "action": "search_material",
  "q": "pump",
  "plant": "1000"
}}


User:
Find valves

Response:
{{
  "action": "search_material",
  "q": "valve"
}}


User:
Give me a material from the database

Response:
{{
  "action": "search_material",
  "q": null
}}


User:
Show materials in plant 1000

Response:
{{
  "action": "search_material",
  "q": null,
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
Hi, I want to create a pump in plant 1000

Response:
{{
  "action": "need_information",
  "material_description": "Pump",
  "plant": "1000",
  "missing_fields": [
    "material_type",
    "material_group",
    "base_unit"
  ]
}}


User:
[RECENT CONVERSATION CONTEXT: previous search used q="pump", plant="1000"]
show me the relevant materials in plant 1000

Response:
{{
  "action": "search_material",
  "q": "pump",
  "plant": "1000"
}}


User:
What does FERT mean?

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


User request:

{message}

Response:
"""

    generated = await generate_async(
        planner_prompt,
        max_new_tokens=200,
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
    # PARSE QWEN RESPONSE
    # ========================================================

    try:

        decision = extract_json_object(
            generated
        )

    except (
        json.JSONDecodeError,
        ValueError,
    ) as exc:

        print(
            "ERROR: Could not parse planner JSON:",
            exc,
        )

        return OUT_OF_SCOPE_MESSAGE

    print(
        "PLANNER DECISION:",
        decision,
    )

    action = decision.get(
        "action"
    )

    # ========================================================
    # ACTION ALLOWLIST
    # ========================================================

    if (
        action
        not in ALLOWED_ACTIONS
    ):

        print(
            "BLOCKED UNKNOWN ACTION:",
            action,
        )

        return OUT_OF_SCOPE_MESSAGE

    # ========================================================
    # REJECT
    # ========================================================

    if (
        action
        == "reject"
    ):

        print(
            "ACTION: Rejected out-of-scope request"
        )

        return OUT_OF_SCOPE_MESSAGE

    # ========================================================
    # SEARCH MATERIAL
    # ========================================================

    if (
        action
        == "search_material"
    ):

        search = normalize_search_decision(
            decision
        )

        # Deterministic backstop alongside the RECENT CONVERSATION
        # CONTEXT prompt block: if the planner returned no keyword
        # but the message reads as a continuation ("show me the
        # relevant materials...") and we have a keyword from the
        # prior turn in this session, carry it forward instead of
        # silently falling back to "everything in this plant."
        if not search["q"] and message_has_continuity_cue(message):
            remembered = get_last_search_entities(
                session_contexts,
                session_id,
            )
            if remembered.get("q"):
                print(
                    "CARRYING FORWARD PRIOR SEARCH KEYWORD:",
                    remembered["q"],
                )
                search["q"] = remembered["q"]
                for field in (
                    "material_type",
                    "material_group",
                ):
                    if not search.get(field) and remembered.get(field):
                        search[field] = remembered[field]

        print(
            "ACTION: Calling MCP search_material_master"
        )

        try:

            materials = (
                await search_material_master(
                    q=search[
                        "q"
                    ],

                    plant=search[
                        "plant"
                    ],

                    material_type=search[
                        "material_type"
                    ],

                    material_group=search[
                        "material_group"
                    ],
                )
            )

        except Exception as exc:

            print(
                "ERROR: MCP material search failed:",
                exc,
            )

            return (
                "I couldn't search the material master because "
                "the material service is unavailable."
            )

        print(
            "MCP SEARCH RESULT COUNT:",
            (
                len(materials)
                if isinstance(
                    materials,
                    list,
                )
                else "non-list"
            ),
        )

        reply = format_search_results(message=message, materials=materials)
        remember_query_context(
            session_contexts,
            session_id,
            message,
            reply,
            bool(materials),
        )
        remember_search_entities(
            session_contexts,
            session_id,
            {
                "q": search["q"],
                "plant": search["plant"],
                "material_type": search["material_type"],
                "material_group": search["material_group"],
            },
        )
        return reply

    # ========================================================
    # DYNAMIC READ-ONLY MATERIAL QUERY
    # ========================================================

    if action == "query_materials":
        try:
            query_plan = validate_query_plan(decision.get("query_plan"))
        except (TypeError, ValueError) as exc:
            print("BLOCKED INVALID QUERY PLAN:", exc)
            return (
                "I understood this as a Material Master analysis, but the "
                "query plan was not safe or valid. Please rephrase the request."
            )
        try:
            result = await query_material_master(query_plan)
        except Exception as exc:
            print("ERROR: MCP analytical query failed:", exc)
            return "I couldn't query the material master because the material service is unavailable."
        reply = format_query_results(result)
        found = bool(result.get("items") or result.get("rows")) if isinstance(result, dict) else False
        if isinstance(result, dict) and result.get("kind") == "aggregate":
            found = result.get("value") is not None or found
        remember_query_context(session_contexts, session_id, message, reply, found)
        return reply

    # ========================================================
    # GET MATERIAL
    # ========================================================

    if (
        action
        == "get_material"
    ):

        material_number = clean_optional_value(
            decision.get(
                "material_number"
            ),
            {
                "material number",
                "material",
                "number",
            },
        )

        if not material_number:

            return (
                "Please provide the material number "
                "you want me to retrieve."
            )

        material_number = (
            material_number.upper()
        )

        print(
            "ACTION: Calling MCP explain_material_master:",
            material_number,
        )

        try:

            material = (
                await explain_material_master(
                    material_number
                )
            )

        except Exception as exc:

            print(
                "ERROR: Material lookup failed:",
                exc,
            )

            return (
                "I couldn't retrieve that material from "
                "the material master."
            )

        return (
            format_material(
                message=message,
                material=material,
            )
        )

    # ========================================================
    # CREATE MATERIAL
    # ========================================================

    if (
        action
        == "create_material"
    ):

        # Deterministic backstop: drop any type/group/base_unit the
        # planner guessed but that isn't actually grounded in what
        # the user typed. See strip_ungrounded_create_fields().
        decision = strip_ungrounded_create_fields(
            decision,
            message,
        )

        # A create request carries the same "what is the user talking
        # about" signal as a search does (e.g. "create a pump in plant
        # 1000" establishes description="pump", plant="1000") -- even
        # when required fields are still missing. Without this, a
        # later "show me the relevant materials" has nothing to carry
        # forward, because remember_search_entities was previously
        # only called from the search_material handler.
        remember_search_entities(
            session_contexts,
            session_id,
            {
                "q": normalize_search_query(
                    decision.get("material_description")
                ),
                "plant": normalize_plant(decision.get("plant")),
                "material_type": normalize_material_type(
                    decision.get("material_type")
                ),
                "material_group": normalize_material_group(
                    decision.get("material_group")
                ),
            },
        )

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

            preview_matches: List[Dict[str, Any]] = []

            try:
                preview_matches = await search_material_master(
                    q=normalize_search_query(
                        decision.get("material_description")
                    ),
                    plant=normalize_plant(decision.get("plant")),
                    material_type=None,
                    material_group=None,
                )
            except Exception as exc:
                print(
                    "WARNING: preview similarity search failed:",
                    exc,
                )
                preview_matches = []

            reply = (
                "I need more information before creating "
                "the material. Missing: "
                + ", ".join(missing)
            )

            if preview_matches:
                reply += (
                    "\n\nWhile you decide, here are existing materials "
                    "that already look similar -- one of these might "
                    "already be what you need:\n"
                    + format_search_results(
                        message=message,
                        materials=preview_matches[:5],
                    )
                )

            return reply

        plant = normalize_plant(
            decision.get(
                "plant"
            )
        )

        if not plant:

            return (
                "I need a valid numeric plant code before "
                "creating the material."
            )

        proposal = {
            "material_description":
                str(
                    decision.get(
                        "material_description"
                    )
                ).strip(),

            "material_type":
                str(
                    decision.get(
                        "material_type"
                    )
                ).strip().upper(),

            "material_group":
                str(
                    decision.get(
                        "material_group"
                    )
                ).strip().upper(),

            "base_unit":
                str(
                    decision.get(
                        "base_unit"
                    )
                ).strip().upper(),

            "plant":
                plant,

            "standard_price":
                decision.get(
                    "standard_price"
                ),

            "currency":
                str(
                    decision.get(
                        "currency",
                        "USD",
                    )
                ).strip().upper(),
        }

        # ----------------------------------------------------
        # REQUIRED DUPLICATE CHECK
        # ----------------------------------------------------

        try:

            matches = (
                await find_similar_materials(
                    proposal
                )
            )

        except Exception:

            return (
                "I couldn't complete the required duplicate-material "
                "check, so I did not proceed with material creation. "
                "Please try again after the material master connection "
                "is available."
            )

        # ----------------------------------------------------
        # POSSIBLE DUPLICATE
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

        missing_fields = decision.get(
            "missing_fields",
            [],
        )

        # The planner may embed whatever it did extract (description,
        # plant, etc.) alongside missing_fields for this same action.
        # Record it as entity memory and use it for a similarity
        # preview, same as the create_material missing-fields branch --
        # otherwise a later "show me the relevant materials" has
        # nothing to carry forward whenever the model picks
        # need_information instead of create_material for an
        # equivalent partial request.
        remember_search_entities(
            session_contexts,
            session_id,
            {
                "q": normalize_search_query(
                    decision.get("material_description")
                ),
                "plant": normalize_plant(decision.get("plant")),
                "material_type": normalize_material_type(
                    decision.get("material_type")
                ),
                "material_group": normalize_material_group(
                    decision.get("material_group")
                ),
            },
        )

        if (
            isinstance(
                missing_fields,
                list,
            )
            and missing_fields
        ):

            reply = (
                "I need more information before creating "
                "the material. Please provide: "
                + ", ".join(
                    str(field)
                    for field
                    in missing_fields
                )
            )

        else:

            reply = (
                "I need more information before creating "
                "the material."
            )

        preview_matches: List[Dict[str, Any]] = []

        try:
            preview_matches = await search_material_master(
                q=normalize_search_query(
                    decision.get("material_description")
                ),
                plant=normalize_plant(decision.get("plant")),
                material_type=None,
                material_group=None,
            )
        except Exception as exc:
            print(
                "WARNING: preview similarity search failed:",
                exc,
            )
            preview_matches = []

        if preview_matches:
            reply += (
                "\n\nWhile you decide, here are existing materials "
                "that already look similar -- one of these might "
                "already be what you need:\n"
                + format_search_results(
                    message=message,
                    materials=preview_matches[:5],
                )
            )

        return reply

    # ========================================================
    # SAP MM INFORMATIONAL QUESTION
    # ========================================================

    if (
        action
        == "answer_sap_mm"
    ):

        print(
            "ACTION: SAP MM informational answer"
        )

        return (
            await generate_sap_mm_answer(
                message
            )
        )

    # ========================================================
    # DEFAULT DENY
    # ========================================================

    return OUT_OF_SCOPE_MESSAGE