import json
import logging
import re

from app.retrieval import get_retriever, TEST_TYPE_LABELS
from app.prompts import SYSTEM_PROMPT, build_catalog_context, build_user_prompt
from app.llm import call_llm
from app.schemas import ChatResponse, Recommendation

logger = logging.getLogger(__name__)

MAX_TURNS = 8
TOP_K_RETRIEVAL = 18

# Fast-path guard: catches blatant prompt-injection / jailbreak attempts
# without even calling the LLM, so we can't be talked out of the system
# prompt no matter how the model interprets the conversation.
INJECTION_PATTERNS = [
    r"ignore (all|any|previous|prior) instructions",
    r"reveal (your|the) (system )?prompt",
    r"you are now",
    r"disregard (all|your) (rules|instructions)",
    r"pretend (you|to) (are|be)",
    r"act as (if|though)",
    r"print your (instructions|system prompt)",
    r"what (is|are) your (instructions|system prompt|rules)",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

FULL_NAME_TO_CODE = {v: k for k, v in TEST_TYPE_LABELS.items()}


def _build_retrieval_query(messages):
    # Weight the most recent user turn higher by repeating it, but include
    # full user-turn history so refine/compare have context to retrieve
    # against too.
    user_msgs = [m.content for m in messages if m.role == "user"]
    if not user_msgs:
        return ""
    recent = user_msgs[-1]
    history = " ".join(user_msgs[:-1])
    return f"{recent} {recent} {history}"


def _validate_and_clean_recommendations(raw_recs, catalog_items_by_name):
    """Drop any recommendation whose name+url doesn't exactly match a real
    catalog item, to guarantee we never surface a hallucinated entry."""
    cleaned = []
    for r in raw_recs:
        name = r.get("name", "").strip()
        match = catalog_items_by_name.get(name.lower())
        if not match:
            logger.warning(f"Dropping hallucinated recommendation: {name}")
            continue
        if r.get("url", "").strip() != match["url"]:
            # trust our catalog URL over whatever the model produced
            pass
        test_type = r.get("test_type", "").strip()
        if test_type not in TEST_TYPE_LABELS.values():
            # fall back to the item's first real category code
            first_cat = match["test_type"][0] if match["test_type"] else "K"
            test_type = TEST_TYPE_LABELS.get(first_cat, "K")
        cleaned.append(
            Recommendation(name=match["name"], url=match["url"], test_type=test_type)
        )
    return cleaned[:10]


def handle_chat(messages) -> ChatResponse:
    turn_count = len(messages)
    last_user_msg = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    # Fast-path refusal for blatant injection attempts — never even reaches
    # the LLM, so it can't be reasoned around.
    if INJECTION_RE.search(last_user_msg):
        return ChatResponse(
            reply=(
                "I can only help with selecting SHL assessments. I'm not able "
                "to change my role or share internal instructions — happy to "
                "help you find the right assessment for a role instead."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    retriever = get_retriever()
    query = _build_retrieval_query(messages)
    candidates = retriever.search(query, top_k=TOP_K_RETRIEVAL) if query else []

    catalog_context = build_catalog_context(candidates)
    system_prompt = SYSTEM_PROMPT.replace("{catalog_context}", catalog_context)
    user_prompt = build_user_prompt(messages)

    try:
        raw = call_llm(system_prompt, user_prompt)
        parsed = json.loads(raw)
    except Exception as e:
        logger.error(f"LLM call/parse failed: {e}")
        return ChatResponse(
            reply=(
                "Sorry, I hit an error processing that. Could you rephrase "
                "what you're looking for?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    catalog_by_name = {item["name"].lower(): item for item in candidates}
    # also allow validation against the full catalog, in case the model
    # (correctly) referenced something outside the retrieved top-K
    full_catalog_by_name = {item["name"].lower(): item for item in retriever.catalog}
    combined_lookup = {**full_catalog_by_name, **catalog_by_name}

    raw_recs = parsed.get("recommendations") or []
    recommendations = _validate_and_clean_recommendations(raw_recs, combined_lookup)

    reply = parsed.get("reply", "").strip() or "Here's what I found."
    end_of_conversation = bool(parsed.get("end_of_conversation", False))

    # Hard turn cap: force wrap-up at MAX_TURNS regardless of model output
    if turn_count >= MAX_TURNS:
        end_of_conversation = True
        if not recommendations:
            # Try one last best-effort retrieval so we don't end empty-handed
            fallback = retriever.search(query, top_k=5) if query else []
            recommendations = [
                Recommendation(
                    name=item["name"],
                    url=item["url"],
                    test_type=TEST_TYPE_LABELS.get(
                        item["test_type"][0] if item["test_type"] else "Knowledge & Skills",
                        "K",
                    ),
                )
                for item in fallback
            ]

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
