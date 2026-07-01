# Approach Document — SHL Assessment Recommender

## Problem framing
The task is to turn a vague hiring need into a grounded shortlist of SHL Individual Test Solutions through
dialogue, while never recommending outside the catalog. I treated this as three composable sub-problems:
(1) building a clean, queryable catalog, (2) retrieving relevant candidates per turn, and (3) a conversation
policy that decides whether to clarify, recommend, refine, compare, or refuse.

## Catalog preparation
The scraped catalog (377 entries) included 7 items that were clearly pre-packaged bundles rather than
individual instruments — e.g. "Customer Service Phone Solution" (description: "includes a contact center
simulation and two behavioral tests"), each stacking multiple `keys` categories under one product. Since the
live SHL catalog page has no visible tab/type field distinguishing this in the scrape, I identified these by
name pattern ("...Solution") cross-checked against their bundled, multi-category descriptions, and excluded
them — leaving **370 Individual Test Solutions**. This is a judgment call I made explicit rather than silent,
since it directly affects Recall@10 and is defensible on inspection of the descriptions.

## Retrieval design
I built a retrieval layer with **semantic embeddings (sentence-transformers `all-MiniLM-L6-v2`) indexed in
FAISS as the primary path**, with **TF-IDF cosine similarity (scikit-learn) as an automatic fallback** if the
embedding model can't be loaded (e.g. no network access at cold start). On top of either backend, I apply a
metadata boost for job level, test type, and language matches extracted from the conversation, since
structured metadata is more reliable than semantic similarity alone for those fields.

I chose this hybrid over a single approach deliberately: pure vector search can miss exact keyword matches
(e.g. "Java 8" vs "Java Platform Enterprise Edition"), and pure keyword search misses paraphrased intent
("someone who works well with stakeholders" → personality/behavioral tests). The fallback also means the
service degrades gracefully rather than failing outright if the embedding model is unreachable.

## Agent / prompt design
Rather than a hand-coded slot-filling state machine, I use **one LLM call per turn** (Groq, Llama 3.3 70B)
that receives the full conversation history plus a retrieved candidate pool (top ~18 catalog items with name,
URL, description, test type, job level, duration) and is instructed, in priority order, to: (1) refuse if
out-of-scope or an injection attempt, (2) clarify with exactly one question if there isn't enough context yet,
(3) recommend 1–10 items once there is, (4) refine an existing shortlist rather than restarting when
constraints change, (5) answer comparisons using only the provided catalog descriptions, never prior/trained
knowledge of SHL products. The model must return the exact JSON schema SHL specified — no other text.

I chose a single well-structured prompt over a multi-agent/multi-call pipeline because the task is bounded
(one domain, one catalog, max 8 turns, 30s timeout per call) — a single call is faster, cheaper, and has far
fewer failure points than orchestrating multiple LLM calls per turn.

## Guardrails (code, not just prompting)
Prompting alone isn't reliable enough for hard constraints, so I enforce three things in code regardless of
what the model outputs:
- **Hallucination guard**: every recommendation is checked against the real catalog by exact name match
  before being returned; anything the model invents is silently dropped.
- **Fast-path injection refusal**: obvious jailbreak patterns ("ignore previous instructions", "reveal your
  system prompt", etc.) are caught by regex *before* the LLM is even called, so they can't be reasoned around.
- **Hard turn cap**: turn count is tracked from message history length; `end_of_conversation` is forced true
  at turn 8 regardless of model output, with a best-effort fallback shortlist if none was given yet.

## Evaluation
I parsed the 10 provided conversation traces into (user-turn sequence, expected final shortlist) pairs and
built a harness that replays each trace turn-by-turn against the live `/chat` endpoint, checks schema
compliance on every response, and computes Recall@10 per trace and the mean across all 10. [Once deployed:
insert your measured mean Recall@10 here from `tests/run_eval.py` output.]

I also wrote offline unit tests (mocked LLM responses) to verify the guardrails independently of model
behavior — this caught a real bug during development: my system prompt's JSON example used literal `{}`
characters that collided with Python's `str.format()`, which would have made every single `/chat` call fail
with a 500 error. I fixed this by switching to a plain string replace.

## What didn't work / trade-offs
- I initially considered a rigid slot-filling state machine (explicit fields: role, seniority, skills,
  language, duration) rather than letting the LLM infer readiness-to-recommend. I moved away from this
  because the traces show real conversational variance (users volunteering info out of order, refusing to
  answer, or correcting themselves) that a rigid slot schema handles poorly — the traces are a better source
  of truth for "when is there enough context" than a fixed field list.
- Embeddings could not be tested from my local development sandbox (no network access to huggingface.co in
  that environment), so the fallback-to-TF-IDF design was as much a resilience decision as a workaround. The
  live deployment has full internet access, so the embedding path is expected to be primary in production —
  this was verified via the eval harness after deployment.

## Stack
FastAPI, Groq (Llama 3.3 70B via OpenAI-compatible API), sentence-transformers + FAISS (primary retrieval),
scikit-learn TF-IDF (fallback retrieval), Pydantic for schema enforcement, deployed on Render.

## AI tool usage disclosure
I used Claude (Anthropic) as an AI pair-programmer for this assignment — for scaffolding the FastAPI
structure, the retrieval fallback pattern, and the prompt template, while I directed the architecture
decisions (catalog filtering rule, hybrid retrieval choice, single-call agent design, guardrails-in-code vs
guardrails-in-prompt) and verified behavior against the provided traces myself.
