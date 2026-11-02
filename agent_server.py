import json
import logging
import os
import sys
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import AsyncGroq
from pydantic import ValidationError

from schemas import (
    ChallengeModel,
    CorrectionModel,
    DebateTurnModel,
    EvaluationReportModel,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORTS = {
    "proponent": 8001,
    "opponent": 8002,
    "evaluator": 8003,
}

# Determine AGENT_ROLE from CLI argument if provided (e.g. `python agent_server.py opponent`),
# otherwise fallback to environment variable or default to 'proponent'
cli_role = None
if len(sys.argv) > 1 and not sys.argv[0].lower().endswith(("uvicorn", "uvicorn.exe")):
    candidate = sys.argv[1].lower()
    if not candidate.startswith("-"):
        if candidate in PORTS:
            cli_role = candidate
        else:
            raise RuntimeError(
                f"Invalid agent role '{sys.argv[1]}'. Valid roles are: {', '.join(PORTS.keys())}"
            )

AGENT_ROLE = (cli_role or os.getenv("AGENT_ROLE", "proponent")).lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

if AGENT_ROLE not in PORTS:
    raise RuntimeError(f"Invalid AGENT_ROLE: {AGENT_ROLE}")

client = AsyncGroq(api_key=GROQ_API_KEY)
app = FastAPI(title=f"A2A {AGENT_ROLE.title()} Agent")


AGENT_CONFIG = {
    "proponent": {
        "name": "Proponent Agent",
        "description": "Supports a proposal and responds to challenges.",
        "instructions": """
You are the PROPONENT in a multi-agent debate.

Argue persuasively in favor of the proposal using sound logical reasoning and factual grounding.
Respond directly to the opponent's counterarguments and critique their weak assumptions.
Identify specific factual errors, contradictions, or unsupported claims when you have justification.
Do not challenge a claim merely because you disagree with the opinion.
Do not invent fake statistics, evidence, or study sources. State uncertainty honestly when applicable.

You must respond in valid JSON matching this schema:
{
  "argument": "Your debate contribution",
  "challenge": null
}

If challenging a specific opponent claim, use:
{
  "argument": "Your debate contribution",
  "challenge": {
    "claim": "The specific claim being challenged",
    "reason": "Why it is questionable, contradictory, or unsupported",
    "confidence": "low, medium, or high",
    "suggested_correction": "A more accurate formulation"
  }
}
""",
    },
    "opponent": {
        "name": "Opponent Agent",
        "description": "Challenges proposals and questionable claims.",
        "instructions": """
You are the OPPONENT in a multi-agent debate.

Challenge the proposal with rigorous logical counterarguments and critical scrutiny.
Address the proponent's latest argument directly and uncover underlying assumptions or risks.
Identify specific factual errors, contradictions, or unsupported claims.
Do not challenge something merely because you disagree with the perspective.
Do not invent fake facts, statistics, or sources. Distinguish false claims from those needing qualification.

You must respond in valid JSON matching this schema:
{
  "argument": "Your counterargument",
  "challenge": null
}

If challenging a specific proponent claim, use:
{
  "argument": "Your counterargument",
  "challenge": {
    "claim": "The specific claim being challenged",
    "reason": "Why it is questionable, contradictory, or unsupported",
    "confidence": "low, medium, or high",
    "suggested_correction": "A more accurate formulation"
  }
}
""",
    },
    "evaluator": {
        "name": "Evaluator Agent",
        "description": "Evaluates debate arguments, scoring both sides and generating a balanced assessment.",
        "instructions": """
You are an impartial, objective DEBATE EVALUATOR and JUDGE.

Evaluate the complete transcript, including arguments, rebuttals, claim challenges, and responses.
Score both sides objectively from 1 to 10 on three criteria:
1. Logic: coherence, valid deductive/inductive reasoning, avoidance of fallacies.
2. Evidence: empirical plausibility, grounding, avoidance of unsubstantiated claims.
3. Rebuttal: responsiveness to points raised and effectiveness in defending or reasonably conceding challenges.

Determine a clear winner based on overall argumentative strength ('proponent', 'opponent', or 'draw').
Identify key concessions made by either side and core unresolved disagreements.
Provide a balanced, detailed final assessment report.

You must respond in valid JSON matching the specified schema.
""",
    },
}

config = AGENT_CONFIG[AGENT_ROLE]


@app.get("/")
async def root():
    return {
        "agent": config["name"],
        "role": AGENT_ROLE,
        "status": "running",
    }


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return {
        "name": config["name"],
        "description": config["description"],
        "url": f"http://127.0.0.1:{PORTS[AGENT_ROLE]}/",
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "capabilities": {"streaming": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": AGENT_ROLE,
                "name": config["name"],
                "description": config["description"],
                "tags": ["debate", "claim-review"],
            }
        ],
    }


def extract_text(message):
    texts = []

    for part in message.get("parts", []):
        if part.get("kind") == "text" and part.get("text"):
            texts.append(part["text"])

    return "\n".join(texts).strip()


def parse_json_response(content):
    """Parse JSON safely, tolerating potential markdown fences or wrappers."""
    text = content.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Recover a JSON object if the model added surrounding text.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        result = json.loads(text[start : end + 1])
        if isinstance(result, dict):
            return result

    raise ValueError("Groq response did not contain valid JSON.")


def build_prompt(topic, history, mode, challenge):
    if mode == "correction":
        return f"""
TOPIC:
{topic}

DEBATE HISTORY:
{history}

A different agent has challenged one of your claims:
{json.dumps(challenge, ensure_ascii=False, indent=2)}

Review the challenge fairly and objectively. Do not accept or reject it automatically.

Choose exactly one decision:
- accepted: you agree the claim was incorrect, inaccurate, or misleading.
- revised: you modify, qualify, or narrow the claim.
- defended: you defend the original claim and explain your justification.

Return ONLY this JSON structure:
{{
  "decision": "accepted, revised, or defended",
  "response": "Your reasoned explanation",
  "corrected_claim": "The modified claim if revised or accepted, otherwise empty string"
}}
"""

    if mode == "evaluation":
        return f"""
TOPIC:
{topic}

COMPLETE DEBATE TRANSCRIPT:
{history}

Evaluate both sides thoroughly.
Assess argument validity, factual grounding, challenges raised, and responses given.

Score each participant on:
- logic (1 to 10)
- evidence (1 to 10)
- rebuttal (1 to 10)

Determine the winner ('proponent', 'opponent', or 'draw').
Identify key concessions made and core unresolved disagreements.
Write a balanced, detailed final assessment report.

Return ONLY this JSON structure:
{{
  "winner": "proponent, opponent, or draw",
  "proponent_scores": {{
    "logic": 8,
    "evidence": 7,
    "rebuttal": 8
  }},
  "opponent_scores": {{
    "logic": 8,
    "evidence": 8,
    "rebuttal": 9
  }},
  "key_concessions": [
    "Specific concession or claim revision made by a participant"
  ],
  "unresolved_disagreements": [
    "Fundamental point of contention that remained unresolved"
  ],
  "report": "Your comprehensive final assessment analyzing both sides"
}}
"""

    if mode == "opening":
        return f"""
TOPIC:
{topic}

Make the opening argument for your assigned debate role.
Provide clear foundational premises and supporting reasoning.
No opposing argument has been made yet, so do not raise a challenge.

Return ONLY this JSON structure:
{{
  "argument": "Your opening argument",
  "challenge": null
}}
"""

    return f"""
TOPIC:
{topic}

DEBATE HISTORY:
{history}

Make your next contribution according to your role.
Address the latest opposing argument directly.
If you identify a specific questionable claim, factual inaccuracy, or contradiction in the opposing argument, raise a challenge. Otherwise set challenge to null.

Return ONLY this JSON structure:
{{
  "argument": "Your debate contribution",
  "challenge": null
}}

Or if raising a challenge:
{{
  "argument": "Your debate contribution",
  "challenge": {{
    "claim": "The specific claim being challenged",
    "reason": "Why it is questionable or unsupported",
    "confidence": "low, medium, or high",
    "suggested_correction": "A more accurate formulation"
  }}
}}
"""


async def generate_response(topic, history, mode, challenge=None):
    prompt = build_prompt(topic, history, mode, challenge)
    last_error = None
    temperature = 0.2 if mode == "evaluation" else 0.7

    for attempt in range(2):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": config["instructions"],
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_completion_tokens=1500,
        )

        content = response.choices[0].message.content

        try:
            if not content:
                raise ValueError("Groq returned an empty response.")

            raw_dict = parse_json_response(content)

            # Validate using Pydantic models
            if mode == "correction":
                validated = CorrectionModel.model_validate(raw_dict)
            elif mode == "evaluation":
                validated = EvaluationReportModel.model_validate(raw_dict)
            else:
                validated = DebateTurnModel.model_validate(raw_dict)

            return validated.model_dump()

        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "%s validation failed on attempt %s: %s",
                config["name"],
                attempt + 1,
                exc,
            )

            prompt += (
                f"\n\nFORMAT RETRY: Your previous response failed schema validation: {exc}. "
                "Return exactly one valid JSON object matching the required schema."
            )

    raise RuntimeError(
        f"{config['name']} failed to return valid structured JSON: {last_error}"
    ) from last_error


@app.post("/")
async def handle_a2a(request: Request):
    request_id = None

    try:
        payload = await request.json()
        request_id = payload.get("id")

        if payload.get("jsonrpc") != "2.0":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32600,
                    "message": "Invalid JSON-RPC request.",
                },
            })

        if payload.get("method") != "message/send":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": "Method not found.",
                },
            })

        params = payload.get("params") or {}
        message = params.get("message") or {}

        topic = params.get("topic", "").strip()
        history = extract_text(message)
        mode = params.get("mode", "debate")
        challenge = params.get("challenge")

        if not topic:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "A debate topic is required.",
                },
            })

        if mode not in {
            "opening",
            "debate",
            "correction",
            "evaluation",
        }:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Unsupported agent mode.",
                },
            })

        logger.info(
            "%s received mode=%s",
            config["name"],
            mode,
        )

        result_data = await generate_response(
            topic=topic,
            history=history,
            mode=mode,
            challenge=challenge,
        )

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "kind": "message",
                "messageId": str(uuid4()),
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": json.dumps(
                            result_data,
                            ensure_ascii=False,
                        ),
                    }
                ],
            },
        }

    except Exception:
        logger.exception("%s request failed", config["name"])

        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": (
                        "Agent processing failed. Check the agent terminal logs."
                    ),
                },
            },
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn

    port = PORTS[AGENT_ROLE]
    print(f"Starting {config['name']} ({AGENT_ROLE}) on http://127.0.0.1:{port}...")
    uvicorn.run(app, host="127.0.0.1", port=port)
