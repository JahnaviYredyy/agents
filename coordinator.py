import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from dotenv import load_dotenv
import httpx
from pydantic import ValidationError

from schemas import (
    CorrectionModel,
    DebateTurnModel,
    EvaluationReportModel,
)

# Load environment variables
env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
load_dotenv()

# Configure stdout for utf-8 if supported (avoids cp1252 crash on Windows)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

AGENTS = {
    "proponent": "http://127.0.0.1:8001",
    "opponent": "http://127.0.0.1:8002",
    "evaluator": "http://127.0.0.1:8003",
}

TIMEOUT_SECONDS = 120
MAX_HISTORY_CHARS = 18000
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
BASE_DIR = Path(__file__).resolve().parent


def extract_result(data, mode="debate"):
    """Extract and validate structured JSON from an A2A response."""
    if "error" in data:
        error = data["error"]
        raise RuntimeError(
            f"A2A error {error.get('code')}: {error.get('message')}"
        )

    result = data.get("result", {})
    parts = result.get("parts", [])

    for part in parts:
        if part.get("kind") != "text":
            continue

        raw = part.get("text", "").strip()
        if not raw:
            continue

        candidates = [raw]
        cleaned = raw

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        candidates.append(cleaned)

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidates.append(cleaned[start : end + 1])

        parsed_dict = None
        for candidate in candidates:
            try:
                candidate_obj = json.loads(candidate)
                if isinstance(candidate_obj, dict):
                    parsed_dict = candidate_obj
                    break
            except json.JSONDecodeError:
                continue

        if parsed_dict is not None:
            try:
                if mode == "correction":
                    return CorrectionModel.model_validate(parsed_dict).model_dump()
                elif mode == "evaluation":
                    return EvaluationReportModel.model_validate(parsed_dict).model_dump()
                else:
                    return DebateTurnModel.model_validate(parsed_dict).model_dump()
            except ValidationError as exc:
                # Tolerant fallback for evaluator if only 'report' text is present
                if mode == "evaluation" and "report" in parsed_dict:
                    fallback = EvaluationReportModel(
                        winner="draw",
                        report=str(parsed_dict["report"]),
                    )
                    return fallback.model_dump()
                raise RuntimeError(
                    f"Agent response failed schema validation ({mode}): {exc}"
                )

        if mode in {"opening", "debate"}:
            print(
                "\n[Warning] Agent returned non-JSON text. Continuing with fallback."
            )
            return {
                "argument": raw,
                "challenge": None,
            }

        raise RuntimeError(
            f"Agent returned invalid structured JSON in {mode} mode. Preview: {raw[:300]}"
        )

    raise RuntimeError("Agent returned no text response.")


async def call_agent(
    client,
    role,
    topic,
    history,
    mode="debate",
    challenge=None,
):
    """Send an A2A-style JSON-RPC message/send request."""
    if role not in AGENTS:
        raise ValueError(f"Unknown agent role: {role}")

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "topic": topic,
            "mode": mode,
            "challenge": challenge,
            "message": {
                "messageId": str(uuid4()),
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": history[-MAX_HISTORY_CHARS:],
                    }
                ],
            },
        },
    }

    url = AGENTS[role] + "/"
    last_error = None

    for attempt in range(2):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return extract_result(response.json(), mode=mode)
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_error = exc
            if attempt == 0:
                print(f"\n[Warning] Request to {role} failed. Retrying once...")
                await asyncio.sleep(1)

    raise RuntimeError(f"Unable to contact {role}: {last_error}")


def add_entry(transcript, speaker, content):
    transcript.append(
        f"\n{'=' * 18} {speaker.upper()} {'=' * 18}\n{content}"
    )


async def handle_challenge(
    client,
    topic,
    transcript,
    challenge,
    challenged_role,
    challenger_role,
    round_number,
    challenges_log,
    debate_events,
):
    """Pause normal progression and ask the challenged speaker to respond."""
    print("\n" + "!" * 60)
    print("CLAIM CHALLENGE DETECTED - PAUSING DEBATE")
    print("!" * 60)
    print(f"Challenger:           {challenger_role.title()} Agent")
    print(f"Challenged:           {challenged_role.title()} Agent")
    print(f"Challenged Claim:     {challenge['claim']}")
    print(f"Reason:               {challenge['reason']}")
    print(f"Confidence:           {challenge['confidence']}")
    print(f"Suggested Correction: {challenge['suggested_correction']}")

    add_entry(
        transcript,
        f"{challenger_role} challenge",
        json.dumps(challenge, ensure_ascii=False, indent=2),
    )

    print(f"\n[PAUSED] Asking {challenged_role.title()} Agent to review the claim...")

    try:
        correction = await call_agent(
            client=client,
            role=challenged_role,
            topic=topic,
            history="\n".join(transcript),
            mode="correction",
            challenge=challenge,
        )
    except (RuntimeError, ValueError) as exc:
        add_entry(transcript, "Correction review failure", str(exc))
        print(f"[Warning] Correction review failed: {exc}")
        print("Resuming debate with challenge unresolved.")
        return

    decision = correction["decision"]
    response = correction["response"]
    corrected_claim = correction.get("corrected_claim", "")

    correction_record = {
        "decision": decision,
        "response": response,
        "corrected_claim": corrected_claim,
    }

    add_entry(
        transcript,
        f"{challenged_role} correction response",
        json.dumps(correction_record, ensure_ascii=False, indent=2),
    )

    challenge_event = {
        "round": round_number,
        "challenger": challenger_role,
        "challenged": challenged_role,
        "claim": challenge["claim"],
        "reason": challenge["reason"],
        "confidence": challenge["confidence"],
        "suggested_correction": challenge["suggested_correction"],
        "decision": decision,
        "response": response,
        "corrected_claim": corrected_claim,
    }
    challenges_log.append(challenge_event)
    debate_events.append({"type": "challenge", "data": challenge_event})

    print(f"\n[{challenged_role.title()} Decision: {decision.upper()}]")
    print(response)

    if corrected_claim:
        print(f"Revised claim: {corrected_claim}")

    print("\n[RESUMING] Continuing debate turn progression...\n")


async def debate_turn(
    client,
    topic,
    transcript,
    speaking_role,
    challenged_role,
    round_number,
    challenges_log,
    debate_events,
    mode="debate",
):
    """Run one debate turn and resolve any challenge before proceeding."""
    result = await call_agent(
        client=client,
        role=speaking_role,
        topic=topic,
        history="\n".join(transcript),
        mode=mode,
    )

    argument = result["argument"].strip()
    add_entry(transcript, speaking_role, argument)

    debate_events.append({
        "type": "turn",
        "round": round_number,
        "speaker": speaking_role,
        "mode": mode,
        "argument": argument,
    })

    print(f"\n[{speaking_role.title()}]")
    print(argument)

    challenge = result.get("challenge")
    if challenge is not None:
        await handle_challenge(
            client=client,
            topic=topic,
            transcript=transcript,
            challenge=challenge,
            challenged_role=challenged_role,
            challenger_role=speaking_role,
            round_number=round_number,
            challenges_log=challenges_log,
            debate_events=debate_events,
        )

    return argument


def calculate_advantage(score_prop: int, score_opp: int) -> str:
    diff = score_prop - score_opp
    if diff > 0:
        return f"Proponent (+{diff})"
    elif diff < 0:
        return f"Opponent (+{abs(diff)})"
    return "Tied"


def generate_markdown_report(
    topic: str,
    rounds: int,
    evaluation: dict,
    debate_events: list,
    challenges_log: list,
) -> str:
    """Generate a clean, professional Markdown report."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    winner = evaluation.get("winner", "draw").lower()
    winner_title = (
        "Proponent Agent"
        if winner == "proponent"
        else "Opponent Agent"
        if winner == "opponent"
        else "Draw / Tie"
    )

    prop_scores = evaluation.get("proponent_scores", {})
    opp_scores = evaluation.get("opponent_scores", {})

    p_logic = prop_scores.get("logic", 0)
    p_evidence = prop_scores.get("evidence", 0)
    p_rebuttal = prop_scores.get("rebuttal", 0)
    p_total = prop_scores.get("total", p_logic + p_evidence + p_rebuttal)

    o_logic = opp_scores.get("logic", 0)
    o_evidence = opp_scores.get("evidence", 0)
    o_rebuttal = opp_scores.get("rebuttal", 0)
    o_total = opp_scores.get("total", o_logic + o_evidence + o_rebuttal)

    logic_adv = calculate_advantage(p_logic, o_logic)
    evidence_adv = calculate_advantage(p_evidence, o_evidence)
    rebuttal_adv = calculate_advantage(p_rebuttal, o_rebuttal)
    total_adv = calculate_advantage(p_total, o_total)

    concessions = evaluation.get("key_concessions", [])
    unresolved = evaluation.get("unresolved_disagreements", [])
    report_text = evaluation.get("report", "").strip()

    md = []
    md.append("# 🏛️ Multi-Agent Debate Report\n")
    md.append("| Metadata | Details |")
    md.append("| :--- | :--- |")
    md.append(f"| **Debate Topic** | {topic} |")
    md.append(f"| **Rounds Completed** | {rounds} |")
    md.append(f"| **Model** | `{MODEL_NAME}` |")
    md.append(f"| **Date & Time** | {now_str} |")
    md.append(f"| **Protocol** | Autonomous A2A with Real-Time Claim Review |\n")

    # Winner Banner
    trophy = "🏆" if winner in {"proponent", "opponent"} else "⚖️"
    md.append("## 🏆 Debate Outcome\n")
    md.append(f"> ### {trophy} Decision: **{winner_title}**")
    md.append(">\n" f"> **Final Score:** Proponent **{p_total}/30** vs Opponent **{o_total}/30** ({total_adv})\n")

    # Scorecard Table
    md.append("## 📊 Evaluator Scorecard\n")
    md.append("| Evaluation Dimension | Proponent Agent | Opponent Agent | Advantage |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **Logic & Argumentation** | {p_logic} / 10 | {o_logic} / 10 | {logic_adv} |")
    md.append(f"| **Evidence & Factuality** | {p_evidence} / 10 | {o_evidence} / 10 | {evidence_adv} |")
    md.append(f"| **Rebuttal & Defense** | {p_rebuttal} / 10 | {o_rebuttal} / 10 | {rebuttal_adv} |")
    md.append(f"| **TOTAL SCORE** | **{p_total} / 30** | **{o_total} / 30** | **{total_adv}** |\n")

    # Key Concessions & Unresolved Disagreements
    md.append("## ⚖️ Concessions & Unresolved Issues\n")
    md.append("### 🤝 Key Concessions Made")
    if concessions:
        for c in concessions:
            md.append(f"- {c}")
    else:
        md.append("- *No explicit concessions recorded during this debate.*")
    md.append("")

    md.append("### ⚡ Core Unresolved Disagreements")
    if unresolved:
        for u in unresolved:
            md.append(f"- {u}")
    else:
        md.append("- *No unresolved fundamental disagreements identified.*")
    md.append("\n---\n")

    # Evaluator Narrative
    md.append("## 📝 Evaluator Narrative Assessment\n")
    md.append(report_text)
    md.append("\n\n---\n")

    # Claim Review / Challenges Log
    md.append("## 🔍 Autonomous Claim Reviews & Challenges\n")
    if challenges_log:
        for idx, ch in enumerate(challenges_log, start=1):
            decision_badge = ch['decision'].upper()
            md.append(f"### Challenge #{idx} (Round {ch['round']}) — Decision: `{decision_badge}`")
            md.append(f"- **Challenger:** {ch['challenger'].title()} Agent")
            md.append(f"- **Challenged:** {ch['challenged'].title()} Agent")
            md.append(f"- **Challenged Claim:** *\"{ch['claim']}\"*")
            md.append(f"- **Reason:** {ch['reason']}")
            md.append(f"- **Confidence:** `{ch['confidence']}`")
            md.append(f"- **Suggested Correction:** {ch['suggested_correction']}")
            md.append(f"> **Speaker Response:** {ch['response']}")
            if ch.get("corrected_claim"):
                md.append(f"> **Corrected Claim Formulation:** *\"{ch['corrected_claim']}\"*")
            md.append("")
    else:
        md.append("*No claim challenges were raised during this debate session.*\n")

    md.append("---\n")

    # Full Debate Transcript
    md.append("## 📜 Full Debate Transcript\n")
    for ev in debate_events:
        if ev["type"] == "turn":
            speaker_title = f"{ev['speaker'].title()} Agent"
            mode_badge = "Opening Statement" if ev.get("mode") == "opening" else f"Round {ev.get('round', 1)}"
            md.append(f"### 🗣️ {speaker_title} ({mode_badge})\n")
            md.append(f"{ev['argument']}\n")
        elif ev["type"] == "challenge":
            ch = ev["data"]
            md.append(f"> ⚠️ **Claim Challenge Raised by {ch['challenger'].title()} Agent**:")
            md.append(f"> - **Claim:** *\"{ch['claim']}\"*")
            md.append(f"> - **Reason:** {ch['reason']}")
            md.append(f"> - **Decision:** **{ch['decision'].upper()}**")
            md.append(f"> - **Response:** {ch['response']}\n")

    return "\n".join(md)


def print_console_scorecard(evaluation: dict):
    """Print an attractive summary table to console."""
    winner = evaluation.get("winner", "draw").upper()
    prop_scores = evaluation.get("proponent_scores", {})
    opp_scores = evaluation.get("opponent_scores", {})

    p_logic = prop_scores.get("logic", 0)
    p_evidence = prop_scores.get("evidence", 0)
    p_rebuttal = prop_scores.get("rebuttal", 0)
    p_total = prop_scores.get("total", p_logic + p_evidence + p_rebuttal)

    o_logic = opp_scores.get("logic", 0)
    o_evidence = opp_scores.get("evidence", 0)
    o_rebuttal = opp_scores.get("rebuttal", 0)
    o_total = opp_scores.get("total", o_logic + o_evidence + o_rebuttal)

    print("\n" + "=" * 65)
    print(f"🏆 DEBATE OUTCOME: WINNER = {winner}")
    print("=" * 65)
    print("EVALUATOR SCORECARD:")
    print(f"{'Dimension':<26} {'Proponent':<12} {'Opponent':<12} {'Advantage'}")
    print("-" * 65)
    print(f"{'Logic & Argumentation':<26} {p_logic:>2} / 10     {o_logic:>2} / 10     {calculate_advantage(p_logic, o_logic)}")
    print(f"{'Evidence & Factuality':<26} {p_evidence:>2} / 10     {o_evidence:>2} / 10     {calculate_advantage(p_evidence, o_evidence)}")
    print(f"{'Rebuttal & Defense':<26} {p_rebuttal:>2} / 10     {o_rebuttal:>2} / 10     {calculate_advantage(p_rebuttal, o_rebuttal)}")
    print("-" * 65)
    print(f"{'TOTAL SCORE':<26} {p_total:>2} / 30     {o_total:>2} / 30     {calculate_advantage(p_total, o_total)}")
    print("=" * 65)

    concessions = evaluation.get("key_concessions", [])
    if concessions:
        print("\nKey Concessions:")
        for c in concessions:
            print(f"  • {c}")

    unresolved = evaluation.get("unresolved_disagreements", [])
    if unresolved:
        print("\nUnresolved Disagreements:")
        for u in unresolved:
            print(f"  • {u}")

    print("\nEvaluator Assessment Summary:")
    print(evaluation.get("report", "").strip())
    print("=" * 65)


async def run_debate(topic, rounds=2):
    if not topic.strip():
        raise ValueError("The topic cannot be empty.")

    if not 1 <= rounds <= 5:
        raise ValueError("Choose between 1 and 5 rounds.")

    transcript = [
        f"DEBATE TOPIC: {topic}",
        f"PLANNED ROUNDS: {rounds}",
        (
            "Debate rule: challenges pause normal progression. "
            "The original speaker reviews the challenge, responds, "
            "and the debate then resumes. A challenge is not proof."
        ),
    ]

    challenges_log = []
    debate_events = []

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        print("\nStarting Groq-powered A2A debate...")
        print(f"Topic: {topic}")

        # Opening argument: proponent opens without opposing challenges
        await debate_turn(
            client=client,
            topic=topic,
            transcript=transcript,
            speaking_role="proponent",
            challenged_role="opponent",
            round_number=0,
            challenges_log=challenges_log,
            debate_events=debate_events,
            mode="opening",
        )

        for round_number in range(1, rounds + 1):
            print("\n" + "#" * 55)
            print(f"ROUND {round_number} OF {rounds}")
            print("#" * 55)

            # Opponent turn
            await debate_turn(
                client=client,
                topic=topic,
                transcript=transcript,
                speaking_role="opponent",
                challenged_role="proponent",
                round_number=round_number,
                challenges_log=challenges_log,
                debate_events=debate_events,
            )

            # Proponent turn
            await debate_turn(
                client=client,
                topic=topic,
                transcript=transcript,
                speaking_role="proponent",
                challenged_role="opponent",
                round_number=round_number,
                challenges_log=challenges_log,
                debate_events=debate_events,
            )

        print("\n[Evaluator] Reviewing complete transcript, arguments, and corrections...")

        evaluation = await call_agent(
            client=client,
            role="evaluator",
            topic=topic,
            history="\n".join(transcript),
            mode="evaluation",
        )

        report_narrative = evaluation.get("report", "")
        add_entry(transcript, "Final Evaluation", report_narrative)

        # 1. Save standard text transcript
        transcript_file = BASE_DIR / "debate_transcript.txt"
        with open(transcript_file, "w", encoding="utf-8") as file:
            file.write("\n".join(transcript))

        # 2. Generate and save rich Markdown report
        markdown_content = generate_markdown_report(
            topic=topic,
            rounds=rounds,
            evaluation=evaluation,
            debate_events=debate_events,
            challenges_log=challenges_log,
        )
        report_file = BASE_DIR / "debate_report.md"
        with open(report_file, "w", encoding="utf-8") as file:
            file.write(markdown_content)

        # 3. Print attractive console scorecard
        print_console_scorecard(evaluation)

        print(f"\nSaved Reports:")
        print(f"  • Rich Markdown Report: {report_file.name}")
        print(f"  • Raw Text Transcript:  {transcript_file.name}")

        return evaluation


async def main():
    print("=" * 45)
    print("GROQ-POWERED A2A DEBATE SYSTEM")
    print("=" * 45)

    topic = input("\nEnter the debate topic: ").strip()
    rounds_text = input("Number of rounds (1-5, default 2): ").strip()

    try:
        rounds = int(rounds_text) if rounds_text else 2
        await run_debate(topic, rounds)

    except ValueError as exc:
        print(f"\nInvalid input: {exc}")

    except (RuntimeError, httpx.HTTPError) as exc:
        print(f"\nDebate failed: {exc}")
        print(
            "Check the agent terminals, Groq API key, model availability, and server logs."
        )


if __name__ == "__main__":
    asyncio.run(main())
