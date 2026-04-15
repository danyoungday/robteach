"""Generate curriculum stage environments using Claude.

Each call produces a complete Python file defining a CurriculumEnv class
that wraps a robosuite task via RobosuiteGymEnv.
"""

import json
from datetime import datetime
from pathlib import Path
from string import Template

import anthropic

PROMPTS_DIR = Path(__file__).parent / "sysprompts"

CURRICULUM_TOOL = {
    "name": "submit_curriculum_env",
    "description": "Submit the curriculum environment code for this stage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python file with imports defining a CurriculumEnv class.",
            },
            "rationale": {
                "type": "string",
                "description": "Short explanation of the design choices for this stage.",
            },
        },
        "required": ["code", "rationale"],
    },
}


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def _build_system_prompt(cfg: dict) -> str:
    """Load curriculum_system.txt and substitute config variables."""
    raw = _load_prompt("curriculum_system.txt")

    # Read and escape source files ($ must be escaped for Template.substitute)
    base_env_source = Path(cfg["base_env_path"]).read_text().replace("$", "$$")
    wrapper_source = (Path(__file__).parent / "robosuite_env.py").read_text().replace("$", "$$")

    return Template(raw).substitute(
        env_name=cfg["env_name"],
        base_env_source=base_env_source,
        wrapper_source=wrapper_source,
    )


def _format_history(history: list[dict]) -> str:
    """Format stage history for the LLM: base-env trajectory, summaries, and code for most recent."""
    if not history:
        return "(no previous stages)"

    # Base-env SR trajectory
    base_rates = ["0%"]
    for entry in history:
        be = entry.get("base_eval")
        base_rates.append(f"{be['success_rate']:.0%}" if be else "?")

    parts = [f"**Base-env success rate: {' -> '.join(base_rates)}**"]

    for i, entry in enumerate(history):
        is_most_recent = (i == len(history) - 1)
        be = entry.get("base_eval", {})
        curr_sr = f"{be['success_rate']:.0%}" if be else "?"
        curriculum_sr = entry.get("curriculum_sr")
        curriculum_str = f"{curriculum_sr:.0%}" if curriculum_sr is not None else "?"

        if is_most_recent:
            section = (
                f"### Stage {entry['stage_num']}\n\n"
                f"Curriculum SR: {curriculum_str}, Base SR: {curr_sr}\n\n"
                f"**Rationale:** {entry['rationale']}\n\n"
                f"**Environment code:**\n```python\n{entry['code']}\n```"
            )
        else:
            prev_be = history[i - 1].get("base_eval", {}) if i > 0 else {}
            prev_sr = f"{prev_be['success_rate']:.0%}" if prev_be else "0%"
            section = (
                f"### Stage {entry['stage_num']}\n\n"
                f"**Rationale:** {entry['rationale']}\n"
                f"Curriculum SR: {curriculum_str}, Base SR: {prev_sr} -> {curr_sr}"
            )

        parts.append(section)

    return "\n\n---\n\n".join(parts)


def _format_content_for_log(content) -> str:
    """Convert a message's content field to a readable string for logging."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(json.dumps(block, indent=2, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _log_call(log_path: str | None, system: str, messages: list[dict],
              tool_result: dict, model: str):
    """Append a human-readable record of one LLM call to the log file."""
    if log_path is None:
        return
    with open(log_path, "a") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"LLM Call — {datetime.now().isoformat()} — model: {model}\n")
        f.write(f"{'=' * 80}\n\n")

        f.write("--- SYSTEM PROMPT ---\n")
        f.write(system)
        f.write("\n\n")

        f.write("--- MESSAGES ---\n")
        for msg in messages:
            f.write(f"[{msg['role']}]\n")
            f.write(_format_content_for_log(msg["content"]))
            f.write("\n\n")

        f.write("--- RESPONSE (tool call) ---\n")
        f.write(json.dumps(tool_result, indent=2))
        f.write("\n")


def _call_claude(
    system: str,
    user: str,
    model: str,
    messages: list[dict] | None = None,
    log_path: str | None = None,
) -> tuple[dict, list[dict]]:
    """Send a request to Claude and extract the tool call result.

    Returns (parsed tool input, full message history including the assistant reply).
    """
    if messages is None:
        messages = [{"role": "user", "content": user}]

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=messages,
        tools=[CURRICULUM_TOOL],
        tool_choice={"type": "tool", "name": "submit_curriculum_env"},
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_curriculum_env":
            _log_call(log_path, system, messages, block.input, model)
            updated_messages = messages + [
                {"role": "assistant", "content": response.content}
            ]
            return block.input, updated_messages
    raise RuntimeError("Claude did not return a tool call")


def append_error_to_messages(messages: list[dict], error: str) -> list[dict]:
    """Append a ``tool_result`` with ``is_error=True`` to the conversation."""
    last_assistant = messages[-1]
    tool_use_id = None
    for block in last_assistant["content"]:
        if getattr(block, "type", None) == "tool_use":
            tool_use_id = block.id
            break
    if tool_use_id is None:
        raise ValueError("No tool_use block found in last assistant message")

    return messages + [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "is_error": True,
                    "content": error,
                }
            ],
        }
    ]


def generate_first_stage(
    cfg: dict,
    model: str = "claude-opus-4-6",
    previous_messages: list[dict] | None = None,
    log_path: str | None = None,
) -> tuple[dict, list[dict]]:
    """Generate the first (easiest) curriculum stage."""
    system = _build_system_prompt(cfg)
    user = _load_prompt("curriculum_initial.txt")
    return _call_claude(system, user, model, messages=previous_messages, log_path=log_path)


def generate_next_stage(
    history: list[dict],
    cfg: dict,
    decision_context: str = "",
    model: str = "claude-opus-4-6",
    previous_messages: list[dict] | None = None,
    log_path: str | None = None,
) -> tuple[dict, list[dict]]:
    """Generate the next curriculum stage given previous results."""
    system = _build_system_prompt(cfg)
    template = _load_prompt("curriculum_next.txt")
    user = template.format(
        decision_context=decision_context,
        history=_format_history(history),
    )
    return _call_claude(system, user, model, messages=previous_messages, log_path=log_path)
