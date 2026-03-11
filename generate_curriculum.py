"""Generate curriculum stage environments using Claude.

Each call produces a complete Python file defining a CurriculumEnv class
that wraps the robosuite Lift task via RobosuiteGymEnv.
"""

import json
from datetime import datetime
from pathlib import Path
from string import Template

import anthropic
import numpy as np

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
            "timesteps": {
                "type": "integer",
                "description": "Recommended total training timesteps for this stage. The system prompt shows how many policy updates this translates to given the current parallelism settings.",
            },
        },
        "required": ["code", "rationale", "timesteps"],
    },
}


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def _build_system_prompt(cfg: dict) -> str:
    """Load curriculum_system.txt and substitute training config variables."""
    raw = _load_prompt("curriculum_system.txt")
    n_envs = cfg["n_envs"]
    n_steps = cfg["ppo_kwargs"]["n_steps"]
    steps_per_update = n_envs * n_steps
    return Template(raw).substitute(
        n_envs=n_envs,
        n_steps=n_steps,
        batch_size=cfg["ppo_kwargs"]["batch_size"],
        n_epochs=cfg["ppo_kwargs"]["n_epochs"],
        horizon=cfg["env_kwargs"]["horizon"],
        steps_per_update=steps_per_update,
    )


def _format_history(history: list[dict]) -> str:
    """Format a list of stage history entries for the LLM prompt."""
    sections = []
    for entry in history:
        section = (
            f"### Stage {entry['stage_num']}\n\n"
            f"**Environment code:**\n```python\n{entry['code']}\n```\n\n"
            f"**Rationale:** {entry['rationale']}\n\n"
            f"**Evaluation results:**\n{entry['eval_summary']}\n"
        )
        sections.append(section)
    return "\n---\n\n".join(sections)


def summarise_eval_log(npz_path: str, requested_timesteps: int | None = None,
                       stop_reason: str | None = None) -> str:
    """Turn an evaluations.npz into a compact text summary for the LLM."""
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]
    lengths = data["ep_lengths"]

    successes = data["successes"]

    lines = [
        "## Previous training run — eval log",
        f"Eval episodes per checkpoint: {results.shape[1]}",
        f"Total timesteps trained: {int(timesteps[-1])}",
        "",
        "| timestep | mean_reward | std_reward | mean_ep_len | success_rate |",
        "|----------|-------------|------------|-------------|--------------|",
    ]
    for i, ts in enumerate(timesteps):
        mr = np.mean(results[i])
        sr = np.std(results[i])
        ml = np.mean(lengths[i])
        success_pct = np.mean(successes[i]) * 100
        lines.append(f"| {int(ts):>8d} | {mr:>11.2f} | {sr:>10.2f} | {ml:>11.0f} | {success_pct:>11.0f}% |")

    summary = "\n".join(lines)

    if requested_timesteps is not None and int(timesteps[-1]) < requested_timesteps:
        actual = int(timesteps[-1])
        if stop_reason == "success":
            summary += (
                f"\n\n**Early stopped (success)**: 100% success rate reached at "
                f"{actual:,} / {requested_timesteps:,} timesteps."
            )
        elif stop_reason == "stagnated":
            summary += (
                f"\n\n**Early stopped (stagnated)**: no reward improvement — "
                f"stopped at {actual:,} / {requested_timesteps:,} timesteps."
            )

    return summary


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
                # anthropic ContentBlock objects
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

    Parameters
    ----------
    system : str
        System prompt.
    user : str
        User prompt (ignored when *messages* is provided).
    model : str
        Claude model identifier.
    messages : list[dict] | None
        Optional pre-built message list (for retries with error context).
        When ``None``, a single user message is built from *user*.
    log_path : str | None
        If set, append a record of this call to the given text file.

    Returns
    -------
    tuple[dict, list[dict]]
        (parsed tool input, full message history including the assistant reply)
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
            # Append assistant response to history
            updated_messages = messages + [
                {"role": "assistant", "content": response.content}
            ]
            return block.input, updated_messages
    raise RuntimeError("Claude did not return a tool call")


def append_error_to_messages(messages: list[dict], error: str) -> list[dict]:
    """Append a ``tool_result`` with ``is_error=True`` to the conversation.

    The ``tool_use_id`` is extracted from the last assistant message so that
    the API sees the error as the result of the tool call it just made.
    """
    # Find the tool_use block in the last assistant message
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
    stage_num: int,
    history: list[dict],
    cfg: dict,
    model: str = "claude-opus-4-6",
    previous_messages: list[dict] | None = None,
    log_path: str | None = None,
) -> tuple[dict, list[dict]]:
    """Generate the next curriculum stage given previous results."""
    system = _build_system_prompt(cfg)
    template = _load_prompt("curriculum_next.txt")
    user = template.format(
        stage_num=stage_num,
        history=_format_history(history),
    )
    return _call_claude(system, user, model, messages=previous_messages, log_path=log_path)
