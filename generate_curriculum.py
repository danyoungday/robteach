"""Generate curriculum stage environments using Claude.

Each call produces a complete Python file defining a CurriculumEnv class
that wraps the robosuite Lift task via RobosuiteGymEnv.
"""

from pathlib import Path

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
                "description": "Recommended training timesteps for this stage.",
            },
        },
        "required": ["code", "rationale", "timesteps"],
    },
}


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def summarise_eval_log(npz_path: str) -> str:
    """Turn an evaluations.npz into a compact text summary for the LLM."""
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]
    lengths = data["ep_lengths"]

    lines = [
        "## Previous training run — eval log",
        f"Eval episodes per checkpoint: {results.shape[1]}",
        f"Total timesteps trained: {int(timesteps[-1])}",
        "",
        "| timestep | mean_reward | std_reward | mean_ep_len |",
        "|----------|-------------|------------|-------------|",
    ]
    for i, ts in enumerate(timesteps):
        mr = np.mean(results[i])
        sr = np.std(results[i])
        ml = np.mean(lengths[i])
        lines.append(f"| {int(ts):>8d} | {mr:>11.2f} | {sr:>10.2f} | {ml:>11.0f} |")

    return "\n".join(lines)


def _call_claude(system: str, user: str, model: str) -> dict:
    """Send a single request to Claude and extract the tool call result."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[CURRICULUM_TOOL],
        tool_choice={"type": "tool", "name": "submit_curriculum_env"},
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_curriculum_env":
            return block.input
    raise RuntimeError("Claude did not return a tool call")


def generate_first_stage(model: str = "claude-opus-4-6") -> dict:
    """Generate the first (easiest) curriculum stage."""
    system = _load_prompt("curriculum_system.txt")
    user = _load_prompt("curriculum_initial.txt")
    return _call_claude(system, user, model)


def generate_next_stage(
    stage_num: int,
    eval_summary: str,
    prev_code: str,
    prev_rationale: str,
    model: str = "claude-opus-4-6",
) -> dict:
    """Generate the next curriculum stage given previous results."""
    system = _load_prompt("curriculum_system.txt")
    template = _load_prompt("curriculum_next.txt")
    user = template.format(
        stage_num=stage_num,
        prev_code=prev_code,
        prev_rationale=prev_rationale,
        eval_summary=eval_summary,
    )
    return _call_claude(system, user, model)
