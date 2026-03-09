"""Generate curriculum stage environments using Claude.

Each call produces a complete Python file defining a CurriculumEnv class
that wraps the robosuite Lift task via RobosuiteGymEnv.
"""

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
            "continue_previous": {
                "type": "boolean",
                "description": "If true, continue training on the previous stage's environment instead of generating a new one. When true, code is ignored.",
            },
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
        "required": ["continue_previous", "code", "rationale", "timesteps"],
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


def generate_first_stage(cfg: dict, model: str = "claude-opus-4-6") -> dict:
    """Generate the first (easiest) curriculum stage."""
    system = _build_system_prompt(cfg)
    user = _load_prompt("curriculum_initial.txt")
    return _call_claude(system, user, model)


def generate_next_stage(
    stage_num: int,
    history: list[dict],
    cfg: dict,
    model: str = "claude-opus-4-6",
) -> dict:
    """Generate the next curriculum stage given previous results."""
    system = _build_system_prompt(cfg)
    template = _load_prompt("curriculum_next.txt")
    user = template.format(
        stage_num=stage_num,
        history=_format_history(history),
    )
    return _call_claude(system, user, model)
