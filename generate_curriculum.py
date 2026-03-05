"""Generate a training curriculum for the Lift task using Claude.

Reads eval logs from a previous PPO run and asks Claude to produce a
sequence of (cube_x_range, cube_y_range) stages that gradually broaden
the cube's spawn region.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python generate_curriculum.py logs/eval/evaluations.npz
    python generate_curriculum.py logs/eval/evaluations.npz --stages 5 --out curriculum.json
"""

import argparse
import json
from pathlib import Path

import anthropic
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a reinforcement learning curriculum designer for a robotic manipulation task.

A Panda robot arm is learning to pick up a cube from a table (the "Lift" task in robosuite).
The cube's starting position is sampled uniformly from a rectangular region on the table,
defined by cube_x_range and cube_y_range (each a [min, max] pair in metres, relative to
the table centre).  The table is roughly 0.8 m wide; practical spawn bounds are about
[-0.15, 0.15] in each axis before the cube risks falling off.

Your job: given training logs from a prior run at a single difficulty, produce a
curriculum — a sequence of stages that gradually widen the spawn region so the
policy can generalise to picking up the cube from anywhere on the table.

Guidelines:
- Start from a range equal to or slightly wider than the previous training run.
- Each stage should be a modest expansion — big jumps cause catastrophic forgetting.
- Include the number of training timesteps for each stage.
- Earlier (easier) stages can be shorter; later (harder) stages should be longer.
- The final stage should cover most of the reachable table surface.
- Use the provided tool to return your curriculum.\
"""

CURRICULUM_TOOL = {
    "name": "submit_curriculum",
    "description": "Submit the generated curriculum stages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "stages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "stage": {"type": "integer", "description": "1-indexed stage number"},
                        "cube_x_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "[min, max] x offset in metres",
                        },
                        "cube_y_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "[min, max] y offset in metres",
                        },
                        "timesteps": {"type": "integer", "description": "Training steps for this stage"},
                        "rationale": {"type": "string", "description": "Short explanation"},
                    },
                    "required": ["stage", "cube_x_range", "cube_y_range", "timesteps", "rationale"],
                },
            }
        },
        "required": ["stages"],
    },
}


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


def generate_curriculum(
    npz_path: str,
    n_stages: int = 5,
    current_x_range: tuple = (0.0, 0.0),
    current_y_range: tuple = (0.0, 0.0),
    model: str = "claude-opus-4-6",
) -> dict:
    """Call Claude to generate a curriculum from eval logs."""
    client = anthropic.Anthropic()

    summary = summarise_eval_log(npz_path)

    user_msg = (
        f"{summary}\n\n"
        f"The run above used:\n"
        f"  cube_x_range = {list(current_x_range)}\n"
        f"  cube_y_range = {list(current_y_range)}\n\n"
        f"Generate a curriculum with {n_stages} stages that progressively widens "
        f"the spawn range toward [-0.15, 0.15] in both axes. "
        f"Use the submit_curriculum tool to return your answer."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        tools=[CURRICULUM_TOOL],
        tool_choice={"type": "tool", "name": "submit_curriculum"},
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_curriculum":
            return block.input

    raise RuntimeError("Claude did not return a tool call")


# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate a training curriculum via Claude")
    parser.add_argument("eval_log", help="Path to evaluations.npz from a previous run")
    parser.add_argument("--stages", type=int, default=5, help="Number of curriculum stages")
    parser.add_argument("--x-range", type=float, nargs=2, default=[0.0, 0.0],
                        help="cube_x_range used in the previous run")
    parser.add_argument("--y-range", type=float, nargs=2, default=[0.0, 0.0],
                        help="cube_y_range used in the previous run")
    parser.add_argument("--out", type=str, default="curriculum.json",
                        help="Output path for the curriculum JSON")
    args = parser.parse_args()

    print(f"Reading eval log: {args.eval_log}")
    curriculum = generate_curriculum(
        args.eval_log,
        n_stages=args.stages,
        current_x_range=tuple(args.x_range),
        current_y_range=tuple(args.y_range),
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(curriculum, indent=2))
    print(f"\nCurriculum ({len(curriculum['stages'])} stages):")
    for s in curriculum["stages"]:
        print(f"  Stage {s['stage']}: x={s['cube_x_range']}  y={s['cube_y_range']}  "
              f"steps={s['timesteps']:,}  — {s['rationale']}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
