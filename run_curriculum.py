"""Orchestrate curriculum learning with fixed-step stages and programmatic decisions.

Usage:
    python run_curriculum.py configs/lift_curriculum.yaml --output-dir results/curriculum
    python run_curriculum.py --resume results/curriculum
"""

import argparse
import shutil
import sys
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from evaluate import evaluate_checkpoint
from generate_curriculum import (
    append_error_to_messages,
    generate_first_stage,
    generate_next_stage,
)
from train_ppo import load_config, make_vec_env, train

# ── Constants ───────────────────────────────────────────────────────────────

STAGE_STEPS = 2_000_000
SOLVED_THRESHOLD = 0.8
TARGET_SR = 0.9
MAX_REWINDS = 2
MAX_TOTAL_STEPS = 50_000_000
MAX_STAGES = 20
MAX_RETRIES = 3
MODEL = "claude-opus-4-6"


# ── Analysis ────────────────────────────────────────────────────────────────


def analyze_training_curve(npz_path: str) -> dict:
    """Read evaluations.npz and return final SR + reward trend.

    Returns {"final_sr": float, "reward_trend": "increasing" | "flat"}.
    """
    data = np.load(npz_path)
    results = data["results"]       # shape: (n_evals, n_episodes)
    successes = data["successes"]   # shape: (n_evals, n_episodes)

    final_sr = float(np.mean(successes[-1]))
    mean_rewards = [float(np.mean(results[i])) for i in range(len(results))]

    # Reward trend: linear regression on last 30% of eval points
    n = len(mean_rewards)
    window = max(3, int(n * 0.3))
    recent = mean_rewards[-window:]

    if len(recent) < 2:
        return {"final_sr": final_sr, "reward_trend": "flat"}

    x = np.arange(len(recent), dtype=float)
    slope = np.polyfit(x, recent, 1)[0]

    # Normalize by reward range across all evals
    reward_range = max(mean_rewards) - min(mean_rewards)
    if reward_range < 1e-8:
        normalized_slope = 0.0
    else:
        normalized_slope = slope / reward_range

    trend = "increasing" if normalized_slope > 0.01 else "flat"
    return {"final_sr": final_sr, "reward_trend": trend}


def decide_action(final_sr: float, reward_trend: str, base_sr: float,
                   rewinds: int, total_steps: int) -> str:
    """Decide next action: 'harder', 'continue', 'easier', or 'done'."""
    if base_sr >= TARGET_SR:
        return "done"
    if rewinds >= MAX_REWINDS:
        return "done"
    if total_steps >= MAX_TOTAL_STEPS:
        return "done"
    if final_sr >= SOLVED_THRESHOLD:
        return "harder"
    if reward_trend == "increasing":
        return "continue"
    return "easier"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _write_results_yaml(stage_dir: Path, data: dict):
    """Write or update results.yaml in a stage directory."""
    results_file = stage_dir / "results.yaml"
    if results_file.exists():
        with open(results_file) as f:
            existing = yaml.safe_load(f) or {}
        existing.update(data)
        data = existing
    with open(results_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def _load_stage_history_entry(stage_dir: Path, stage_num: int) -> dict:
    """Load a completed stage's data as a history entry dict."""
    code = (stage_dir / "curriculum_env.py").read_text()

    results_file = stage_dir / "results.yaml"
    with open(results_file) as f:
        results = yaml.safe_load(f) or {}

    return {
        "stage_num": stage_num,
        "stage_dir": stage_dir,
        "code": code,
        "rationale": results.get("rationale", ""),
        "curriculum_sr": results.get("curriculum_sr"),
        "reward_trend": results.get("reward_trend"),
        "base_eval": results.get("base_eval"),
        "action": results.get("action"),
        "resumed_from": results.get("resumed_from"),
    }


def _detect_completed_stages(output_dir: Path) -> int:
    """Return the number of fully completed stages."""
    stage = 0
    while (output_dir / f"stage_{stage}" / "ppo_final.zip").exists():
        stage += 1
    return stage


def validate_curriculum_env(py_path: str, cfg: dict) -> str | None:
    """Smoke-test: build VecEnv + PPO and run one update cycle.

    Returns None on success or an error string on failure.
    """
    env = None
    try:
        from stable_baselines3 import PPO
        env = make_vec_env(cfg, n_envs=2, env_cls_path=py_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PPO(
                "MlpPolicy", env, device=cfg["device"],
                tensorboard_log=tmpdir,
                **cfg["ppo_kwargs"],
            )
            model.learn(total_timesteps=cfg["ppo_kwargs"]["n_steps"] * 2)
        return None
    except Exception:
        return traceback.format_exc()
    finally:
        if env is not None:
            env.close()


def _generate_and_validate(stage_dir: Path, generate_fn, cfg: dict,
                            llm_log_path: str) -> dict:
    """Call the LLM, validate the code, retry on failure. Returns the result dict.

    generate_fn should be a callable that takes (previous_messages, log_path) and
    returns (result_dict, messages).
    """
    attempts_dir = stage_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    previous_messages = None

    for attempt in range(MAX_RETRIES):
        attempt_dir = attempts_dir / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        try:
            result, previous_messages = generate_fn(previous_messages, llm_log_path)
        except Exception as e:
            print(f"  LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            (attempt_dir / "error.txt").write_text(traceback.format_exc())
            if attempt == MAX_RETRIES - 1:
                raise
            continue

        # Save attempt
        attempt_env_path = attempt_dir / "curriculum_env.py"
        attempt_env_path.write_text(result["code"])
        _write_results_yaml(attempt_dir, {"rationale": result["rationale"]})

        print(f"  Rationale: {result['rationale']}")

        # Validate
        print("  Validating environment...")
        error = validate_curriculum_env(str(attempt_env_path), cfg)
        if error is None:
            print("  Validation passed.")
            shutil.copy2(attempt_env_path, stage_dir / "curriculum_env.py")
            _write_results_yaml(stage_dir, {"rationale": result["rationale"]})
            return result

        print(f"  Validation failed (attempt {attempt + 1}/{MAX_RETRIES}):")
        print(f"  {error[:500]}")
        (attempt_dir / "error.txt").write_text(error)
        if attempt < MAX_RETRIES - 1:
            print("  Retrying with error feedback...")
            error_with_code = (
                f"Your submitted code:\n```python\n{result['code']}\n```\n\n"
                f"Validation error:\n{error}"
            )
            previous_messages = append_error_to_messages(previous_messages, error_with_code)
        else:
            print(f"  FATAL: Could not generate valid env after {MAX_RETRIES} attempts.")
            sys.exit(1)


# ── Main loop ───────────────────────────────────────────────────────────────


def run_curriculum(config_path: str | None, output_dir: str, resume: bool = False):
    output_dir = Path(output_dir)

    # Load config
    if resume:
        saved_config = output_dir / "base_config.yaml"
        assert saved_config.exists(), f"No base_config.yaml found in {output_dir}"
        cfg = load_config(str(saved_config))
    else:
        assert config_path is not None, "config_path required when not resuming"
        cfg = load_config(config_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, output_dir / "base_config.yaml")

    llm_log_path = str(output_dir / "llm_calls.log")

    # Load state from completed stages
    stage_history = []
    start_stage = _detect_completed_stages(output_dir)
    if start_stage > 0:
        print(f"Detected {start_stage} completed stage(s), resuming from stage {start_stage}")
        for s in range(start_stage):
            entry = _load_stage_history_entry(output_dir / f"stage_{s}", s)
            stage_history.append(entry)

    stage = start_stage
    rewinds = 0
    total_steps = stage * STAGE_STEPS
    # Figure out the initial action from the last stage's recorded action, or "first"
    if stage_history:
        last_action = stage_history[-1].get("action", "harder")
        action = last_action  # the action that was decided after the last completed stage
    else:
        action = "first"

    while action != "done" and stage < MAX_STAGES:
        stage_dir = output_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"STAGE {stage} — action: {action}")
        print(f"{'=' * 60}")

        # ── 1. Get curriculum env ───────────────────────────────────────
        if action == "first":
            print("Generating first stage environment...")

            def gen_fn(prev_msgs, log_path):
                return generate_first_stage(
                    cfg=cfg, model=MODEL,
                    previous_messages=prev_msgs, log_path=log_path,
                )
            _generate_and_validate(stage_dir, gen_fn, cfg, llm_log_path)
            resumed_from = None

        elif action == "continue":
            # Reuse the same curriculum env, resume from ppo_final
            prev_dir = stage_history[-1]["stage_dir"]
            shutil.copy2(prev_dir / "curriculum_env.py", stage_dir / "curriculum_env.py")
            _write_results_yaml(stage_dir, {"rationale": stage_history[-1]["rationale"]})
            resumed_from = {
                "ppo_path": str(prev_dir / "ppo_final"),
                "vec_norm_path": str(prev_dir / "vec_normalize.pkl"),
            }

        elif action == "harder":
            prev_dir = stage_history[-1]["stage_dir"]
            context = (
                f"This is stage {stage}. The previous curriculum (stage {stage - 1}) was SOLVED "
                f"(curriculum SR: {stage_history[-1].get('curriculum_sr', '?'):.0%}). "
                f"Make it harder — push closer to the full default task."
            )
            print(f"Generating harder environment...")

            def gen_fn(prev_msgs, log_path, _ctx=context):
                return generate_next_stage(
                    history=list(stage_history),
                    cfg=cfg, decision_context=_ctx, model=MODEL,
                    previous_messages=prev_msgs, log_path=log_path,
                )
            _generate_and_validate(stage_dir, gen_fn, cfg, llm_log_path)
            resumed_from = {
                "ppo_path": str(prev_dir / "best" / "best_model"),
                "vec_norm_path": str(prev_dir / "best" / "vec_normalize.pkl"),
            }

        elif action == "easier":
            prev_entry = stage_history[-1]
            context = (
                f"This is stage {stage}. The previous curriculum (stage {stage - 1}) was TOO HARD — "
                f"the agent couldn't learn it (reward trend: {prev_entry.get('reward_trend', '?')}, "
                f"curriculum SR: {prev_entry.get('curriculum_sr', 0):.0%}). "
                f"Generate an EASIER curriculum. Reduce difficulty significantly."
            )
            print(f"Generating easier environment (rewind {rewinds + 1}/{MAX_REWINDS})...")

            def gen_fn(prev_msgs, log_path, _ctx=context):
                return generate_next_stage(
                    history=list(stage_history),
                    cfg=cfg, decision_context=_ctx, model=MODEL,
                    previous_messages=prev_msgs, log_path=log_path,
                )
            _generate_and_validate(stage_dir, gen_fn, cfg, llm_log_path)
            # Rewind: resume from the same checkpoint the previous stage started from
            resumed_from = prev_entry.get("resumed_from")

        # ── 2. Train ────────────────────────────────────────────────────
        stage_cfg = deepcopy(cfg)
        stage_cfg["save_dir"] = str(stage_dir)
        stage_cfg["total_timesteps"] = STAGE_STEPS
        stage_cfg["early_stop"] = True

        env_cls_path = str(stage_dir / "curriculum_env.py")

        print(f"\n  Training for {STAGE_STEPS:,} steps...")
        if resumed_from:
            print(f"  Resuming from: {resumed_from['ppo_path']}")

        train(stage_cfg, config_path=None, resume_from=resumed_from, env_cls_path=env_cls_path)

        # ── 3. Analyze training curve ───────────────────────────────────
        eval_npz = stage_dir / "eval" / "evaluations.npz"
        if eval_npz.exists():
            curve = analyze_training_curve(str(eval_npz))
        else:
            curve = {"final_sr": 0.0, "reward_trend": "flat"}

        _write_results_yaml(stage_dir, {
            "curriculum_sr": curve["final_sr"],
            "reward_trend": curve["reward_trend"],
            "resumed_from": resumed_from,
        })

        # ── 4. Evaluate on base env ─────────────────────────────────────
        best_model = stage_dir / "best" / "best_model.zip"
        base_eval = {"success_rate": 0.0, "mean_reward": 0.0, "std_reward": 0.0}
        if best_model.exists():
            n_eval = cfg.get("n_eval_episodes", 50)
            print(f"\n  Evaluating on base environment ({n_eval} episodes)...")
            eval_results = evaluate_checkpoint(
                str(stage_dir / "best" / "best_model"),
                n_episodes=n_eval,
                config_override=str(output_dir / "base_config.yaml"),
            )
            base_eval = {
                "success_rate": float(eval_results["success_rate"]),
                "mean_reward": float(eval_results["mean_reward"]),
                "std_reward": float(eval_results["std_reward"]),
            }

        _write_results_yaml(stage_dir, {"base_eval": base_eval})

        # ── 5. Decide next action ───────────────────────────────────────
        total_steps += STAGE_STEPS
        action = decide_action(
            curve["final_sr"], curve["reward_trend"],
            base_eval["success_rate"], rewinds, total_steps,
        )
        if action == "easier":
            rewinds += 1

        _write_results_yaml(stage_dir, {"action": action})

        print(f"\n  Curriculum SR: {curve['final_sr']:.0%}, Reward trend: {curve['reward_trend']}")
        print(f"  Base SR: {base_eval['success_rate']:.0%}")
        print(f"  -> Next action: {action}")

        # ── 6. Update state ─────────────────────────────────────────────
        stage_history.append(_load_stage_history_entry(stage_dir, stage))
        stage += 1

    # ── Done ────────────────────────────────────────────────────────────────
    reason = action if action == "done" else f"max_stages ({MAX_STAGES})"
    best_sr = max((e.get("base_eval", {}).get("success_rate", 0) for e in stage_history), default=0)
    print(f"\n{'=' * 60}")
    print(f"Curriculum finished: {reason}")
    print(f"Stages: {stage}, Total steps: {total_steps:,}, Best base SR: {best_sr:.0%}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Run curriculum learning loop")
    parser.add_argument("config", nargs="?", default=None,
                        help="Path to base YAML config (not needed with --resume)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for all stages")
    parser.add_argument("--resume", type=str, default=None, metavar="DIR",
                        help="Resume from an existing output directory")
    args = parser.parse_args()

    if args.resume:
        if args.config:
            parser.error("config and --resume are mutually exclusive")
        output_dir = args.output_dir or args.resume
        run_curriculum(config_path=None, output_dir=output_dir, resume=True)
    else:
        if not args.config:
            parser.error("config is required when not using --resume")
        output_dir = args.output_dir or "results/curriculum"
        run_curriculum(config_path=args.config, output_dir=output_dir)


if __name__ == "__main__":
    main()
