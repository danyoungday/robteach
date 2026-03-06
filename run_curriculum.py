"""Orchestrate a multi-stage curriculum learning loop.

Usage:
    python run_curriculum.py configs/basic.yaml --stages 6 --output-dir results/curriculum
    python run_curriculum.py --resume results/curriculum --stages 8
"""

import argparse
import shutil
import sys
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path

from generate_curriculum import (
    generate_first_stage,
    generate_next_stage,
    summarise_eval_log,
)
from train_ppo import load_config, make_vec_env, train


def _detect_completed_stages(output_dir: Path) -> int:
    """Return the number of fully completed stages."""
    stage = 0
    while (output_dir / f"stage_{stage}" / "ppo_final.zip").exists():
        stage += 1
    return stage


def _load_stage_state(stage_dir: Path) -> tuple[str, str]:
    """Read prev_code and prev_rationale from a completed stage."""
    code = (stage_dir / "curriculum_env.py").read_text()
    rationale = (stage_dir / "rationale.txt").read_text()
    return code, rationale


def validate_curriculum_env(py_path: str, cfg: dict) -> str | None:
    """Smoke-test: build VecEnv + PPO and run one update cycle.

    Returns None on success or an error string on failure.
    """
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
        env.close()
        return None
    except Exception:
        return traceback.format_exc()


def run_curriculum(
    config_path: str | None,
    n_stages: int,
    output_dir: str,
    model: str = "claude-opus-4-6",
    max_retries: int = 3,
    resume: bool = False,
):
    output_dir = Path(output_dir)
    start_stage = 0
    prev_stage_dir = None
    prev_code = None
    prev_rationale = None

    if resume:
        saved_config = output_dir / "base_config.yaml"
        assert saved_config.exists(), f"No base_config.yaml found in {output_dir}"
        base_cfg = load_config(str(saved_config))
        start_stage = _detect_completed_stages(output_dir)
        print(f"Detected {start_stage} completed stage(s), resuming from stage {start_stage}")
        if start_stage > 0:
            last_stage_dir = output_dir / f"stage_{start_stage - 1}"
            prev_code, prev_rationale = _load_stage_state(last_stage_dir)
            prev_stage_dir = last_stage_dir
    else:
        assert config_path is not None, "config_path is required when not resuming"
        base_cfg = load_config(config_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, output_dir / "base_config.yaml")

    for stage in range(start_stage, n_stages):
        stage_dir = output_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"STAGE {stage}")
        print(f"{'='*60}")

        # Generate environment code
        attempts_dir = stage_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)

        for attempt in range(max_retries):
            attempt_dir = attempts_dir / f"attempt_{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            try:
                if stage == 0:
                    print("Generating first stage environment...")
                    result = generate_first_stage(model=model)
                else:
                    eval_npz = prev_stage_dir / "eval" / "evaluations.npz"
                    eval_summary = summarise_eval_log(str(eval_npz))
                    print(f"Generating stage {stage} environment...")
                    result = generate_next_stage(
                        stage_num=stage,
                        eval_summary=eval_summary,
                        prev_code=prev_code,
                        prev_rationale=prev_rationale,
                        model=model,
                    )
            except Exception as e:
                print(f"  LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
                (attempt_dir / "error.txt").write_text(traceback.format_exc())
                if attempt == max_retries - 1:
                    raise
                continue

            # Save attempt files
            attempt_env_path = attempt_dir / "curriculum_env.py"
            attempt_env_path.write_text(result["code"])
            (attempt_dir / "rationale.txt").write_text(result["rationale"])
            (attempt_dir / "timesteps.txt").write_text(str(result["timesteps"]))

            print(f"  Rationale: {result['rationale']}")
            print(f"  Timesteps: {result['timesteps']:,}")

            # Validate
            print("  Validating environment...")
            error = validate_curriculum_env(str(attempt_env_path), base_cfg)
            if error is None:
                print("  Validation passed.")
                # Copy winning attempt to stage dir
                shutil.copy2(attempt_env_path, stage_dir / "curriculum_env.py")
                (stage_dir / "rationale.txt").write_text(result["rationale"])
                (stage_dir / "timesteps.txt").write_text(str(result["timesteps"]))
                break
            else:
                print(f"  Validation failed (attempt {attempt+1}/{max_retries}):")
                print(f"  {error[:500]}")
                (attempt_dir / "error.txt").write_text(error)
                if attempt < max_retries - 1:
                    print("  Retrying with error feedback...")
                else:
                    print(f"  FATAL: Could not generate valid env after {max_retries} attempts.")
                    sys.exit(1)

        # Train
        stage_cfg = deepcopy(base_cfg)
        stage_cfg["save_dir"] = str(stage_dir)
        stage_cfg["total_timesteps"] = result["timesteps"]

        resume_from = str(prev_stage_dir) if prev_stage_dir is not None else None
        env_cls_path = str(stage_dir / "curriculum_env.py")

        print(f"\n  Training stage {stage} for {result['timesteps']:,} timesteps...")
        if resume_from:
            print(f"  Resuming from: {resume_from}")

        train(
            stage_cfg,
            config_path=None,
            resume_from=resume_from,
            env_cls_path=env_cls_path,
        )

        prev_stage_dir = stage_dir
        prev_code = result["code"]
        prev_rationale = result["rationale"]

    print(f"\n{'='*60}")
    print(f"Curriculum complete! {n_stages} stages in {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Run curriculum learning loop")
    parser.add_argument("config", nargs="?", default=None,
                        help="Path to base YAML config (not needed with --resume)")
    parser.add_argument("--stages", type=int, default=2, help="Total number of curriculum stages")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for all stages")
    parser.add_argument("--model", type=str, default="claude-opus-4-6",
                        help="Claude model to use for generation")
    parser.add_argument("--resume", type=str, default=None, metavar="DIR",
                        help="Resume from an existing output directory")
    args = parser.parse_args()

    if args.resume:
        if args.config:
            parser.error("config and --resume are mutually exclusive")
        output_dir = args.output_dir or args.resume
        run_curriculum(
            config_path=None,
            n_stages=args.stages,
            output_dir=output_dir,
            model=args.model,
            resume=True,
        )
    else:
        if not args.config:
            parser.error("config is required when not using --resume")
        output_dir = args.output_dir or "results/curriculum"
        run_curriculum(
            config_path=args.config,
            n_stages=args.stages,
            output_dir=output_dir,
            model=args.model,
        )


if __name__ == "__main__":
    main()
