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
    append_error_to_messages,
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


def _load_stage_history_entry(stage_dir: Path, stage_num: int) -> dict:
    """Load a completed stage's data as a history entry dict."""
    code = (stage_dir / "curriculum_env.py").read_text()
    rationale = (stage_dir / "rationale.txt").read_text()
    eval_npz = stage_dir / "eval" / "evaluations.npz"
    ts_file = stage_dir / "timesteps.txt"
    requested_ts = int(ts_file.read_text().strip()) if ts_file.exists() else None
    stop_reason_file = stage_dir / "stop_reason.txt"
    stop_reason = stop_reason_file.read_text().strip() if stop_reason_file.exists() else None
    if eval_npz.exists():
        eval_summary = summarise_eval_log(str(eval_npz), requested_timesteps=requested_ts, stop_reason=stop_reason)
    else:
        eval_summary = "(no evaluation data available)"
    return {
        "stage_num": stage_num,
        "stage_dir": stage_dir,
        "code": code,
        "rationale": rationale,
        "eval_summary": eval_summary,
        "stop_reason": stop_reason,
    }


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


class CurriculumRunner:
    """Manages state and execution for a multi-stage curriculum run."""

    def __init__(
        self,
        config_path: str | None,
        n_stages: int,
        output_dir: str,
        model: str = "claude-opus-4-6",
        max_retries: int = 3,
        resume: bool = False,
    ):
        self.n_stages = n_stages
        self.model = model
        self.max_retries = max_retries

        # Mutable per-stage state
        self.output_dir = Path(output_dir)
        self.start_stage = 0
        self.stage_history: list[dict] = []

        # If we are resuming a run, detect completed stages and load this config and history.
        # Otherwise, load config from provided path and prepare an empty directory to save to.
        if resume:
            saved_config = self.output_dir / "base_config.yaml"
            assert saved_config.exists(), f"No base_config.yaml found in {self.output_dir}"
            self.base_cfg = load_config(str(saved_config))
            self.start_stage = _detect_completed_stages(self.output_dir)
            print(f"Detected {self.start_stage} completed stage(s), resuming from stage {self.start_stage}")
            if self.start_stage > 0:
                for s in range(self.start_stage):
                    entry = _load_stage_history_entry(self.output_dir / f"stage_{s}", s)
                    self.stage_history.append(entry)
        else:
            assert config_path is not None, "config_path is required when not resuming"
            self.base_cfg = load_config(config_path)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, self.output_dir / "base_config.yaml")

        self.llm_log_path = str(self.output_dir / "llm_calls.log")

    def _get_resume_from(self) -> dict | None:
        """Find the last non-stagnated stage checkpoint to resume from."""
        for entry in reversed(self.stage_history):
            if entry["stop_reason"] != "stagnated":
                best_dir = entry["stage_dir"] / "best"
                return {
                    "ppo_path": str(best_dir / "best_model"),
                    "vec_norm_path": str(best_dir / "vec_normalize.pkl"),
                }
        return None

    def run(self):
        """Execute the main stage loop."""
        for stage in range(self.start_stage, self.n_stages):
            stage_dir = self.output_dir / f"stage_{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"STAGE {stage}")
            print(f"{'='*60}")

            self._run_stage(stage, stage_dir)

            self.stage_history.append(_load_stage_history_entry(stage_dir, stage))

        print(f"\n{'='*60}")
        print(f"Curriculum complete! {self.n_stages} stages in {self.output_dir}")
        print(f"{'='*60}")

    def _generate_and_validate_code(
        self, stage: int, stage_dir: Path, previous_messages: list[dict] | None,
    ) -> tuple[dict, list[dict]]:
        """Run the LLM generate -> validate retry loop.

        Returns (result, previous_messages).  Calls sys.exit(1) if retries exhausted.
        """
        attempts_dir = stage_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)

        for attempt in range(self.max_retries):
            attempt_dir = attempts_dir / f"attempt_{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            try:
                if stage == 0:
                    print("Generating first stage environment...")
                    result, previous_messages = generate_first_stage(
                        cfg=self.base_cfg, model=self.model,
                        previous_messages=previous_messages,
                        log_path=self.llm_log_path,
                    )
                else:
                    print(f"Generating stage {stage} environment...")
                    result, previous_messages = generate_next_stage(
                        stage_num=stage,
                        history=list(self.stage_history),
                        cfg=self.base_cfg,
                        model=self.model,
                        previous_messages=previous_messages,
                        log_path=self.llm_log_path,
                    )
            except Exception as e:
                print(f"  LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                (attempt_dir / "error.txt").write_text(traceback.format_exc())
                if attempt == self.max_retries - 1:
                    raise
                continue

            # Save attempt files
            attempt_env_path = attempt_dir / "curriculum_env.py"
            attempt_env_path.write_text(result["code"])
            (attempt_dir / "rationale.txt").write_text(result["rationale"])
            (attempt_dir / "timesteps.txt").write_text(str(result["timesteps"]))

            print(f"  Rationale: {result['rationale']}")
            print(f"  Timesteps: {result['timesteps']:,}")

            print("  Validating environment...")
            error = validate_curriculum_env(str(attempt_env_path), self.base_cfg)
            if error is None:
                print("  Validation passed.")
                shutil.copy2(attempt_env_path, stage_dir / "curriculum_env.py")
                (stage_dir / "rationale.txt").write_text(result["rationale"])
                (stage_dir / "timesteps.txt").write_text(str(result["timesteps"]))
                break
            else:
                print(f"  Validation failed (attempt {attempt+1}/{self.max_retries}):")
                print(f"  {error[:500]}")
                (attempt_dir / "error.txt").write_text(error)
                if attempt < self.max_retries - 1:
                    print("  Retrying with error feedback...")
                    error_with_code = (
                        f"Your submitted code:\n```python\n{result['code']}\n```\n\n"
                        f"Validation error:\n{error}"
                    )
                    previous_messages = append_error_to_messages(
                        previous_messages, error_with_code
                    )
                else:
                    print(f"  FATAL: Could not generate valid env after {self.max_retries} attempts.")
                    sys.exit(1)

        return result, previous_messages

    def _train_stage(self, stage: int, stage_dir: Path, timesteps: int) -> str:
        """Deep-copy config, train, write stop_reason.txt.  Returns the stop reason."""
        stage_cfg = deepcopy(self.base_cfg)
        stage_cfg["save_dir"] = str(stage_dir)
        stage_cfg["total_timesteps"] = timesteps

        resume_from = self._get_resume_from()
        env_cls_path = str(stage_dir / "curriculum_env.py")

        print(f"\n  Training stage {stage} for {timesteps:,} timesteps...")
        if resume_from:
            print(f"  Resuming from: {resume_from['ppo_path']}")

        train_result = train(
            stage_cfg,
            config_path=None,
            resume_from=resume_from,
            env_cls_path=env_cls_path,
        )

        stop_reason = train_result["stop_reason"]
        (stage_dir / "stop_reason.txt").write_text(stop_reason)
        return stop_reason

    def _run_stage(self, stage: int, stage_dir: Path) -> None:
        """Generate, validate, and train a single stage."""
        result, _ = self._generate_and_validate_code(stage, stage_dir, previous_messages=None)
        self._train_stage(stage, stage_dir, result["timesteps"])


def run_curriculum(
    config_path: str | None,
    n_stages: int,
    output_dir: str,
    model: str = "claude-opus-4-6",
    max_retries: int = 3,
    resume: bool = False,
):
    runner = CurriculumRunner(
        config_path, n_stages, output_dir, model=model, max_retries=max_retries,
        resume=resume,
    )
    runner.run()


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
