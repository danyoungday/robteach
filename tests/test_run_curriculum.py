"""Tests for run_curriculum.py."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from generate_curriculum import _build_system_prompt, append_error_to_messages
from run_curriculum import (
    _detect_completed_stages,
    _load_stage_history_entry,
    analyze_training_curve,
    decide_action,
    run_curriculum,
    SOLVED_THRESHOLD,
    TARGET_SR,
    MAX_REWINDS,
    MAX_TOTAL_STEPS,
)


# ── Helpers ────────────────────────────────────────────────────────────────

_FAKE_MESSAGES = [{"role": "user", "content": "fake"}]


def _gen_result(result_dict):
    """Wrap a result dict into the (dict, messages) tuple that generate_* returns."""
    return (result_dict, _FAKE_MESSAGES)


@pytest.fixture
def output_dir(tmp_path):
    """Create a fake curriculum output directory with a base config."""
    cfg = {
        "robots": "Panda",
        "env_name": "Lift",
        "base_env_path": "/dev/null",
        "env_kwargs": {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_object_obs": True,
            "use_camera_obs": False,
            "reward_shaping": True,
            "control_freq": 20,
            "horizon": 500,
            "hard_reset": False,
        },
        "n_envs": 2,
        "device": "cpu",
        "ppo_kwargs": {
            "learning_rate": 3.0e-4,
            "n_steps": 128,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.001,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1,
            "policy_kwargs": {
                "log_std_init": -2,
                "ortho_init": False,
                "activation_fn": "ReLU",
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        },
        "total_timesteps": 1_000_000,
        "save_dir": "unused",
        "checkpoint_freq": 50_000,
        "eval_freq": 50_000,
        "early_stop": False,
        "wandb": {"project": None, "entity": None},
    }
    with open(tmp_path / "base_config.yaml", "w") as f:
        yaml.dump(cfg, f)
    return tmp_path


def _make_completed_stage(output_dir: Path, stage: int, code: str = "",
                          rationale: str = "", base_eval: dict | None = None,
                          curriculum_sr: float = 0.0, reward_trend: str = "flat",
                          action: str = "harder", resumed_from: dict | None = None):
    """Create a completed stage directory with all expected files."""
    stage_dir = output_dir / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "ppo_final.zip").write_text("fake model")
    (stage_dir / "vec_normalize.pkl").write_text("fake norm")
    (stage_dir / "curriculum_env.py").write_text(code or f"# stage {stage} env code")

    best_dir = stage_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    (best_dir / "best_model.zip").write_text("fake best model")
    (best_dir / "vec_normalize.pkl").write_text("fake best norm")

    results = {
        "rationale": rationale or f"rationale for stage {stage}",
        "curriculum_sr": curriculum_sr,
        "reward_trend": reward_trend,
        "action": action,
        "resumed_from": resumed_from,
    }
    if base_eval is not None:
        results["base_eval"] = base_eval
    with open(stage_dir / "results.yaml", "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    eval_dir = stage_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)


def _make_eval_npz(tmp_dir, timesteps, results, successes):
    """Create a fake evaluations.npz and return its path."""
    path = Path(tmp_dir) / "evaluations.npz"
    np.savez(
        path,
        timesteps=np.array(timesteps),
        results=np.array(results),
        successes=np.array(successes),
    )
    return str(path)


# ── analyze_training_curve ──────────────────────────────────────────────────


class TestAnalyzeTrainingCurve:
    def test_increasing_reward(self, tmp_path):
        """Reward going up should return 'increasing'."""
        n_eps = 10
        timesteps = [100_000 * i for i in range(1, 11)]
        # Rewards steadily increasing from 10 to 100
        results = [[10 + 10 * i] * n_eps for i in range(10)]
        successes = [[False] * n_eps for _ in range(10)]

        npz_path = _make_eval_npz(tmp_path, timesteps, results, successes)
        curve = analyze_training_curve(npz_path)
        assert curve["reward_trend"] == "increasing"
        assert curve["final_sr"] == 0.0

    def test_flat_reward(self, tmp_path):
        """Flat reward should return 'flat'."""
        n_eps = 10
        timesteps = [100_000 * i for i in range(1, 11)]
        results = [[50.0] * n_eps for _ in range(10)]
        successes = [[False] * n_eps for _ in range(10)]

        npz_path = _make_eval_npz(tmp_path, timesteps, results, successes)
        curve = analyze_training_curve(npz_path)
        assert curve["reward_trend"] == "flat"

    def test_high_success_rate(self, tmp_path):
        """High SR at the end should be reflected in final_sr."""
        n_eps = 10
        timesteps = [100_000 * i for i in range(1, 6)]
        results = [[1.0] * n_eps for _ in range(5)]
        successes = [
            [False] * n_eps,
            [False] * n_eps,
            [True] * 5 + [False] * 5,
            [True] * 8 + [False] * 2,
            [True] * 9 + [False] * 1,
        ]

        npz_path = _make_eval_npz(tmp_path, timesteps, results, successes)
        curve = analyze_training_curve(npz_path)
        assert curve["final_sr"] == 0.9

    def test_decreasing_reward(self, tmp_path):
        """Decreasing reward should return 'flat' (not increasing)."""
        n_eps = 10
        timesteps = [100_000 * i for i in range(1, 11)]
        results = [[100 - 10 * i] * n_eps for i in range(10)]
        successes = [[False] * n_eps for _ in range(10)]

        npz_path = _make_eval_npz(tmp_path, timesteps, results, successes)
        curve = analyze_training_curve(npz_path)
        assert curve["reward_trend"] == "flat"

    def test_few_eval_points(self, tmp_path):
        """Should handle very few eval points without crashing."""
        n_eps = 10
        timesteps = [100_000]
        results = [[5.0] * n_eps]
        successes = [[False] * n_eps]

        npz_path = _make_eval_npz(tmp_path, timesteps, results, successes)
        curve = analyze_training_curve(npz_path)
        assert curve["reward_trend"] == "flat"
        assert curve["final_sr"] == 0.0


# ── decide_action ───────────────────────────────────────────────────────────


class TestDecideAction:
    def test_done_on_target_sr(self):
        assert decide_action(0.5, "increasing", TARGET_SR, 0, 0) == "done"

    def test_done_on_max_rewinds(self):
        assert decide_action(0.1, "flat", 0.0, MAX_REWINDS, 0) == "done"

    def test_done_on_max_steps(self):
        assert decide_action(0.5, "increasing", 0.3, 0, MAX_TOTAL_STEPS) == "done"

    def test_harder_when_solved(self):
        assert decide_action(SOLVED_THRESHOLD, "flat", 0.3, 0, 0) == "harder"

    def test_continue_when_improving(self):
        assert decide_action(0.1, "increasing", 0.0, 0, 0) == "continue"

    def test_easier_when_stuck(self):
        assert decide_action(0.1, "flat", 0.0, 0, 0) == "easier"

    def test_harder_beats_continue(self):
        """If solved AND reward increasing, should still go harder."""
        assert decide_action(SOLVED_THRESHOLD + 0.1, "increasing", 0.3, 0, 0) == "harder"

    def test_done_beats_harder(self):
        """If solved but base SR at target, should be done."""
        assert decide_action(SOLVED_THRESHOLD + 0.1, "increasing", TARGET_SR, 0, 0) == "done"


# ── _detect_completed_stages ────────────────────────────────────────────────


class TestDetectCompletedStages:
    def test_no_stages(self, output_dir):
        assert _detect_completed_stages(output_dir) == 0

    def test_one_completed(self, output_dir):
        _make_completed_stage(output_dir, 0)
        assert _detect_completed_stages(output_dir) == 1

    def test_multiple_completed(self, output_dir):
        for i in range(3):
            _make_completed_stage(output_dir, i)
        assert _detect_completed_stages(output_dir) == 3

    def test_gap_in_stages(self, output_dir):
        """If stage_0 and stage_2 exist but not stage_1, only count 1."""
        _make_completed_stage(output_dir, 0)
        _make_completed_stage(output_dir, 2)
        assert _detect_completed_stages(output_dir) == 1

    def test_incomplete_stage_not_counted(self, output_dir):
        """A stage dir without ppo_final.zip is not complete."""
        _make_completed_stage(output_dir, 0)
        stage_1 = output_dir / "stage_1"
        stage_1.mkdir(parents=True, exist_ok=True)
        (stage_1 / "curriculum_env.py").write_text("code")
        assert _detect_completed_stages(output_dir) == 1


# ── _load_stage_history_entry ────────────────────────────────────────────────


class TestLoadStageHistoryEntry:
    def test_loads_entry(self, output_dir):
        _make_completed_stage(output_dir, 0, code="code0", rationale="rat0",
                              curriculum_sr=0.85, reward_trend="increasing",
                              action="harder")
        entry = _load_stage_history_entry(output_dir / "stage_0", 0)
        assert entry["stage_num"] == 0
        assert entry["code"] == "code0"
        assert entry["rationale"] == "rat0"
        assert entry["curriculum_sr"] == 0.85
        assert entry["reward_trend"] == "increasing"
        assert entry["action"] == "harder"
        assert entry["base_eval"] is None

    def test_loads_entry_with_base_eval(self, output_dir):
        base_eval = {"success_rate": 0.45, "mean_reward": 3.2, "std_reward": 2.1}
        _make_completed_stage(output_dir, 0, base_eval=base_eval)
        entry = _load_stage_history_entry(output_dir / "stage_0", 0)
        assert entry["base_eval"]["success_rate"] == 0.45

    def test_missing_results_yaml_raises(self, output_dir):
        stage_dir = output_dir / "stage_0"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "curriculum_env.py").write_text("code")
        with pytest.raises(FileNotFoundError):
            _load_stage_history_entry(stage_dir, 0)


# ── append_error_to_messages ────────────────────────────────────────────────


class TestAppendErrorToMessages:
    def test_appends_tool_result_with_error(self):
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "toolu_123"

        messages = [
            {"role": "user", "content": "generate env"},
            {"role": "assistant", "content": [tool_use_block]},
        ]

        result = append_error_to_messages(messages, "ImportError: no module")

        assert len(messages) == 2  # original unchanged
        assert len(result) == 3
        block = result[2]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_123"
        assert block["is_error"] is True

    def test_raises_if_no_tool_use(self):
        text_block = MagicMock()
        text_block.type = "text"

        messages = [{"role": "assistant", "content": [text_block]}]

        with pytest.raises(ValueError, match="No tool_use block"):
            append_error_to_messages(messages, "some error")


# ── _build_system_prompt ────────────────────────────────────────────────────


class TestBuildSystemPrompt:
    def test_no_unresolved_placeholders(self):
        cfg = {
            "env_name": "Lift",
            "base_env_path": __file__,  # any readable file
        }
        prompt = _build_system_prompt(cfg)
        import re
        unresolved = re.findall(r"\$[a-z_]+", prompt)
        assert unresolved == [], f"Unresolved placeholders: {unresolved}"

    def test_contains_env_name(self):
        cfg = {
            "env_name": "Lift",
            "base_env_path": __file__,
        }
        prompt = _build_system_prompt(cfg)
        assert "Lift" in prompt


# ── Integration: run_curriculum resume ──────────────────────────────────────


class TestRunCurriculumResume:
    def test_resume_no_config_file_raises(self, tmp_path):
        with pytest.raises(AssertionError, match="No base_config.yaml"):
            run_curriculum(config_path=None, output_dir=str(tmp_path), resume=True)
