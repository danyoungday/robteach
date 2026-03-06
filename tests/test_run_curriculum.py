"""Tests for run_curriculum.py resume functionality."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from run_curriculum import _detect_completed_stages, _load_stage_state, run_curriculum


@pytest.fixture
def output_dir(tmp_path):
    """Create a fake curriculum output directory with a base config."""
    cfg = {
        "robots": "Panda",
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
        "wandb": {"project": None, "entity": None},
    }
    with open(tmp_path / "base_config.yaml", "w") as f:
        yaml.dump(cfg, f)
    return tmp_path


def _make_completed_stage(output_dir: Path, stage: int, code: str = "", rationale: str = ""):
    """Create a completed stage directory with all expected files."""
    stage_dir = output_dir / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "ppo_final.zip").write_text("fake model")
    (stage_dir / "vec_normalize.pkl").write_text("fake norm")
    (stage_dir / "curriculum_env.py").write_text(code or f"# stage {stage} env code")
    (stage_dir / "rationale.txt").write_text(rationale or f"rationale for stage {stage}")
    (stage_dir / "timesteps.txt").write_text("100000")
    # Create eval dir with fake npz
    eval_dir = stage_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)


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
        # stage_1 exists but has no ppo_final.zip
        stage_1 = output_dir / "stage_1"
        stage_1.mkdir(parents=True, exist_ok=True)
        (stage_1 / "curriculum_env.py").write_text("code")
        assert _detect_completed_stages(output_dir) == 1


# ── _load_stage_state ───────────────────────────────────────────────────────


class TestLoadStageState:
    def test_loads_code_and_rationale(self, output_dir):
        _make_completed_stage(output_dir, 0, code="class CurriculumEnv: ...", rationale="easy start")
        code, rationale = _load_stage_state(output_dir / "stage_0")
        assert code == "class CurriculumEnv: ..."
        assert rationale == "easy start"

    def test_missing_file_raises(self, output_dir):
        stage_dir = output_dir / "stage_0"
        stage_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            _load_stage_state(stage_dir)


# ── run_curriculum resume integration ───────────────────────────────────────


class TestRunCurriculumResume:
    @patch("run_curriculum.train")
    @patch("run_curriculum.validate_curriculum_env", return_value=None)
    @patch("run_curriculum.generate_next_stage")
    @patch("run_curriculum.generate_first_stage")
    @patch("run_curriculum.summarise_eval_log", return_value="eval summary")
    def test_resume_skips_completed_stages(
        self, mock_summarise, mock_gen_first, mock_gen_next, mock_validate, mock_train, output_dir
    ):
        """Resuming with 2 completed stages and n_stages=3 should only run stage 2."""
        _make_completed_stage(output_dir, 0, code="code0", rationale="rat0")
        _make_completed_stage(output_dir, 1, code="code1", rationale="rat1")

        mock_gen_next.return_value = {
            "code": "code2",
            "rationale": "rat2",
            "timesteps": 50000,
        }

        run_curriculum(
            config_path=None,
            n_stages=3,
            output_dir=str(output_dir),
            resume=True,
        )

        # Should not generate first stage (already done)
        mock_gen_first.assert_not_called()
        # Should generate exactly one next stage (stage 2)
        mock_gen_next.assert_called_once()
        call_kwargs = mock_gen_next.call_args
        assert call_kwargs.kwargs["stage_num"] == 2
        assert call_kwargs.kwargs["prev_code"] == "code1"
        assert call_kwargs.kwargs["prev_rationale"] == "rat1"
        # Should train exactly once
        mock_train.assert_called_once()

    @patch("run_curriculum.train")
    @patch("run_curriculum.validate_curriculum_env", return_value=None)
    @patch("run_curriculum.generate_next_stage")
    @patch("run_curriculum.generate_first_stage")
    def test_resume_all_complete_does_nothing(
        self, mock_gen_first, mock_gen_next, mock_validate, mock_train, output_dir
    ):
        """If all stages are already done, nothing runs."""
        _make_completed_stage(output_dir, 0)
        _make_completed_stage(output_dir, 1)

        run_curriculum(
            config_path=None,
            n_stages=2,
            output_dir=str(output_dir),
            resume=True,
        )

        mock_gen_first.assert_not_called()
        mock_gen_next.assert_not_called()
        mock_train.assert_not_called()

    @patch("run_curriculum.train")
    @patch("run_curriculum.validate_curriculum_env", return_value=None)
    @patch("run_curriculum.generate_first_stage")
    def test_resume_zero_stages_generates_first(
        self, mock_gen_first, mock_validate, mock_train, output_dir
    ):
        """Resuming with 0 completed stages generates from stage 0."""
        mock_gen_first.return_value = {
            "code": "code0",
            "rationale": "rat0",
            "timesteps": 50000,
        }

        run_curriculum(
            config_path=None,
            n_stages=1,
            output_dir=str(output_dir),
            resume=True,
        )

        mock_gen_first.assert_called_once()
        mock_train.assert_called_once()

    @patch("run_curriculum.train")
    @patch("run_curriculum.validate_curriculum_env", return_value=None)
    @patch("run_curriculum.generate_next_stage")
    @patch("run_curriculum.generate_first_stage")
    @patch("run_curriculum.summarise_eval_log", return_value="eval summary")
    def test_resume_passes_prev_stage_dir_for_training(
        self, mock_summarise, mock_gen_first, mock_gen_next, mock_validate, mock_train, output_dir
    ):
        """When resuming from stage 1, train should get resume_from pointing to stage_0."""
        _make_completed_stage(output_dir, 0, code="code0", rationale="rat0")

        mock_gen_next.return_value = {
            "code": "code1",
            "rationale": "rat1",
            "timesteps": 50000,
        }

        run_curriculum(
            config_path=None,
            n_stages=2,
            output_dir=str(output_dir),
            resume=True,
        )

        train_call = mock_train.call_args
        assert train_call.kwargs["resume_from"] == str(output_dir / "stage_0")

    def test_resume_no_config_file_raises(self, tmp_path):
        """Resuming from a dir without base_config.yaml should fail."""
        with pytest.raises(AssertionError, match="No base_config.yaml"):
            run_curriculum(
                config_path=None,
                n_stages=3,
                output_dir=str(tmp_path),
                resume=True,
            )

    @patch("run_curriculum.train")
    @patch("run_curriculum.validate_curriculum_env", return_value=None)
    @patch("run_curriculum.generate_next_stage")
    @patch("run_curriculum.generate_first_stage")
    @patch("run_curriculum.summarise_eval_log", return_value="eval summary")
    def test_resume_chains_stages_correctly(
        self, mock_summarise, mock_gen_first, mock_gen_next, mock_validate, mock_train, output_dir
    ):
        """Resume from 1 completed stage, run 2 more. Second new stage should use first new stage's state."""
        _make_completed_stage(output_dir, 0, code="code0", rationale="rat0")

        mock_gen_next.side_effect = [
            {"code": "code1", "rationale": "rat1", "timesteps": 50000},
            {"code": "code2", "rationale": "rat2", "timesteps": 60000},
        ]

        run_curriculum(
            config_path=None,
            n_stages=3,
            output_dir=str(output_dir),
            resume=True,
        )

        assert mock_gen_next.call_count == 2
        # First call uses state from completed stage_0
        first_call = mock_gen_next.call_args_list[0]
        assert first_call.kwargs["prev_code"] == "code0"
        assert first_call.kwargs["prev_rationale"] == "rat0"
        assert first_call.kwargs["stage_num"] == 1
        # Second call uses state from just-generated stage_1
        second_call = mock_gen_next.call_args_list[1]
        assert second_call.kwargs["prev_code"] == "code1"
        assert second_call.kwargs["prev_rationale"] == "rat1"
        assert second_call.kwargs["stage_num"] == 2
