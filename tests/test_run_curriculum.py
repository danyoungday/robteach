"""Tests for run_curriculum.py resume functionality."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from generate_curriculum import _build_system_prompt, append_error_to_messages, summarise_eval_log
from run_curriculum import (
    _detect_completed_stages,
    _load_stage_history_entry,
    _load_stage_state,
    run_curriculum,
)


# Helper: generate functions now return (dict, list) tuples
_FAKE_MESSAGES = [{"role": "user", "content": "fake"}]


def _gen_result(result_dict):
    """Wrap a result dict into the (dict, messages) tuple that generate_* returns."""
    return (result_dict, _FAKE_MESSAGES)


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


def _make_completed_stage(output_dir: Path, stage: int, code: str = "", rationale: str = "",
                          stop_reason: str = "completed"):
    """Create a completed stage directory with all expected files."""
    stage_dir = output_dir / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "ppo_final.zip").write_text("fake model")
    (stage_dir / "vec_normalize.pkl").write_text("fake norm")
    (stage_dir / "curriculum_env.py").write_text(code or f"# stage {stage} env code")
    (stage_dir / "rationale.txt").write_text(rationale or f"rationale for stage {stage}")
    (stage_dir / "timesteps.txt").write_text("100000")
    (stage_dir / "stop_reason.txt").write_text(stop_reason)
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


# ── _load_stage_history_entry ────────────────────────────────────────────────


class TestLoadStageHistoryEntry:
    def test_loads_entry_without_eval(self, output_dir):
        _make_completed_stage(output_dir, 0, code="code0", rationale="rat0")
        entry = _load_stage_history_entry(output_dir / "stage_0", 0)
        assert entry["stage_num"] == 0
        assert entry["stage_dir"] == output_dir / "stage_0"
        assert entry["code"] == "code0"
        assert entry["rationale"] == "rat0"
        assert entry["eval_summary"] == "(no evaluation data available)"
        assert entry["stop_reason"] == "completed"

    def test_missing_files_raises(self, output_dir):
        stage_dir = output_dir / "stage_0"
        stage_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            _load_stage_history_entry(stage_dir, 0)


# ── run_curriculum resume integration ───────────────────────────────────────


class TestRunCurriculumResume:
    @patch("run_curriculum.train", return_value={"stop_reason": "completed"})
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

        mock_gen_next.return_value = _gen_result({
            "code": "code2",
            "rationale": "rat2",
            "timesteps": 50000,
        })

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
        # Should receive full history as list of dicts
        history = call_kwargs.kwargs["history"]
        assert isinstance(history, list)
        assert len(history) == 2
        assert history[0]["code"] == "code0"
        assert history[1]["code"] == "code1"
        # Should train exactly once
        mock_train.assert_called_once()

    @patch("run_curriculum.train", return_value={"stop_reason": "completed"})
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

    @patch("run_curriculum.train", return_value={"stop_reason": "completed"})
    @patch("run_curriculum.validate_curriculum_env", return_value=None)
    @patch("run_curriculum.generate_first_stage")
    def test_resume_zero_stages_generates_first(
        self, mock_gen_first, mock_validate, mock_train, output_dir
    ):
        """Resuming with 0 completed stages generates from stage 0."""
        mock_gen_first.return_value = _gen_result({
            "code": "code0",
            "rationale": "rat0",
            "timesteps": 50000,
        })

        run_curriculum(
            config_path=None,
            n_stages=1,
            output_dir=str(output_dir),
            resume=True,
        )

        mock_gen_first.assert_called_once()
        mock_train.assert_called_once()

    @patch("run_curriculum.train", return_value={"stop_reason": "completed"})
    @patch("run_curriculum.validate_curriculum_env", return_value=None)
    @patch("run_curriculum.generate_next_stage")
    @patch("run_curriculum.generate_first_stage")
    @patch("run_curriculum.summarise_eval_log", return_value="eval summary")
    def test_resume_passes_prev_stage_dir_for_training(
        self, mock_summarise, mock_gen_first, mock_gen_next, mock_validate, mock_train, output_dir
    ):
        """When resuming from stage 1, train should get resume_from pointing to stage_0."""
        _make_completed_stage(output_dir, 0, code="code0", rationale="rat0")

        mock_gen_next.return_value = _gen_result({
            "code": "code1",
            "rationale": "rat1",
            "timesteps": 50000,
        })

        run_curriculum(
            config_path=None,
            n_stages=2,
            output_dir=str(output_dir),
            resume=True,
        )

        train_call = mock_train.call_args
        best_dir = output_dir / "stage_0" / "best"
        assert train_call.kwargs["resume_from"] == {
            "ppo_path": str(best_dir / "best_model"),
            "vec_norm_path": str(best_dir / "vec_normalize.pkl"),
        }

    def test_resume_no_config_file_raises(self, tmp_path):
        """Resuming from a dir without base_config.yaml should fail."""
        with pytest.raises(AssertionError, match="No base_config.yaml"):
            run_curriculum(
                config_path=None,
                n_stages=3,
                output_dir=str(tmp_path),
                resume=True,
            )

    @patch("run_curriculum.train", return_value={"stop_reason": "completed"})
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
            _gen_result({"code": "code1", "rationale": "rat1", "timesteps": 50000}),
            _gen_result({"code": "code2", "rationale": "rat2", "timesteps": 60000}),
        ]

        run_curriculum(
            config_path=None,
            n_stages=3,
            output_dir=str(output_dir),
            resume=True,
        )

        assert mock_gen_next.call_count == 2
        # First call uses history from completed stage_0
        first_call = mock_gen_next.call_args_list[0]
        history_0 = first_call.kwargs["history"]
        assert len(history_0) == 1
        assert history_0[0]["code"] == "code0"
        assert first_call.kwargs["stage_num"] == 1
        # Second call includes stage_0 + stage_1 history
        second_call = mock_gen_next.call_args_list[1]
        history_1 = second_call.kwargs["history"]
        assert len(history_1) == 2
        assert history_1[0]["code"] == "code0"
        assert history_1[1]["stage_num"] == 1
        assert second_call.kwargs["stage_num"] == 2


# ── cfg kwarg tests ──────────────────────────────────────────────────────────


class TestCfgPassedToGenerate:
    @patch("run_curriculum.train", return_value={"stop_reason": "completed"})
    @patch("run_curriculum.validate_curriculum_env", return_value=None)
    @patch("run_curriculum.generate_first_stage")
    def test_first_stage_receives_cfg(
        self, mock_gen_first, mock_validate, mock_train, output_dir
    ):
        """generate_first_stage should receive cfg kwarg."""
        mock_gen_first.return_value = _gen_result({
            "code": "code0",
            "rationale": "rat0",
            "timesteps": 50000,
        })

        run_curriculum(
            config_path=None,
            n_stages=1,
            output_dir=str(output_dir),
            resume=True,
        )

        call_kwargs = mock_gen_first.call_args
        assert "cfg" in call_kwargs.kwargs
        assert call_kwargs.kwargs["cfg"]["robots"] == "Panda"

    @patch("run_curriculum.train", return_value={"stop_reason": "completed"})
    @patch("run_curriculum.validate_curriculum_env", return_value=None)
    @patch("run_curriculum.generate_next_stage")
    @patch("run_curriculum.generate_first_stage")
    @patch("run_curriculum.summarise_eval_log", return_value="eval summary")
    def test_next_stage_receives_cfg(
        self, mock_summarise, mock_gen_first, mock_gen_next, mock_validate, mock_train, output_dir
    ):
        """generate_next_stage should receive cfg kwarg."""
        _make_completed_stage(output_dir, 0, code="code0", rationale="rat0")

        mock_gen_next.return_value = _gen_result({
            "code": "code1",
            "rationale": "rat1",
            "timesteps": 50000,
        })

        run_curriculum(
            config_path=None,
            n_stages=2,
            output_dir=str(output_dir),
            resume=True,
        )

        call_kwargs = mock_gen_next.call_args
        assert "cfg" in call_kwargs.kwargs
        assert call_kwargs.kwargs["cfg"]["robots"] == "Panda"


# ── _build_system_prompt tests ───────────────────────────────────────────────


class TestBuildSystemPrompt:
    def test_substitutes_variables(self):
        cfg = {
            "n_envs": 8,
            "ppo_kwargs": {"n_steps": 512, "batch_size": 1024, "n_epochs": 10},
            "env_kwargs": {"horizon": 500},
        }
        prompt = _build_system_prompt(cfg)
        assert "**n_envs** (parallel workers): 8" in prompt
        assert "**n_steps** (steps per env per update): 512" in prompt
        assert "**batch_size**: 1024" in prompt
        assert "**n_epochs** (SGD passes per update): 10" in prompt
        assert "**horizon** (max episode length): 500" in prompt
        assert "8 × 512 = **4096 timesteps**" in prompt

    def test_no_unresolved_placeholders(self):
        cfg = {
            "n_envs": 4,
            "ppo_kwargs": {"n_steps": 128, "batch_size": 256, "n_epochs": 5},
            "env_kwargs": {"horizon": 1000},
        }
        prompt = _build_system_prompt(cfg)
        # No leftover $variable placeholders
        import re
        unresolved = re.findall(r"\$[a-z_]+", prompt)
        assert unresolved == [], f"Unresolved placeholders: {unresolved}"


# ── append_error_to_messages tests ──────────────────────────────────────────


class TestAppendErrorToMessages:
    def test_appends_tool_result_with_error(self):
        """Should append a tool_result with is_error=True."""
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "toolu_123"

        messages = [
            {"role": "user", "content": "generate env"},
            {"role": "assistant", "content": [tool_use_block]},
        ]

        result = append_error_to_messages(messages, "ImportError: no module")

        # Original messages unchanged
        assert len(messages) == 2
        # New list has 3 entries
        assert len(result) == 3
        tool_result = result[2]
        assert tool_result["role"] == "user"
        block = tool_result["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_123"
        assert block["is_error"] is True
        assert block["content"] == "ImportError: no module"

    def test_raises_if_no_tool_use(self):
        """Should raise ValueError when no tool_use block found."""
        text_block = MagicMock()
        text_block.type = "text"

        messages = [
            {"role": "assistant", "content": [text_block]},
        ]

        with pytest.raises(ValueError, match="No tool_use block"):
            append_error_to_messages(messages, "some error")


# ── retry with error feedback tests ─────────────────────────────────────────


class TestRetryWithErrorFeedback:
    @patch("run_curriculum.train", return_value={"stop_reason": "completed"})
    @patch("run_curriculum.validate_curriculum_env")
    @patch("run_curriculum.generate_first_stage")
    def test_error_passed_on_retry(
        self, mock_gen_first, mock_validate, mock_train, output_dir
    ):
        """On validation failure, second call should receive previous_messages."""
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "toolu_abc"

        first_messages = [
            {"role": "user", "content": "initial"},
            {"role": "assistant", "content": [tool_use_block]},
        ]

        bad_result = {
            "code": "bad code",
            "rationale": "attempt 1",
            "timesteps": 50000,
        }
        good_result = {
            "code": "good code",
            "rationale": "attempt 2",
            "timesteps": 50000,
        }

        mock_gen_first.side_effect = [
            (bad_result, first_messages),
            (good_result, first_messages),
        ]
        # First validation fails, second passes
        mock_validate.side_effect = ["Traceback: some error", None]

        run_curriculum(
            config_path=None,
            n_stages=1,
            output_dir=str(output_dir),
            resume=True,
        )

        assert mock_gen_first.call_count == 2
        # First call: previous_messages should be None
        first_call = mock_gen_first.call_args_list[0]
        assert first_call.kwargs["previous_messages"] is None
        # Second call: previous_messages should be populated with error
        second_call = mock_gen_first.call_args_list[1]
        prev_msgs = second_call.kwargs["previous_messages"]
        assert prev_msgs is not None
        # Should have original messages + error tool_result
        assert len(prev_msgs) == 3
        error_msg = prev_msgs[-1]
        assert error_msg["content"][0]["is_error"] is True
        assert "some error" in error_msg["content"][0]["content"]

    @patch("run_curriculum.train", return_value={"stop_reason": "completed"})
    @patch("run_curriculum.validate_curriculum_env")
    @patch("run_curriculum.generate_next_stage")
    @patch("run_curriculum.generate_first_stage")
    @patch("run_curriculum.summarise_eval_log", return_value="eval summary")
    def test_error_passed_on_retry_next_stage(
        self, mock_summarise, mock_gen_first, mock_gen_next, mock_validate, mock_train, output_dir
    ):
        """Same as above but for generate_next_stage."""
        _make_completed_stage(output_dir, 0, code="code0", rationale="rat0")

        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "toolu_xyz"

        messages = [
            {"role": "user", "content": "next stage"},
            {"role": "assistant", "content": [tool_use_block]},
        ]

        mock_gen_next.side_effect = [
            ({"code": "bad", "rationale": "r1", "timesteps": 50000}, messages),
            ({"code": "good", "rationale": "r2", "timesteps": 50000}, messages),
        ]
        mock_validate.side_effect = ["error trace", None]

        run_curriculum(
            config_path=None,
            n_stages=2,
            output_dir=str(output_dir),
            resume=True,
        )

        assert mock_gen_next.call_count == 2
        first_call = mock_gen_next.call_args_list[0]
        assert first_call.kwargs["previous_messages"] is None
        second_call = mock_gen_next.call_args_list[1]
        assert second_call.kwargs["previous_messages"] is not None


# ── summarise_eval_log tests ─────────────────────────────────────────────────


def _make_eval_npz(tmp_dir, timesteps, results, ep_lengths, successes):
    """Create a fake evaluations.npz and return its path."""
    path = Path(tmp_dir) / "evaluations.npz"
    np.savez(
        path,
        timesteps=np.array(timesteps),
        results=np.array(results),
        ep_lengths=np.array(ep_lengths),
        successes=np.array(successes),
    )
    return str(path)


class TestSummariseEvalLog:
    def test_completed_zero_success_no_false_claim(self, tmp_path):
        """Completed run with 0% success must NOT claim 100% success."""
        n_checkpoints = 5
        n_eps = 10
        timesteps = [10000 * (i + 1) for i in range(n_checkpoints)]
        results = [[0.1] * n_eps for _ in range(n_checkpoints)]
        lengths = [[500] * n_eps for _ in range(n_checkpoints)]
        successes = [[False] * n_eps for _ in range(n_checkpoints)]

        npz_path = _make_eval_npz(tmp_path, timesteps, results, lengths, successes)
        # Last eval at 50000, requested 60000 — simulates normal completion
        summary = summarise_eval_log(npz_path, requested_timesteps=60000, stop_reason="completed")

        assert "100% success" not in summary
        assert "Early stopped" not in summary
        assert "success_rate" in summary
        assert "0%" in summary

    def test_success_stop_reason_annotates_correctly(self, tmp_path):
        """stop_reason='success' should annotate early stop for success."""
        n_checkpoints = 3
        n_eps = 10
        timesteps = [10000 * (i + 1) for i in range(n_checkpoints)]
        results = [[1.0] * n_eps for _ in range(n_checkpoints)]
        lengths = [[200] * n_eps for _ in range(n_checkpoints)]
        successes = [[True] * n_eps for _ in range(n_checkpoints)]

        npz_path = _make_eval_npz(tmp_path, timesteps, results, lengths, successes)
        summary = summarise_eval_log(npz_path, requested_timesteps=100000, stop_reason="success")

        assert "Early stopped (success)" in summary
        assert "100% success rate reached" in summary

    def test_stagnated_stop_reason_annotates_correctly(self, tmp_path):
        """stop_reason='stagnated' should annotate stagnation, not success."""
        n_checkpoints = 3
        n_eps = 10
        timesteps = [10000 * (i + 1) for i in range(n_checkpoints)]
        results = [[0.05] * n_eps for _ in range(n_checkpoints)]
        lengths = [[500] * n_eps for _ in range(n_checkpoints)]
        successes = [[False] * n_eps for _ in range(n_checkpoints)]

        npz_path = _make_eval_npz(tmp_path, timesteps, results, lengths, successes)
        summary = summarise_eval_log(npz_path, requested_timesteps=100000, stop_reason="stagnated")

        assert "Early stopped (stagnated)" in summary
        assert "100% success" not in summary

    def test_success_rate_column_values(self, tmp_path):
        """Success rate column should show correct percentages per checkpoint."""
        n_eps = 10
        timesteps = [10000, 20000, 30000]
        results = [[0.5] * n_eps] * 3
        lengths = [[400] * n_eps] * 3
        # 0%, 50%, 100% success rates
        successes = [
            [False] * n_eps,
            [True] * 5 + [False] * 5,
            [True] * n_eps,
        ]

        npz_path = _make_eval_npz(tmp_path, timesteps, results, lengths, successes)
        summary = summarise_eval_log(npz_path)

        lines = summary.split("\n")
        # Data rows start after header (6 header lines: title, episodes, timesteps, blank, header, separator)
        data_lines = [l for l in lines if l.startswith("|") and "timestep" not in l and "---" not in l]
        assert len(data_lines) == 3
        assert "0%" in data_lines[0]
        assert "50%" in data_lines[1]
        assert "100%" in data_lines[2]

    def test_none_stop_reason_no_annotation(self, tmp_path):
        """stop_reason=None with last eval < requested should not annotate."""
        n_eps = 5
        timesteps = [5000, 10000]
        results = [[0.1] * n_eps] * 2
        lengths = [[500] * n_eps] * 2
        successes = [[False] * n_eps] * 2

        npz_path = _make_eval_npz(tmp_path, timesteps, results, lengths, successes)
        summary = summarise_eval_log(npz_path, requested_timesteps=20000, stop_reason=None)

        assert "Early stopped" not in summary
        assert "100% success" not in summary
