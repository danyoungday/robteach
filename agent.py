import base64
from datetime import datetime
import io
import json
import os
import re

import anthropic
import imageio
import numpy as np
from PIL import Image


class LLMAgent():
    """
    Simple wrapper around Claude API with logging of interactions.
    """
    def __init__(self, sysprompt_path: str, log_path: str):
        with open(sysprompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        self.log_path = log_path

        self.model = "claude-opus-4-6"

        self.client = anthropic.Anthropic()

        self.kill_switch = 0

    @staticmethod
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
                    parts.append(str(block))
            return "\n".join(parts)
        return str(content)

    def _log_call(self, messages: list[dict], response: str):
        """Append a human-readable record of one LLM call to the log file."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"LLM Call — {datetime.now().isoformat()}\n")
            f.write(f"{'=' * 80}\n\n")

            f.write("--- SYSTEM PROMPT ---\n")
            f.write(self.system_prompt)
            f.write("\n\n")

            f.write("--- MESSAGES ---\n")
            for msg in messages:
                f.write(f"[{msg['role']}]\n")
                f.write(self._format_content_for_log(msg["content"]))
                f.write("\n\n")

            f.write("--- RESPONSE ---\n")
            f.write(response)
            f.write("\n\n")

            f.write("\n")

    def call_claude(
        self,
        user,
    ) -> str:
        """
        Calls the Claude API with given user message and logs the interaction. `user` may be a
        plain string (text-only) or a list of Anthropic content blocks (for vision input).
        """
        # Construct messages
        messages = [{"role": "user", "content": user}]

        # Call API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            messages=messages
        )
        response_content = response.content[0].text

        # Log the output
        self._log_call(messages, response_content)

        self.kill_switch += 1
        if self.kill_switch >= 20:
            raise RuntimeError("Kill switch triggered: too many LLM calls. Check logs for details.")

        return response_content


class CurriculumAgent(LLMAgent):
    """
    Agent that generates curriculum by taking in training logs
    """
    def __init__(self, log_path: str):
        super().__init__(sysprompt_path="sysprompts/curriculum_system.txt", log_path=log_path)

    @staticmethod
    def format_metrics(training_metrics: list[dict]) -> str:
        """
        Convert a list of training metrics to a string readable by the LLM.
        Currently the training metrics are assumed to have keys:
            - stop_reason
            - success_rate
            - entropy_loss
            - value_loss
        """
        STOP_EXPLANATIONS = {
            "plateau": "the agent's performance plateaued and is no longer improving",
            "success": "the agent successfully completed the curriculum"
        }
        lines = []
        for i, metrics in enumerate(training_metrics):
            line = f"Curriculum {i+1} ended with reason: {STOP_EXPLANATIONS[metrics['stop_reason']]}\n"
            line += f"\tBest curriculum success rate: {metrics['success_rate']:.2%}\n"
            line += f"\tFinal entropy loss: {metrics['entropy_loss']:.4f}\n"
            line += f"\tFinal value loss: {metrics['value_loss']:.4f}\n"
            lines.append(line)
        return "\n\n".join(lines)

    def generate_curriculum(self, training_metrics: list[dict] | None, past_curriculum: list[str] | None) -> dict:
        """
        Generate curriculum based on training metrics. Returns parsed curriculum dict.
        """
        if training_metrics is None:
            user_msg = "This is the first stage of curriculum generation, so there are no training metrics or previous curriculum yet."

        else:
            assert len(training_metrics) == len(past_curriculum), "Length of training metrics and past curriculum must match"
            user_msg = "Past curriculum:\n"
            for i, curriculum in enumerate(past_curriculum):
                user_msg += f"```Curriculum {i+1}:\n{curriculum}\n```\n\n"

            training_metrics_str = self.format_metrics(training_metrics)
            user_msg += f"\nTraining metrics from past curriculum:\n{training_metrics_str}"

        response_content = self.call_claude(user_msg)
        return response_content

    def parse_curriculum_dict(self, response_content: str) -> dict:
        """
        Parses the output of the LLM into a curriculum dict.
        """
        # Parse the final json block from the response as the curriculum output
        json_blocks = re.findall(r"\{[\s\S]*\}", response_content)
        if not json_blocks:
            raise ValueError("No JSON block found in LLM response")
        curriculum_json = json_blocks[-1]  # Take the last JSON block in the response
        curriculum_dict = json.loads(curriculum_json)
        return curriculum_dict


class VideoCurriculumAgent(LLMAgent):
    """
    Curriculum agent that receives behavior video (not training metrics) and outputs a
    reward-weight dict. Uses a dedicated system prompt. Optionally writes the frames it sends
    to the LLM as an mp4 in video_log_dir so each call is debuggable after the fact.
    """
    def __init__(self, log_path: str,
                 sysprompt_path: str = "sysprompts/curriculum_video_system.txt",
                 video_log_dir: str | None = None,
                 video_log_fps: int = 2):
        super().__init__(sysprompt_path=sysprompt_path, log_path=log_path)
        self.video_log_dir = video_log_dir
        self.video_log_fps = video_log_fps
        if self.video_log_dir is not None:
            os.makedirs(self.video_log_dir, exist_ok=True)

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        """Encode a uint8 HxWx3 array as base64 PNG for the Anthropic vision API."""
        img = Image.fromarray(frame.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    def _build_user_content(self, past_curriculum, frames) -> list[dict]:
        """Build the Anthropic `content` list: past-curriculum text + interleaved image blocks."""
        blocks: list[dict] = []
        if past_curriculum:
            text = "Past curriculum (oldest first):\n\n"
            for i, curriculum in enumerate(past_curriculum):
                text += f"```Curriculum {i+1}:\n{curriculum}\n```\n\n"
            blocks.append({"type": "text", "text": text})
        else:
            blocks.append({"type": "text", "text": "This is the first call. No past curriculum."})

        if frames:
            blocks.append({
                "type": "text",
                "text": f"Below are {len(frames)} frames of the current policy's behavior, "
                        "in chronological order:",
            })
            for frame in frames:
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self._encode_frame(frame),
                    },
                })
        else:
            blocks.append({
                "type": "text",
                "text": "No behavior video is available (training has not started yet). "
                        "Output the simplest possible initial curriculum.",
            })

        return blocks

    def _log_video(self, frames: list) -> str | None:
        """Write `frames` as an mp4 to video_log_dir, returning the path (or None if disabled)."""
        if self.video_log_dir is None or not frames:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.video_log_dir, f"{ts}.mp4")
        imageio.mimsave(path, [f.astype(np.uint8) for f in frames], fps=self.video_log_fps)
        return path

    def generate_curriculum(self, past_curriculum=None, frames=None) -> str:
        """
        Generate curriculum from (optional) past curriculum and (optional) behavior video.
        Signature intentionally differs from the text-only parent: no training_metrics.
        """
        self._log_video(frames)
        content = self._build_user_content(past_curriculum, frames)
        return self.call_claude(content)

    def parse_response(self, response_content: str) -> dict:
        """
        Parse the LLM response into a reward-weights dict. Raises ValueError if no reward-weights
        JSON block (identified by the required 'reach_weight' key) is found.
        """
        blocks = re.findall(r"\{[\s\S]*?\}", response_content)
        for block in blocks:
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "reach_weight" in parsed:
                return parsed
        raise ValueError("No reward-weights JSON block (missing 'reach_weight') in response")
