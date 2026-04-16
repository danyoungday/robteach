from datetime import datetime
import json
import re

import anthropic


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
        user: str,
    ) -> tuple[dict, list[dict]]:
        """
        Calls the Claude API with given user message and logs the interaction.
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
