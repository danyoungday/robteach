import anthropic

CURRICULUM_TOOL = {
    "name": "submit_curriculum_env",
    "description": "Submit the curriculum environment code for this stage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python file with imports defining a CurriculumEnv class.",
            },
            "rationale": {
                "type": "string",
                "description": "Short explanation of the design choices for this stage.",
            },
        },
        "required": ["code", "rationale"],
    },
}


class LLMAgent():
    def _call_claude(
        self,
        system: str,
        user: str,
        model: str,
        messages: list[dict] | None = None,
        log_path: str | None = None,
    ) -> tuple[dict, list[dict]]:
        """Send a request to Claude and extract the tool call result.

        Returns (parsed tool input, full message history including the assistant reply).
        """
        if messages is None:
            messages = [{"role": "user", "content": user}]

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=[CURRICULUM_TOOL],
            tool_choice={"type": "tool", "name": "submit_curriculum_env"},
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_curriculum_env":
                _log_call(log_path, system, messages, block.input, model)
                updated_messages = messages + [
                    {"role": "assistant", "content": response.content}
                ]
                return block.input, updated_messages
        raise RuntimeError("Claude did not return a tool call")