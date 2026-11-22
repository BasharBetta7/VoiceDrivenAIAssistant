import asyncio
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage

from my_langsmith.config import ls_client
from my_langsmith.dataset import DATASET_NAME
from my_langsmith.judge import helpfulness
from src.llm_model import LLMChat

async def main():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}
    agent = LLMChat(config=config.get("llm_model"))

    async def target(inputs: dict) -> dict:
        agent.flush_messages()

        messages = await agent.llm_chat(inputs['question'])
        final = messages[-1]

        if not isinstance(final, AIMessage) or final.tool_calls:
            raise RuntimeError("Agent did not produce a final answer")
        return {"response": final.text}

    async with agent.session():
        await ls_client.aevaluate(
            target,
            data=DATASET_NAME,
            evaluators=[helpfulness],
            experiment_prefix="voice-agent-text-baseline",
            max_concurrency=1,
            metadata= {
                "model":config.get("llm_model", {}).get("model"),
                "scope":"text_agent",
            },
        )

if __name__ == '__main__':
    asyncio.run(main())
    