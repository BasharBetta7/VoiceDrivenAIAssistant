from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit import HTML
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
import sys
import asyncio
import os
import logging
from pathlib import Path
import yaml
from src.TTS_model import TTSModel
import gc
import torch

os.environ["PROMPT_TOOLKIT_NO_CPR"] = "1"

from src.llm_model import LLMChat
from dotenv import load_dotenv
from src.events import *
STATUS_MSG_STYLE = "dim"
CONTENT_MSG_STYLE = "blue"
console = Console(file=sys.stdout)
status = console.status("", spinner_style='cyan', spinner='dots')

stream_display = None
stream_text = ""
VOICE_INPUT = object()



def configure_cli_logging():
    # SpeechBrain assigns levels to individual loggers as modules are imported.
    os.environ["SB_LOG_LEVEL"] = "ERROR"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    libraries = ("speechbrain", "transformers")
    for name in (*libraries, *tuple(logging.root.manager.loggerDict)):
        if any(name == library or name.startswith(library + ".") for library in libraries):
            logging.getLogger(name).setLevel(logging.ERROR)


def input_bindings():
    bindings = KeyBindings()

    @bindings.add("f2")
    def request_voice(event):
        event.app.exit(result=VOICE_INPUT)

    return bindings


def finish_stream() -> None:
    global stream_display, stream_text
    if stream_display is not None:
        stream_display.stop()
        stream_display = None
    stream_text = ""


def render_event(event: AgentEvent) -> None:
    global stream_display, stream_text, tts_model
    data = event["data"]
    if event["type"] == "message_delta":
        status.stop()
        stream_text += data["text"]
        if stream_display is None:
            stream_display = Live(
                console=console,
                auto_refresh=False,
                vertical_overflow="visible",
            )
            stream_display.start()
        stream_display.update(
            Markdown(stream_text, style=CONTENT_MSG_STYLE), refresh=True
        )
        return

    if event["type"] == "status":
        finish_stream()
        status.update(f"[dim]{data["text"]}[/dim]", spinner_style="dim")
        status.start()
    else:
        status.stop()

        if event["type"] == "message":
            if data.get("streamed") and stream_display is not None:
                finish_stream()  
                #tts_model.speak(data['content'])
                return
            content = data["content"]
            if isinstance(content, str):
                console.print(Markdown(content), style=CONTENT_MSG_STYLE)
            else:
                console.print(content, style=CONTENT_MSG_STYLE)
        elif event["type"] == "user_prompt":
            console.print(data["text"], markup=False)
        elif event["type"] == "error_message":
            console.print(data["text"], style="red", markup=False)
        elif event["type"] == "tool_started":
           pass

        elif event["type"] == "tool_result":
            pass




async def main():
    load_dotenv()
    configure_cli_logging()
   

   
    voice = None
    draft = Document()

    session = PromptSession(
        completer=WordCompleter(["/help","/new","/quit", "/chathistory"]),
        key_bindings=input_bindings(),
    )
    console.print("[bold cyan]Voice Assistant[/bold cyan]")
    console.print("Type a prompt, or press F2 to speak one command. /help for commands. Ctrl-D to exit.")
    async with agent.session():
        with patch_stdout(raw=True):
            while True:
                try:
                    result = await session.prompt_async(
                        HTML("<ansigreen><b>You &gt; </b></ansigreen>"), default=draft)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    draft = Document()
                    continue

                if result is VOICE_INPUT:
                    draft = session.default_buffer.document
                    try:
                        if voice is None:
                            console.print("Loading voice models. First use requires voice enrollment; follow the prompts.")
                            from src.pipeline import VoiceAssistant
                            render_event({
                                "type":"status",
                                "data":{"text":"Loading Speaker Verification Module..."}
                            })
                            voice = await asyncio.to_thread(VoiceAssistant, config, llm=agent)
                        text = await voice.transcribe_once(on_event=render_event)
                    except Exception as e:
                        console.print("Voice input failed:", str(e), markup=False)
                        continue
                    finally:
                        status.stop()
                    if not text:
                        continue
                else:
                    text = result.strip()
                    draft = Document()

                if not text:
                    continue
                if text == "/quit":
                    break
                if text == "/help":
                    console.print("F2 - listen for one voice command (stops after a pause).\nFirst use: load models and enroll your voice.\n/new - reset conversation.\n/quit - exit.\n/chathistory - get session history.\n/clear - clear screen.\n")
                    continue
                if text == "/new":
                    agent.flush_messages()
                    console.clear()
                    console.print("Started a new conversation.")
                    continue
                if text == "/chathistory":
                    console.print(Markdown(str(agent)))
                    continue
                if text == "/clear":
                    console.clear()
                    continue
                if text.startswith("/"):
                    console.print("Unkown command. Use /help.")
                    continue
                try:
                    await agent.llm_chat(message=text, on_event=render_event)
                except Exception as e:
                    finish_stream()
                    console.print("Request Failed:", str(e), markup=False)
                finally:
                    status.stop()
                    finish_stream()





if __name__ == "__main__":
    try:
        load_dotenv()
        with (Path(__file__).resolve().parents[1] / "config.yaml").open() as handle:
                config = yaml.safe_load(handle) or {}
        agent = LLMChat(config.get("llm_model"))
        tts_model = TTSModel(config.get('tts_model'))
        asyncio.run(main())
    finally:
        gc.collect()
        torch.cuda.empty_cache()
