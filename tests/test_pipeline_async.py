"""Exercise microphone-thread handoff without microphones or model downloads."""

import asyncio
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import threading
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch


def load_pipeline():
    # The ASR module loads GPU weights on import; substitute only model imports.
    stubs = {}
    for module_name, class_name in (
        ("src.speaker_classifier", "SpeakerClassifier"),
        ("src.llm_model", "LLMChat"),
        ("src.asr_model", "SpeechRecognizer"),
    ):
        module = ModuleType(module_name)
        setattr(module, class_name, object)
        stubs[module_name] = module
    # Audio conversion is mocked below; no native audio/tensor imports are needed.
    stubs["numpy"] = ModuleType("numpy")
    stubs["torch"] = ModuleType("torch")
    stubs["torch"].Tensor = object
    stubs["speech_recognition"] = ModuleType("speech_recognition")
    stubs["speech_recognition"].AudioSource = object
    stubs["speech_recognition"].Recognizer = object
    path = Path(__file__).parents[1] / "src" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with ExitStack() as cleanup:
        for name, stub in stubs.items():
            if name in sys.modules:
                cleanup.callback(sys.modules.__setitem__, name, sys.modules[name])
            else:
                cleanup.callback(sys.modules.pop, name, None)
            sys.modules[name] = stub
        spec.loader.exec_module(module)
    return module


class PipelineAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def run_phrases(self, authorization, fail_first=False):
        pipeline = load_pipeline()
        loops = []
        messages = []
        stop = Mock()
        listener_thread = None

        def start_listener(mic, callback, phrase_time_limit):
            nonlocal listener_thread
            def produce():
                for phrase in ("first", "second"):
                    callback(None, phrase)
            listener_thread = threading.Thread(target=produce)
            listener_thread.start()
            return stop

        async def llm_chat(text):
            loop = asyncio.get_running_loop()
            self.assertFalse(loop.is_closed())
            loops.append(loop)
            messages.append(text)
            if fail_first and len(messages) == 1:
                raise RuntimeError("temporary request failure")
            if text == "second":
                raise asyncio.CancelledError()

        assistant = pipeline.VoiceAssistant.__new__(pipeline.VoiceAssistant)
        assistant.recognizer = SimpleNamespace(listen_in_background=start_listener)
        assistant.mic = object()
        assistant.speaker_classifier = SimpleNamespace(
            authorized_speaker=Mock(side_effect=authorization)
        )
        assistant.asr_model = SimpleNamespace(transcribe=Mock(side_effect=lambda audio: audio))
        assistant.llm = SimpleNamespace(llm_chat=llm_chat)
        with (
            patch.object(pipeline, "convert_audio_to_tensor", side_effect=lambda audio: audio),
            patch.object(pipeline, "convert_audio_to_numpy", side_effect=lambda audio: audio),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(assistant._listen_async(), timeout=5)
        listener_thread.join(timeout=1)
        self.assertFalse(listener_thread.is_alive())
        stop.assert_called_once_with(wait_for_stop=True)
        return loops, messages, assistant

    async def test_consecutive_phrases_use_the_same_running_loop(self):
        loops, messages, _ = await self.run_phrases([True, True])
        self.assertEqual(messages, ["first", "second"])
        self.assertIs(loops[0], loops[1])
        self.assertIs(loops[0], asyncio.get_running_loop())

    async def test_request_failure_does_not_stop_next_phrase(self):
        with self.assertLogs(level="ERROR") as logs:
            loops, messages, _ = await self.run_phrases([True, True], fail_first=True)
        self.assertEqual(messages, ["first", "second"])
        self.assertIs(loops[0], loops[1])
        self.assertIn("temporary request failure", "\n".join(logs.output))

    async def test_rejected_speaker_does_not_reach_asr_or_llm(self):
        _, messages, assistant = await self.run_phrases([False, True])
        self.assertEqual(messages, ["second"])
        assistant.asr_model.transcribe.assert_called_once_with("second")


if __name__ == "__main__":
    unittest.main()
