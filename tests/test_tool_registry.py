"""Registry tests use disposable projects and never invoke an LLM."""

import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tool_registry import RegistryError, ScriptRegistry, build_registry_tools


SOURCE = '''def sum_values(a: float, b: float = 2) -> float:
    """Add two values and return their sum."""
    print("Adding values")
    return a + b
'''


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.registry = ScriptRegistry(self.root)

    def register(self, source=SOURCE, name="sum_values", entry="sum_values"):
        return self.registry.register(name, source, entry)

    def test_register_extracts_schema_without_executing_source(self):
        source = 'raise RuntimeError("must not run during registration")\n' + SOURCE
        metadata = self.register(source)
        self.assertEqual(metadata["status"], "pending_validation")
        self.assertEqual(metadata["description"], "Add two values and return their sum.")
        self.assertEqual(metadata["parameters"]["required"], ["a"])
        self.assertEqual(set(metadata["parameters"]["properties"]), {"a", "b"})
        self.assertEqual(metadata["return_type"], "float")
        self.assertTrue((self.root / metadata["path"]).is_file())

    def test_registry_survives_restart_and_returns_same_id(self):
        metadata = self.register()
        reloaded = ScriptRegistry(self.root)
        self.assertEqual(reloaded.search("sum")[0]["id"], metadata["id"])

    def test_pending_scripts_cannot_execute(self):
        metadata = self.register()
        with self.assertRaisesRegex(RegistryError, "not enabled"):
            self.registry.execute(metadata["id"], {"a": 1})

    def test_enabled_function_executes_with_default_and_captured_stdout(self):
        metadata = self.register()
        self.registry.enable_script(metadata["id"])
        self.assertEqual(
            self.registry.execute(metadata["id"], {"a": 3.0}),
            {"result": 5.0, "stdout": "Adding values\n"},
        )

    def test_bad_arguments_rejected(self):
        metadata = self.register()
        self.registry.enable_script(metadata["id"])
        for arguments in ({}, {"a": "3"}, {"a": 3, "extra": 2}):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(RegistryError, "Invalid arguments"):
                self.registry.execute(metadata["id"], arguments)

    def test_modified_source_rejected(self):
        metadata = self.register()
        self.registry.enable_script(metadata["id"])
        (self.root / metadata["path"]).write_text(SOURCE + "\n# modified")
        with self.assertRaisesRegex(RegistryError, "changed after registration"):
            self.registry.execute(metadata["id"], {"a": 1})

    def test_duplicate_names_and_paths_do_not_overwrite(self):
        metadata = self.register()
        with self.assertRaisesRegex(RegistryError, "already registered"):
            self.register()
        with self.assertRaisesRegex(RegistryError, "already exists"):
            self.register(SOURCE.replace("sum_values", "another_function"), entry="another_function")
        self.assertEqual((self.root / metadata["path"]).read_text(), SOURCE)
        self.assertEqual(len(self.registry.search("")), 1)

    def test_invalid_source_and_paths_are_rejected(self):
        cases = [
            ("../escape", SOURCE, "sum_values"),
            ("sum_values", "def broken(", "broken"),
            ("sum_values", SOURCE, "missing"),
            ("sum_values", SOURCE.replace("a: float", "a"), "sum_values"),
            ("sum_values", SOURCE.replace("a: float", "a: object"), "sum_values"),
        ]
        for name, source, entry in cases:
            with self.subTest(name=name, source=source), self.assertRaises(RegistryError):
                self.register(source, name=name, entry=entry)

    def test_registry_write_failure_rolls_back_script_and_preserves_registry(self):
        self.register()
        before = self.registry.registry_path.read_bytes()
        with patch("src.tool_registry.os.replace", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                self.register(SOURCE.replace("sum_values", "second"), name="second", entry="second")
        self.assertFalse((self.registry.scripts_dir / "second.py").exists())
        self.assertEqual(self.registry.registry_path.read_bytes(), before)
        self.assertEqual(list(self.registry.registry_path.parent.glob("*.tmp")), [])

    def test_execution_timeout(self):
        metadata = self.register('''import time
def wait_forever() -> None:
    """Wait longer than the permitted execution time."""
    time.sleep(5)
''', name="wait_forever", entry="wait_forever")
        self.registry.enable_script(metadata["id"])
        with self.assertRaisesRegex(RegistryError, "timeout"):
            self.registry.execute(metadata["id"], {}, timeout=0.1)

    def test_langchain_tools_register_search_and_execute(self):
        tools = {item.name: item for item in build_registry_tools(self.registry)}
        metadata = tools["write_script_tool"].invoke({
            "script_name": "sum_values", "source": SOURCE, "entry_point": "sum_values",
        })
        self.assertEqual(tools["search_scripts"].invoke({"query": "sum"})[0]["id"], metadata["id"])
        self.registry.enable_script(metadata["id"])
        result = tools["execute_script"].invoke({"script_id": metadata["id"], "arguments": {"a": 4}})
        self.assertEqual(result["result"], 6)
        self.assertIn("error", tools["execute_script"].invoke({"script_id": "missing", "arguments": {}}))

    def test_existing_legacy_entries_are_preserved(self):
        self.registry.registry_path.parent.mkdir()
        legacy = {"id": "old-id", "name": "old-tool", "parameters": []}
        self.registry.registry_path.write_text(json.dumps({"tools": [legacy]}))
        self.register()
        self.assertEqual(self.registry.search("old")[0], legacy)

    def test_notebook_agent_can_search_then_execute_and_report(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        metadata = self.register()
        self.registry.enable_script(metadata["id"])
        notebook = json.loads((Path(__file__).parents[1] / "baseline_pipeline.ipynb").read_text())
        source = next(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
            if "def llm_chat(" in "".join(cell.get("source", []))
        )
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "llm_chat")
        replies = [
            AIMessage(content="I will find the tool.", tool_calls=[{
                "name": "search_scripts", "args": {"query": "sum"}, "id": "search-1",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "execute_script", "args": {"script_id": metadata["id"], "arguments": {"a": 3}}, "id": "execute-1",
            }]),
            AIMessage(content="The sum is 5."),
        ]

        class FakeChat:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                return replies.pop(0)

        scope = {
            "chat": FakeChat(), "SystemPrompt": "Use registered tools.",
            "SystemMessage": SystemMessage, "ToolMessage": ToolMessage, "json": json,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook-agent>", "exec"), scope)
        with contextlib.redirect_stdout(io.StringIO()):
            result = scope["llm_chat"]([HumanMessage(content="Add 3 and 2.")], build_registry_tools(self.registry))
        observations = [message for message in result if isinstance(message, ToolMessage)]
        self.assertEqual(len(observations), 2)
        self.assertEqual(json.loads(observations[1].content)["result"], 5)
        self.assertEqual(result[-1].content, "The sum is 5.")


if __name__ == "__main__":
    unittest.main()
