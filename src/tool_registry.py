"""Persistent registration and execution of generated Python function tools."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import keyword
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Union
from uuid import uuid4

from pydantic import ConfigDict, ValidationError, create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RegistryError(ValueError):
    """An actionable registration, validation, or execution failure."""


def _annotation(node):
    """Resolve a small JSON-compatible type vocabulary without eval()."""
    primitives = {"str": str, "int": int, "float": float, "bool": bool}
    if isinstance(node, ast.Constant):
        if node.value is None:
            return type(None)
        if isinstance(node.value, str):
            return _annotation(ast.parse(node.value, mode="eval").body)
    if isinstance(node, ast.Name) and node.id in primitives:
        return primitives[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return Union[_annotation(node.left), _annotation(node.right)]
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id == "list":
            return list[_annotation(node.slice)]
        if node.value.id == "dict" and isinstance(node.slice, ast.Tuple):
            keys, values = node.slice.elts
            if _annotation(keys) is str:
                return dict[str, _annotation(values)]
    raise RegistryError(
        "Use JSON-compatible annotations: str, int, float, bool, list[T], "
        "dict[str, T], or unions such as str | None."
    )


def extract_source_metadata(source: str, entry_point: str):
    """Return function metadata and a strict argument validator; never import source."""
    try:
        tree = ast.parse(source)
        compile(tree, "<generated-script>", "exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise RegistryError(f"Invalid Python source: {exc}") from exc
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
    ]
    if len(matches) != 1:
        raise RegistryError("Source must define exactly one top-level entry-point function.")
    function = matches[0]
    if isinstance(function, ast.AsyncFunctionDef) or function.decorator_list:
        raise RegistryError("Use a synchronous, undecorated entry-point function.")
    args = function.args
    if args.posonlyargs or args.vararg or args.kwarg:
        raise RegistryError("Entry points must use named parameters, without *args or **kwargs.")
    description = ast.get_docstring(function)
    if not description:
        raise RegistryError("The entry-point function needs a descriptive docstring.")
    fields = {}
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
    parameters = list(zip(args.args, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
    try:
        for parameter, default in parameters:
            if parameter.annotation is None:
                raise RegistryError(f"Parameter {parameter.arg!r} needs a type annotation.")
            value = ... if default is None else ast.literal_eval(default)
            if default is not None:
                json.dumps(value, allow_nan=False)
            fields[parameter.arg] = (_annotation(parameter.annotation), value)
        argument_model = create_model(
            "ScriptArguments",
            __config__=ConfigDict(extra="forbid", strict=True, validate_default=True),
            **fields,
        )
        if function.returns is None:
            raise RegistryError("The entry-point function needs a return annotation.")
        _annotation(function.returns)
    except RegistryError:
        raise
    except (ValueError, TypeError, SyntaxError) as exc:
        raise RegistryError(f"Unsupported parameter annotation or default: {exc}") from exc
    return {
        "name": function.name,
        "description": description,
        "parameters": argument_model.model_json_schema(),
        "return_type": ast.unparse(function.returns),
    }, argument_model


class ScriptRegistry:
    """A project-local JSON registry. Registration does not enable execution."""

    def __init__(self, project_root: Path | str = PROJECT_ROOT):
        self.root = Path(project_root).resolve()
        self.registry_path = self.root / "tools" / "registry.json"
        self.scripts_dir = self.root / "agent_generated_scripts"

    @contextmanager
    def _lock(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self):
        if not self.registry_path.exists():
            return {"tools": []}
        try:
            data = json.loads(self.registry_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"Cannot read registry: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("tools"), list):
            raise RegistryError("Registry must contain a 'tools' list.")
        if not all(isinstance(entry, dict) for entry in data["tools"]):
            raise RegistryError("Every registry entry must be an object.")
        return data

    def _write(self, data):
        """Replace the JSON file only after a complete, flushed write."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.registry_path.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.registry_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def register(self, script_name: str, source: str, entry_point: str) -> dict:
        if not script_name.isidentifier() or keyword.iskeyword(script_name):
            raise RegistryError("script_name must be a Python identifier, without a path or extension.")
        metadata, _ = extract_source_metadata(source, entry_point)
        source_bytes = source.encode("utf-8")
        with self._lock():
            data = self._read()
            if any(entry.get("name") == metadata["name"] for entry in data["tools"]):
                raise RegistryError(f"A tool named {metadata['name']!r} is already registered.")
            self.scripts_dir.mkdir(parents=True, exist_ok=True)
            path = self.scripts_dir / f"{script_name}.py"
            metadata.update({
                "id": str(uuid4()),
                "path": str(path.relative_to(self.root)),
                "entry_point": entry_point,
                "status": "pending_validation",
                "version": 1,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            })
            try:
                with path.open("xb") as stream:
                    try:
                        stream.write(source_bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                    except BaseException:
                        path.unlink(missing_ok=True)
                        raise
            except FileExistsError as exc:
                raise RegistryError(f"Script already exists: {path.name}") from exc
            try:
                data["tools"].append(metadata)
                self._write(data)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        return metadata

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if not 1 <= limit <= 20:
            raise RegistryError("limit must be between 1 and 20.")
        words = set(re.findall(r"\w+", query.lower()))
        entries = self._read()["tools"]
        def score(entry):
            text = f"{entry.get('name', '')} {entry.get('description', '')}".lower()
            return sum(word in text for word in words)
        matches = [entry for entry in entries if not words or score(entry)]
        return sorted(matches, key=score, reverse=True)[:limit]

    def _find(self, data, script_id):
        for entry in data["tools"]:
            if entry.get("id") == script_id:
                return entry
        raise RegistryError(f"Unknown script ID: {script_id}")

    def _verified_path(self, entry):
        if not entry.get("path") or not entry.get("sha256"):
            raise RegistryError("Legacy metadata must be registered with source before execution.")
        path = (self.root / entry["path"]).resolve()
        if not path.is_relative_to(self.scripts_dir.resolve()) or path.suffix != ".py":
            raise RegistryError("Registered script path is outside the scripts directory.")
        if not path.is_file():
            raise RegistryError("Registered script file is missing.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RegistryError("Script changed after registration; review and register it again.")
        return path

    def enable_script(self, script_id: str) -> dict:
        """Enable a reviewed script. Deliberately not exposed as an LLM tool."""
        with self._lock():
            data = self._read()
            entry = self._find(data, script_id)
            self._verified_path(entry)
            entry["status"] = "enabled"
            self._write(data)
            return entry

    def execute(self, script_id: str, arguments: dict, timeout: float = 10) -> dict:
        entry = self._find(self._read(), script_id)
        if entry.get("status") != "enabled":
            raise RegistryError("Script is not enabled. Review it and call registry.enable_script(id).")
        path = self._verified_path(entry)
        _, validator = extract_source_metadata(path.read_text(), entry["entry_point"])
        try:
            validated = validator.model_validate(arguments, strict=True)
            payload = json.dumps({
                "path": str(path), "entry_point": entry["entry_point"],
                "arguments": validated.model_dump(mode="json"),
            }, allow_nan=False)
        except (ValidationError, TypeError, ValueError) as exc:
            raise RegistryError(f"Invalid arguments: {exc}") from exc
        try:
            process = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).with_name("tool_runner.py"))],
                input=payload, text=True, capture_output=True,
                cwd=self.root, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RegistryError(f"Script exceeded the {timeout}-second timeout.") from exc
        if process.returncode:
            raise RegistryError(f"Script failed: {process.stderr.strip()[-4000:]}")
        try:
            return json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RegistryError("Script runner returned invalid JSON.") from exc


def build_registry_tools(registry: ScriptRegistry | None = None):
    """Build LangChain tools backed by one persistent registry instance."""
    from langchain_core.tools import tool

    registry = registry or ScriptRegistry()

    @tool
    def write_script_tool(script_name: str, source: str, entry_point: str) -> dict:
        """Save a Python function tool with typed parameters, return type and docstring.

        script_name is a simple name without .py. entry_point names a top-level
        synchronous function. Registration returns an ID and pending_validation
        status; a reviewed script must be enabled before it can execute.
        """
        try:
            return registry.register(script_name, source, entry_point)
        except (RegistryError, OSError) as exc:
            return {"error": str(exc)}

    @tool
    def search_scripts(query: str, limit: int = 5) -> list[dict]:
        """Search saved script names and descriptions; return IDs, schemas and status."""
        try:
            return registry.search(query, limit)
        except RegistryError as exc:
            return [{"error": str(exc)}]

    @tool
    def execute_script(script_id: str, arguments: dict) -> dict:
        """Execute an enabled registered function by ID with JSON arguments.

        Returns its JSON result and captured stdout. Search the registry first
        to discover IDs and argument schemas. Execution has a ten-second timeout.
        """
        try:
            return registry.execute(script_id, arguments)
        except (RegistryError, OSError) as exc:
            return {"error": str(exc)}

    return [write_script_tool, search_scripts, execute_script]
