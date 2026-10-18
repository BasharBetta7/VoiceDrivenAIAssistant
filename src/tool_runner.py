"""Subprocess entry point for invoking one registered function.

This separates function execution from the agent process; it is not a sandbox.
"""

import contextlib
import importlib.util
import io
import json
import sys


def main():
    request = json.load(sys.stdin)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        spec = importlib.util.spec_from_file_location("registered_tool", request["path"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result = getattr(module, request["entry_point"])(**request["arguments"])
    json.dump({"result": result, "stdout": output.getvalue()}, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
