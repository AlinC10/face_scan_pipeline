"""
Runs plain test_* functions (pytest style) under unittest: `python -m unittest discover -s tests -t .`
A function with a `tmp_path` parameter receives a fresh temporary directory, as with pytest.

Use at the bottom of a test module:
    from tests._functional import function_tests
    load_tests = function_tests(globals())
"""

import inspect
import tempfile
import unittest
from pathlib import Path


def function_tests(namespace: dict):
    funcs = {name: fn for name, fn in namespace.items() if name.startswith("test_") and inspect.isfunction(fn)}

    def make(fn):
        def method(self):
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(tmp_path=Path(tmp))
            else:
                fn()
        method.__doc__ = fn.__doc__
        return method

    case = type("FunctionTests", (unittest.TestCase,), {name: make(fn) for name, fn in funcs.items()})
    case.__module__ = namespace.get("__name__", case.__module__)

    def load_tests(loader, tests, pattern):
        suite = unittest.TestSuite()
        suite.addTests(tests)
        suite.addTests(loader.loadTestsFromTestCase(case))
        return suite

    return load_tests
