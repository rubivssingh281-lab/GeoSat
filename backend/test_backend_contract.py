import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class BackendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "flask_backend.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(source)
        cls.routes = {
            decorator.args[0].value
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.FunctionDef)
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "route"
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
        }

    def test_required_routes_are_declared(self):
        self.assertTrue({
            "/api/health",
            "/api/status",
            "/api/metrics",
            "/api/retrieve",
        }.issubset(self.routes))

    def test_project_relative_paths_are_used(self):
        names = {
            node.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Name)
        }
        self.assertIn("PROJECT_DIR", names)
        self.assertIn("CACHE_METADATA_PATH", names)


if __name__ == "__main__":
    unittest.main()
