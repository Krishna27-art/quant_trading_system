import ast
import json
import os


def analyze_file(filepath):
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        return [{"Severity": "Low", "Problem": f"Could not read file: {e}"}]

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [{"Severity": "High", "Problem": f"Syntax Error: {e}"}]

    class AuditVisitor(ast.NodeVisitor):
        def __init__(self):
            self.findings = []

        def add_finding(
            self, severity, node, problem, why_it_matters, best_practice, recommended_fix
        ):
            self.findings.append(
                {
                    "Severity": severity,
                    "File(s)": filepath,
                    "Line": node.lineno if hasattr(node, "lineno") else 0,
                    "Problem description": problem,
                    "Why it matters": why_it_matters,
                    "Institutional best practice": best_practice,
                    "Recommended fix": recommended_fix,
                    "Estimated implementation difficulty": "Low",
                    "Estimated performance/risk impact": "Medium",
                }
            )

        def visit_ExceptHandler(self, node):
            if node.type is None or (
                isinstance(node.type, ast.Name) and node.type.id == "Exception"
            ):
                has_pass = any(isinstance(s, ast.Pass) for s in node.body)
                if has_pass:
                    self.add_finding(
                        "Critical",
                        node,
                        "Bare except or swallowed Exception with pass.",
                        "Swallowing exceptions silently hides critical runtime failures, leading to data corruption, missed executions, or undetected state inconsistencies.",
                        "Explicitly catch specific exceptions and log them with stack traces. Never swallow exceptions silently in a trading platform.",
                        "Replace 'except Exception: pass' with specific exception handling and logging.",
                    )
                else:
                    self.add_finding(
                        "High",
                        node,
                        "Broad Exception catching without specific type.",
                        "Catching base Exception can mask unrelated bugs like KeyboardInterrupt or MemoryError, making debugging difficult.",
                        "Catch only the specific exceptions expected to fail.",
                        "Narrow down the exception type.",
                    )
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            # Check for pass or ...
            for stmt in node.body:
                if isinstance(stmt, ast.Pass) or (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is Ellipsis
                ):
                    self.add_finding(
                        "Medium",
                        node,
                        f"Function '{node.name}' is unimplemented (contains pass/...).",
                        "Dead or stub code left in production can lead to unexpected behavior if invoked.",
                        "Production code should not contain stub methods unless clearly marked as abstract/interfaces.",
                        "Implement or remove the method.",
                    )

            # Check for mutable defaults
            for default in node.args.defaults + node.args.kw_defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    self.add_finding(
                        "High",
                        node,
                        f"Mutable default argument in function '{node.name}'.",
                        "Mutable default arguments maintain state across function calls, leading to subtle data leakage and state contamination bugs.",
                        "Use None as the default value and initialize the mutable object inside the function body.",
                        "Change default to None and initialize inside.",
                    )

            # Complex functions (simple heuristic: > 10 branching statements)
            branches = [
                n for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While, ast.Try))
            ]
            if len(branches) > 10:
                self.add_finding(
                    "Medium",
                    node,
                    f"Function '{node.name}' has high cyclomatic complexity (>{len(branches)} branches).",
                    "Highly complex functions are difficult to test, prone to bugs, and hard to maintain.",
                    "Functions should do one thing and be small enough to easily unit test. Use SOLID principles.",
                    "Refactor into smaller, focused helper methods.",
                )

            # Global variables usage
            has_global = any(isinstance(n, ast.Global) for n in ast.walk(node))
            if has_global:
                self.add_finding(
                    "High",
                    node,
                    f"Function '{node.name}' uses global variables.",
                    "Global state in a concurrent trading system causes race conditions and unpredictable behavior.",
                    "Pass state explicitly or encapsulate within classes. Avoid global state entirely.",
                    "Refactor to remove global usage.",
                )

            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                self.add_finding(
                    "Low",
                    node,
                    "Usage of print() instead of logging.",
                    "Print statements are not captured by observability tools (e.g., ELK, Splunk) and lack severity levels.",
                    "All output should go through a structured logging framework (e.g., JSON logging) for traceability.",
                    "Replace print() with structured logger calls.",
                )
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "sleep":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "time":
                    self.add_finding(
                        "High",
                        node,
                        "Blocking time.sleep() used.",
                        "Blocking sleep calls in potentially async or high-frequency contexts stall the entire thread, delaying critical market data and execution.",
                        "Use async sleep (asyncio.sleep) or event-driven timers instead of blocking threads.",
                        "Refactor to asyncio.sleep or event-driven architecture.",
                    )

            self.generic_visit(node)

    visitor = AuditVisitor()
    visitor.visit(tree)
    return visitor.findings


def main():
    root_dir = "/Users/pandu/Desktop/quant"
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "frontend", "docs", "venv"}
    all_findings = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for file in filenames:
            if file.endswith(".py"):
                filepath = os.path.join(dirpath, file)
                findings = analyze_file(filepath)
                all_findings.extend(findings)

    output_path = "/Users/pandu/.gemini/antigravity/brain/886943b5-4300-4fed-b1da-6b96832de624/scratch/ast_findings.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_findings, f, indent=2)
    print(f"AST Analysis complete. Found {len(all_findings)} issues.")


if __name__ == "__main__":
    main()
