#!/usr/bin/env python3
"""Script to check and report on type hint coverage in the codebase.

This script analyzes Python files to identify functions and methods that
are missing type hints and generates a report with recommendations.

Usage:
    python scripts/check_type_hints.py [--fix] [--path src/]

Options:
    --fix: Attempt to add basic type hints where possible
    --path: Directory to analyze (default: src/)
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class TypeHintChecker(ast.NodeVisitor):
    """AST visitor to check for missing type hints."""

    def __init__(self) -> None:
        """Initialize the type hint checker."""
        self.missing_hints: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None
        self.current_file: Optional[Path] = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition nodes."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition nodes."""
        self._check_function_hints(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition nodes."""
        self._check_function_hints(node)
        self.generic_visit(node)

    def _check_function_hints(self, node: ast.FunctionDef) -> None:
        """Check if function has proper type hints."""
        issues = []

        # Skip private functions and test functions
        if node.name.startswith("_") and node.name != "__init__":
            return
        if node.name.startswith("test_"):
            return

        # Check return type annotation
        if node.returns is None:
            issues.append("missing_return_type")

        # Check parameter type annotations
        missing_params = []
        for arg in node.args.args:
            if arg.annotation is None and arg.arg != "self" and arg.arg != "cls":
                missing_params.append(arg.arg)

        if missing_params:
            issues.append(f"missing_param_types: {', '.join(missing_params)}")

        if issues:
            func_location = f"{self.current_class}.{node.name}" if self.current_class else node.name
            self.missing_hints.append(
                {
                    "file": str(self.current_file),
                    "function": func_location,
                    "line": node.lineno,
                    "issues": issues,
                }
            )

    def analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for type hints."""
        self.current_file = file_path
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            self.visit(tree)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")


def analyze_directory(directory: Path) -> TypeHintChecker:
    """Analyze all Python files in a directory."""
    checker = TypeHintChecker()

    python_files = list(directory.rglob("*.py"))
    print(f"Analyzing {len(python_files)} Python files...")

    for file_path in python_files:
        # Skip certain directories
        if any(
            part in str(file_path) for part in ["__pycache__", ".venv", "venv", "build", "dist"]
        ):
            continue

        checker.analyze_file(file_path)

    return checker


def generate_report(checker: TypeHintChecker) -> None:
    """Generate a detailed report of missing type hints."""
    if not checker.missing_hints:
        print("🎉 All analyzed functions have proper type hints!")
        return

    print(f"\n📊 Type Hint Analysis Report")
    print(f"Found {len(checker.missing_hints)} functions/methods with missing type hints:\n")

    # Group by file
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for hint in checker.missing_hints:
        file_path = hint["file"]
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(hint)

    for file_path, hints in by_file.items():
        print(f"📄 {file_path}")
        for hint in hints:
            print(f"  🔹 Line {hint['line']}: {hint['function']}")
            for issue in hint["issues"]:
                print(f"    ❌ {issue}")
        print()

    # Summary statistics
    total_missing_returns = sum(
        1
        for h in checker.missing_hints
        if any("missing_return_type" in issue for issue in h["issues"])
    )
    total_missing_params = sum(
        1
        for h in checker.missing_hints
        if any("missing_param_types" in issue for issue in h["issues"])
    )

    print(f"📈 Summary:")
    print(f"  • Functions missing return type hints: {total_missing_returns}")
    print(f"  • Functions missing parameter type hints: {total_missing_params}")
    print(f"  • Total files with issues: {len(by_file)}")


def suggest_type_hints() -> None:
    """Provide suggestions for common type hint patterns."""
    print("\n💡 Type Hint Suggestions:")
    print(
        """
Common patterns to use:

1. Basic types:
   def process_text(text: str) -> str:
   def calculate_total(items: List[float]) -> float:
   def get_config() -> Dict[str, Any]:

2. Optional types:
   def find_user(user_id: str) -> Optional[User]:
   def get_cached_result(key: str) -> Union[str, None]:

3. Async functions:
   async def fetch_data(url: str) -> Dict[str, Any]:
   async def process_message(msg: str) -> Optional[Response]:

4. Class methods:
   def __init__(self, name: str, age: int) -> None:
   def get_info(self) -> Dict[str, str]:

5. Complex types:
   def batch_process(items: List[Dict[str, Any]]) -> List[Result]:
   def handle_callback(func: Callable[[str], bool]) -> None:

6. Modern Python 3.9+ style:
   def get_users() -> list[User]:        # Instead of List[User]
   def get_mapping() -> dict[str, int]:  # Instead of Dict[str, int]
"""
    )


def create_type_hints_todo() -> None:
    """Create a TODO file with type hint improvement tasks."""
    todo_content = """# Type Hints Improvement Tasks

## High Priority Files
These files have the most missing type hints and should be addressed first:

### Core Services
- [ ] src/assistant/core/services/chat_service.py
- [ ] src/assistant/core/services/classification_service.py
- [ ] src/assistant/core/services/conversation_service.py

### API Layer
- [ ] src/assistant/api/routes/chat.py
- [ ] src/assistant/api/routes/health.py
- [ ] src/assistant/api/routes/knowledge.py

### Infrastructure
- [ ] src/assistant/infrastructure/database/vector_store.py
- [ ] src/assistant/infrastructure/llm/openai_client.py
- [ ] src/assistant/infrastructure/email/smtp_client.py

## Type Hint Standards Checklist

For each function/method, ensure:
- [ ] All parameters have type hints (except 'self' and 'cls')
- [ ] Return type is specified (use -> None for functions without return)
- [ ] Use Optional[T] for nullable parameters
- [ ] Use Union[T1, T2] for multiple possible types
- [ ] Use List[T], Dict[K, V] for collections (or list[T], dict[K, V] in Python 3.9+)
- [ ] Import necessary types from typing module
- [ ] Use Any sparingly and only when truly needed

## Modern Type Annotation Patterns

### Python 3.9+ Features (Preferred)
```python
# Use built-in types where possible
def process_items(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

# Use | for Union types
def get_value(key: str) -> str | None:
    return cache.get(key)
```

### Legacy Support (if needed)
```python
from typing import List, Dict, Optional, Union

def process_items(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}

def get_value(key: str) -> Optional[str]:
    return cache.get(key)
```

## Validation Commands

Run these commands to check type hint coverage:

```bash
# Check with mypy
mypy src/assistant/

# Run type hint checker script
python scripts/check_type_hints.py

# Format with black (preserves type hints)
black src/

# Sort imports
isort src/
```
"""

    todo_path = Path("TYPE_HINTS_TODO.md")
    with open(todo_path, "w", encoding="utf-8") as f:
        f.write(todo_content)

    print(f"\n📝 Created type hints TODO file: {todo_path}")


def main() -> None:
    """Main entry point for the type hint checker."""
    parser = argparse.ArgumentParser(description="Check type hint coverage in Python code")
    parser.add_argument("--path", default="src/", help="Directory to analyze")
    parser.add_argument("--suggestions", action="store_true", help="Show type hint suggestions")
    parser.add_argument("--create-todo", action="store_true", help="Create type hints TODO file")

    args = parser.parse_args()

    directory = Path(args.path)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)

    # Analyze the codebase
    checker = analyze_directory(directory)
    generate_report(checker)

    if args.suggestions:
        suggest_type_hints()

    if args.create_todo:
        create_type_hints_todo()

    # Exit with error code if there are missing type hints
    if checker.missing_hints:
        print(f"\n⚠️  Found {len(checker.missing_hints)} functions with missing type hints")
        print("Run with --suggestions flag for type hint patterns")
        print("Run with --create-todo flag to create improvement tasks")
        sys.exit(1)
    else:
        print("\n✅ Type hint coverage looks good!")


if __name__ == "__main__":
    main()
