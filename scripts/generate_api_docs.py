import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "forecast_combo"
API_DOC = PROJECT_ROOT / "docs" / "api.md"

PUBLIC_NAMESPACES = (
    ("Forecast combination", "forecast_combo", PACKAGE_ROOT / "__init__.py"),
    ("Weighting methods", "forecast_combo.combinations", PACKAGE_ROOT / "combinations" / "__init__.py"),
    ("Weight visualisations", "forecast_combo.visualisations", PACKAGE_ROOT / "visualisations" / "__init__.py"),
)


def read_exports(path: Path) -> list[str]:
    """Read a module's literal ``__all__`` declaration without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            exports = ast.literal_eval(node.value)
            if not isinstance(exports, list) or not all(isinstance(name, str) for name in exports):
                raise ValueError(f"{path}: __all__ must be a literal list of strings")
            return exports
    raise ValueError(f"{path}: no literal __all__ declaration found")


def read_literal(path: Path, variable_name: str) -> object:
    """Read a literal module-level variable without importing its module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path}: no literal {variable_name} declaration found")


def public_objects() -> list[tuple[str, list[str]]]:
    """Return canonical public object paths grouped by documentation section."""
    namespace_exports = [(section, namespace, read_exports(path)) for section, namespace, path in PUBLIC_NAMESPACES]
    child_exports = {
        name for _, namespace, exports in namespace_exports if namespace != "forecast_combo" for name in exports
    }

    sections = []
    for section, namespace, exports in namespace_exports:
        if namespace == "forecast_combo":
            exports = [name for name in exports if name not in child_exports]
        if namespace == "forecast_combo.visualisations":
            module_names = read_literal(PACKAGE_ROOT / "visualisations" / "__init__.py", "_PLOT_MODULES")
            if not isinstance(module_names, dict) or set(module_names) != set(exports):
                raise ValueError("visualisations._PLOT_MODULES must map every name in __all__")
            object_paths = [f"{namespace}.{module_names[name]}.{name}" for name in exports]
        else:
            object_paths = [f"{namespace}.{name}" for name in exports]
        sections.append((section, object_paths))
    return sections


def render_api_page(sections: list[tuple[str, list[str]]]) -> str:
    """Render the Markdown manifest that mkdocstrings consumes."""
    lines = [
        "# API Reference",
        "",
        (
            "This script writes the API manifest from each package's `__all__` declaration. "
            "Zensical renders the API content from the current source."
        ),
        "",
    ]
    for section, objects in sections:
        lines.extend((f"## {section}", ""))
        for object_path in objects:
            lines.extend(
                (
                    f"::: {object_path}",
                    "    options:",
                    "      show_source: false",
                    "      show_root_heading: true",
                    "",
                )
            )
    return "\n".join(lines)


def main() -> None:
    """Update the generated API documentation manifest."""
    API_DOC.write_text(render_api_page(public_objects()), encoding="utf-8")


if __name__ == "__main__":
    main()
