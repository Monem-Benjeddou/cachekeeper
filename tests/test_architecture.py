import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "cachekeeper"


def module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module_name(path).rsplit(".", node.level - (path.name == "__init__.py"))[0]
                names.add(f"{base}.{node.module}" if node.module else base)
            else:
                names.add(node.module or "")
    return names


MODULES = {module_name(path): imported_modules(path) for path in PACKAGE.rglob("*.py")}

# Each layer may import only from the layers listed; core may also import nothing outside the stdlib.
FORBIDDEN = {
    "cachekeeper.core": ("cachekeeper.adapters", "cachekeeper.container", "cachekeeper.config"),
    "cachekeeper.adapters.backends": ("cachekeeper.adapters.inbound", "cachekeeper.container"),
    "cachekeeper.adapters.storage": ("cachekeeper.adapters.inbound", "cachekeeper.container"),
    "cachekeeper.container": ("cachekeeper.adapters.inbound",),
    "cachekeeper.adapters.inbound.http": ("cachekeeper.adapters.inbound.cli",),
}


def test_package_has_modules() -> None:
    assert "cachekeeper.core.prompt" in MODULES


@pytest.mark.parametrize("module", sorted(m for m in MODULES if m.startswith("cachekeeper.core")))
def test_core_imports_only_stdlib_and_core(module: str) -> None:
    outside = {
        name
        for name in MODULES[module]
        if not name.startswith("cachekeeper.core")
        and name.split(".")[0] not in sys.stdlib_module_names | {"__future__"}
    }
    assert not outside, f"{module} imports {sorted(outside)}"


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_dependencies_point_inward(layer: str) -> None:
    violations = {
        (module, name)
        for module, names in MODULES.items()
        if module.startswith(layer)
        for name in names
        if name.startswith(FORBIDDEN[layer])
    }
    assert not violations, sorted(violations)
