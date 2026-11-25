"""Load feature dependencies only when a workflow needs them."""

from importlib import import_module


def require_optional(module, extra):
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        # Preserve errors from broken installations and missing transitive
        # dependencies instead of misreporting them as an absent extra.
        if exc.name != module.split(".")[0]:
            raise
        raise ImportError(
            f"{module} is required for this feature; "
            f"install it with: python -m pip install 'python-intan[{extra}]'"
        ) from exc
