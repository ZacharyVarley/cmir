"""Tools for homography transformations."""


__all__ = [
    "HomographyLieTransform",
    "HomographyBasisTransform",
]

def __dir__():
    return sorted(__all__)


def __getattr__(name):
    _import_mapping = {
        "HomographyLieTransform": "homography_lie",
        "HomographyBasisTransform": "homography_basis",
    }
    if name in __all__:
        import importlib

        if name in _import_mapping.keys():
            import_path = f"{__name__}.{_import_mapping.get(name)}"
            return getattr(importlib.import_module(import_path), name)
        else:  # pragma: no cover
            return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
