"""Lazy import hooks with idempotent method installation and status reporting."""
import logging
import sys
from functools import wraps
from importlib.abc import MetaPathFinder

logger = logging.getLogger(__name__)
HOOK_REGISTRY = {}
PATCH_STATUS = {}


def register_hook(module_name):
    def register(func):
        HOOK_REGISTRY[module_name] = func
        return func
    return register


def apply_method_patch(cls, method_name, wrapper_factory):
    original = getattr(cls, method_name, None)
    key = f"{cls.__module__}.{cls.__name__}.{method_name}:{wrapper_factory.__qualname__}"
    if original is None:
        PATCH_STATUS[key] = "missing"
        return False
    installed = set(getattr(original, "_de_latency_wrappers", ()))
    if key in installed:
        return True
    patched = wrapper_factory(original)
    if patched is original:
        # A component may recognize an inherited wrapper by its own marker.
        # wraps(original)(original) would create a cyclic __wrapped__ chain.
        PATCH_STATUS[key] = "inherited"
        return True
    patched = wraps(original)(patched)
    patched._de_latency_wrappers = installed | {key}
    setattr(cls, method_name, patched)
    PATCH_STATUS[key] = "installed"
    return True


class PatchingLoader:
    def __init__(self, original_loader, patch_func):
        self.original_loader = original_loader
        self.patch_func = patch_func

    def __getattr__(self, name):
        return getattr(self.original_loader, name)

    def create_module(self, spec):
        if hasattr(self.original_loader, "create_module"):
            return self.original_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self.original_loader.exec_module(module)
        apply_hook(module.__name__, module, self.patch_func)


def apply_hook(name, module, hook):
    try:
        hook(module)
        PATCH_STATUS[name] = "installed"
    except Exception:
        PATCH_STATUS[name] = "failed"
        logger.exception("Cannot install tracing hook for %s", name)


class VllmUniversalPatcher(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in HOOK_REGISTRY:
            return None
        for finder in tuple(sys.meta_path):
            if isinstance(finder, VllmUniversalPatcher):
                continue
            find = getattr(finder, "find_spec", None)
            spec = find(fullname, path, target) if find else None
            if spec and spec.loader:
                spec.loader = PatchingLoader(spec.loader, HOOK_REGISTRY[fullname])
                return spec
        return None


def install():
    for name, hook in tuple(HOOK_REGISTRY.items()):
        if name in sys.modules:
            apply_hook(name, sys.modules[name], hook)
    if not any(isinstance(finder, VllmUniversalPatcher) for finder in sys.meta_path):
        sys.meta_path.insert(0, VllmUniversalPatcher())
