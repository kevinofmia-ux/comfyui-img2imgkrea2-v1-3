"""ComfyUI registration and node contract."""

from __future__ import annotations

import inspect

import torch

import chromagrade
from chromagrade.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, ChromaGradeColorMatch
from tests import fixtures


def test_package_exports_the_comfy_mappings():
    assert NODE_CLASS_MAPPINGS == {"ChromaGradeColorMatch": ChromaGradeColorMatch}
    assert NODE_DISPLAY_NAME_MAPPINGS["ChromaGradeColorMatch"]
    assert chromagrade.NODE_CLASS_MAPPINGS is NODE_CLASS_MAPPINGS


def test_top_level_module_registers_cleanly():
    """Simulate ComfyUI importing the custom-node directory as a package."""
    import importlib
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(root.name)
    assert "ChromaGradeColorMatch" in module.NODE_CLASS_MAPPINGS
    assert module.NODE_DISPLAY_NAME_MAPPINGS["ChromaGradeColorMatch"]
    assert isinstance(module.__version__, str)


def test_node_contract():
    spec = ChromaGradeColorMatch.INPUT_TYPES()
    required = spec["required"]
    assert required["TARGET_IMAGE"][0] == "IMAGE"
    assert required["COLOR_REFERENCE"][0] == "IMAGE"
    assert ChromaGradeColorMatch.RETURN_TYPES == ("IMAGE",)
    assert ChromaGradeColorMatch.RETURN_NAMES == ("IMAGE",)
    assert ChromaGradeColorMatch.CATEGORY
    assert ChromaGradeColorMatch.DESCRIPTION
    assert hasattr(ChromaGradeColorMatch, ChromaGradeColorMatch.FUNCTION)


def test_every_declared_input_reaches_the_function():
    spec = ChromaGradeColorMatch.INPUT_TYPES()["required"]
    sig = inspect.signature(getattr(ChromaGradeColorMatch, ChromaGradeColorMatch.FUNCTION))
    for name in spec:
        assert name in sig.parameters, f"{name} is declared in INPUT_TYPES but not accepted"


def test_every_widget_is_documented_and_bounded():
    for name, (kind, opts) in ChromaGradeColorMatch.INPUT_TYPES()["required"].items():
        assert opts.get("tooltip"), f"{name} has no tooltip"
        if kind == "FLOAT":
            assert "default" in opts and "min" in opts and "max" in opts, name
            assert opts["min"] <= opts["default"] <= opts["max"], name


def test_mode_widget_offers_exactly_the_implemented_modes():
    from chromagrade.pipeline import MODES

    kind, opts = ChromaGradeColorMatch.INPUT_TYPES()["required"]["mode"]
    assert tuple(kind) == tuple(MODES)
    assert opts["default"] in MODES


def test_node_runs_with_defaults():
    node = ChromaGradeColorMatch()
    defaults = {
        name: opts["default"]
        for name, (kind, opts) in ChromaGradeColorMatch.INPUT_TYPES()["required"].items()
        if "default" in opts
    }
    (out,) = node.match(fixtures.gradient_scene(), fixtures.sepia_reference(), **defaults)
    assert isinstance(out, torch.Tensor)
    assert out.shape == fixtures.gradient_scene().shape
    assert torch.isfinite(out).all()


def test_node_wraps_failures_in_a_readable_runtime_error():
    node = ChromaGradeColorMatch()
    try:
        node.match(torch.rand(1, 4, 4, 7), fixtures.sepia_reference())
    except RuntimeError as exc:
        assert "ChromaGrade" in str(exc)
        assert "channels" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError")


def test_example_workflow_matches_the_node():
    """The shipped workflow must stay in step with INPUT_TYPES.

    Widget values in a ComfyUI workflow are positional, so reordering or adding
    an input silently corrupts every saved graph. This catches that.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "workflows" / "chromagrade_basic.json"
    graph = json.loads(path.read_text(encoding="utf-8"))

    node = next(n for n in graph["nodes"] if n["type"] == "ChromaGradeColorMatch")
    spec = ChromaGradeColorMatch.INPUT_TYPES()["required"]
    widgets = [n for n, (kind, _) in spec.items() if kind != "IMAGE"]
    sockets = [n for n, (kind, _) in spec.items() if kind == "IMAGE"]

    assert len(node["widgets_values"]) == len(widgets), (
        f"workflow has {len(node['widgets_values'])} widget values, node declares {len(widgets)}: {widgets}"
    )
    assert [i["name"] for i in node["inputs"]] == sockets

    for name, value in zip(widgets, node["widgets_values"], strict=True):
        kind, opts = spec[name]
        if kind == "FLOAT":
            assert opts["min"] <= value <= opts["max"], f"{name}={value} is out of range"
        else:
            assert value in kind, f"{name}={value!r} is not one of {kind}"

    ids = {n["id"] for n in graph["nodes"]}
    for link_id, src, _, dst, _, _ in graph["links"]:
        assert src in ids and dst in ids, f"link {link_id} points at a missing node"


def test_node_does_not_leak_gradients():
    node = ChromaGradeColorMatch()
    target = fixtures.gradient_scene().requires_grad_(False)
    (out,) = node.match(target, fixtures.sepia_reference())
    assert not out.requires_grad
