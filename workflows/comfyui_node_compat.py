"""ComfyUI node compatibility validation.

Queries ComfyUI's /object_info endpoint to validate that workflow builder
params match the actual node schemas. Prevents silent breaking changes
like LTXVTiledVAEDecode overlap param range change (64->8).

Each check function targets a specific node class and param. New checks
are added to _CHECK_REGISTRY and picked up by run_all_checks().
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# ── Node class name constants ──────────────────────────────────────

LTXV_TILED_VAE_DECODE = "LTXVTiledVAEDecode"
VAE_DECODE_TILED = "VAEDecodeTiled"

# ── Param value constants (our hardcoded values) ──────────────────

LTXV_OVERLAP_VALUE = 8
VAE_TILED_OVERLAP_VALUE = 64
VAE_TILED_TEMPORAL_OVERLAP_VALUE = 8

# ── Timeout constants ──────────────────────────────────────────────

OBJECT_INFO_TIMEOUT_S = 10


class NodeCompatibilityError(RuntimeError):
    """Raised when node param values are incompatible with ComfyUI schema."""


def get_node_schema(comfyui_base: str, node_class: str) -> dict | None:
    """Query /object_info for a node's input schema.

    Args:
        comfyui_base: ComfyUI server base URL (e.g. http://127.0.0.1:8199).
        node_class: ComfyUI node class name (e.g. "LTXVTiledVAEDecode").

    Returns:
        The node's schema dict from /object_info, or None on failure.
    """
    url = f"{comfyui_base}/object_info"
    try:
        with urllib.request.urlopen(url, timeout=OBJECT_INFO_TIMEOUT_S) as resp:
            all_nodes: dict[str, Any] = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as e:
        logger.warning("Failed to fetch /object_info from %s: %s", comfyui_base, e)
        return None

    node_info = all_nodes.get(node_class)
    if node_info is None:
        logger.warning("Node %s not found in /object_info (may not be installed)", node_class)
        return None
    return node_info


def _extract_param_bounds(schema: dict, param_name: str) -> tuple[int | None, int | None] | None:
    """Extract min/max bounds for a param from a node schema.

    Args:
        schema: Node schema dict from get_node_schema().
        param_name: Parameter name to look up.

    Returns:
        (min, max) tuple if bounds found, None if param not in schema.
    """
    inputs = schema.get("input", {})
    required = inputs.get("required", {})
    optional = inputs.get("optional", {})

    param_def = required.get(param_name) or optional.get(param_name)
    if param_def is None:
        return None

    # Param def format: ["TYPE", {"min": N, "max": M, "default": D, ...}]
    if not isinstance(param_def, list) or len(param_def) < 2:
        return None

    constraints = param_def[1]
    if not isinstance(constraints, dict):
        return None

    min_val = constraints.get("min")
    max_val = constraints.get("max")
    return (min_val, max_val)


def validate_node_params(
    comfyui_base: str,
    node_class: str,
    params: dict[str, int | float],
) -> list[str]:
    """Validate param values against ComfyUI's live node schema.

    Args:
        comfyui_base: ComfyUI server base URL.
        node_class: Node class name to validate.
        params: Dict of param_name -> our_hardcoded_value.

    Returns:
        List of violation strings. Empty list means all params are valid.
    """
    schema = get_node_schema(comfyui_base, node_class)
    if schema is None:
        return [f"{node_class}: schema unavailable (node may not be installed)"]

    violations: list[str] = []
    for param_name, our_value in params.items():
        bounds = _extract_param_bounds(schema, param_name)
        if bounds is None:
            violations.append(
                f"{node_class}.{param_name}: param not found in schema "
                f"(our value={our_value}, node schema may have changed)"
            )
            continue

        min_val, max_val = bounds
        if min_val is not None and our_value < min_val:
            violations.append(
                f"{node_class}.{param_name}: our value {our_value} < min {min_val}"
            )
        elif max_val is not None and our_value > max_val:
            violations.append(
                f"{node_class}.{param_name}: our value {our_value} > max {max_val}"
            )

    return violations


def check_ltxv_tiled_vae_decode(comfyui_base: str) -> list[str]:
    """Check LTXVTiledVAEDecode overlap param compatibility.

    The overlap param silently changed from range 0-64 to 1-8 in a
    ComfyUI-LTXVideo update. Our workflow hardcodes overlap=8.

    Args:
        comfyui_base: ComfyUI server base URL.

    Returns:
        List of violation strings.
    """
    return validate_node_params(
        comfyui_base,
        LTXV_TILED_VAE_DECODE,
        {"overlap": LTXV_OVERLAP_VALUE},
    )


def check_vae_decode_tiled(comfyui_base: str) -> list[str]:
    """Check VAEDecodeTiled overlap params compatibility.

    ComfyUI core VAEDecodeTiled uses overlap=64 and temporal_overlap=8.
    Validate these are still within acceptable ranges.

    Args:
        comfyui_base: ComfyUI server base URL.

    Returns:
        List of violation strings.
    """
    return validate_node_params(
        comfyui_base,
        VAE_DECODE_TILED,
        {
            "overlap": VAE_TILED_OVERLAP_VALUE,
            "temporal_overlap": VAE_TILED_TEMPORAL_OVERLAP_VALUE,
        },
    )


# ── Check registry ─────────────────────────────────────────────────

_CHECK_REGISTRY: list[tuple[str, str]] = [
    ("LTXVTiledVAEDecode overlap", LTXV_TILED_VAE_DECODE),
    ("VAEDecodeTiled overlap", VAE_DECODE_TILED),
]


def run_all_checks(comfyui_base: str) -> list[str]:
    """Run all registered compatibility checks against ComfyUI.

    Args:
        comfyui_base: ComfyUI server base URL.

    Returns:
        List of all violation strings across all checks. Empty means clean.

    Raises:
        NodeCompatibilityError: If any check finds violations.
    """
    all_violations: list[str] = []

    for check_name, node_class in _CHECK_REGISTRY:
        if node_class == LTXV_TILED_VAE_DECODE:
            violations = check_ltxv_tiled_vae_decode(comfyui_base)
        elif node_class == VAE_DECODE_TILED:
            violations = check_vae_decode_tiled(comfyui_base)
        else:
            violations = validate_node_params(comfyui_base, node_class, {})

        if violations:
            logger.error("COMPAT CHECK FAILED [%s]: %s", check_name, "; ".join(violations))
            all_violations.extend(violations)
        else:
            logger.info("COMPAT CHECK OK [%s]", check_name)

    if all_violations:
        raise NodeCompatibilityError(
            f"Node compatibility check failed with {len(all_violations)} violation(s): "
            + "; ".join(all_violations)
        )

    return all_violations
