"""Versioned local prompt provenance; runtime text remains compatible with str."""

from .registry import bind_prompt, bind_source_prompt, compose_prompt, describe_registry, render_prompt

__all__ = ["bind_prompt", "bind_source_prompt", "compose_prompt", "describe_registry", "render_prompt"]
