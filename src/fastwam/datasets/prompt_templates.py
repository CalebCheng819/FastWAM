"""Lightweight prompt templates shared by robot datasets.

Keep this module dependency-free: direct dataset adapters import it during
startup without pulling optional Hugging Face dataset machinery into the
process.
"""

DEFAULT_ROBOT_VIDEO_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
