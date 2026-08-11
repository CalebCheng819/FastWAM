"""Match legacy tensor-only text caches to task names without computing hashes."""

from pathlib import Path

import torch

from fastwam.models.wan22.helpers.loader import _load_registered_model
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer


CACHE_ROOT = Path(
    "/oss-chengjuntao/cpfs-user-chengjuntao/datasets/"
    "robofactory_multi_robot/text_embeds_cache_n234"
)
MODEL_PATH = (
    "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache/"
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
    "models_t5_umt5-xxl-enc-bf16.safetensors"
)
TOKENIZER_PATH = (
    "/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache/"
    "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
)
TASKS = {
    "StrikeCubeHard-rf": "two robots collaboratively strike the cube to the target",
    "PlaceFood-rf": "two robots collaboratively place the food in the target location",
    "PlaceCubeInCup-rf": "two robots collaboratively place the cube in the cup",
    "ThreeRobotsPlaceShoes-rf": (
        "three robots collaboratively place the shoes in their target locations"
    ),
    "ThreeRobotsStackCube-rf": "three robots collaboratively stack the cubes",
    "FourRobotsStackCube-rf": "four robots collaboratively stack the cubes",
}
PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def main() -> None:
    files = sorted(CACHE_ROOT.glob("*.pt"))
    cached = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    print("LOAD_MODEL_START", flush=True)
    encoder = _load_registered_model(
        MODEL_PATH,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device="cuda",
        checkpoint_integrity_mode="metadata_no_hash",
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=TOKENIZER_PATH,
        seq_len=128,
        clean="whitespace",
    )
    prompts = [PROMPT.format(task=value) for value in TASKS.values()]
    ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
    mask = mask.to(device="cuda", dtype=torch.bool)
    with torch.no_grad():
        context = encoder(ids.to("cuda"), mask).cpu().to(torch.bfloat16)
    mask = mask.cpu()

    used: set[int] = set()
    for row, task in enumerate(TASKS):
        matches = [
            idx
            for idx, item in enumerate(cached)
            if torch.equal(mask[row], item["mask"])
            and torch.equal(context[row], item["context"])
        ]
        if len(matches) != 1:
            raise RuntimeError(f"task {task!r} has candidates {matches!r}")
        idx = matches[0]
        if idx in used:
            raise RuntimeError("cache matching is not bijective")
        used.add(idx)
        print("MATCH", task, files[idx].name, flush=True)
    if len(used) != len(files):
        raise RuntimeError("not all cache tensors were matched")
    print("MATCH_COMPLETE", len(used), flush=True)


if __name__ == "__main__":
    main()
