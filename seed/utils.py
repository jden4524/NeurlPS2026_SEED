# seed/utils.py
"""
General utility functions (SEED Project)
Includes multi-GPU environment setup, path resolution, result parsing, sharded data merging, etc.
"""

import json
import os
import time
import re
import random
import glob
import numpy as np
from typing import Dict, List, Optional, Any

import torch

# ==========================================
# 🧪 Experiment & Multi-GPU support
# ==========================================

def seed_everything(seed: int = 42):
    """Fix all random seeds to ensure absolute reproducibility for single/multi-GPU experiments."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_device(device_str: str = "auto", gpu_id: int = None) -> torch.device:
    """Resolve the target device from a string description, supporting multi-GPU specification."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
        else:
            device_str = "cpu"
    return torch.device(device_str)

def get_gpu_memory_info(gpu_id: int = 0) -> Optional[Dict[str, float]]:
    """Get VRAM usage information for the specified GPU."""
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_gb": torch.cuda.memory_allocated(gpu_id) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(gpu_id) / 1e9,
        "total_gb": torch.cuda.get_device_properties(gpu_id).total_mem / 1e9,
    }

def enforce_min_new_tokens(args, min_new_tokens: int = 512) -> None:
    """Guard production runs against stale short-generation commands."""
    if getattr(args, "max_new_tokens", min_new_tokens) >= min_new_tokens:
        return
    raise SystemExit(
        f"--max_new_tokens must be >= {min_new_tokens} for this run "
        f"(got {args.max_new_tokens})."
    )

def partition_indices(total: int, gpu_id: int, world_size: int) -> List[int]:
    """Split contiguous sample chunks by GPU weight to reduce tail-waiting caused by slow GPUs."""
    if world_size <= 1:
        return list(range(total))

    weights_env = os.environ.get("SEED_GPU_WEIGHTS") or os.environ.get("GPU_PARTITION_WEIGHTS")
    weights: List[float]
    if weights_env:
        try:
            weights = [max(0.0, float(x.strip())) for x in weights_env.split(",")]
        except ValueError:
            weights = []
    elif world_size == 4:
        weights = [1.0, 0.25, 1.0, 1.0]
    else:
        weights = [1.0] * world_size

    if len(weights) != world_size or sum(weights) <= 0:
        weights = [1.0] * world_size

    raw = [total * w / sum(weights) for w in weights]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(world_size), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1

    start = sum(counts[:gpu_id])
    end = start + counts[gpu_id]
    return list(range(start, end))

def resize_image_for_seed(image, max_pixels: int = 200704):
    """Downscale large PIL images before processor tokenization."""
    if image is None:
        return None
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels:
        return image

    scale = (max_pixels / pixels) ** 0.5
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    try:
        from PIL import Image
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = 1
    return image.resize(new_size, resample)

# ==========================================
# 📂 I/O & Data Parsing
# ==========================================

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def parse_mcq_answer(text: str) -> Optional[str]:
    """Use regex to precisely extract the final option letter from the model's long-text output."""
    if not text:
        return None
    match = re.search(r"[Aa]nswer\s*[:：]\s*([A-Za-z])\b", text)
    return match.group(1).upper() if match else None

# ==========================================
# 💾 File Writers & Mergers
# ==========================================

def append_jsonl(record: Dict, path: str) -> None:
    """[Recommended for multi-process] Write JSONL in append mode to prevent OOM and mid-run crashes."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def collect_processed_jsonl_ids(
    output_dir: str,
    prefix: str,
    mode_str: str,
    include_merged: bool = True,
) -> set[str]:
    """Collect completed record ids from all shards and the merged file."""
    paths = glob.glob(os.path.join(output_dir, f"{prefix}_{mode_str}_gpu*.jsonl"))
    if include_merged:
        merged_path = os.path.join(output_dir, f"{prefix}_{mode_str}_merged.jsonl")
        if os.path.exists(merged_path):
            paths.append(merged_path)

    ids: set[str] = set()
    for path in sorted(set(paths)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(str(json.loads(line)["id"]))
    return ids

def dedupe_records_by_id(records: List[Dict]) -> List[Dict]:
    """Keep one record per id while preserving first-seen output order."""
    order: List[str] = []
    by_id: Dict[str, Dict] = {}
    for record in records:
        record_id = str(record.get("id", ""))
        if not record_id:
            order.append(record_id)
            continue
        if record_id not in by_id:
            order.append(record_id)
        by_id[record_id] = record
    return [by_id[record_id] for record_id in order if record_id in by_id]

def read_complete_jsonl_shards(
    output_dir: str,
    prefix: str,
    mode_str: str,
    world_size: int,
    expected_total: Optional[int] = None,
    wait_seconds: float = 60.0,
    dedupe_by_id: bool = False,
) -> List[Dict]:
    """Read all GPU result shards, failing fast if a run is incomplete."""
    expected_shard_paths = [
        os.path.join(output_dir, f"{prefix}_{mode_str}_gpu{gpu_id}.jsonl")
        for gpu_id in range(world_size)
    ]
    extra_shard_paths = glob.glob(os.path.join(output_dir, f"{prefix}_{mode_str}_gpu*.jsonl"))
    shard_paths = sorted(set(expected_shard_paths + extra_shard_paths))

    deadline = time.time() + wait_seconds
    while True:
        missing = [path for path in expected_shard_paths if not os.path.exists(path)]
        if not missing:
            break
        if time.time() >= deadline:
            missing_str = ", ".join(missing)
            raise RuntimeError(f"Missing GPU shard(s), refusing partial merge: {missing_str}")
        print(f"Waiting for {len(missing)} GPU shard(s) before merge...", flush=True)
        time.sleep(2.0)

    merged_data: List[Dict] = []
    for shard in shard_paths:
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    merged_data.append(json.loads(line))

    if dedupe_by_id:
        before = len(merged_data)
        merged_data = dedupe_records_by_id(merged_data)
        if len(merged_data) != before:
            print(f"Deduped merged records by id: {before} -> {len(merged_data)}", flush=True)

    if not merged_data:
        raise RuntimeError("No results to merge.")

    ids = [str(item.get("id", "")) for item in merged_data]
    if any(not item_id for item_id in ids):
        raise RuntimeError("At least one result record is missing an id.")
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate result ids detected, refusing partial merge.")
    if expected_total is not None and len(merged_data) != expected_total:
        raise RuntimeError(
            f"Incomplete merged results: expected {expected_total}, got {len(merged_data)}."
        )

    return merged_data

def merge_multi_gpu_results(base_path: str, num_gpus: int, save_merged: bool = True, cleanup: bool = True) -> List[Dict]:
    """
    [Multi-GPU killer feature] Merge the sharded JSONL files produced by each GPU into a single complete list.
    After a successful merge, automatically clean up the temporary shard files.
    """
    merged_data = []
    files_to_delete = [] # Track the paths of successfully read shard files

    for i in range(num_gpus):
        gpu_file = base_path.replace(".jsonl", f"_gpu{i}.jsonl")
        if os.path.exists(gpu_file):
            with open(gpu_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        merged_data.append(json.loads(line.strip()))
            files_to_delete.append(gpu_file) # Record for later deletion
        else:
            print(f"⚠️ Warning: GPU {i} result file not found: {gpu_file}")

    if save_merged and merged_data:
        # 1. Save the complete file first to ensure data is safely persisted to disk
        save_jsonl(merged_data, base_path)
        print(f"🔗 Successfully merged {len(merged_data)} records into {base_path}")

        # 2. After confirming the merge succeeded, perform cleanup
        if cleanup:
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"⚠️ Warning: Failed to delete {file_path}. Error: {e}")
            if files_to_delete:
                print(f"🧹 Cleaned up {len(files_to_delete)} temporary GPU shard files.")
        
    return merged_data

def save_jsonl(data: List[Dict], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def save_json(data: Any, path: str, indent: int = 4) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==========================================
# ⏱️ Timing
# ==========================================

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"

class Timer:
    """Simple timer"""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return format_duration(self.elapsed)
