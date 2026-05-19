import json
import os
from pathlib import Path
from typing import List, Dict, Any
from seed.utils import get_project_root


PROJECT_ROOT = Path(get_project_root())

class SEEDDataLoader:
    def __init__(self, mode: str, **kwargs):
        """
        SEED multimodal dataset loader
        """
        self.mode = mode.lower()
        self.dataset = []
        
        if self.mode == "local":
            default_data_path = PROJECT_ROOT / "dataset" / "train.jsonl"
            default_image_root = PROJECT_ROOT / "dataset"
            
            self.data_path = Path(kwargs.get("data_path", default_data_path))
            self.image_root = Path(kwargs.get("image_root", default_image_root))
            
            self.dataset = self._load_local_data()
            print(f"📦 [Local Mode] Loaded {len(self.dataset)} samples from {self.data_path}")
            
        elif self.mode == "hf":
            dataset_name = "LightChen2333/M3CoT"
            split = "train"
            self._load_hf_data(dataset_name, split)
            print(f"☁️ [HF Mode] Loaded {len(self.dataset)} samples from HuggingFace ({dataset_name}[{split}])")
            
        else:
            raise ValueError("Mode must be either 'local' or 'hf'")

    def _load_local_data(self) -> List[Dict[str, Any]]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"❌ Local dataset file not found: {self.data_path}")

        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def _load_hf_data(self, dataset_name: str, split: str):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("❌ Please install datasets library: pip install datasets")
            
        # HuggingFace datasets automatically handles image downloading and caching
        hf_dataset = load_dataset(dataset_name, split=split)

        # Convert HF Dataset object to list to keep indexing consistent with local mode
        # Note: we are not loading all images into memory here; HF's Image feature is lazy-loaded
        self.dataset = [item for item in hf_dataset]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Unified external interface: regardless of whether the backend is local or hf, the output data format must be absolutely consistent!
        """
        raw_item = self.dataset[idx]
        
        # ====================================================
        # ⚠️ Note: the "question" and "answer" keys here need to be adjusted to match your actual keys
        # Since these two columns were truncated in the screenshot, I'm using generic names as placeholders
        # ====================================================
        item_id = raw_item.get("id", str(idx))
        context = raw_item.get("context", "")
        question = raw_item.get("question", "")
        ground_truth = raw_item.get("answer", "")
        choices = raw_item.get("choices", None)
        rationale = raw_item.get("rationale", None)  
        
        image_obj_or_path = None

        if self.mode == "local":
            image_path = raw_item.get("image_path", "")
            if image_path:
                # Build the full local path, e.g.: ../dataset/images/commonsense-physical-commonsense-5091.png
                img_path = self.image_root / f"{image_path}"
                if img_path.exists():
                    # Pass the path string directly here; Qwen3's processor supports reading from a path
                    image_obj_or_path = str(img_path.resolve())
                else:
                    print(f"⚠️ Warning: Image not found -> {img_path}")
                    
        elif self.mode == "hf":
            # HuggingFace's image field is usually a PIL.Image object directly
            # Qwen3's processor perfectly supports passing in a PIL Image!
            if "image" in raw_item:
                image_obj_or_path = raw_item["image"]
            else:
                print(f"⚠️ Warning: No 'image' field found in HF dataset item")

        return {
            "id": item_id,
            "image": image_obj_or_path,  # Could be a local absolute path URI, or a PIL.Image object
            "question": question,
            "context": context,
            "choices": choices,
            "ground_truth": ground_truth,
            "rationale": rationale
        }