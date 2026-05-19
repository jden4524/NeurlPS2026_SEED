# seed/evaluator.py
import json
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForImageTextToText

from .config import SEEDConfig
from .data_loader import SEEDDataLoader
from .generation_core import SEEDGenerator
from .prompts import PromptManager
from .utils import append_jsonl, parse_mcq_answer, partition_indices

class SEEDEvaluator:
    def __init__(self, 
                 model_path: str, 
                 config: SEEDConfig, 
                 data_loader: SEEDDataLoader, 
                 gpu_id: int = 0,
                 world_size: int = 1,
                 output_dir: str = "outputs"):
        
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.data_loader = data_loader
        
        self.output_dir = Path(__file__).resolve().parent.parent / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[GPU {gpu_id}] ⏳ Loading Model & Processor: {model_path}")
        
        # 🌟 1. Model-family identification
        self.is_llama = "llama" in model_path.lower()

        # Load the unified AutoModel and Processor
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map={"": self.device}, 
            attn_implementation="sdpa", 
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        
        # 🌟 2. Dynamically assign the Adapter
        if self.is_llama:
            from .model_adapters.mllama_adapter import MllamaAdapter
            print(f"[GPU {gpu_id}] 🔌 Activating Mllama cross-attention adapter")
            self.adapter = MllamaAdapter(self.model, self.processor)
        else:
            from .model_adapters.qwen_vl_adapter import QwenVLAdapter
            print(f"[GPU {gpu_id}] 🔌 Activating Qwen self-attention adapter")
            self.adapter = QwenVLAdapter(self.model, self.processor)
            
        self.generator = SEEDGenerator(self.model, self.processor, self.adapter, config)

    def evaluate(self, use_seed: bool = True, max_new_tokens: int = 1024, resume: bool = True):
        mode_str = "seed" if use_seed else "base"
        out_file = self.output_dir / f"m3cot_results_{mode_str}_gpu{self.gpu_id}.jsonl"
        
        processed_ids = set()
        correct_count = 0
        total_processed = 0
        
        if resume and out_file.exists():
            with open(out_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line.strip())
                        processed_ids.add(str(record["id"]))
                        total_processed += 1
                        if record.get("correct", False):
                            correct_count += 1
            print(f"[GPU {self.gpu_id}] 🔄 Resume: Found {len(processed_ids)} completed samples.")
        
        total_samples = len(self.data_loader)
        my_indices = partition_indices(total_samples, self.gpu_id, self.world_size)
        my_processed_count = sum(1 for idx in my_indices if str(self.data_loader[idx]["id"]) in processed_ids)

        for idx in tqdm(
            my_indices, 
            total=len(my_indices),
            initial=my_processed_count,
            desc=f"GPU {self.gpu_id} Eval", 
            position=self.gpu_id,
            leave=True,
            dynamic_ncols=True
        ):
            sample = self.data_loader[idx]
            sample_id_str = str(sample["id"])
            
            if sample_id_str in processed_ids:
                continue

            formatted_text = PromptManager.get_m3cot_prompt(
                question=sample["question"], choices=sample["choices"], context=sample["context"]
            )
            
            # ==========================================
            # 🌟 3. Differentiated input assembly (Chat Template)
            # ==========================================
            if self.is_llama:
                # 🦙 Llama 3.2 Vision official assembly format
                messages = [{"role": "user", "content": []}]
                if sample.get("image"):
                    messages[0]["content"].append({"type": "image"}) # The Mllama template does not pass the actual image object
                messages[0]["content"].append({"type": "text", "text": formatted_text})

                # Mllama standard preprocessing: first obtain the prompt text, then let the processor handle the image
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

                if sample.get("image"):
                    inputs = self.processor(images=sample["image"], text=prompt, return_tensors="pt").to(self.device)
                else:
                    inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)

                # 🚨 Critically important: force-cast Mllama's pixel_values to bfloat16, otherwise cross-attention will definitely error out!
                for k, v in inputs.items():
                    if v.dtype in [torch.float32, torch.float16]:
                        inputs[k] = v.to(torch.bfloat16)

            else:
                # 🚀 Qwen official assembly format
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": PromptManager.get_system_prompt()}]},
                    {"role": "user", "content": []}
                ]
                if sample.get("image"):
                    messages[1]["content"].append({"type": "image", "image": sample["image"], "max_pixels": 200704})
                messages[1]["content"].append({"type": "text", "text": formatted_text})

                # Qwen does it in one shot
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                ).to(self.device)

            # ==========================================
            # 4. Generation and post-processing
            # ==========================================
            try:
                if use_seed:
                    # 🟢 [Intervention mode] The SEED engine automatically calls back the Adapter's injection logic
                    output_ids, stats = self.generator.generate(
                        inputs, max_new_tokens=max_new_tokens, return_stats=True
                    )
                    injections = stats.get("injections_triggered", 0)
                    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
                else:
                    # 🔴 [Baseline mode] Call the official API directly
                    with torch.no_grad():
                        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                    injections = 0
                    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

                # Differentiated decoding: Llama (processor only) vs Qwen (nested tokenizer)
                if self.is_llama:
                    generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
                else:
                    generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
            except Exception as e:
                print(f"\n[GPU {self.gpu_id}] ❌ Error on sample {sample_id_str}: {e}")
                generated_text = f"[ERROR] {str(e)}"
                injections = 0

            # Answer evaluation and result saving
            pred_answer = parse_mcq_answer(generated_text)
            gt_answer = str(sample["ground_truth"]).strip().upper()
            is_correct = (pred_answer == gt_answer)
            
            if is_correct: correct_count += 1
            total_processed += 1

            res_dict = {
                "id": sample_id_str, "question": sample["question"], "ground_truth": gt_answer,
                "predicted": pred_answer, "correct": is_correct, "generated_text": generated_text,
                "injections_triggered": injections
            }
            append_jsonl(res_dict, str(out_file))
            processed_ids.add(sample_id_str)

        acc = correct_count / total_processed if total_processed > 0 else 0
        print(f"\n[GPU {self.gpu_id}] ✅ Finished. Accuracy: {acc:.4f} ({correct_count}/{total_processed})")
