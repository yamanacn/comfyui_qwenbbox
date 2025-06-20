import os
import torch
import folder_paths
from dataclasses import dataclass
from typing import Dict, Any, List
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
import json
import ast
from PIL import Image
import requests
import traceback


def parse_json(json_output: str) -> str:
    """Extract the JSON payload from a model response string."""
    if "```json" in json_output:
        json_output = json_output.split("```json", 1)[1]
        json_output = json_output.split("```", 1)[0]

    try:
        parsed = json.loads(json_output)
        if isinstance(parsed, dict) and "content" in parsed:
            inner = parsed["content"]
            if isinstance(inner, str):
                json_output = inner
    except Exception:
        pass
    return json_output


@dataclass
class DetectedBox:
    bbox: List[int]
    score: float
    label: str = ""


def parse_boxes(
    text: str,
    img_width: int,
    img_height: int,
    input_w: int,
    input_h: int,
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Return bounding boxes parsed from the model's raw JSON output."""
    text = parse_json(text)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            end_idx = text.rfind('"}') + len('"}')
            truncated = text[:end_idx] + "]"
            data = ast.literal_eval(truncated)
    if isinstance(data, dict):
        inner = data.get("content")
        if isinstance(inner, str):
            try:
                data = ast.literal_eval(inner)
            except Exception:
                data = []
        else:
            data = []
    items: List[DetectedBox] = []
    x_scale = img_width / input_w
    y_scale = img_height / input_h

    for item in data:
        box = item.get("bbox_2d") or item.get("bbox") or item
        label = item.get("label", "")
        score = float(item.get("score", 1.0))
        y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        abs_y1 = int(y1 * y_scale)
        abs_x1 = int(x1 * x_scale)
        abs_y2 = int(y2 * y_scale)
        abs_x2 = int(x2 * x_scale)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        if score >= score_threshold:
            items.append(DetectedBox([abs_x1, abs_y1, abs_x2, abs_y2], score, label))
    items.sort(key=lambda x: x.score, reverse=True)
    return [
        {"score": b.score, "bbox": b.bbox, "label": b.label}
        for b in items
    ]


@dataclass
class QwenModel:
    model: Any
    processor: Any
    device: str
    original_params: Dict[str, Any] = None

class LoadQwenModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                ],),
                "device": (["auto", "cuda:0", "cuda:1", "cpu"],),
                "precision": (["FP16", "INT8"],),
                "attention": (["flash_attention_2", "sdpa"],),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load"
    CATEGORY = "qwen_object_v2"

    def load(self, model_name: str, device: str, precision: str, attention: str):
        original_params = {
            "model_name": model_name,
            "device": device,
            "precision": precision,
            "attention": attention,
        }
        model_dir = os.path.join(folder_paths.models_dir, "Qwen", model_name.replace("/", "_"))
        
        if not os.path.exists(model_dir) or not any(os.listdir(model_dir)):
            print(f"本地模型不存在 '{model_dir}'，将自动下载...")
            snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        
        device_map = "auto" if device == "auto" else {"": device}
        
        torch_dtype = torch.float16  # Default to float16 for both FP16 and INT8 loading
        
        quant_config = None
        if precision == "INT8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        attn_impl = attention

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            device_map=device_map,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

        return (QwenModel(model, processor, device, original_params),)


class QwenBbox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "image": ("IMAGE",),
                "target": ("STRING", {"default": "object"}),
                "bbox_selection": ("STRING", {"default": "all"}),
                "score_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "merge_boxes": ("BOOLEAN", {"default": False}),
                "unload_after_detection": ("BOOLEAN", {"default": True}),
                "sort_method": (["none", "left_to_right", "top_to_bottom", "right_to_left", "bottom_to_top"], {"default": "none"}),
            },
        }

    RETURN_TYPES = ("JSON", "BBOX")
    RETURN_NAMES = ("text", "bboxes")
    FUNCTION = "detect"
    CATEGORY = "qwen_object_v2"

    def detect(
        self,
        qwen_model: QwenModel,
        image,
        target: str,
        bbox_selection: str = "all",
        score_threshold: float = 0.0,
        merge_boxes: bool = False,
        unload_after_detection: bool = True,
        sort_method: str = "none",
    ):
        model = qwen_model.model
        processor = qwen_model.processor
        
        if model is None or processor is None:
            # For simplicity in this new version, we'll just raise an error.
            # In a real-world robust scenario, we'd implement the auto-reload logic.
            raise ValueError("Qwen Model is not loaded. Please connect a valid model.")

        device = qwen_model.device
        if device == "auto":
            device = str(next(model.parameters()).device)
        
        prompt = f"Locate the {target} and output bbox in JSON"

        if isinstance(image, torch.Tensor):
            image = (image.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"image": image}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        
        gen_ids = [output_ids[len(inp):] for inp, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        
        # Estimate input dimensions for scaling
        try:
            input_h = inputs['image_grid_thw'][0][1] * 14
            input_w = inputs['image_grid_thw'][0][2] * 14
        except (KeyError, IndexError):
            input_h, input_w = 336, 336 # Fallback
            
        items = parse_boxes(
            output_text, image.width, image.height, input_w, input_h, score_threshold
        )

        boxes = items
        if bbox_selection.strip().lower() != "all":
            try:
                idxs = [int(p) for p in bbox_selection.replace(",", " ").split() if p.isdigit()]
                boxes = [boxes[i] for i in idxs if 0 <= i < len(boxes)]
            except:
                pass # Ignore parsing errors, return all

        if merge_boxes and boxes:
            x1 = min(b["bbox"][0] for b in boxes)
            y1 = min(b["bbox"][1] for b in boxes)
            x2 = max(b["bbox"][2] for b in boxes)
            y2 = max(b["bbox"][3] for b in boxes)
            score = max(b["score"] for b in boxes)
            label = boxes[0].get("label", target)
            boxes = [{"bbox": [x1, y1, x2, y2], "score": score, "label": label}]

        bboxes_only = [b["bbox"] for b in boxes]

        if sort_method != "none" and bboxes_only:
            sort_key = None
            reverse = False
            if sort_method == "left_to_right": sort_key = lambda bbox: bbox[0]
            elif sort_method == "right_to_left": sort_key, reverse = (lambda bbox: bbox[0]), True
            elif sort_method == "top_to_bottom": sort_key = lambda bbox: bbox[1]
            elif sort_method == "bottom_to_top": sort_key, reverse = (lambda bbox: bbox[1]), True
            
            if sort_key:
                sorted_indices = sorted(range(len(boxes)), key=lambda i: sort_key(boxes[i]['bbox']), reverse=reverse)
                boxes = [boxes[i] for i in sorted_indices]
                bboxes_only = [b["bbox"] for b in boxes]

        json_output = json.dumps(boxes, ensure_ascii=False)
        
        if unload_after_detection and device.startswith("cuda"):
            import gc
            qwen_model.model = None
            qwen_model.processor = None
            gc.collect()
            torch.cuda.empty_cache()
            
        return (json_output, bboxes_only)


class BBoxToSAM_v2:
    """Convert a list of bounding boxes to the format expected by SAM nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"bboxes": ("BBOX",)}}

    RETURN_TYPES = ("BBOXES",)
    RETURN_NAMES = ("sam2_bboxes",)
    FUNCTION = "convert"
    CATEGORY = "qwen_object_v2"

    def convert(self, bboxes):
        if not isinstance(bboxes, list):
            return ([],) # Return empty batched list on error

        # If already batched, return as-is
        if bboxes and isinstance(bboxes[0], (list, tuple)) and len(bboxes[0]) > 0 and isinstance(bboxes[0][0], (list, tuple)):
            return (bboxes,)

        return ([bboxes],)


NODE_CLASS_MAPPINGS = {
    "LoadQwenModel_v2": LoadQwenModel,
    "QwenBbox": QwenBbox,
    "BBoxToSAM_v2": BBoxToSAM_v2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadQwenModel_v2": "Load Qwen Model (v2)",
    "QwenBbox": "Qwen Bbox Detection",
    "BBoxToSAM_v2": "Prepare BBox for SAM (v2)",
} 