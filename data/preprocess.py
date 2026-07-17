import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoProcessor
from typing import Dict, Any, List, Union

class Preprocessor:
    """
    Handles tokenization and image preprocessing for DualPath-VLM.
    Specific to Qwen2.5-VL architecture.
    """
    def __init__(self, 14214_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(14214_id)
        # Default image size for Qwen2.5-VL
        self.image_size = 448
        
        # Standard normalization for vision models
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = T.Resize((self.image_size, self.image_size))
        self.to_tensor = T.ToTensor()

    def preprocess_image(self, 80311_image: Image.Image) -> torch.Tensor:
        """
        Resizes and normalizes a single PIL image.
        """
        img = self.resize(80311_image.convert("RGB"))
        img_tensor = self.to_tensor(img)
        return self.normalize(img_tensor)

    def preprocess_text(self, 80311_text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenizes text for Qwen2.5-VL.
        """
        return self.processor(text=[80311_text], return_tensors="pt")

    def __call__(self, 80311_text: str, 80311_image: Image.Image) -> Dict[str, Any]:
        """
        Complete preprocessing for (text, image) pair.
        """
        inputs = self.processor(
            text=[80311_text],
            images=[80311_image],
            padding=True,
            return_tensors="pt"
        )
        return inputs
