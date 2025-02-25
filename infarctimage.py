"""
    Author: Gavit0
    Date: 20250205
    This file is a library for generating infarct images.
    InfarctImage is a LoRA-based model fine-tuned on Stable Diffusion 2.1 to generate realistic images of individuals simulating a heart attack. This model was developed to facilitate synthetic dataset generation for human activity recognition and medical emergency monitoring applications.
    License: MIT
"""
__version__ = "0.0.1"

import os
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from torch.utils.data import Dataset

class InfarctImageCreator:
    """
    InfarctImage class to handle model loading, image generation, and dataset management.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the InfarctImage model.
        
        Args:
            device: Device to load the model on ("cuda" or "cpu").
        """
        self.device = device
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load the base Stable Diffusion model and apply LoRA fine-tuned weights.            
        
        Returns:
            Loaded model with LoRA weights applied.
        """
        print("Loading base model with device:", self.device, "...")
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, use_safetensors=True).to(self.device)

        print("Applying LoRA fine-tuned weights...")
        pipe.load_lora_weights("Gavit0/InfarctImage")
        
        print("Model loaded successfully.")
        return pipe
    
    def generate_image(self, prompt, negative_prompt = (
                "blurry, deformed face, bad anatomy, poorly drawn face, out of focus, ugly, noisy, extra fingers, "
                "distorted, grainy, worst quality, low quality, low resolution, illustration, "
                "dull, watermark, close-up, 3d, 2d, painting, sketch, render, cartoon, grain, kitsch"
                ), verbose = False
      ):
        """
        Generate a single image based on a given prompt.            
        
        Args:
            prompt: The text prompt to generate images.            
            negative_prompt: Optional negative prompt to avoid unwanted features.
            verbose: Print the generated prompt.

        Returns:
            Image: The generated image.
        """
        trigger = "Person with expression of pain due to a heart attack"
        full_prompt = f"{trigger}, {prompt}"
        if verbose: print(f"Generating imagewith prompt: {full_prompt}")
        image = self.model(prompt=full_prompt, 
                            negative_prompt=negative_prompt,
                            guidance_scale=4, num_inference_steps=40, 
                            num_images_per_prompt=1).images[0]
        return image

    def generate_images(self, prompts, num_images=1, 
            negative_prompt = (
                "blurry, deformed face, bad anatomy, poorly drawn face, out of focus, ugly, noisy, extra fingers, "
                "distorted, grainy, worst quality, low quality, low resolution, illustration, "
                "dull, watermark, close-up, 3d, 2d, painting, sketch, render, cartoon, grain, kitsch"
                ), verbose = False
      ):
        """
        Generate images based on a given prompt.            
        
        Args:
            prompts: The text prompt to generate images. Could be a single string or a list of prompts.
            num_images: Number of images to generate.            
            negative_prompt: Optional negative prompt to avoid unwanted features.
            verbose: Print the generated prompt.
        
        Returns:
            List: of generated images.
        """
        trigger = "Person with expression of pain due to a heart attack"

        images = []
        if isinstance(prompts, list):
            if len(prompts) == 0: prompts = [""] # Empty prompt
                
            for _ in range(num_images):
                prompt = prompts[torch.randint(0, len(prompts), (1,)).item()]
                images += [self.generate_image(prompt, negative_prompt, verbose=verbose)]
        else:
            full_prompt = f"{trigger}, {prompts}"
            if verbose: print(f"Generating {num_images} image(s) with prompt: {full_prompt}")
            images = self.model(prompt=full_prompt, 
                                negative_prompt=negative_prompt,
                                guidance_scale=4, num_inference_steps=40, 
                                num_images_per_prompt=num_images).images
        return images
    
    def generate_to_disk(self, prompts, num_images=100, save_dir="./InfarctImage_Dataset", verbose=False):
        """
        Generate and save a dataset of images based on a given prompt.            
        
        Args:
            prompts: The text prompt to generate images. Could be a single string or a list of prompts.           
            num_images: Number of images to generate.            
            save_dir: Directory to save the generated images.
            verbose: Print the generated prompt.
        """
        os.makedirs(save_dir, exist_ok=True)
        if len(prompts) == 0: prompts = [""] # Empty prompt            
        for i in range(num_images):
            prompt = prompts[torch.randint(0, len(prompts), (1,)).item()]
            image = self.generate_image(prompt, verbose=verbose)
            image.save(os.path.join(save_dir, f"infarct_image_{i}.png"))
        print(f"Generated {num_images} images saved to {save_dir}")

    def generate_dataset(self, prompts, num_images=100, transform=None):
        """
        Generate and return a torch Dataset of images based on a given prompt.

        Args:
            prompts: The text prompt to generate images. Could be a single string or a list of prompts.
            num_images: Number of images to generate.
            transform: Optional transformations to apply to the images.

        Returns:
            InfarctImageDataset: A custom dataset of generated images.
        """
        class InfarctImageDataset(Dataset):
            def __init__(self, creator:InfarctImageCreator, prompts, num_images=100, transform=None):
                self.creator = creator
                self.prompts = prompts
                self.num_images = num_images
                self.transform = transform

            def __len__(self):
                return num_images
            
            def __getitem__(self, idx):
                #image = Image.open(img_path).convert('RGB')
                image = self.creator.generate_image(self.prompts[idx % len(self.prompts)])                
                if self.transform:
                    image = self.transform(image)
                return image
            
        return InfarctImageDataset(self, prompts, num_images, transform)

class InfarctImageDataset(Dataset):
    """
    A custom dataset for loading infarct images from a specified directory.
    The dataset supports train, validation, and test sets, and applies optional transformations.
    """
    
    def __init__(self, data_dir="./InfarctImage_Dataset", split='train', transform=None, force_download=False):
        """
        Args:
            data_dir (str): Path to the base dataset directory.
            split (str): The dataset split to use ('train', 'val', or 'test').
            transform (callable, optional): A function/transform to apply to the images.
            force_download (bool): Replaces the dataset if it exists.
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        if force_download or not os.path.exists(self.data_dir):
            self.download_dataset(data_dir)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset not found in {self.data_dir}. Run download_dataset() first.")
        
        self.image_files = [f for f in os.listdir(self.data_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Loads and returns an image at the specified index.
        
        Args:
            idx (int): Index of the image to load.
        
        Returns:
            Tensor: Transformed image tensor.
        """
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image

    def download_dataset(self, save_path="./InfarctImage_Dataset"):
        """
        Download and extract the dataset from Kaggle.

        Args:
            save_path (str): Directory to store the downloaded dataset.
        """
        try:
            import kaggle
        except ImportError:
            print("Please install the Kaggle API to download the dataset.")
            print("Run the following command: !pip install kaggle")
            return
        
        print("Downloading dataset from Kaggle...")
        os.makedirs(save_path, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files("gavit0/infarct-image", path=save_path, unzip=True)



        print(f"Dataset downloaded and extracted to {save_path}")
