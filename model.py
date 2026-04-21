import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torchvision.transforms as transforms

class BEVControlNetGenerator:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing BEVControlNetGenerator on {self.device}...")
        self._load_models()

    def _load_models(self):
        print("Loading ControlNet models (Depth & Segmentation)...")
        controlnet_depth = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", 
            torch_dtype=torch.float16
        )

        controlnet_seg = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", 
            torch_dtype=torch.float16
        )

        controlnets = [controlnet_depth, controlnet_seg]

        print("Loading Stable Diffusion Pipeline...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnets,
            torch_dtype=torch.float16
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()
        print("ControlNet Pipeline ready!")

    def generate(self, prompt, depth_img, seg_img, depth_scale=1.0, seg_scale=0.8, num_inference_steps=20):
        """
        Generates an image based on prompt, depth, and segmentation inputs.
        Expects depth_img and seg_img to be PIL Images or tensors convertible to PIL.
        """
        # Convert tensors to PIL if needed
        if isinstance(depth_img, torch.Tensor):
            depth_img = transforms.ToPILImage()(depth_img)
        if isinstance(seg_img, torch.Tensor):
            seg_img = transforms.ToPILImage()(seg_img)

        print(f"Generating image for prompt: '{prompt}'...")
        generated_image = self.pipe(
            prompt,
            image=[depth_img, seg_img],
            controlnet_conditioning_scale=[depth_scale, seg_scale],
            num_inference_steps=num_inference_steps
        ).images[0]
        
        return generated_image

# Example usage:
# generator = BEVControlNetGenerator()
# batch = next(iter(dataloader))
# depth_tensor = batch['depth'][0]
# seg_tensor = batch['seg'][0]
# generated_bev = generator.generate(
#     prompt="A highly detailed top-down view of a city street, photorealistic",
#     depth_img=depth_tensor,
#     seg_img=seg_tensor
# )
# display(generated_bev)