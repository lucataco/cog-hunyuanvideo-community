# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from diffusers.utils import export_to_video
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/hunyuanvideo-community/HunyuanVideo/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        # Download weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        
        self.transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            MODEL_CACHE,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self.pipe = HunyuanVideoPipeline.from_pretrained(
            MODEL_CACHE,
            transformer=self.transformer,
            torch_dtype=torch.float16,
        )
        self.pipe.vae.enable_tiling()
        self.pipe.to("cuda")

    def predict(
        self,
        prompt: str = Input(
            description="The prompt to guide the video generation",
            default="A cat walks on the grass, realistic",
        ),
        width: int = Input(
            description="Width of the video in pixels", 
            default=960, ge=512, le=1280
        ),
        height: int = Input(
            description="Height of the video in pixels",
            default=544, ge=320, le=720
        ),
        num_frames: int = Input(
            description="Number of frames to generate (must be 4k+1, ex: 49 or 129)",
            default=61, ge=49, le=129
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=30, ge=1, le=50
        ),
        guidance_scale: float = Input(
            description="Guidance scale",
            default=6.0, ge=1.0, le=10.0
        ),
        fps: int = Input(
            description="Frames per second of the output video",
            default=15, ge=1, le=30
        ),
        seed: int = Input(
            description="Random seed (0 for random)",
            default=0
        ),
    ) -> Path:
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        ).frames[0]

        output_path = "/tmp/output.mp4"
        export_to_video(output, output_path, fps=fps)
        return Path(output_path)
