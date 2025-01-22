from pathlib import Path
import time
import subprocess
import requests
import sys
import signal

import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

from server.multigpu_util import AsyncGenerationServer

def run_server(model, port, world_size):
    server = AsyncGenerationServer(model, port, world_size)
    server.run()

class HunyuanPredictor:
    """
    A class that wraps the HunyuanVideoPipeline
    setup and predict are both executed in a process per GPU by the multigpu http server
    """

    def __init__(self, model_cache: str, model_url: str, http_port: int, world_size: int):
        self.model_cache = model_cache
        self.model_url = model_url
        self.world_size = world_size

    def setup(self) -> None:
        
        # initialize the distributed process group
        # set the current device to the current rank
        # this means that hereonafter tensor.to("cuda") is equivalent to tensor.to(f"cuda:{dist.get_rank()}")
        dist.init_process_group("nccl")
        torch.cuda.set_device(dist.get_rank())

        # load and parallelize the transformer model
        self.transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            self.model_cache,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,

        )
        self.pipe = HunyuanVideoPipeline.from_pretrained(
            self.model_cache,
            transformer=self.transformer,
            torch_dtype=torch.float16,
        ).to("cuda")

        self.pipe.vae.enable_tiling()

        mesh = init_context_parallel_mesh(
            self.pipe.device.type,
        )
        parallelize_pipe(
            self.pipe,
            mesh=mesh,
        )

        parallelize_vae(self.pipe.vae, mesh=mesh._flatten())

    def predict(
        self,
        prompt: str,
        width: int,
        height: int,
        video_length: int,
        infer_steps: int,
        embedded_guidance_scale: float,
        fps: int,
        seed: int,
        output_path: str,
    ) -> Path:

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # for stress testing
        # if dist.get_rank() == 2 and seed == 1234:
        #     raise Exception("Worker 2 failed :(")

        output = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=video_length,
            num_inference_steps=infer_steps,
            generator=generator,
            guidance_scale=embedded_guidance_scale,
            output_type="pil" if dist.get_rank() == 0 else "pt",
        ).frames[0]

        # all ranks end up with a full copy of the video, but only rank 0 will save it
        if dist.get_rank() == 0:
            export_to_video(output, output_path, fps=fps)

        return Path(output_path)
