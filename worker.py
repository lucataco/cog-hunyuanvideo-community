
import torch.multiprocessing as mp
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
import os
import torch
import torch.distributed as dist
from enum import Enum

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

class InferenceWorkerStatus(Enum):
    GOOD = "GOOD"
    ERROR = "ERROR"

    def __init__(self, status):
        self._error_message = None

    def set_error_message(self, message: str):
        if self == InferenceWorkerStatus.ERROR:
            self._error_message = message

    def get_error_message(self) -> str:
        if self == InferenceWorkerStatus.ERROR:
            return self._error_message
        return ""

def inference_worker(
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    barrier: mp.Barrier,
    world_size: int,
    rank: int,
    model_cache: str,
):

    # normally torch.distributed does this for us but we are managing process creation
    # so we need to set these environment variables manually
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')

    cache_threshold = os.environ.get("CACHE_THRESHOLD")
    assert cache_threshold is not None, "CACHE_THRESHOLD must be set"
    cache_threshold = float(cache_threshold)
    print(f"Cache threshold: {cache_threshold}")
    
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_cache,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        revision="refs/pr/18"
    )
    
    pipe = HunyuanVideoPipeline.from_pretrained(
        model_cache,
        transformer=transformer,
        torch_dtype=torch.float16,
        revision="refs/pr/18"
    ).to("cuda")

    mesh = init_context_parallel_mesh(
        pipe.device.type,
    )
    pipe = parallelize_pipe(pipe, mesh=mesh)
    pipe.vae = parallelize_vae(pipe.vae, mesh=mesh._flatten())
    pipe = apply_cache_on_pipe(pipe, residual_diff_threshold=cache_threshold)
    pipe.vae.enable_tiling()

    while True:
        try:
            predict_args = in_queue.get()
            barrier.wait()

            generator = torch.Generator("cuda").manual_seed(predict_args["seed"])

            output = pipe(
                prompt=predict_args["prompt"],
                height=predict_args["height"],
                width=predict_args["width"],
                num_frames=predict_args["num_frames"],
                num_inference_steps=predict_args["num_inference_steps"],
                guidance_scale=predict_args["guidance_scale"],
                generator=generator
            ).frames[0]

            if rank == 0:
                export_to_video(output, predict_args["save_path"], fps=predict_args["fps"])
            out_queue.put(InferenceWorkerStatus.GOOD)
        
        except Exception as e:
            print(f"Worker {rank} failed with error: {e}")
            error_status = InferenceWorkerStatus.ERROR
            error_status.set_error_message(str(e))
            out_queue.put(error_status)
