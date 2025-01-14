# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import torch.multiprocessing as mp
import subprocess
from diffusers.utils import export_to_video
import signal
import sys
from worker import inference_worker, InferenceWorkerStatus

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/hunyuanvideo-community/HunyuanVideo/model.tar"

# set start method to spawn rather than fork, fork does not work with cuda device contexts
WORLD_SIZE = torch.cuda.device_count()
# WORLD_SIZE = 2
if WORLD_SIZE > 1:
    current_method = mp.get_start_method(allow_none=True)
    if current_method != "spawn":
        print(f"{os.getpid()} setting start method to spawn")
        mp.set_start_method('spawn', force=True)

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)



# def handle_sigusr1(signum, frame):
#     print(f"Received signal: {signum}")
#     # Add your custom handling logic here
#     # For example, clean up resources, log the event, etc.
#     sys.exit(0)  # Optionally, exit the program
# # Register the signal handler
# signal.signal(signal.SIGUSR1, handle_sigusr1)


class MultiGPUPredictor(BasePredictor):
    def setup(self) -> None:
        
        # mp primitives for communication between main process and worker processes
        self.in_queue = [mp.Queue() for _ in range(WORLD_SIZE)]
        self.out_queue = [mp.Queue() for _ in range(WORLD_SIZE)]
        self.barrier = mp.Barrier(WORLD_SIZE)
        self.processes = []
        
        # Download weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
    
        for rank in range(WORLD_SIZE):

            worker_args = (
                self.in_queue[rank],
                self.out_queue[rank],
                self.barrier,
                WORLD_SIZE,
                rank,
                MODEL_CACHE
            )

            p = mp.Process(target=inference_worker, args=worker_args)
            p.start()
            self.processes.append(p)
        
        signal.signal(signal.SIGUSR1, self.handle_sigusr1)
    
    
    def handle_sigusr1(self, signum, frame):
        print(f"Received signal: {signum}")
        # Add your custom handling logic here
        # For example, clean up resources, log the event, etc.
        sys.exit(0)  # Optionally, exit the program


    def predict(
        self,
        prompt: str = Input(
            description="The prompt to guide the video generation",
            default="A cat walks on the grass, realistic style",
        ),
        width: int = Input(
            description="Width of the video in pixels (must be divisible by 16)", 
            default=864, ge=16
        ),
        height: int = Input(
            description="Height of the video in pixels (must be divisible by 16)",
            default=480, ge=16
        ),
        video_length: int = Input(
            description="Number of frames to generate (must be 4k+1, ex: 49 or 129)",
            default=129, ge=1
        ),
        infer_steps: int = Input(
            description="Number of denoising steps",
            default=50, ge=1
        ),
        embedded_guidance_scale: float = Input(
            description="Guidance scale",
            default=6.0, ge=1.0, le=10.0
        ),
        fps: int = Input(
            description="Frames per second of the output video",
            default=24, ge=1
        ),
        seed: int = Input(
            description="Random seed (leave empty for random)",
            default=None
        ),
    ) -> Path:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # enforce constraints on width, height, and video_length
        if width % 16 != 0:
            new_width = (width // 16 + 1) * 16
            print(f"Warning: width {width} is not divisible by 16, rounding up to {new_width}")
            width = new_width
        if height % 16 != 0:
            new_height = (height // 16 + 1) * 16
            print(f"Warning: height {height} is not divisible by 16, rounding up to {new_height}")
            height = new_height
        if video_length % 4 != 1:
            new_video_length = (video_length // 4 + 1) * 4 + 1
            print(f"Warning: video_length {video_length} is not divisible by 4, rounding up to {new_video_length}")
            video_length = new_video_length

        save_path = f"output_{seed}.mp4"
        predict_args = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": video_length,
            "num_inference_steps": infer_steps,
            "guidance_scale": embedded_guidance_scale,
            "seed": seed,
            "fps": fps,
            "save_path": save_path
        }
        print(f"Predict args: {predict_args}")

        # send the predict args to each worker
        for rank in range(WORLD_SIZE):
            self.in_queue[rank].put(predict_args)

        
        worker_status = [self.out_queue[rank].get() for rank in range(WORLD_SIZE)]
        if any([status == InferenceWorkerStatus.ERROR for status in worker_status]):
            for status in worker_status:
                if status == InferenceWorkerStatus.ERROR:
                    raise Exception(f"Predict: Worker {rank} failed with error: {status.get_error_message()}")

        assert os.path.exists(save_path), f"Video not found at {save_path}"
        return Path(save_path)
