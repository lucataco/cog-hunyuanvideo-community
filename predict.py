# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import signal
import sys
import requests
import subprocess

from cog import Input, Path
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from hunyuan_predictor import HunyuanPredictor, run_server


class Predictor:

    def setup(self) -> None:
        """
        setup the predictor and server
        """

        # when the endpoint http://<host>:<port>/predictions/<prediction_id>/cancel is called
        # (user presses the cancel button on the website)
        # cog will send SIGUSR1 to the process that is running predict. When this happens we must 
        # terminate the current prediction and return from predict as soon as possible.
        # this signal handler passes along the cancellation to the multigpu inference server 
        signal.signal(signal.SIGUSR1, self._get_signal_handler())

        # stuff that the server needs to know
        self.WORLD_SIZE = torch.cuda.device_count()
        self.MODEL_CACHE = "checkpoints"
        self.MODEL_URL = "https://weights.replicate.delivery/default/hunyuanvideo-community/HunyuanVideo/model.tar"
        self.HTTP_PORT = 5001
        print(f"inference server will use {self.WORLD_SIZE} devices")

        if not os.path.exists(self.MODEL_CACHE):
            self._download_weights()
        
        # initialize the model specific predictor and server
        # the predictor is a wrapper around the HunyuanVideoPipeline
        # the server is an inference server that runs the predictor
        self.model = HunyuanPredictor(self.MODEL_CACHE, self.WORLD_SIZE, self.HTTP_PORT, self.WORLD_SIZE)
        self.server_process = mp.Process(target=run_server, args=(self.model, self.HTTP_PORT, self.WORLD_SIZE))
        self.server_process.start()
        self.request_cancelled = False # internal flag that is used to manage cancellation of inference requests
        
        # warmup
        predict_args = {
            'prompt': 'A cat walks on the grass, realistic style',
            'width': 864,
            'height': 480,
            'video_length': 61,
            'infer_steps': 1,
            'embedded_guidance_scale': 6.0,
            'fps': 24,
            'seed': 42,
            'output_path': 'warmup.mp4'
        }
        self._request_and_wait(predict_args, timeout=float('inf'))
        print("Server is up and ready")
    

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
        self.request_cancelled = False

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
            print(f"Warning: video_length {video_length}%4 != 1, rounding up to {new_video_length}")
            video_length = new_video_length

        # ensure that height/width/video length result in a sequence length that is divisible by WORLD_SIZE
        seq_length = self._get_seq_length(height, width, video_length)
        if seq_length % self.WORLD_SIZE != 0:
            new_height = self._adjust_height(height, width, video_length, self.WORLD_SIZE)
            print(f"Warning: sequence length {seq_length} is not divisible by WORLD_SIZE {self.WORLD_SIZE}, rounding height up to {new_height}")
            height = new_height

        # send request to the server and wait
        save_path = f"/tmp/output_{seed}.mp4"
        predict_args = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "video_length": video_length,
            "infer_steps": infer_steps,
            "embedded_guidance_scale": embedded_guidance_scale,
            "seed": seed,
            "fps": fps,
            "output_path": save_path
        }

        self._request_and_wait(predict_args)

        return Path(save_path)


    def _get_signal_handler(self):
        
        def signal_handler(signum, frame):
            print(f"Received signal: {signum}")

            # pass on cancellation message to the server
            # this will result in the server's worker processes stopping
            # and returning once the current diffusion step has finished
            cancel_url = f"http://localhost:{self.HTTP_PORT}/cancel"
            response = requests.post(cancel_url)
            print(f"Cancel response: {response.json()}")
            
            # set internal flag to true, this causes us to stop polling the server
            self.request_cancelled = True
            assert response.status_code == 200
        
        return signal_handler
    

    def _request_and_wait(self, predict_args, timeout=20):
        """
        sent a prediction request to the server and wait, wait for the server to finish generating the video
        once the server has finished generating the video, it will return a 200 response and the video will be
        saved to predict_args['output_path']
        """
        # make sure server is running, if not restart it
        if not self.server_process.is_alive():
            self._restart_server()
        
        # make sure server is ready to accept requests
        # if it is not up for 20 seconds, restart it
        seconds_waited = 0
        while not self._does_endpoint_exist(f"http://localhost:{self.HTTP_PORT}/server_status"):
            print(f"waiting for inference server to start")
            time.sleep(2)
            seconds_waited += 2

            if seconds_waited > timeout:
                self._restart_server()
                seconds_waited = 0
        
        
        # call generate endpoint, we expect a 202 response meaning that the request was accepted
        print(f"Requesting generation with args: {predict_args}")
        generate_url = f"http://localhost:{self.HTTP_PORT}/generate"
        response = requests.post(generate_url, json=predict_args)
        print(f"Response: {response.json()}")
        assert response.status_code == 202
        assert response.json()['request_id'] is not None
        request_id = response.json()['request_id']

        # poll the status endpoint until we get a 200 response meaning that the request is complete
        # if a cancellation signal is recieved from cog, the signal handler will pass this on to the inference
        # and set self.request_cancelled to true. Once self.request_cancelled is true, we should exit immediately
        while not self.request_cancelled:
            print(f"Requesting status for {request_id}")
            status_url = f"http://localhost:{self.HTTP_PORT}/request_status/{request_id}"
            response = requests.post(status_url)
            print(f"Status response: {response.json()}")
            
            # if the request is still in progress, sleep for 1 second and continue
            if response.status_code == 202:
                time.sleep(1)
                continue
            
            # if the request is complete, break out of the loop
            # the video will be saved to predict_args['output_path']
            elif response.status_code == 200:
                break

            # something went wrong, restart the server and fail this request
            elif response.status_code == 500:
                self._restart_server()
                raise Exception(f"Recieved error resonse from server {response.json()}")
            
            # something unexpected happened, fail this request
            else:
                self._restart_server()
                raise Exception(f"Recieved unexpected response from server {response.json()}")
        
    def _does_endpoint_exist(self, url):
        """
        check if an endpoint exists
        """
        try:
            server_status_url = f"http://localhost:{self.HTTP_PORT}/server_status"
            response = requests.get(server_status_url)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _restart_server(self):
        """
        restart the server
        """
        print(f"Restarting server")
        self.server_process.terminate()
        self.server_process = mp.Process(target=run_server, args=(self.model, self.HTTP_PORT, self.WORLD_SIZE))
        self.server_process.start()

    def _download_weights(self):
        start = time.time()
        print("downloading url: ", self.MODEL_URL)
        print("downloading to: ", self.MODEL_CACHE)
        subprocess.check_call(["pget", "-xf", self.MODEL_URL, self.MODEL_CACHE], close_fds=False)
        print("downloading took: ", time.time() - start)

    def _get_seq_length(self, height, width, num_frames):
        """
        get the sequence length of the latent space
        """
        latent_height = height // 16
        latent_width = width // 16
        latent_frames = (num_frames - 1) // 4 + 1
        seq_length = latent_height * latent_width * latent_frames
        return seq_length

    def _adjust_height(self, height, width, num_frames, W):
        """
        adjust the height to ensure that the sequence length is divisible by WORLD_SIZE
        """
        
        # if sequence length is divisible by world size, no adjustment needed
        if self._get_seq_length(height, width, num_frames) % W == 0:
            return height

        # otherwise, round up height to the nearest multiple of 16
        adjusted_height = ((height + 15) // 16) * 16
        
        # Keep incrementing by 16 until divisibility condition is met
        while self._get_seq_length(adjusted_height, width, num_frames) % W != 0:
            adjusted_height += 16
        
        return adjusted_height