"""
A handy utility for verifying image generation locally.
To set up, first run a local cog server using:
   cog run -p 5000 python -m cog.server.http
Then, in a separate terminal, generate samples
   python samples.py
"""

import base64
import sys
import time
from pathlib import Path
import requests
import shutil
import os
import random
def gen(output_fn, **kwargs):
    st = time.time()
    print("Generating", output_fn)
    url = "http://localhost:5000/predictions/abc123"
    response = requests.put(url, json={"input": kwargs})
    data = response.json()
    print("Generated in: ", time.time() - st)
    print(data)

    if data['status'] == "succeeded":
        datauri = data["output"]
        base64_encoded_data = datauri.split(",")[1]
        content = base64.b64decode(base64_encoded_data)
        Path(output_fn).write_bytes(content)
        return data['status']

    else:
        return data['status']

random.seed(1234)
width_range = [128, 1270]
height_range = [128, 780]
video_length_range = [1, 130]
infer_steps_range = [1, 50]

fps = 24
from video_prompts import video_prompts

def test_prompts():
    """
    runs generations in fp8 and bf16 on the same node! wow!
    """
    

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for prompt in video_prompts:
        
        filename = prompt.split(" ")[3:6]
        filename = "_".join(filename)
        filename = f"{filename}.mp4"
        output_fn = output_dir / filename

        width = random.randint(width_range[0], width_range[1])
        height = random.randint(height_range[0], height_range[1])
        video_length = random.randint(video_length_range[0], video_length_range[1])
        infer_steps = random.randint(infer_steps_range[0], infer_steps_range[1])
        embedded_guidance_scale = 8.0
        
        # Generating Sailing ship in storm with width 2031, height 1291, video_length 118, infer_steps 43, embedded_guidance_scale 5.269843510825848

        print(f"Generating {prompt} with width {width}, height {height}, video_length {video_length}, infer_steps {infer_steps}, embedded_guidance_scale {embedded_guidance_scale}")

        status = gen(
            output_fn=output_fn,
            prompt=prompt,
            width=width,
            height=height,
            video_length=video_length,
            infer_steps=infer_steps,
            embedded_guidance_scale=embedded_guidance_scale,
            fps=24,
        )
        assert status == "succeeded"
        assert output_fn.exists()


if __name__ == "__main__":
    test_prompts()
