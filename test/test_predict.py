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

def gen(output_fn, **kwargs):
    st = time.time()
    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()
    print("Generated in: ", time.time() - st)

    if data['status'] == "succeeded":
        datauri = data["output"]
        base64_encoded_data = datauri.split(",")[1]
        content = base64.b64decode(base64_encoded_data)
        Path(output_fn).write_bytes(content)
        return data['status']

    else:
        return data['status']


def test_prompts():
    """
    runs generations in fp8 and bf16 on the same node! wow!
    """


    status = gen(
        output_fn=f"cool_cat.mp4",
        prompt="a cool cat walking around",
        width=846,
        height=762,
        num_frames=47,
        num_inference_steps=30,
        guidance_scale=6.0,
        fps=15,
        seed=1234,
    )
    assert status == "succeeded"
    assert os.path.exists(f"cool_cat.mp4")

    status = gen(
        output_fn=f"cool_dog.mp4",
        prompt="cool_dog.mp4",
        width=711,
        height=400,
        num_frames=43,
        num_inference_steps=30,
        guidance_scale=6.0,
        fps=15,
        seed=1234,
    )
    assert status == "succeeded"
    assert os.path.exists(f"cool_dog.mp4")

if __name__ == "__main__":
    test_prompts()
