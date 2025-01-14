# cog predict -e "CACHE_THRESHOLD=0.000125" --gpus '"device=0,1,2,3"' -o "cat_in_sky/output_000125.mp4" -i "prompt='a cat running around on clouds in the sky' seed=1234"

from video_prompts import video_prompts
import os
cache_thresholds = [0.0, 0.06, 0.2]

def get_folder_name(prompt):
    words = prompt.split(" ")
    return "_".join(words[:3])

for prompt in video_prompts:
    folder_name = get_folder_name(prompt)
    folder_path = f"threshold_test_videos/{folder_name}"
    os.makedirs(folder_path, exist_ok=True)
    for cache_threshold in cache_thresholds:
        print(f"Running {prompt} with cache threshold {cache_threshold}")
        cmd = f"""
        cog predict -e 'CACHE_THRESHOLD={cache_threshold}' -e 'FOLDER_PATH={folder_path}' --gpus '"device=4,5,6,7"' -o '{folder_path}/output_{cache_threshold}.mp4' -i prompt="{prompt}" -i seed=1234
        """
        print(cmd)
        os.system(cmd)
