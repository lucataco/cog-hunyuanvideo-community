import os
cmd = f"""cog predict --debug -e "CACHE_THRESHOLD=0.000125" --gpus '"device=0,1,2,3"' -o "threshold_test_videos/cat_in_sky/output_000125.mp4" -i prompt="a cat running around on clouds in the sky" -i seed=1234
"""
print(cmd)
os.system(cmd)
