import time
import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())
SEED = 123
RESIDUAL_DIFF_THRESHOLD = 0.035

PROMPT = "A flock of colorful birds soaring through a windswept mountain landscape at sunset, swirling and diving in intricate patterns, realistic"


# [rank1]: RuntimeError: Expected mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
torch.backends.cuda.enable_cudnn_sdp(False)

model_id = "checkpoints"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

mesh = init_context_parallel_mesh(
    pipe.device.type,
)
parallelize_pipe(
    pipe,
    mesh=mesh,
)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe, residual_diff_threshold=RESIDUAL_DIFF_THRESHOLD)

# from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only
#
# torch._inductor.config.reorder_for_compute_comm_overlap = True
#
# quantize_(pipe.text_encoder, float8_weight_only())
# quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
# pipe.transformer = torch.compile(
#    pipe.transformer, mode="max-autotune-no-cudagraphs",
# )

pipe.vae.enable_tiling()
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
generator = torch.Generator("cuda").manual_seed(SEED)

for i in range(2):
    begin = time.time()
    output = pipe(
        prompt=PROMPT,
        height=720,
        width=1280,
        num_frames=129,
        num_inference_steps=1 if i == 0 else 30,
        generator=generator,
        output_type="pil" if dist.get_rank() == 0 else "pt",
    ).frames[0]
    end = time.time()
    if dist.get_rank() == 0:
        if i == 0:
            print(f"Warm up time: {end - begin:.2f}s")
        else:
            world_size = dist.get_world_size()
            print("=========== TIME FOR WORLD SIZE", world_size, "===========")
            print(f"Time: {end - begin:.2f}s")
            print("================================================")

if dist.get_rank() == 0:
    print("Saving video to hunyuan_video.mp4")
    # export_to_video(output, f"hunyuan_video_motion_nocache.mp4", fps=30)
    export_to_video(output, f"hunyuan_video_motion_layer_cache.mp4", fps=30)

dist.destroy_process_group()