

{
    python -m torch.distributed.run --nproc_per_node=8 run.py
    python -m torch.distributed.run --nproc_per_node=4 run.py
    python -m torch.distributed.run --nproc_per_node=2 run.py
    python -m torch.distributed.run --nproc_per_node=1 run.py
} 2>&1 | tee benchmark_output.txt