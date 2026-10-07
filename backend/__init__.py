import os

# PyTorch and FAISS each ship their own OpenMP runtime. On macOS, loading both
# and letting FAISS run multi-threaded crashes the process (segfault) on a
# 32k-vector index. One OpenMP thread avoids it; this must be set before torch
# or faiss is imported, which is why it lives here. The models run on the GPU
# (MPS/CUDA) anyway, and an exact FAISS search over 32k vectors stays fast.
os.environ.setdefault("OMP_NUM_THREADS", "1")
