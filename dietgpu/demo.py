import torch

torch.ops.load_library("/ocean/projects/asc200010p/jjia1/TOOLS/dietgpu/build/lib/libdietgpu.so")
dev = torch.device("cuda:0")
