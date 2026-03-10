from __future__ import annotations
import torch

try:
    import Performance_Fundamentals.arch_config
except ImportError:
    pass


from core.utils.compile_utils import compile_model
from typing import Optional

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.harness.verification_mixin import VerificationMixin
from workload1 import WORKLOAD

def resolve_device()->torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        torch.zeros(1,device="cuda")
        return torch.device("cuda")
    except Exception as e:
        print(f"CUDA is not available: {e}")
        return torch.device('cpu')

def _should_use_compile(device:torch.device)->bool:
    if device.type !== "cuda":
        return False
    return False


class BaselinePerformanceBenchmark(BaseBenchmark, VerificationMixin):
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.device = resolve_device()
        self.model = None
        self.data = None
        self.target = None
        self.optimizer = None
        self.workload = WORKLOAD
        self.batch_size = self.workload.batch_size
        self.num_microbatches = self.workload.performance_microbatches
        self.fusion = 8
        self.dim_hidden = self.workload.dim_hidden
        self.microbatch_size = None
        self.targets = None
        self._verify_input = None
        self.verify_input = None
        self.parameter_count = 0
        samples = float(self.batch_size * self.num_microbatches)
        self.register_workload_metadata(samples_per_iteration=samples)

    def setup(self) -> None:
        torch.manual_seed(14)
        torch.cuda.manual_seed_all(14)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.dim_hidden, self.dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_hidden, self.dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_hidden, 10),
        )
        if _should_use_compile(self.device):
            self.model = compile_model(
                self.model.to(self.device),
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False
            )
        else:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())