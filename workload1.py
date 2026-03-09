from __future__ import annotations
from dataclasses import dataclass


@dataclass
class workload1:
    hidden_dim:int=512
    batch_size:int=64
    tokens_per_step:int=4
    decode_steps:int=256
    speculative_chunk:int=8
    request_pool:int=256
    data_parallel_chunk:int=32
    prefill_chunk:int=8
    micro_batch_size:int=32
    performance_microbatches:int=64
    performance_hidden_dim:int=8192
 
    @property
    def total_requests(self) -> int:
        return self.request_pool

    @property
    def total_tokens(self)-> int:
        return self.decode_steps * self.tokens_per_step * self.batch_size
WORKLOAD = workload1()