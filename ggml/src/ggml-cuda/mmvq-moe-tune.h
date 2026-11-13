#pragma once
// local patch 0011: lane groups for tiny-K MoE MMVQ calls (bit-identical). All types: the lane mapping is type-independent.
#define MMVQ_MOE_LANEGROUPS 1
static constexpr __host__ __device__ bool mmvq_moe_listed(ggml_type) {
    return true;
}
