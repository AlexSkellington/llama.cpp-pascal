#include "common.cuh"
#include "fattn-common.cuh"

// nbatch_fa == number of KQ rows to process per iteration
// nbatch_K == number of K columns to load in parallel for KQ calculation

// TODO optimize kernel parameters for FP16 NVIDIA (P100)
// TODO optimize kernel parameters for head sizes 40, 72, 80, 96, 112

// The ROCm compiler cannot handle templating in __launch_bounds__.
// As a workaround, define a macro to package the kernel parameters as uint32_t:
#define GGML_CUDA_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        return ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_nvidia_fp16(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2, 128, 3, 128,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2, 128,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 128, 3, 128,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  64,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 256, 3, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2, 128, 2,  64,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 3, 128,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  64,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  64,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  64,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2, 128, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2, 128,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2, 128,  24)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2, 128, 3,  64,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  64,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 3,  64, 112)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32, 112)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2, 128,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 16, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  64,  64)

    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_nvidia_fp32(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  64,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 3,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 128, 3,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 128, 3,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  4, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  8, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 16, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  2,  64, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  32,  64)

    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 64, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 64, 256, 2,  32,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 64, 256, 2,  32,  48)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 64, 256, 2,  32,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 64, 256, 2,  64,  32)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  2, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  4, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  32, 128)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 32, 512, 1, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 32, 512, 1, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 32, 512, 1, 128,  64)

    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd_rdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 5, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 128, 5, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 128, 4,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 64, 128, 5,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 64, 256, 2,  32,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 64, 256, 2,  32,  48)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 64, 256, 2,  32,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 8,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 128, 8,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 3, 128, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 3, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 64, 256, 3,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  4, 128, 6,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  8, 128, 6,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 5,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 3,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 6,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 128, 6,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 5,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 3,  64, 128)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 32, 256, 2, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 4,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 32, 256, 2, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 4,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 32, 256, 2, 128,  64)

    return 0;
}

static __host__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA(cc)) {
            return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
        }
        return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
    }
    if (fast_fp16_available(cc)) {
        return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
    }
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
}

static constexpr __device__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols) {
#ifdef GGML_USE_HIP
#ifdef RDNA
    return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
#endif // RDNA
#else
#ifdef FAST_FP16_AVAILABLE
    return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
#endif // FAST_FP16_AVAILABLE
#endif // GGML_USE_HIP
}

static __host__ int ggml_cuda_fattn_tile_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_nbatch_K(const int DKQ, const int DV, const int ncols, const int cc) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_nbatch_K(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

// TODO: deduplicate with mma-f16
template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int stride_KV, const int i_sup) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    auto load = [&] __device__ (const int n) {
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             ((J/2)/cpy_ne) - ((J/2)/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_j == warp_size ? 0 : threadIdx.x / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0*cpy_ne + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*cpy_ne;

                    const __align__(16) half2 zero[cpy_ne] = {{0.0f, 0.0f}};
                    ggml_cuda_memcpy_1<cpy_nb>(
                        tile_KV + i*(J/2 + J_padding) + j,
                        !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);
                }
            }
        }
    };
    // 1: max 64*16=512 bytes, 512 half
    // 2: max 32*16=512 bytes, 256 half
    // 3: max 16*16=256 bytes, 128 half
    // 4: max  8*16=128 bytes,  64 half
    // 5: max  4*16= 64 bytes,  32 half
    // 6: max  2*16= 32 bytes,  16 half
    // 7: max  1*16= 16 bytes,   8 half
    static_assert(J % 8 == 0, "bad J");
    static_assert((J/2) % cpy_ne == 0, "bad J");
    ggml_cuda_unroll<7>{}(load);
}

template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_load_tile(
        const half2 * const __restrict__ KV, float * const __restrict__ tile_KV, const int stride_KV, const int i_sup) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    auto load = [&] __device__ (const int n) {
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_j == warp_size ? 0 : threadIdx.x / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0*(cpy_ne/2) + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*(cpy_ne/2);

                    const half2 zero[cpy_ne/2] = {{0.0f, 0.0f}};
                    __align__(16) half2 tmp_h2[cpy_ne/2];
                    ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __align__(16) float2 tmp_f2[cpy_ne/2];
#pragma unroll
                    for (int l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = __half22float2(tmp_h2[l]);
                    }
                    ggml_cuda_memcpy_1<sizeof(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
                }
            }
        }
    };
    // 1: max 32*16=512 bytes, 128 float
    // 2: max 16*16=256 bytes,  64 float
    // 3: max  8*16=128 bytes,  32 float
    // 4: max  4*16= 64 bytes,  16 float
    // 5: max  2*16= 32 bytes,   8 float
    static_assert(J % 8 == 0, "bad J");
    static_assert(J % cpy_ne == 0, "bad J");
    ggml_cuda_unroll<5>{}(load);
}

// Lab patch: q8_0 -> FP32 tile, fused dequantization (no f16 conversion buffer). KV points at row 0 and the first
// q8_0 block of the J-column slab; stride_KV is in blocks. Each work item converts 8 elements (a quarter block):
// 4 consecutive threads share one block, so a warp reads 8 consecutive blocks (272 contiguous bytes).
template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_load_tile_q8(
        const block_q8_0 * const __restrict__ KV, float * const __restrict__ tile_KV, const int stride_KV, const int i_sup) {
    static_assert(J % QK8_0 == 0, "bad J for q8_0");
    // Store granularity follows the tile row stride (J + J_padding floats); offsets within a row are 32-byte multiples.
    constexpr int row_bytes = (J + J_padding)*int(sizeof(float));
    constexpr int st_nb = row_bytes % 16 == 0 ? 16 : (row_bytes % 8 == 0 ? 8 : 4);
    constexpr int nblk     = J/QK8_0;
    constexpr int items    = I*nblk*4;
    constexpr int nthreads = nwarps*warp_size;
    const int tid = threadIdx.y*warp_size + threadIdx.x;
#pragma unroll
    for (int it0 = 0; it0 < items; it0 += nthreads) {
        const int it = it0 + tid;
        if (it0 + nthreads > items && it >= items) {
            break;
        }
        const int q = it % 4;
        const int b = (it / 4) % nblk;
        const int i = it / (4*nblk);

        __align__(16) float out[8];
        if (!oob_check || i < i_sup) {
            const block_q8_0 * blk = KV + int64_t(i)*stride_KV + b;
            const float dk = __half2float(__ldg((const half *) &blk->d));
            const uint16_t * qs = (const uint16_t *) (blk->qs + 8*q); // q8_0 blocks are only 2-byte aligned
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint16_t v = __ldg(qs + l);
                out[2*l + 0] = dk * (float) (int8_t) (v & 0xFF);
                out[2*l + 1] = dk * (float) (int8_t) (v >> 8);
            }
        } else {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                out[l] = 0.0f;
            }
        }
        float * dst = tile_KV + i*(J + J_padding) + b*QK8_0 + q*8;
#pragma unroll
        for (int l = 0; l < 8; l += st_nb/int(sizeof(float))) {
            ggml_cuda_memcpy_1<st_nb>(dst + l, out + l);
        }
    }
}

// Function that performs a single iteration in for the KQ matrix multiplication:
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check, bool kv_q8, typename T_vec_dot>
static __device__ __forceinline__ void flash_attn_tile_iter_KQ(
        T_vec_dot   * const Q_tmp,
        const void  * const __restrict__ K_h2,
        T_vec_dot   * const KV_tmp,
        const int stride_K2,
        const int k_VKQ_0,
        const int k_VKQ_sup,
        const int k_KQ_0,
        float * KQ_acc) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

#ifndef FAST_FP16_AVAILABLE
    if constexpr (kv_q8) {
        flash_attn_tile_load_tile_q8<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
            ((const block_q8_0 *) K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/QK8_0, KV_tmp, stride_K2, k_VKQ_sup);
    } else
#endif // FAST_FP16_AVAILABLE
    {
        flash_attn_tile_load_tile<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
            ((const half2 *) K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    }
    __syncthreads();

#ifdef FAST_FP16_AVAILABLE
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __align__(16) half2 K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __align__(16) half2 Q_k[cpw][cpy_ne];
#else
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __align__(16) float K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __align__(16) float Q_k[cpw][cpy_ne];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

#ifdef FAST_FP16_AVAILABLE
            ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#else
            ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
        }
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y / np)*cpw;

#ifdef FAST_FP16_AVAILABLE
            ggml_cuda_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#else
            ggml_cuda_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                for (int k = 0; k < cpy_ne; ++k) {
                    ggml_cuda_mad(KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0], K_k[i_KQ_0/(np*warp_size)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    if (k_KQ_0 + nbatch_K < DKQ) {
        __syncthreads(); // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check, bool kv_q8, bool fold_gqa, typename T_vec_dot, typename T_KQ, typename T_acc>
static __device__ __forceinline__ void flash_attn_tile_iter(
        T_vec_dot * const Q_tmp,
        const void  * const __restrict__ K_h2,
        const void  * const __restrict__ V_h2,
        const half  * const __restrict__ mask,
        const uint3 ne01,
        const float logit_softcap,
        const float slope,
        T_KQ      * const KQ,
        T_vec_dot * const KV_tmp,
        const int stride_K2,
        const int stride_V2,
        const int stride_mask,
        float * const KQ_max,
        float * const KQ_sum,
        T_acc * const VKQ,
        const int k_VKQ_0,
        const int k_VKQ_max,
        const int col_Q_0) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.

    // KQ_cs == KQ chunk size, number of KQ values in j direction to store as one contiguous chunk in memory.
    // KQ is originally 2D but uses a Z-shaped 3D memory pattern like KQ[ncols/KQ_cs][DVp][KQ_cs].
#ifdef FAST_FP16_AVAILABLE
    constexpr int KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#else
    constexpr int KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#endif // FAST_FP16_AVAILABLE
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    float KQ_max_new[cpw];
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    float KQ_acc[nbatch_fa/(np*warp_size) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr int nbatch_K_last = DKQ % nbatch_K;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, kv_q8>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    if (nbatch_K_last > 0) {
        constexpr int k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, use_logit_softcap, oob_check, kv_q8>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        // lp0024: the padded eight-head tile packs six real heads per query.
        constexpr int ncols2_eff = fold_gqa ? 6 : ncols2;
        const int j = fastmodulo(col_Q_0 + (jc0 + (threadIdx.y / np)*cpw)/ncols2_eff, ne01);

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0] *= 4.0f;
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)

            if (use_logit_softcap) {
                KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] = logit_softcap * tanhf(KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0]);
            }

            if (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] += (ncols2 > 1 || mask) ?
                    slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] + FATTN_KQ_MAX_OFFSET);
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
    }

    if constexpr (np == 1) {
        __syncthreads();
    } else {
        static_assert(cpw == 1, "bad cpw");
        __shared__ float KQ_max_new_shared[nwarps];
        if (threadIdx.x == 0) {
            KQ_max_new_shared[threadIdx.y] = KQ_max_new[0];
        }
        __syncthreads();
        KQ_max_new[0] = KQ_max_new_shared[(threadIdx.y & ~(np-1)) + threadIdx.x % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#ifdef FAST_FP16_AVAILABLE
        __align__(16) half  tmp[nbatch_fa/(np*warp_size)][KQ_cs];
#else
        __align__(16) float tmp[nbatch_fa/(np*warp_size)][KQ_cs];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
        for (int jc1 = 0; jc1 < KQ_cs; ++jc1) {
            const int jc = jc0 + jc1;

            const float KQ_max_scale = expf(KQ_max[jc] - KQ_max_new[jc]);
            KQ_max[jc] = KQ_max_new[jc];

            float KQ_sum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                const float val = !oob_check || i0 + (threadIdx.y % np)*warp_size + threadIdx.x < static_cast<uint32_t>(k_VKQ_sup) ?
                    expf(KQ_acc[(i0/(np*warp_size))*cpw + jc] - KQ_max[jc]) : 0.0f;
                KQ_sum_add += val;
                tmp[i0/(np*warp_size)][jc1] = val;
            }
            KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
            const int i = i0 + (threadIdx.y % np)*warp_size + threadIdx.x;

            ggml_cuda_memcpy_1<sizeof(tmp[0])>(
                KQ + (jc0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs))*(nbatch_fa*KQ_cs) + i*KQ_cs,
                tmp[i0/(np*warp_size)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr int nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
#ifndef FAST_FP16_AVAILABLE
        if constexpr (kv_q8) {
            flash_attn_tile_load_tile_q8<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
                ((const block_q8_0 *) V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        } else
#endif // FAST_FP16_AVAILABLE
        {
            flash_attn_tile_load_tile<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
                ((const half2 *) V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        }
        __syncthreads();

#ifdef FAST_FP16_AVAILABLE
#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
            __align__(16) half2 V_k[(DVp/2)/warp_size];
            __align__(16) half2 KQ_k[cpw];

            constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/warp_size], &KV_tmp[(k1 + threadIdx.y % np)*(DV/2) + i0 + threadIdx.x*cpy_ne_D]);
            }
#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);

                __align__(16) half tmp[KQ_cs];
                ggml_cuda_memcpy_1<KQ_cs*sizeof(half)>(
                    &tmp, KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
#pragma unroll
                for (int jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0+jc_VKQ_1] = __half2half2(tmp[jc_VKQ_1]);
                }
            }

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += V_k[i0/warp_size]*KQ_k[jc_VKQ_0];
                }
            }
        }
#else
#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
            __align__(16) float2 V_k[(DVp/2)/warp_size];
            __align__(16) float  KQ_k[cpw];

            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_size)], &KV_tmp[(k1 + threadIdx.y % np)*DV + i0 + threadIdx.x*cpy_ne_D]);
            }
#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);

                ggml_cuda_memcpy_1<KQ_cs*sizeof(float)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
            }

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x += V_k[i0/warp_size].x*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y += V_k[i0/warp_size].y*KQ_k[jc_VKQ_0];
                }
            }
        }
#endif // FAST_FP16_AVAILABLE

        __syncthreads();
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool kv_q8 = false, bool fold_gqa = false> // D == head size
__launch_bounds__(ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_tile_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_tile(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        const int  * KV_rows_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;
    GGML_UNUSED(KV_rows_ptr);

    // Skip unused kernel variants for faster compilation:

#ifdef FAST_FP16_AVAILABLE
    constexpr bool kv_q8_unsupported = kv_q8; // fused q8_0 only exists for the FP32 tile layout
#else
    constexpr bool kv_q8_unsupported = kv_q8 && !((DKQ == 128 || DKQ == 256) && DKQ == DV);
#endif // FAST_FP16_AVAILABLE
    if ((use_logit_softcap && !(DV == 128 || DV == 256 || DV == 512)) || kv_q8_unsupported) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    static_assert(ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr int ncols     = ncols1*ncols2;
    constexpr int warp_size = 32;
    constexpr int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_size;
    constexpr int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr int nbatch_K  = ggml_cuda_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    // lp0024: retain power-of-two tile storage, fold six real GQA heads into it.
    static_assert(!fold_gqa || (DKQ == 256 && DV == 256 && ncols2 == 8 && kv_q8 && !use_logit_softcap),
        "lp0024 is restricted to the FP32 Q8 head256 GQA6 path");
    constexpr int ncols2_eff = fold_gqa ? 6 : ncols2;
    constexpr int ncols1_eff = fold_gqa ? ncols/ncols2_eff : ncols1;
    constexpr int ncols_valid = ncols1_eff*ncols2_eff;
    const int col_Q_0 = blockIdx.x * ncols1_eff;

    const int sequence = fold_gqa ? blockIdx.z/ne12 : blockIdx.z/(ne02/ncols2);
    const int head0 = fold_gqa ? (blockIdx.z % ne12)*6 : blockIdx.z*ncols2 - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0);
    const void * K_h2 = (const void *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const void * V_h2 = (const void *) (V + nb23*sequence + nb22*(head0 / gqa_ratio)); // K and V have same shape

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33)) : nullptr;

    // q8_0: strides in blocks, f16: strides in half2.
    const int stride_K2   = kv_q8 ? nb11 / int(sizeof(block_q8_0)) : nb11 / int(sizeof(half2));
    const int stride_V2   = kv_q8 ? nb21 / int(sizeof(block_q8_0)) : nb21 / int(sizeof(half2));
    const int stride_mask = nb31 / sizeof(half);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.
    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_size) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr int DKQp = (DKQ + 2*warp_size - 1) & ~(2*warp_size - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr int DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.
#ifdef FAST_FP16_AVAILABLE
    __shared__ half2 Q_tmp[ncols * DKQ/2];
    __shared__ half2 KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV];
    __shared__ half  KQ[ncols * nbatch_fa];
    __align__(16) half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
#else
    __shared__ float Q_tmp[ncols * DKQ];
    __shared__ float KV_tmp[nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV];
    __shared__ float KQ[ncols * nbatch_fa];
    __align__(16) float2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
#endif // FAST_FP16_AVAILABLE

    float KQ_max[cpw];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

    ggml_cuda_pdl_sync();

    // Load Q data, convert to FP16 if fast:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y / np)*cpw;

        const int j = jc / ncols2_eff;
        const int c = jc % ncols2_eff;

        constexpr int cpy_ne_D = cpy_ne < DKQp/warp_size ? cpy_ne : DKQp/warp_size;

#pragma unroll
        for (int i0 = 0; i0 < DKQp; i0 += np*warp_size*cpy_ne_D) {
            if (i0 + np*warp_size*cpy_ne_D <= DKQ || i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D < DKQ) {
                __align__(16) float tmp_f[cpy_ne_D] = {0.0f};
                if (!fold_gqa || (jc < ncols_valid && col_Q_0 + j < int(ne01.z))) {
                    ggml_cuda_memcpy_1<sizeof(tmp_f)>
                        (tmp_f, &Q_f[c*(nb02/sizeof(float)) + fastmodulo(col_Q_0 + j, ne01)*(nb01/sizeof(float))
                                     + i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D]);
                }

#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#ifdef FAST_FP16_AVAILABLE
                __align__(16) half2 tmp_h2[cpy_ne_D/2];
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1/2] *= make_half2(0.25f, 0.25f);
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
                }
                ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                    &Q_tmp[jc*(DKQ/2) + i0/2 + (threadIdx.y % np)*(warp_size*cpy_ne_D/2) + threadIdx.x*(cpy_ne_D/2)],
                    tmp_h2);
#else
                ggml_cuda_memcpy_1<sizeof(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (threadIdx.y % np)*(warp_size*cpy_ne_D)   + threadIdx.x* cpy_ne_D],
                    tmp_f);
#endif // FAST_FP16_AVAILABLE
            }
        }
    }

    __syncthreads();

    // Main loop over KV cache:
    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    if (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        int k_VKQ_0 = blockIdx.y*nbatch_fa;
        while (k_VKQ_0 <= k_VKQ_max - nbatch_fa) {
            constexpr bool oob_check = false;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, kv_q8, fold_gqa>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0);
            k_VKQ_0 += gridDim.y*nbatch_fa;
        }
        if (k_VKQ_0 < k_VKQ_max) {
            constexpr bool oob_check = true;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, kv_q8, fold_gqa>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0);
        }
    } else {
        // Branch without out-of-bounds checks.
        for (int k_VKQ_0 = blockIdx.y*nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nbatch_fa) {
            constexpr bool oob_check = false;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, kv_q8, fold_gqa>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0);
        }
    }

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum<warp_size>(KQ_sum[jc0]);
    }

    if constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#ifdef FAST_FP16_AVAILABLE
        half2 * VKQ_combine    = (half2 *) KV_tmp;
#else
        float * VKQ_combine    = (float *) KV_tmp;
#endif // FAST_FP16_AVAILABLE
        float * KQ_sum_combine = (float *) Q_tmp;

        if (threadIdx.y % np != 0) {
#ifdef FAST_FP16_AVAILABLE
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&VKQ_combine[threadIdx.y*(DVp/2) + i0 + threadIdx.x*cpy_ne_D], &VKQ[i0/warp_size]);
            }
#else
            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[threadIdx.y*DVp + i0 + threadIdx.x*cpy_ne_D], ((const float *) VKQ) + i0/warp_size);
            }
#endif // FAST_FP16_AVAILABLE

            if (threadIdx.x == 0) {
                KQ_sum_combine[threadIdx.y] = KQ_sum[0];
            }

            return;
        }

        __syncthreads();

#pragma unroll
        for (int ip = 1; ip < np; ++ip) {
#ifdef FAST_FP16_AVAILABLE
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                __align__(16) half2 tmp[cpy_ne_D];
                ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(threadIdx.y + ip)*(DVp/2) + i0 + threadIdx.x*cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_size + i1] += tmp[i1];
                }
            }
#else
            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                __align__(16) float tmp[cpy_ne_D];
                ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(threadIdx.y + ip)*DVp + i0 + threadIdx.x*cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((float *)VKQ)[i0/warp_size + i1] += tmp[i1];
                }
            }
#endif // FAST_FP16_AVAILABLE

            KQ_sum[0] += KQ_sum_combine[threadIdx.y + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y/np)*cpw;
            const float sink = ((const float *) sinks)[head0 + jc % ncols2];

            float KQ_max_new_j = fmaxf(KQ_max[jc0], sink);
            const float KQ_max_scale = expf(KQ_max[jc0] - KQ_max_new_j);
            KQ_max[jc0] = KQ_max_new_j;

            const float val = expf(sink - KQ_max[jc0]);
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

    // Write back results:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y/np)*cpw;

        const int j = jc / ncols2_eff;
        const int c = jc % ncols2_eff;

        if (fold_gqa ? (jc >= ncols_valid || col_Q_0 + j >= int(ne01.z))
                     : (ncols1 > 1 && col_Q_0 + j >= int(ne01.z))) {
            return;
        }

        const float scale = gridDim.y == 1 ? 1.0f/KQ_sum[jc0] : 1.0f;

        const int j_dst_unrolled = ((sequence*int(ne01.z) + col_Q_0 + j)*ne02 + head0 + c)*gridDim.y + blockIdx.y;

#ifdef FAST_FP16_AVAILABLE
        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
            __align__(16) float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = __half22float2(VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size + i1]);
                tmp[i1].x *= scale;
                tmp[i1].y *= scale;
            }
            if (i0 + warp_size*cpy_ne_D <= DV/2 || i0 + threadIdx.x*cpy_ne_D < DV/2) {
                ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*DV + 2*i0 + threadIdx.x*(2*cpy_ne_D)], tmp);
            }
        }
#else
        constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
            if (i0 + warp_size*cpy_ne_D <= DV || i0 + threadIdx.x*cpy_ne_D < DV) {
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].x *= scale;
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].y *= scale;
                }
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + threadIdx.x*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size)]);
            }
        }
#endif // FAST_FP16_AVAILABLE

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check, bool kv_q8, bool fold_gqa, typename T_vec_dot, typename T_KQ, typename T_acc>
static __device__ __forceinline__ void lpdp4a_tile_iter(
        T_vec_dot * const Q_tmp,
        const void  * const __restrict__ K_h2,
        const void  * const __restrict__ V_h2,
        const half  * const __restrict__ mask,
        const uint3 ne01,
        const float logit_softcap,
        const float slope,
        T_KQ      * const KQ,
        T_vec_dot * const KV_tmp,
        const int stride_K2,
        const int stride_V2,
        const int stride_mask,
        float * const KQ_max,
        float * const KQ_sum,
        T_acc * const VKQ,
        const int k_VKQ_0,
        const int k_VKQ_max,
        const int col_Q_0) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.

    // KQ_cs == KQ chunk size, number of KQ values in j direction to store as one contiguous chunk in memory.
    // KQ is originally 2D but uses a Z-shaped 3D memory pattern like KQ[ncols/KQ_cs][DVp][KQ_cs].
#ifdef FAST_FP16_AVAILABLE
    constexpr int KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#else
    constexpr int KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#endif // FAST_FP16_AVAILABLE
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    float KQ_max_new[cpw];
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    float KQ_acc[nbatch_fa/(np*warp_size) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.


    // LP_DP4A_PREFILL: load packed Q8 keys cooperatively, transposed in shared memory.
    static_assert(DKQ == 256 && DV == 256 && ncols == 32 && np == 1 && !fold_gqa, "prototype layout");
    constexpr int words = DKQ/4;
    constexpr int pitch = nbatch_fa+1;
    int * const packed_K = (int *) KV_tmp;
    const block_q8_0 * const keys = (const block_q8_0 *) K_h2;
    for (int t = threadIdx.y*warp_size+threadIdx.x; t < nbatch_fa*words; t += nwarps*warp_size) {
        const int row=t/words, word=t%words, block=word/8;
        int packed=0; float d=0.0f;
        if (!oob_check || row < k_VKQ_sup) {
            const block_q8_0 * key=keys+int64_t(k_VKQ_0+row)*stride_K2+block;
            ggml_cuda_memcpy_1<4,2>(&packed, key->qs+4*(word%8));
            if (word%8 == 0) d=__half2float(key->d);
        }
        packed_K[word*pitch+row]=packed;
        if (word%8 == 0) KV_tmp[(words+block)*pitch+row]=d;
    }
    __syncthreads();
#pragma unroll
    for (int block=0; block<DKQ/32; ++block) {
        int sums[cpw]={0};
#pragma unroll
        for (int word=0; word<8; ++word) {
            const int k=packed_K[(block*8+word)*pitch+threadIdx.x];
#pragma unroll
            for (int col=0; col<cpw; ++col) {
                const int jc=col+threadIdx.y*cpw;
                const int q=((const int *)Q_tmp)[jc*(words+DKQ/32)+block*8+word];
                sums[col]=ggml_cuda_dp4a(k,q,sums[col]);
            }
        }
        const float kd=KV_tmp[(words+block)*pitch+threadIdx.x];
#pragma unroll
        for (int col=0; col<cpw; ++col) {
            const int jc=col+threadIdx.y*cpw;
            const float qd=((const float *)Q_tmp)[jc*(words+DKQ/32)+words+block];
            KQ_acc[col] += float(sums[col])*(kd*qd);
        }
    }
    // Apply logit softcap + mask, update KQ_max:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        // lp0024: the padded eight-head tile packs six real heads per query.
        constexpr int ncols2_eff = fold_gqa ? 6 : ncols2;
        const int j = fastmodulo(col_Q_0 + (jc0 + (threadIdx.y / np)*cpw)/ncols2_eff, ne01);

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0] *= 4.0f;
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)

            if (use_logit_softcap) {
                KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] = logit_softcap * tanhf(KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0]);
            }

            if (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] += (ncols2 > 1 || mask) ?
                    slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] + FATTN_KQ_MAX_OFFSET);
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
    }

    if constexpr (np == 1) {
        __syncthreads();
    } else {
        static_assert(cpw == 1, "bad cpw");
        __shared__ float KQ_max_new_shared[nwarps];
        if (threadIdx.x == 0) {
            KQ_max_new_shared[threadIdx.y] = KQ_max_new[0];
        }
        __syncthreads();
        KQ_max_new[0] = KQ_max_new_shared[(threadIdx.y & ~(np-1)) + threadIdx.x % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    constexpr bool lp50 = (!kv_q8 && DV == 256 && DVp == DV && np == 1 && nbatch_fa == 32 && cpw == 4 && warp_size == 32); // local patch 0050
    float P50[cpw];
    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#ifdef FAST_FP16_AVAILABLE
        __align__(16) half  tmp[nbatch_fa/(np*warp_size)][KQ_cs];
#else
        __align__(16) float tmp[nbatch_fa/(np*warp_size)][KQ_cs];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
        for (int jc1 = 0; jc1 < KQ_cs; ++jc1) {
            const int jc = jc0 + jc1;

            const float KQ_max_scale = expf(KQ_max[jc] - KQ_max_new[jc]);
            KQ_max[jc] = KQ_max_new[jc];

            float KQ_sum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                const float val = !oob_check || i0 + (threadIdx.y % np)*warp_size + threadIdx.x < static_cast<uint32_t>(k_VKQ_sup) ?
                    expf(KQ_acc[(i0/(np*warp_size))*cpw + jc] - KQ_max[jc]) : 0.0f;
                KQ_sum_add += val;
                tmp[i0/(np*warp_size)][jc1] = val;
            }
            KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;
            if constexpr (lp50) {
                P50[jc] = tmp[0][jc1];
            }

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }

#pragma unroll
        for (int i0 = 0; i0 < (lp50 ? 0 : nbatch_fa); i0 += np*warp_size) {
            const int i = i0 + (threadIdx.y % np)*warp_size + threadIdx.x;

            ggml_cuda_memcpy_1<sizeof(tmp[0])>(
                KQ + (jc0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs))*(nbatch_fa*KQ_cs) + i*KQ_cs,
                tmp[i0/(np*warp_size)]);
        }
    }


    // local patch 0050: DP4A P.V with P from registers and the pre-pass V8T tile (see apply0050.py)
    if constexpr (lp50) {
        constexpr int nkg = nbatch_fa/4;                 // 8 groups of 4 keys
        int   * const V8T = (int *) KV_tmp;              // [nkg][DV]
        float * const Vsd = (float *) KV_tmp + nkg*DV;   // [DV]
        const int tid = threadIdx.y*warp_size + threadIdx.x;
        const int4 * src = (const int4 *) ((const int *) V_h2 + int64_t(k_VKQ_0/nbatch_fa)*(nkg*DV + DV));
        for (int e = tid; e < (nkg*DV + DV)/4; e += nwarps*warp_size) {
            ((int4 *) KV_tmp)[e] = src[e];
        }
        // P (lane = key) -> int8 per (column, tile), packed 4 keys per int: lane 4*kg holds the pack of group kg
        int   pk[cpw];
        float ps[cpw];
#pragma unroll
        for (int jc = 0; jc < cpw; ++jc) {
            const float m = warp_reduce_max<warp_size>(P50[jc]);
            ps[jc] = m*(1.0f/127.0f);
            const int q = m > 0.0f ? __float2int_rn(P50[jc]*(127.0f/m)) : 0;
            const int v0 = q & 0xFF;
            const int v1 = __shfl_down_sync(0xffffffff, v0, 1);
            const int v2 = __shfl_down_sync(0xffffffff, v0, 2);
            const int v3 = __shfl_down_sync(0xffffffff, v0, 3);
            pk[jc] = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
        }
        __syncthreads();
        int acc[DV/64][cpw][2];
#pragma unroll
        for (int i = 0; i < DV/64; ++i) {
#pragma unroll
            for (int jc = 0; jc < cpw; ++jc) {
                acc[i][jc][0] = 0;
                acc[i][jc][1] = 0;
            }
        }
#pragma unroll
        for (int kg = 0; kg < nkg; ++kg) {
            int pb[cpw];
#pragma unroll
            for (int jc = 0; jc < cpw; ++jc) {
                pb[jc] = __shfl_sync(0xffffffff, pk[jc], 4*kg);
            }
#pragma unroll
            for (int i = 0; i < DV/64; ++i) {
                const int2 v = *(const int2 *) (V8T + kg*DV + 64*i + 2*threadIdx.x);
#pragma unroll
                for (int jc = 0; jc < cpw; ++jc) {
                    acc[i][jc][0] = ggml_cuda_dp4a(pb[jc], v.x, acc[i][jc][0]);
                    acc[i][jc][1] = ggml_cuda_dp4a(pb[jc], v.y, acc[i][jc][1]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < DV/64; ++i) {
            const float2 sd = *(const float2 *) (Vsd + 64*i + 2*threadIdx.x);
#pragma unroll
            for (int jc = 0; jc < cpw; ++jc) {
                VKQ[jc*((DVp/2)/warp_size) + i].x += ps[jc]*sd.x*float(acc[i][jc][0]);
                VKQ[jc*((DVp/2)/warp_size) + i].y += ps[jc]*sd.y*float(acc[i][jc][1]);
            }
        }
        __syncthreads();
        return;
    }
    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr int nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
#ifndef FAST_FP16_AVAILABLE
        if constexpr (kv_q8) {
            flash_attn_tile_load_tile_q8<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
                ((const block_q8_0 *) V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        } else
#endif // FAST_FP16_AVAILABLE
        {
            flash_attn_tile_load_tile<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
                ((const half2 *) V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        }
        __syncthreads();

#ifdef FAST_FP16_AVAILABLE
#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
            __align__(16) half2 V_k[(DVp/2)/warp_size];
            __align__(16) half2 KQ_k[cpw];

            constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/warp_size], &KV_tmp[(k1 + threadIdx.y % np)*(DV/2) + i0 + threadIdx.x*cpy_ne_D]);
            }
#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);

                __align__(16) half tmp[KQ_cs];
                ggml_cuda_memcpy_1<KQ_cs*sizeof(half)>(
                    &tmp, KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
#pragma unroll
                for (int jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                    KQ_k[jc_VKQ_0+jc_VKQ_1] = __half2half2(tmp[jc_VKQ_1]);
                }
            }

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += V_k[i0/warp_size]*KQ_k[jc_VKQ_0];
                }
            }
        }
#else
#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
            __align__(16) float2 V_k[(DVp/2)/warp_size];
            __align__(16) float  KQ_k[cpw];

            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_size)], &KV_tmp[(k1 + threadIdx.y % np)*DV + i0 + threadIdx.x*cpy_ne_D]);
            }
#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);

                ggml_cuda_memcpy_1<KQ_cs*sizeof(float)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
            }

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x += V_k[i0/warp_size].x*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y += V_k[i0/warp_size].y*KQ_k[jc_VKQ_0];
                }
            }
        }
#endif // FAST_FP16_AVAILABLE

        __syncthreads();
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool kv_q8 = false, bool fold_gqa = false> // D == head size
__launch_bounds__(ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, ncols1*ncols2), (kv_q8 && ncols2 == 8) ? 2 : 3) // local patch 0044
static __global__ void lpdp4a_tile(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        const int  * KV_rows_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;
    GGML_UNUSED(KV_rows_ptr);

    // Skip unused kernel variants for faster compilation:

#ifdef FAST_FP16_AVAILABLE
    constexpr bool kv_q8_unsupported = kv_q8; // fused q8_0 only exists for the FP32 tile layout
#else
    constexpr bool kv_q8_unsupported = kv_q8 && !((DKQ == 128 || DKQ == 256) && DKQ == DV);
#endif // FAST_FP16_AVAILABLE
    if ((use_logit_softcap && !(DV == 128 || DV == 256 || DV == 512)) || kv_q8_unsupported) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    static_assert(ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr int ncols     = ncols1*ncols2;
    constexpr int warp_size = 32;
    constexpr int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_size;
    constexpr int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr int nbatch_K  = ggml_cuda_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    // lp0024: retain power-of-two tile storage, fold six real GQA heads into it.
    static_assert(!fold_gqa || (DKQ == 256 && DV == 256 && ncols2 == 8 && kv_q8 && !use_logit_softcap),
        "lp0024 is restricted to the FP32 Q8 head256 GQA6 path");
    constexpr int ncols2_eff = fold_gqa ? 6 : ncols2;
    constexpr int ncols1_eff = fold_gqa ? ncols/ncols2_eff : ncols1;
    constexpr int ncols_valid = ncols1_eff*ncols2_eff;
    const int col_Q_0 = blockIdx.x * ncols1_eff;

    const int sequence = fold_gqa ? blockIdx.z/ne12 : blockIdx.z/(ne02/ncols2);
    const int head0 = fold_gqa ? (blockIdx.z % ne12)*6 : blockIdx.z*ncols2 - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0);
    const void * K_h2 = (const void *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const void * V_h2 = (const void *) (V + nb23*sequence + nb22*(head0 / gqa_ratio)); // K and V have same shape

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33)) : nullptr;

    // q8_0: strides in blocks, f16: strides in half2.
    const int stride_K2 = nb11 / int(sizeof(block_q8_0));
    const int stride_V2   = kv_q8 ? nb21 / int(sizeof(block_q8_0)) : nb21 / int(sizeof(half2));
    const int stride_mask = nb31 / sizeof(half);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.
    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_size) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr int DKQp = (DKQ + 2*warp_size - 1) & ~(2*warp_size - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr int DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.
#ifdef FAST_FP16_AVAILABLE
    __shared__ half2 Q_tmp[ncols * DKQ/2];
    __shared__ half2 KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV];
    __shared__ half  KQ[ncols * nbatch_fa];
    __align__(16) half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
#else
    __shared__ float Q_tmp[ncols * (DKQ/4 + DKQ/32)];
    __shared__ float KV_tmp[(DKQ/4+DKQ/32)*(nbatch_fa+1)];
    __shared__ float KQ[ncols * nbatch_fa];
    __align__(16) float2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
#endif // FAST_FP16_AVAILABLE

    float KQ_max[cpw];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

    ggml_cuda_pdl_sync();


    static_assert(DKQ == 256 && DV == 256 && ncols == 32 && np == 1 && !fold_gqa, "prototype layout");
    // A warp owns cpw query columns. Each 32-element block has its own F32 scale.
    // No FP16 query scale or probability/value accumulation is introduced.
#pragma unroll
    for (int col=0; col<cpw; ++col) {
        const int jc=col+threadIdx.y*cpw, j=jc/ncols2, head=jc%ncols2;
#pragma unroll
        for (int block=0; block<DKQ/32; ++block) {
            const float val=Q_f[head*(nb02/sizeof(float))+fastmodulo(col_Q_0+j,ne01)*(nb01/sizeof(float))+block*32+threadIdx.x]*scale;
            const float maxval=warp_reduce_max<32>(fabsf(val));
            const float d=maxval/127.0f;
            const int qi=d>0.0f ? max(-127,min(127,int(roundf(val/d)))) : 0;
            const unsigned int q0=unsigned(qi)&255u;
            const unsigned int q1=__shfl_sync(0xffffffffu,q0,(threadIdx.x&~3)+1);
            const unsigned int q2=__shfl_sync(0xffffffffu,q0,(threadIdx.x&~3)+2);
            const unsigned int q3=__shfl_sync(0xffffffffu,q0,(threadIdx.x&~3)+3);
            if (threadIdx.x%4==0) ((int *)Q_tmp)[jc*(DKQ/4+DKQ/32)+block*8+threadIdx.x/4]=int(q0|(q1<<8)|(q2<<16)|(q3<<24));
            if (threadIdx.x==0) Q_tmp[jc*(DKQ/4+DKQ/32)+DKQ/4+block]=d;
        }
    }
    __syncthreads();
    // Main loop over KV cache:
    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    if (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        int k_VKQ_0 = blockIdx.y*nbatch_fa;
        while (k_VKQ_0 <= k_VKQ_max - nbatch_fa) {
            constexpr bool oob_check = false;
            lpdp4a_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, kv_q8, fold_gqa>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0);
            k_VKQ_0 += gridDim.y*nbatch_fa;
        }
        if (k_VKQ_0 < k_VKQ_max) {
            constexpr bool oob_check = true;
            lpdp4a_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, kv_q8, fold_gqa>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0);
        }
    } else {
        // Branch without out-of-bounds checks.
        for (int k_VKQ_0 = blockIdx.y*nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nbatch_fa) {
            constexpr bool oob_check = false;
            lpdp4a_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, kv_q8, fold_gqa>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0);
        }
    }

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum<warp_size>(KQ_sum[jc0]);
    }

    if constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#ifdef FAST_FP16_AVAILABLE
        half2 * VKQ_combine    = (half2 *) KV_tmp;
#else
        float * VKQ_combine    = (float *) KV_tmp;
#endif // FAST_FP16_AVAILABLE
        float * KQ_sum_combine = (float *) Q_tmp;

        if (threadIdx.y % np != 0) {
#ifdef FAST_FP16_AVAILABLE
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&VKQ_combine[threadIdx.y*(DVp/2) + i0 + threadIdx.x*cpy_ne_D], &VKQ[i0/warp_size]);
            }
#else
            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[threadIdx.y*DVp + i0 + threadIdx.x*cpy_ne_D], ((const float *) VKQ) + i0/warp_size);
            }
#endif // FAST_FP16_AVAILABLE

            if (threadIdx.x == 0) {
                KQ_sum_combine[threadIdx.y] = KQ_sum[0];
            }

            return;
        }

        __syncthreads();

#pragma unroll
        for (int ip = 1; ip < np; ++ip) {
#ifdef FAST_FP16_AVAILABLE
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                __align__(16) half2 tmp[cpy_ne_D];
                ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(threadIdx.y + ip)*(DVp/2) + i0 + threadIdx.x*cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_size + i1] += tmp[i1];
                }
            }
#else
            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                __align__(16) float tmp[cpy_ne_D];
                ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(threadIdx.y + ip)*DVp + i0 + threadIdx.x*cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((float *)VKQ)[i0/warp_size + i1] += tmp[i1];
                }
            }
#endif // FAST_FP16_AVAILABLE

            KQ_sum[0] += KQ_sum_combine[threadIdx.y + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y/np)*cpw;
            const float sink = ((const float *) sinks)[head0 + jc % ncols2];

            float KQ_max_new_j = fmaxf(KQ_max[jc0], sink);
            const float KQ_max_scale = expf(KQ_max[jc0] - KQ_max_new_j);
            KQ_max[jc0] = KQ_max_new_j;

            const float val = expf(sink - KQ_max[jc0]);
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

    // Write back results:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y/np)*cpw;

        const int j = jc / ncols2_eff;
        const int c = jc % ncols2_eff;

        if (fold_gqa ? (jc >= ncols_valid || col_Q_0 + j >= int(ne01.z))
                     : (ncols1 > 1 && col_Q_0 + j >= int(ne01.z))) {
            return;
        }

        const float scale = gridDim.y == 1 ? 1.0f/KQ_sum[jc0] : 1.0f;

        const int j_dst_unrolled = ((sequence*int(ne01.z) + col_Q_0 + j)*ne02 + head0 + c)*gridDim.y + blockIdx.y;

#ifdef FAST_FP16_AVAILABLE
        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
            __align__(16) float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = __half22float2(VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size + i1]);
                tmp[i1].x *= scale;
                tmp[i1].y *= scale;
            }
            if (i0 + warp_size*cpy_ne_D <= DV/2 || i0 + threadIdx.x*cpy_ne_D < DV/2) {
                ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*DV + 2*i0 + threadIdx.x*(2*cpy_ne_D)], tmp);
            }
        }
#else
        constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
            if (i0 + warp_size*cpy_ne_D <= DV || i0 + threadIdx.x*cpy_ne_D < DV) {
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].x *= scale;
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].y *= scale;
                }
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + threadIdx.x*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size)]);
            }
        }
#endif // FAST_FP16_AVAILABLE

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

// Lab patch: fused q8_0 K/V in the FP32 tile kernel (Pascal and older NVIDIA, i.e. no fast FP16).
// MUST stay consistent between the launch below and ggml_cuda_flash_attn_ext_get_alloc_size (conversion buffer).
static bool ggml_cuda_fattn_tile_fused_q8(const ggml_tensor * dst, const int cc) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_TILE_FUSED_Q8");
        return e == nullptr || atoi(e) != 0;
    }();
    // Small batches only (decode, MTP verify, a few slots at once): there the per-call f16 conversion of the whole
    // cache dominates. For prefill-sized batches each Q block would re-dequantize the same K/V, which measured slower
    // than converting once (P5000: -5% at 16K, -9% at 131K), so those keep the stock conversion path.
    static const int max_tokens = [] {
        const char * e = getenv("GGML_CUDA_TILE_FUSED_Q8_MAX_TOKENS");
        return e ? atoi(e) : 16;
    }();
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    return enabled && Q->ne[1] <= max_tokens && GGML_CUDA_CC_IS_NVIDIA(cc) && !fast_fp16_available(cc) &&
        K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 &&
        K->ne[0] == V->ne[0] && (K->ne[0] == 128 || K->ne[0] == 256) &&
        K->nb[1] % sizeof(block_q8_0) == 0 && V->nb[1] % sizeof(block_q8_0) == 0;
}

// lp0024 v2: default ON (GGML_CUDA_LP0024=0 disables), KV >= GGML_CUDA_LP0024_MIN_KV (1024); all other cases keep the production path.
static bool lp0024_eligible(const ggml_tensor * dst, const int cc) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_LP0024");
        return !(e && strcmp(e, "0") == 0);
    }();
    // v2: below ~1k KV cells the folded grid has too few blocks and loses (micro: 256 +28%, 768 +5%, 1024 -9%, 3072 -26%).
    static const long long min_kv = [] {
        const char * e = getenv("GGML_CUDA_LP0024_MIN_KV");
        return e && *e ? atoll(e) : 1024LL;
    }();
    if (!enabled || cc != 610 || !ggml_cuda_fattn_tile_fused_q8(dst, cc)) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    return Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 &&
        Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256 &&
        Q->ne[1] >= 1 && Q->ne[1] <= 3 && Q->ne[3] >= 1 &&
        K->ne[1] >= min_kv && K->ne[1] > 0 && K->ne[1] == V->ne[1] && K->ne[1] % FATTN_KQ_STRIDE == 0 &&
        K->ne[2] > 0 && K->ne[2] == V->ne[2] && Q->ne[2] == 6*K->ne[2] &&
        Q->ne[3] == K->ne[3] && K->ne[3] == V->ne[3] &&
        max_bias == 0.0f && logit_softcap == 0.0f && dst->src[4] == nullptr &&
        mask && mask->type == GGML_TYPE_F16 && mask->ne[0] >= K->ne[1] &&
        mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 && mask->ne[3] >= 1 &&
        mask->nb[0] == sizeof(half) && mask->nb[1] >= size_t(mask->ne[0])*sizeof(half) &&
        mask->nb[2] >= size_t(mask->ne[1])*mask->nb[1] &&
        mask->nb[3] >= size_t(mask->ne[2])*mask->nb[2];
}

template <int ncols>
static void lp0024_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int cc) {
    static_assert(ncols == 8 || ncols == 16 || ncols == 32, "invalid lp0024 tile");
    constexpr int warp_size = 32;
    const int nwarps = ggml_cuda_fattn_tile_get_nthreads(256, 256, ncols, cc) / warp_size;
    const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(256, 256, ncols, cc);
    static const bool trace = [] {
        const char * e = getenv("GGML_CUDA_LP0024_TRACE");
        return e && strcmp(e, "1") == 0;
    }();
    static thread_local unsigned traced = 0;
    const int id = ggml_cuda_get_device();
    if (trace && id >= 0 && id < 32 && !(traced & (1u << id))) {
        traced |= 1u << id;
        GGML_LOG_WARN("lp0024 SELECT gpu=%d cc=%d q=%lld head=256 gqa=6 kv=q8_0 cols=%d qcols=%d groups=3->1 threads=%d nbatch_fa=%d nbatch_K=%d kernel=flash_attn_tile<256,256,%d,8,false,true,true>\n",
            id, cc, (long long) dst->src[0]->ne[1], ncols, ncols/6, 32*nwarps, nbatch_fa,
            ggml_cuda_fattn_tile_get_nbatch_K(256, 256, ncols, cc), ncols/8);
    }
    // No conversion, softcap, sparse indices, Stream-K or async KV preloading.
    launch_fattn<256, ncols/8, 8, true>(ctx, dst,
        flash_attn_tile<256, 256, ncols/8, 8, false, true, true>, nwarps, 0, nbatch_fa,
        false, false, false, false, warp_size);
}


// local patch 0050: per KV head and 32-key tile: V8T[kg][dim] (int8 of keys 4kg..4kg+3, one scale per (tile, dim))
// followed by the 256 per-dim scales; 2304 words per tile. Keys >= KV are zero.
static __global__ void lp50_v8t(const char * __restrict__ V, const int64_t KV, const int64_t ntiles, const int64_t nh,
        const int64_t nb1, const int64_t nb2, const int64_t nb3, int * __restrict__ out) {
    constexpr int DV = 256, NKG = 8;
    const int64_t hs = blockIdx.y, seq = hs / nh, head = hs % nh;
    const char * const vh = V + seq*nb3 + head*nb2;
    for (int64_t e = int64_t(blockIdx.x)*blockDim.x + threadIdx.x; e < ntiles*DV; e += int64_t(gridDim.x)*blockDim.x) {
        const int64_t tile = e / DV;
        const int dim = int(e % DV), b = dim / 32, w = dim % 32;
        float x[32];
        float m = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int64_t k = tile*32 + j;
            x[j] = 0.0f;
            if (k < KV) {
                const block_q8_0 * blk = (const block_q8_0 *) (vh + k*nb1) + b;
                x[j] = float(blk->qs[w])*__half2float(blk->d);
            }
            m = fmaxf(m, fabsf(x[j]));
        }
        const float inv = m > 0.0f ? 127.0f/m : 0.0f;
        int * const o = out + (hs*ntiles + tile)*(NKG*DV + DV);
#pragma unroll
        for (int kg = 0; kg < NKG; ++kg) {
            int pkd = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pkd |= (__float2int_rn(x[4*kg + j]*inv) & 0xFF) << (8*j);
            }
            o[kg*DV + dim] = pkd;
        }
        ((float *) o)[NKG*DV + dim] = m*(1.0f/127.0f);
    }
}

template <int DKQ, int DV, int ncols2, bool use_logit_softcap>
static void launch_fattn_tile_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    const int id        = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[id].cc;
    const int warp_size = 32;

    constexpr size_t nbytes_shared = 0;

    constexpr bool q8_capable = (DKQ == 128 || DKQ == 256) && DKQ == DV;
    const bool fused_q8 = q8_capable && ggml_cuda_fattn_tile_fused_q8(dst, cc);
    const bool need_f16 = !fused_q8;

    // LP_DP4A_PREFILL (local patch 0027): default ON, GGML_CUDA_LP0027=0 disables; decode/vision paths unchanged.
    if constexpr (DKQ == 256 && DV == 256 && !use_logit_softcap && ncols2 <= 8) {
        static const bool enabled=[] { const char *e=getenv("GGML_CUDA_LP0027"); return !(e && atoi(e)==0); }();
        const ggml_tensor *K=dst->src[1], *V=dst->src[2];
        const bool alias=V->view_src && (V->view_src==K || (V->view_src==K->view_src && V->view_offs==K->view_offs));
        float bias=0.0f; memcpy(&bias,(const float *)dst->op_params+1,sizeof(float));
        if (enabled && cc==610 && Q->ne[1]>=128 && K->ne[1]>=1024 && K->type==GGML_TYPE_Q8_0 && V->type==GGML_TYPE_Q8_0 &&
            K->nb[1]%sizeof(block_q8_0)==0 && V->nb[1]%sizeof(block_q8_0)==0 && !alias && bias==0.0f && !dst->src[4]) {
            static thread_local unsigned logged=0;
            if (getenv("GGML_CUDA_LP0027_TRACE") && logged++<4) GGML_LOG_WARN("LP_DP4A_PREFILL gpu=%d q=%lld kv=%lld group=%d K=q8 V=f16 Q=block32int8 softmax=f32\n", id,(long long)Q->ne[1],(long long)K->ne[1],ncols2);
            { // local patch 0050: V re-quantized per (32-key tile, dim) and transposed once per call; P.V runs in DP4A
                const int64_t KV = V->ne[1], ntiles = (KV + 31)/32, nh = V->ne[2], ns = V->ne[3];
                ggml_cuda_pool_alloc<int> v8t(ctx.pool(), size_t(ns*nh*ntiles*2304));
                const int64_t nwork = ntiles*256;
                const dim3 cg(unsigned((nwork + 255)/256 < 1024 ? (nwork + 255)/256 : 1024), unsigned(ns*nh));
                lp50_v8t<<<cg, 256, 0, ctx.stream()>>>((const char *) V->data, KV, ntiles, nh, V->nb[1], V->nb[2], V->nb[3], v8t.get());
                CUDA_CHECK(cudaGetLastError());
                ggml_tensor Vt = *V;
                Vt.data  = v8t.get();
                Vt.nb[2] = size_t(ntiles*2304)*sizeof(int);
                Vt.nb[3] = size_t(nh*ntiles*2304)*sizeof(int);
                ggml_tensor * const V_orig = dst->src[2];
                dst->src[2] = &Vt;
                launch_fattn<256,32/ncols2,ncols2>(ctx,dst,lpdp4a_tile<256,256,32/ncols2,ncols2,false,false,false>,8,0,32,false,false,false,false,warp_size);
                dst->src[2] = V_orig;
            }
            return;
        }
        // local patch 0031: DP4A Q.K for decode / MTP verify; V through the fused q8 loader, no F16 copies needed.
        static const bool on31=[] { const char *e=getenv("GGML_CUDA_LP0031"); return !(e && atoi(e)==0); }();
        static const long long min_kv31=[] { const char *e=getenv("GGML_CUDA_LP0031_MIN_KV"); return e ? atoll(e) : -1LL; }();
        const long long kv_gate31 = min_kv31 >= 0 ? min_kv31 : (Q->ne[1] >= 16 ? 1024LL : Q->ne[1] >= 4 ? 2048LL : 4096LL);
        static const long long max_q31=[] { const char *e=getenv("GGML_CUDA_LP0031_MAX_TOKENS"); return e ? atoll(e) : 127LL; }();
        if (on31 && enabled && cc==610 && Q->ne[1]<=max_q31 && Q->ne[1]<128 && K->ne[1]>=kv_gate31 && K->ne[1]>=1024 &&
            K->type==GGML_TYPE_Q8_0 && V->type==GGML_TYPE_Q8_0 &&
            K->nb[1]%sizeof(block_q8_0)==0 && V->nb[1]%sizeof(block_q8_0)==0 && !alias && bias==0.0f && !dst->src[4]) {
            static thread_local unsigned logged31=0;
            if (getenv("GGML_CUDA_LP0027_TRACE") && logged31++<4) GGML_LOG_WARN("LP0031_DP4A_DECODE gpu=%d q=%lld kv=%lld group=%d\n", id,(long long)Q->ne[1],(long long)K->ne[1],ncols2);
            launch_fattn<256,32/ncols2,ncols2>(ctx,dst,lpdp4a_tile<256,256,32/ncols2,ncols2,false,true,false>,8,0,32,false,false,false,false,warp_size);
            return;
        }
    }

#ifdef GGML_USE_HIP
    if constexpr (DKQ <= 128) {
        if (Q->ne[1] > 32/ncols2) {
            constexpr int cols_per_block = 64;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
            fattn_kernel_t fattn_kernel = fused_q8 ?
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, q8_capable> :
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, false>;
            launch_fattn<DV, cols_per_block/ncols2, ncols2>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16, need_f16, false, false, warp_size);
            return;
        }
    }
#endif // GGML_USE_HIP

#ifndef GGML_USE_HIP
    if constexpr (DKQ <= 256)
#endif // GGML_USE_HIP
    {
        if (Q->ne[1] > 16/ncols2) {
            constexpr int cols_per_block = 32;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
            fattn_kernel_t fattn_kernel = fused_q8 ?
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, q8_capable> :
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, false>;
            launch_fattn<DV, cols_per_block/ncols2, ncols2>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16, need_f16, false, false, warp_size);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] > 8/ncols2) {
            constexpr int cols_per_block = 16;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
            fattn_kernel_t fattn_kernel = fused_q8 ?
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, q8_capable> :
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, false>;
            launch_fattn<DV, cols_per_block/ncols2, ncols2>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16, need_f16, false, false, warp_size);
            return;
        }
    }

    if constexpr (ncols2 <= 8) {
        if (Q->ne[1] > 4/ncols2) {
            constexpr int cols_per_block = 8;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
            fattn_kernel_t fattn_kernel = fused_q8 ?
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, q8_capable> :
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, false>;
            launch_fattn<DV, cols_per_block/ncols2, ncols2>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16, need_f16, false, false, warp_size);
            return;
        }
    }

    if constexpr (ncols2 <= 4) {
        if (Q->ne[1] > 2/ncols2) {
            constexpr int cols_per_block = 4;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
            fattn_kernel_t fattn_kernel = fused_q8 ?
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, q8_capable> :
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, false>;
            launch_fattn<DV, cols_per_block/ncols2, ncols2>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16, need_f16, false, false, warp_size);
            return;
        }
    }

    if constexpr (ncols2 <= 2) {
        constexpr int cols_per_block = 2;
        const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;
        const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);
        fattn_kernel_t fattn_kernel = fused_q8 ?
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, q8_capable> :
                flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, false>;
        launch_fattn<DV, cols_per_block/ncols2, ncols2>
            (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16, need_f16, false, false, warp_size);
        return;
    }

    GGML_ABORT("fatal error");
}

template <int DKQ, int DV, bool use_logit_softcap>
static void launch_fattn_tile_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if constexpr (DKQ == 256 && DV == 256 && !use_logit_softcap) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (lp0024_eligible(dst, cc)) {
            if (Q->ne[1] == 1) { lp0024_launch< 8>(ctx, dst, cc); }
            if (Q->ne[1] == 2) { lp0024_launch<16>(ctx, dst, cc); }
            if (Q->ne[1] == 3) { lp0024_launch<32>(ctx, dst, cc); }
            return;
        }
    }

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    const bool nvidia = GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
    const int gqa_limit = nvidia && gqa_ratio <= 4 && DV <= 256 ? 16 : INT_MAX;
    const bool use_gqa_opt = mask && max_bias == 0.0f && Q->ne[1] <= gqa_limit && K->ne[1] % FATTN_KQ_STRIDE == 0;

    if constexpr (DKQ == 320) {
        // This branch is only used for Mistral Small 4 which has a GQA ratio of 32.
        // On AMD, simply use that GQA ratio with 32 columns / block since we always have enough SRAM.
        // On NVIDIA however, the tile kernel is only used for GPUs that can't use the mma kernel (Pascal and older).
        // Therefore, use a GQA ratio of 16 with 16 columns / block to stay below 48 kiB of SRAM / block.
#ifdef GGML_USE_HIP
        if (use_gqa_opt && gqa_ratio % 32 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 32, use_logit_softcap>(ctx, dst);
            return;
        }
#else
        if (use_gqa_opt && gqa_ratio % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            return;
        }
#endif // GGML_USE_HIP
        GGML_ABORT("flash-attn tile (320/256): expected GQA ratio multiple of 32");
    }

    if constexpr (DKQ == 576) {
        if (use_gqa_opt && gqa_ratio % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            return;
        }
        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            return;
        }
    }

    if constexpr (DKQ == 192) {
        // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
        if (use_gqa_opt && gqa_ratio % 16 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(ctx, dst);
            return;
        }
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV,  8, use_logit_softcap>(ctx, dst);
            return;
        }
        GGML_ABORT("flash-attn tile (192/128): expected GQA ratio multiple of 8");
    }

    if constexpr (DKQ <= 512 && DKQ != 320 && DKQ != 192) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(ctx, dst);
            return;
        }

        if constexpr (DV <= 256) {
            launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(ctx, dst);
            return;
        }
    }
    GGML_ABORT("fatal error");
}

template <int DKQ, int DV>
void ggml_cuda_flash_attn_ext_tile_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap>(ctx, dst);
    }
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_TILE_CASE(DKQ, DV)                             \
    template void ggml_cuda_flash_attn_ext_tile_case              \
    <DKQ, DV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_TILE_CASE( 40,  40);
extern DECL_FATTN_TILE_CASE( 64,  64);
extern DECL_FATTN_TILE_CASE( 72,  72);
extern DECL_FATTN_TILE_CASE( 80,  80);
extern DECL_FATTN_TILE_CASE( 96,  96);
extern DECL_FATTN_TILE_CASE(112, 112);
extern DECL_FATTN_TILE_CASE(128, 128);
extern DECL_FATTN_TILE_CASE(192, 128);
extern DECL_FATTN_TILE_CASE(256, 256);
extern DECL_FATTN_TILE_CASE(320, 256);
extern DECL_FATTN_TILE_CASE(512, 512);
extern DECL_FATTN_TILE_CASE(576, 512);
