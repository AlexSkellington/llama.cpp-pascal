#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>

constexpr int gdn_cols_per_warp = 4;

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

// local patch 0049 v2: chunked GDN prefill for Pascal. See apply0049.py for the algebra.
#define LP49_C  16   // chunk length
#define LP49_D  128  // S_k == S_v
#define LP49_NC 64   // state columns per scan block (v6: 512 threads, W/K/Q chunk read by 2 blocks per head)

static __global__ void __launch_bounds__(256)
gdn49_prep(const float * q, const float * k, const float * v, const float * g, const float * beta,
           float * W, float * U, float * QMg, float * cumg, int64_t H, int64_t n_main, int64_t nch,
           int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
           int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic) {
    constexpr int C = LP49_C, D = LP49_D, D4 = D/4 + 1;
    __shared__ float4 Ks4[C][D4];
    __shared__ float4 Qs4[C][D4];
    __shared__ float  Vs[C][D + 1];
    __shared__ float  cum[C];
    __shared__ float  bet[C];
    __shared__ float  A[C][C + 1];
    __shared__ float  T[C][C + 1];
    __shared__ float  bg[C];   // beta_j * gamma_j

    const int tid = threadIdx.x;
    const int64_t ch = blockIdx.x;
    const uint32_t h = blockIdx.y, sequence = blockIdx.z;
    const int64_t t0 = ch*C;
    const int Cn = (int) (n_main - t0 < C ? n_main - t0 : C);
    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);
    const float * qb = q + iq3*sq3 + iq1*sq1;
    const float * kb = k + iq3*sq3 + iq1*sq1;
    const float * vb = v + sequence*sv3 + h*sv1;
    const int64_t gbb = sequence*sb3 + h*sb1;
    const int64_t base = ((int64_t) sequence*H + h)*nch + ch;

    for (int e = tid; e < C*(D/4); e += 256) {
        const int t = e / (D/4), w = e % (D/4);
        float4 kz = make_float4(0.0f, 0.0f, 0.0f, 0.0f), qz = kz;
        if (t < Cn) {
            kz = ((const float4 *) (kb + (t0 + t)*sq2))[w];
            qz = ((const float4 *) (qb + (t0 + t)*sq2))[w];
        }
        Ks4[t][w] = kz;
        Qs4[t][w] = qz;
    }
    for (int e = tid; e < C*D; e += 256) {
        const int t = e / D, i = e % D;
        Vs[t][i] = t < Cn ? vb[(t0 + t)*sv2 + i] : 0.0f;
    }
    if (tid < 32) {
        float lg = 0.0f, b = 0.0f;
        if (tid < Cn) {
            lg = g[gbb + (t0 + tid)*sb2];
            b  = beta[gbb + (t0 + tid)*sb2];
        }
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float y = __shfl_up_sync(0xffffffff, lg, o);
            if (tid >= o) {
                lg += y;
            }
        }
        if (tid < C) {
            cum[tid] = lg;
            bet[tid] = b;
            cumg[base*C + tid] = lg;
        }
    }
    __syncthreads();

    // v6: one thread per (t, j): A[t][j] (j < t) and QM[t][j] (j <= t) from one pass over k_j
    {
        const int t = tid / C, j = tid % C;   // 256 == C*C
        float ak = 0.0f, aq = 0.0f;
        if (t < Cn && j <= t) {
#pragma unroll 8
            for (int w = 0; w < D/4; ++w) {
                const float4 y = Ks4[j][w], x = Ks4[t][w], z = Qs4[t][w];
                ak += x.x*y.x + x.y*y.y + x.z*y.z + x.w*y.w;
                aq += z.x*y.x + z.y*y.y + z.z*y.z + z.w*y.w;
            }
        }
        const float dec = (t < Cn && j <= t) ? expf(cum[t] - cum[j]) : 0.0f;
        A[t][j] = j < t ? bet[t]*dec*ak : 0.0f;
        QMg[base*C*C + t*C + j] = dec*aq;
        if (tid < C) {
            bg[tid] = bet[tid]*expf(cum[tid]);
        }
    }
    __syncthreads();

    // v8: lane l solves column l of T = (I + A)^-1 with the column held in registers
    if (tid < C) {
        const int l = tid;
        float Tc[C];
#pragma unroll
        for (int t = 0; t < C; ++t) {
            float x = t == l ? 1.0f : 0.0f;
#pragma unroll
            for (int j = 0; j < t; ++j) {
                if (j >= l) {
                    x -= A[t][j]*Tc[j];
                }
            }
            Tc[t] = t < l ? 0.0f : x;
        }
#pragma unroll
        for (int t = 0; t < C; ++t) {
            T[t][l] = Tc[t];
        }
    }
    __syncthreads();

    // v8: thread = one column i of W (threads 0..127) or U (128..255), all 16 rows accumulated in registers
    {
        const bool isU = tid >= D;
        const int i = tid % D;
        float acc[C];
#pragma unroll
        for (int t = 0; t < C; ++t) {
            acc[t] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < C; ++j) {
            const float x = isU ? bet[j]*Vs[j][i] : bg[j]*((const float *) Ks4[j])[i];
#pragma unroll
            for (int t = j; t < C; ++t) {
                acc[t] += T[t][j]*x;
            }
        }
        float * dstm = isU ? U : W;
#pragma unroll
        for (int t = 0; t < C; ++t) {
            dstm[(base*C + t)*D + i] = t < Cn ? acc[t] : 0.0f;
        }
    }
}

template <bool keep_rs_t>
static __global__ void __launch_bounds__(LP49_NC*8, 1)
gdn49_scan(const float * q, const float * k, const float * v, const float * g, const float * beta,
           const float * W, const float * U, const float * QMg, const float * cumg,
           const float * curr_state, float * dst, float * state, int64_t H, int64_t n_tokens, int64_t n_main, int64_t nch,
           int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
           int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic, float scale,
           int64_t state_slot_stride, int K) {
    constexpr int C = LP49_C, D = LP49_D, NC = LP49_NC, D4 = D/4 + 1, NT = NC*8;
    static_assert(C*(D/4) % NT == 0 && (C*NC) % NT == 0 && C*C <= NT, "v6 fetch layout");
    __shared__ float4 Ws4[C][D4];
    __shared__ float4 Ks4[C][D4];
    __shared__ float4 Qs4[C][D4];
    __shared__ float  Us[C][NC];
    __shared__ float  QMs[C][C + 1];
    __shared__ float  cum[C];
    __shared__ float  Dl[C][NC + 1];
    __shared__ float  Os[C][NC + 1];

    const int tid = threadIdx.x;
    const int cl  = tid >> 3;              // local column (8 consecutive lanes per column)
    const int rg  = tid & 7;               // key rows: float4 blocks rg, rg+8, rg+16, rg+24 (v4)
    const int c   = blockIdx.z*NC + cl;
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);
    const float * qb = q + iq3*sq3 + iq1*sq1;
    const float * kb = k + iq3*sq3 + iq1*sq1;
    const int64_t gbb = sequence*sb3 + h_idx*sb1;
    const int64_t hb  = ((int64_t) sequence*H + h_idx)*nch;
    state += (sequence*H + h_idx)*D*D;
    // v4: lane rg owns the float4 row blocks rg, rg+8, rg+16, rg+24 (8 lanes of a column read 128 contiguous bytes)
    const float * cs_in = curr_state + sequence*H*D*D + h_idx*D*D + (int64_t) c*D;
    float * attn = dst + (sequence*n_tokens*H + h_idx)*D;

    float s[16];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float4 x = ((const float4 *) cs_in)[rg + 8*r];
        s[4*r + 0] = x.x; s[4*r + 1] = x.y; s[4*r + 2] = x.z; s[4*r + 3] = x.w;
    }

    // v5: chunk ch+1's global inputs are fetched into registers while chunk ch computes
    constexpr int NKQ = C*(D/4)/NT, NU = C*NC/NT;
    float4 pk[NKQ], pq[NKQ], pw[NKQ];
    float  pu[NU], pqm = 0.0f, pcum = 0.0f;
    auto fetch = [&](const int64_t ch) {
        const int64_t t0 = ch*C;
        const int Cn = (int) (n_main - t0 < C ? n_main - t0 : C);
        const int64_t base = hb + ch;
#pragma unroll
        for (int x = 0; x < NKQ; ++x) {
            const int e = tid + NT*x;
            const int t = e / (D/4), w = e % (D/4);
            pk[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            pq[x] = pk[x];
            if (t < Cn) {
                pk[x] = ((const float4 *) (kb + (t0 + t)*sq2))[w];
                pq[x] = ((const float4 *) (qb + (t0 + t)*sq2))[w];
            }
            pw[x] = ((const float4 *) (W + (base*C + t)*D))[w];
        }
#pragma unroll
        for (int x = 0; x < NU; ++x) {
            const int e = tid + NT*x;
            pu[x] = U[(base*C + e / NC)*D + blockIdx.z*NC + e % NC];
        }
        if (tid < C*C) {
            pqm = QMg[base*C*C + tid];
        }
        if (tid < C) {
            pcum = cumg[base*C + tid];
        }
    };
    if (nch > 0) {
        fetch(0);
    }
    for (int64_t ch = 0; ch < nch; ++ch) {
        const int64_t t0 = ch*C;
        const int Cn = (int) (n_main - t0 < C ? n_main - t0 : C);
        __syncthreads();
#pragma unroll
        for (int x = 0; x < NKQ; ++x) {
            const int e = tid + NT*x;
            const int t = e / (D/4), w = e % (D/4);
            Ks4[t][w] = pk[x];
            Qs4[t][w] = pq[x];
            Ws4[t][w] = pw[x];
        }
#pragma unroll
        for (int x = 0; x < NU; ++x) {
            const int e = tid + NT*x;
            Us[e / NC][e % NC] = pu[x];
        }
        if (tid < C*C) {
            QMs[tid / C][tid % C] = pqm;
        }
        if (tid < C) {
            cum[tid] = pcum;
        }
        __syncthreads();
        if (ch + 1 < nch) {
            fetch(ch + 1);
        }

        // v3: all 16 rows' partial dots first (split accumulators), then the 32 reductions, then the stores
        float pw[C], pq[C];
#pragma unroll
        for (int t = 0; t < C; ++t) {
            float aw0 = 0.0f, aw1 = 0.0f, aq0 = 0.0f, aq1 = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float4 a = Ws4[t][rg + 8*r], b = Qs4[t][rg + 8*r];
                aw0 += a.x*s[4*r + 0] + a.y*s[4*r + 1];
                aw1 += a.z*s[4*r + 2] + a.w*s[4*r + 3];
                aq0 += b.x*s[4*r + 0] + b.y*s[4*r + 1];
                aq1 += b.z*s[4*r + 2] + b.w*s[4*r + 3];
            }
            pw[t] = aw0 + aw1;
            pq[t] = aq0 + aq1;
        }
        // v7: reduce-scatter over the 8 lanes of a column (28 shuffles instead of 32 x 3): lane rg ends with the 4 sums
        // of value block 4*rg .. 4*rg+3 of [pw[0..15], pq[0..15]] (bit 2 of rg picks pw/pq, bits 1..0 the t quarter)
        const bool b2 = rg & 4, b1 = rg & 2, b0 = rg & 1;
        float r1[16], r2[8], r3[4];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float keep = b2 ? pq[i] : pw[i];
            const float send = b2 ? pw[i] : pq[i];
            r1[i] = keep + __shfl_xor_sync(0xffffffff, send, 4, 8);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float keep = b1 ? r1[8 + i] : r1[i];
            const float send = b1 ? r1[i] : r1[8 + i];
            r2[i] = keep + __shfl_xor_sync(0xffffffff, send, 2, 8);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float keep = b0 ? r2[4 + i] : r2[i];
            const float send = b0 ? r2[i] : r2[4 + i];
            r3[i] = keep + __shfl_xor_sync(0xffffffff, send, 1, 8);
        }
        {
            const int tq = (b1 ? 8 : 0) + (b0 ? 4 : 0);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int t = tq + i;
                if (!b2) {
                    Dl[t][cl] = t < Cn ? Us[t][cl] - r3[i] : 0.0f;
                } else {
                    Os[t][cl] = expf(cum[t])*r3[i];
                }
            }
        }
        __syncthreads();

        for (int e = tid; e < C*NC; e += NT) {
            const int t = e / NC, cc = e % NC;
            if (t < Cn) {
                float x = Os[t][cc];
                for (int j = 0; j <= t; ++j) {
                    x += QMs[t][j]*Dl[j][cc];
                }
                attn[(t0 + t)*D*H + blockIdx.z*NC + cc] = x*scale;
            }
        }
        {
            const float cend = cum[Cn - 1];
            const float gl = expf(cend);
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                s[r] *= gl;
            }
            float dj[C];
#pragma unroll
            for (int j = 0; j < C; ++j) {
                dj[j] = j < Cn ? expf(cend - cum[j])*Dl[j][cl] : 0.0f;
            }
#pragma unroll
            for (int j = 0; j < C; ++j) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float4 a = Ks4[j][rg + 8*r];
                    s[4*r + 0] += a.x*dj[j]; s[4*r + 1] += a.y*dj[j]; s[4*r + 2] += a.z*dj[j]; s[4*r + 3] += a.w*dj[j];
                }
            }
        }
    }

    // sequential tail (rollback snapshots of the last K tokens)
    for (int64_t tt = n_main; tt < n_tokens; ++tt) {
        const float4 * k4 = (const float4 *) (kb + tt*sq2);
        const float4 * q4 = (const float4 *) (qb + tt*sq2);
        float kr[16], qr[16];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float4 a = k4[rg + 8*r], b = q4[rg + 8*r];
            kr[4*r + 0] = a.x; kr[4*r + 1] = a.y; kr[4*r + 2] = a.z; kr[4*r + 3] = a.w;
            qr[4*r + 0] = b.x; qr[4*r + 1] = b.y; qr[4*r + 2] = b.z; qr[4*r + 3] = b.w;
        }
        const float bv = beta[gbb + tt*sb2];
        const float gv = expf(g[gbb + tt*sb2]);
        const float vv = v[sequence*sv3 + tt*sv2 + h_idx*sv1 + c];
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            kv += s[r]*kr[r];
        }
        kv = warp_reduce_sum<8>(kv);
        const float delta = (vv - gv*kv)*bv;
        float at = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            s[r] = gv*s[r] + kr[r]*delta;
            at  += s[r]*qr[r];
        }
        at = warp_reduce_sum<8>(at);
        if (rg == 0) {
            attn[tt*D*H + c] = at*scale;
        }
        if constexpr (keep_rs_t) {
            const int64_t slot = n_tokens - 1 - tt;
            if (slot >= 0 && slot < K) {
                float4 * cs = (float4 *) (state + slot*state_slot_stride + (int64_t) c*D);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    cs[rg + 8*r] = make_float4(s[4*r + 0], s[4*r + 1], s[4*r + 2], s[4*r + 3]);
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        float4 * cs = (float4 *) (state + (int64_t) c*D);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            cs[rg + 8*r] = make_float4(s[4*r + 0], s[4*r + 1], s[4*r + 2], s[4*r + 3]);
        }
    }
}

// returns true when the chunked path handled the op
static bool lp49_chunked(ggml_backend_cuda_context & ctx, const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d, float * dst_d, float * state_d,
        int64_t S_k, int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, bool keep_rs, cudaStream_t stream) {
    static const bool on49 = [] { const char * e = getenv("GGML_CUDA_LP0049"); return !(e && atoi(e) == 0); }();
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const bool aligned = ((uintptr_t) q_d % 16 == 0) && ((uintptr_t) k_d % 16 == 0) && ((uintptr_t) s_d % 16 == 0) &&
        ((uintptr_t) state_d % 16 == 0) && sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0 && state_slot_stride % 4 == 0;
    // v8 gate (gdnbench, P5000): the scan is throughput-bound with one 512-thread block per SM, so it wins only while
    // its grid (2 blocks per head and sequence) fits one resident wave: 16 value heads -10..-18% (256..4096 tokens),
    // 24 heads +19..26% (3 waves), 4 heads under-fills the GPU. Short batches stay sequential too.
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int64_t nblk = H*n_seqs*(LP49_D/LP49_NC);
    if (!on49 || S_v != LP49_D || S_k != LP49_D || n_tokens < 256 || !aligned || warp_size != 32 ||
            nblk < 16 || nblk > 2*nsm) {
        return false;
    }
    const int64_t n_tail = keep_rs ? (K < n_tokens ? (int64_t) K : n_tokens) : 0;
    const int64_t n_main = n_tokens - n_tail;
    const int64_t nch    = (n_main + LP49_C - 1) / LP49_C;
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
    const int64_t nbase = n_seqs*H*(nch > 0 ? nch : 1);
    ggml_cuda_pool_alloc<float> W(ctx.pool(), nbase*LP49_C*LP49_D);
    ggml_cuda_pool_alloc<float> U(ctx.pool(), nbase*LP49_C*LP49_D);
    ggml_cuda_pool_alloc<float> QM(ctx.pool(), nbase*LP49_C*LP49_C);
    ggml_cuda_pool_alloc<float> cum(ctx.pool(), nbase*LP49_C);
    if (nch > 0) {
        gdn49_prep<<<dim3(nch, H, n_seqs), 256, 0, stream>>>(q_d, k_d, v_d, g_d, b_d, W.get(), U.get(), QM.get(), cum.get(),
            H, n_main, nch, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic);
        CUDA_CHECK(cudaGetLastError());
    }
    const dim3 grid(H, n_seqs, LP49_D / LP49_NC);
    if (keep_rs) {
        gdn49_scan<true><<<grid, LP49_NC*8, 0, stream>>>(q_d, k_d, v_d, g_d, b_d, W.get(), U.get(), QM.get(), cum.get(), s_d, dst_d,
            state_d, H, n_tokens, n_main, nch, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale,
            state_slot_stride, K);
    } else {
        gdn49_scan<false><<<grid, LP49_NC*8, 0, stream>>>(q_d, k_d, v_d, g_d, b_d, W.get(), U.get(), QM.get(), cum.get(), s_d, dst_d,
            state_d, H, n_tokens, n_main, nch, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale,
            state_slot_stride, K);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    // local patch 0049: chunked prefill on GPUs without the MMA path
    if (!kda && lp49_chunked(ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, neq0, S_v, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, keep_rs, stream)) {
        return;
    }
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
