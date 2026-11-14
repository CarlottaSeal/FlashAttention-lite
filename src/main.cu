#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <functional>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "attention.cuh"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t e_ = (call);                                             \
        if (e_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "cuda %s:%d: %s\n",                         \
                         __FILE__, __LINE__, cudaGetErrorString(e_));        \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)


// Double-precision stable-softmax reference for correctness checks.
static void attention_cpu_ref(const float* Q, const float* K, const float* V,
                              float* O, int N, int D)
{
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<double> s(N);
    for (int i = 0; i < N; ++i) {
        double mx = -1e30;
        for (int j = 0; j < N; ++j) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d) dot += (double)Q[i*D+d] * (double)K[j*D+d];
            s[j] = dot * scale;
            if (s[j] > mx) mx = s[j];
        }
        double sum = 0.0;
        for (int j = 0; j < N; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
        double inv = 1.0 / sum;
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int j = 0; j < N; ++j) acc += s[j] * (double)V[j*D+d];
            O[i*D+d] = (float)(acc * inv);
        }
    }
}

struct Stats { float max_abs, mean_abs; };
static Stats diff_stats(const std::vector<float>& a, const std::vector<float>& b)
{
    double mx = 0.0, sum = 0.0;
    for (size_t k = 0; k < a.size(); ++k) {
        double d = std::fabs((double)a[k] - (double)b[k]);
        if (d > mx) mx = d;
        sum += d;
    }
    return { (float)mx, (float)(sum / a.size()) };
}

static float time_kernel_ms(int warmup, int repeats, std::function<void()> run)
{
    for (int i = 0; i < warmup; ++i) run();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < repeats; ++i) run();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / repeats;
}

int main(int argc, char** argv)
{
    int N = 512;
    int D = 64;
    int B = 1, H = 1;
    int BC = 0;    // v5 K/V tile, 0 = per-D default
    int seed = 42;
    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--N") == 0 && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--D") == 0 && i + 1 < argc) D = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--B") == 0 && i + 1 < argc) B = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--H") == 0 && i + 1 < argc) H = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--BC") == 0 && i + 1 < argc) BC = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = std::atoi(argv[++i]);
    }
    const int BH = B * H;
    const size_t ND = (size_t)N * D, all = BH * ND;
    std::printf("B=%d  H=%d  N=%d  D=%d\n", B, H, N, D);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt((float)D));

    // [B, H, N, D]; the single-head kernels (naive..v4) only see head 0
    std::vector<float> Q(all), K(all), V(all);
    for (auto& x : Q) x = dist(rng);
    for (auto& x : K) x = dist(rng);
    for (auto& x : V) x = dist(rng);

    float *dQ, *dK, *dV, *dO, *dS;
    __half *dQh, *dKh, *dVh;
    CUDA_CHECK(cudaMalloc(&dQ, all * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, all * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, all * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, all * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dS, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQh, all * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dKh, all * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dVh, all * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice));

    int nQ = (int)all;
    int bt = 256;
    float_to_half_kernel<<<(nQ + bt - 1) / bt, bt>>>(dQ, dQh, nQ);
    float_to_half_kernel<<<(nQ + bt - 1) / bt, bt>>>(dK, dKh, nQ);
    float_to_half_kernel<<<(nQ + bt - 1) / bt, bt>>>(dV, dVh, nQ);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto run_naive   = [&]() { run_attention_naive(dQ, dK, dV, dO, dS, N, D); };
    auto run_flash   = [&]() { run_attention_flash(dQ, dK, dV, dO, N, D); };
    auto run_qtile   = [&]() { run_attention_flash_qtile(dQ, dK, dV, dO, N, D); };
    auto run_wmma    = [&]() { run_attention_flash_wmma(dQh, dKh, dVh, dO, N, D); };
    auto run_wmma_v4 = [&]() { run_attention_flash_wmma_v4(dQh, dKh, dVh, dO, N, D); };
    auto run_v5      = [&]() { run_attention_flash_v5(dQh, dKh, dVh, dO, BH, N, D, BC); };
    auto run_v4_loop = [&]() {   // v4 launched once per head, for comparison with v5
        for (int h = 0; h < BH; ++h)
            run_attention_flash_wmma_v4(dQh + h * ND, dKh + h * ND, dVh + h * ND, dO + h * ND, N, D);
    };

    std::vector<float> O_cpu(N * D), O_naive(N * D), O_flash(N * D), O_qtile(N * D),
                       O_wmma(N * D), O_wmma_v4(N * D), O_cpu_all(all), O_v5(all);

    // CPU reference
    auto t0 = std::chrono::steady_clock::now();
    attention_cpu_ref(Q.data(), K.data(), V.data(), O_cpu.data(), N, D);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::copy(O_cpu.begin(), O_cpu.end(), O_cpu_all.begin());
    for (int h = 1; h < BH; ++h)
        attention_cpu_ref(&Q[h * ND], &K[h * ND], &V[h * ND], &O_cpu_all[h * ND], N, D);

    // naive
    run_naive();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_naive.data(), dO, O_naive.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // flash
    run_flash();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_flash.data(), dO, O_flash.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // q-tile flash
    run_qtile();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_qtile.data(), dO, O_qtile.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // wmma flash v3 (FP16 input, Tensor Core matmul, 1 warp / block)
    run_wmma();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_wmma.data(), dO, O_wmma.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // wmma flash v4 (BR=64, 4 warps / block)
    run_wmma_v4();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_wmma_v4.data(), dO, O_wmma_v4.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // v5, all heads
    run_v5();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_v5.data(), dO, O_v5.size() * sizeof(float), cudaMemcpyDeviceToHost));

    Stats s_n  = diff_stats(O_naive,   O_cpu);
    Stats s_f  = diff_stats(O_flash,   O_cpu);
    Stats s_q  = diff_stats(O_qtile,   O_cpu);
    Stats s_w  = diff_stats(O_wmma,    O_cpu);
    Stats s_w4 = diff_stats(O_wmma_v4, O_cpu);
    Stats s_v5 = diff_stats(O_v5,      O_cpu_all);
    std::printf("naive   vs cpu-ref:  max|Δ|=%.3e  mean|Δ|=%.3e\n", s_n.max_abs,  s_n.mean_abs);
    std::printf("flash   vs cpu-ref:  max|Δ|=%.3e  mean|Δ|=%.3e\n", s_f.max_abs,  s_f.mean_abs);
    std::printf("qtile   vs cpu-ref:  max|Δ|=%.3e  mean|Δ|=%.3e\n", s_q.max_abs,  s_q.mean_abs);
    std::printf("wmma v3 vs cpu-ref:  max|Δ|=%.3e  mean|Δ|=%.3e  (FP16 input)\n",
                s_w.max_abs, s_w.mean_abs);
    std::printf("wmma v4 vs cpu-ref:  max|Δ|=%.3e  mean|Δ|=%.3e  (FP16 input, BR=64)\n",
                s_w4.max_abs, s_w4.mean_abs);
    std::printf("v5      vs cpu-ref:  max|Δ|=%.3e  mean|Δ|=%.3e  (all %d heads)\n",
                s_v5.max_abs, s_v5.mean_abs, BH);

    float t_naive   = time_kernel_ms(3, 10, run_naive);
    float t_flash   = time_kernel_ms(3, 10, run_flash);
    float t_qtile   = time_kernel_ms(3, 10, run_qtile);
    float t_wmma    = time_kernel_ms(3, 10, run_wmma);
    float t_wmma_v4 = time_kernel_ms(3, 10, run_wmma_v4);
    float t_v4_loop = time_kernel_ms(3, 10, run_v4_loop);
    float t_v5      = time_kernel_ms(3, 10, run_v5);
    std::printf("naive:   %.3f ms\n", t_naive);
    std::printf("flash:   %.3f ms  (vs naive %.2fx)\n", t_flash, t_naive / t_flash);
    std::printf("qtile:   %.3f ms  (vs naive %.2fx)\n", t_qtile, t_naive / t_qtile);
    std::printf("wmma v3: %.3f ms  (vs naive %.2fx, vs qtile %.2fx)\n",
                t_wmma, t_naive / t_wmma, t_qtile / t_wmma);
    std::printf("wmma v4: %.3f ms  (vs naive %.2fx, vs v3 %.2fx)\n",
                t_wmma_v4, t_naive / t_wmma_v4, t_wmma / t_wmma_v4);
    std::printf("%d heads: v4 x%d launches %.3f ms, v5 one launch %.3f ms (%.2fx)\n",
                BH, BH, t_v4_loop, t_v5, t_v4_loop / t_v5);
    std::printf("cpu ref: %.1f ms (head 0)\n", cpu_ms);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dS);
    cudaFree(dQh); cudaFree(dKh); cudaFree(dVh);
    return 0;
}
