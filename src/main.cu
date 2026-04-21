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


// CPU reference: numerically stable, double precision internally.
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
    int seed = 42;
    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--N") == 0 && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--D") == 0 && i + 1 < argc) D = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = std::atoi(argv[++i]);
    }
    std::printf("N=%d  D=%d\n", N, D);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt((float)D));

    std::vector<float> Q(N * D), K(N * D), V(N * D);
    for (auto& x : Q) x = dist(rng);
    for (auto& x : K) x = dist(rng);
    for (auto& x : V) x = dist(rng);

    float *dQ, *dK, *dV, *dO, *dS;
    CUDA_CHECK(cudaMalloc(&dQ, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dS, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto run_naive = [&]() { run_attention_naive(dQ, dK, dV, dO, dS, N, D); };
    auto run_flash = [&]() { run_attention_flash(dQ, dK, dV, dO, N, D); };

    std::vector<float> O_cpu(N * D), O_naive(N * D), O_flash(N * D);

    // CPU reference
    auto t0 = std::chrono::steady_clock::now();
    attention_cpu_ref(Q.data(), K.data(), V.data(), O_cpu.data(), N, D);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // naive
    run_naive();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_naive.data(), dO, O_naive.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // flash
    run_flash();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_flash.data(), dO, O_flash.size() * sizeof(float), cudaMemcpyDeviceToHost));

    Stats s_n = diff_stats(O_naive, O_cpu);
    Stats s_f = diff_stats(O_flash, O_cpu);
    std::printf("naive vs cpu-ref:  max|Δ|=%.3e  mean|Δ|=%.3e\n", s_n.max_abs, s_n.mean_abs);
    std::printf("flash vs cpu-ref:  max|Δ|=%.3e  mean|Δ|=%.3e\n", s_f.max_abs, s_f.mean_abs);

    float t_naive = time_kernel_ms(3, 10, run_naive);
    float t_flash = time_kernel_ms(3, 10, run_flash);
    std::printf("naive:  %.3f ms\n", t_naive);
    std::printf("flash:  %.3f ms  (speedup %.2fx)\n", t_flash, t_naive / t_flash);
    std::printf("cpu ref: %.1f ms\n", cpu_ms);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dS);
    return 0;
}
