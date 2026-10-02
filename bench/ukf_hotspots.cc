// What one UKF filter step spends its time on, piece by piece, at Stanford HARDI's size
// (N = 150 gradients, doubled to 300 measurement rows by UKF; state n = 10; 21 sigma points).
// Mirrors pnlbwh/ukftractography 2d2b661 (ukf/unscented_kalman_filter.cc, filter_Simple2T.cc):
//
//   rcopy    `const ukfMatrixType temp = localConstFilterModel->R();` copies the dense 300x300
//            measurement-noise matrix (720 KB) every step, to read temp(0,0)
//   allocs   the step's other heap matrices: X, dim_dimext, X_, P, Yk, Z, signaldim_dimext, Z_,
//            Pxz, Ht, the signal vector... (Eigen dynamic matrices, freed at step end)
//   H        the signal model as written: per sigma point, per row of 300, two 3x3 mat-vec products
//            u.(D u) and two exps
//   H_lean   the same values from 150 rows (the antipodal duplicates are identical) and
//            u'Du = l2 + (l1 - l2)(u.m)^2
//
// Build and run (natively, and as x86_64 under Rosetta, as the Slicer extension runs):
//   clang++ -O2 -std=c++17 ukf_hotspots.cc -o hot_arm64 && ./hot_arm64
//   clang++ -O2 -std=c++17 -arch x86_64 ukf_hotspots.cc -o hot_x86 && ./hot_x86
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static const int N = 150, M = 2 * N, n = 10, S = 2 * n + 1;
static volatile double sink;

static double rcopy(int iters) {
  std::vector<double> R(M * M, 0.0);
  for (int i = 0; i < M; ++i) R[i * M + i] = 0.02;
  double acc = 0;
  for (int it = 0; it < iters; ++it) {
    double* t = static_cast<double*>(std::malloc(sizeof(double) * M * M));
    std::memcpy(t, R.data(), sizeof(double) * M * M);
    acc += 1.0 / t[0];
    std::free(t);
  }
  return acc;
}

static double allocs(int iters) {
  const size_t sizes[] = {n * S, n * n, n * n, n * S, n * S, n * n, n * n, n, M * S, M * S, M * S,
                          n * M, n * M, n * n, n, M, M, n};
  double acc = 0;
  for (int it = 0; it < iters; ++it) {
    double* p[sizeof(sizes) / sizeof(sizes[0])];
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); ++k) {
      p[k] = static_cast<double*>(std::malloc(sizeof(double) * sizes[k]));
      p[k][0] = 1.0; p[k][sizes[k] - 1] = 2.0;            // touch both ends, as setConstant/assign would
    }
    for (auto q : p) { acc += q[0]; std::free(q); }
  }
  return acc;
}

struct Model { std::vector<double> g; std::vector<double> b; };

static Model model() {
  Model m; m.g.resize(3 * M); m.b.resize(M, 2000.0);
  unsigned s = 1;
  for (int j = 0; j < N; ++j) {
    double v[3]; for (double& c : v) { s = s * 1103515245u + 12345u; c = (s >> 8) / double(1 << 24) - 0.5; }
    double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int c = 0; c < 3; ++c) { m.g[3 * j + c] = v[c] / r; m.g[3 * (j + N) + c] = -v[c] / r; }
  }
  return m;
}

static void diffusion(const double m[3], double l1, double l2, double D[9]) {   // common/linalg.h:67
  double R[9] = {m[0], m[1], m[2],
                 m[1], m[1] * m[1] / (1 + m[0]) - 1, m[1] * m[2] / (1 + m[0]),
                 m[2], m[1] * m[2] / (1 + m[0]), m[2] * m[2] / (1 + m[0]) - 1};
  double L[3] = {l1, l2, l2};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      double s = 0; for (int k = 0; k < 3; ++k) s += R[r * 3 + k] * L[k] * R[c * 3 + k];
      D[r * 3 + c] = s * 1e-6;
    }
}

static double H(const Model& md, int iters) {
  double acc = 0;
  double m1[3] = {0.8, 0.36, 0.48}, m2[3] = {0.6, -0.64, 0.48};
  for (int it = 0; it < iters; ++it)
    for (int s = 0; s < S; ++s) {
      double D1[9], D2[9];
      diffusion(m1, 1700 + s, 300, D1); diffusion(m2, 1600 + s, 350, D2);
      for (int j = 0; j < M; ++j) {                                    // all 2N rows, as written
        const double* u = &md.g[3 * j];
        double a = 0, c = 0;
        for (int r = 0; r < 3; ++r) {
          double Du1 = D1[r * 3] * u[0] + D1[r * 3 + 1] * u[1] + D1[r * 3 + 2] * u[2];
          double Du2 = D2[r * 3] * u[0] + D2[r * 3 + 1] * u[1] + D2[r * 3 + 2] * u[2];
          a += u[r] * Du1; c += u[r] * Du2;
        }
        acc += 0.5 * std::exp(-md.b[j] * a) + 0.5 * std::exp(-md.b[j] * c);
      }
    }
  return acc;
}

static double H_lean(const Model& md, int iters) {
  double acc = 0;
  double m1[3] = {0.8, 0.36, 0.48}, m2[3] = {0.6, -0.64, 0.48};
  for (int it = 0; it < iters; ++it)
    for (int s = 0; s < S; ++s) {
      double l11 = (1700 + s) * 1e-6, l21 = 300e-6, l12 = (1600 + s) * 1e-6, l22 = 350e-6;
      for (int j = 0; j < N; ++j) {                                    // N rows: duplicates dropped
        const double* u = &md.g[3 * j];
        double p1 = u[0] * m1[0] + u[1] * m1[1] + u[2] * m1[2];
        double p2 = u[0] * m2[0] + u[1] * m2[1] + u[2] * m2[2];
        acc += 0.5 * std::exp(-md.b[j] * (l21 + (l11 - l21) * p1 * p1))
             + 0.5 * std::exp(-md.b[j] * (l22 + (l12 - l22) * p2 * p2));
      }
    }
  return acc;
}

template <class Fn>
static double per_step_us(Fn fn, int iters, int threads) {
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back([&] { sink = fn(iters); });
  for (auto& th : pool) th.join();
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return s / iters * 1e6;                       // wall time per step per thread
}

int main() {
  Model md = model();
  const int it = 4000;
  for (int threads : {1, 8}) {
    double a = per_step_us([](int i) { return rcopy(i); }, it, threads);
    double b = per_step_us([](int i) { return allocs(i); }, it, threads);
    double c = per_step_us([&](int i) { return H(md, i); }, it, threads);
    double d = per_step_us([&](int i) { return H_lean(md, i); }, it, threads);
    std::printf("{\"threads\": %d, \"rcopy_us\": %.1f, \"allocs_us\": %.1f, \"H_as_written_us\": %.1f, \"H_lean_us\": %.1f}\n",
                threads, a, b, c, d);
  }
}
