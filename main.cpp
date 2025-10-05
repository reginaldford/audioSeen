#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <fftw3.h>
#include <iostream>
#include <pulse/error.h>
#include <pulse/simple.h>
#include <vector>

volatile std::sig_atomic_t g_stop = 0;
void sigint_handler(int) { g_stop = 1; }

static const unsigned SAMPLE_RATE = 44100;
static const unsigned CHANNELS = 2;
static const unsigned FRAMES_PER_BUFFER = 512*2;
static const int BAR_WIDTH = 256;

// Simple processing: elementwise gain
inline void processingBlock(Eigen::ArrayXf &buffer) { buffer *= 0.8f; }

int main() {
  std::signal(SIGINT, sigint_handler);

  pa_sample_spec sample_spec;
  sample_spec.format = PA_SAMPLE_FLOAT32LE;
  sample_spec.rate = SAMPLE_RATE;
  sample_spec.channels = CHANNELS;

  int error;
  pa_simple *pa_in =
      pa_simple_new(nullptr, "pa_eigen_fft", PA_STREAM_RECORD, nullptr,
                    "record", &sample_spec, nullptr, nullptr, &error);
  if (!pa_in) {
    std::cerr << "Failed to open capture: " << pa_strerror(error) << "\n";
    return 1;
  }

  const size_t samples_per_buffer = FRAMES_PER_BUFFER * CHANNELS;

  // Triple buffers
  Eigen::ArrayXf buffer1(samples_per_buffer);
  Eigen::ArrayXf buffer2(samples_per_buffer);
  Eigen::ArrayXf buffer3(samples_per_buffer);
  Eigen::ArrayXf *buffers[3] = {&buffer1, &buffer2, &buffer3};

  // FFT preallocation
  unsigned nfft = FRAMES_PER_BUFFER;
  unsigned n_bins = nfft / 2 + 1;
  Eigen::ArrayXd fft_in(nfft);
  fft_in.setZero();
  fftw_complex *fft_out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_bins);
  fftw_plan fft_plan =
      fftw_plan_dft_r2c_1d(nfft, fft_in.data(), fft_out, FFTW_ESTIMATE);

  double max_seen_power = 1.0;
  const double alpha = 0.99;

  unsigned flipCount = 0;
  const std::string block_utf8 = u8"█"; // full UTF-8 block

  std::cout << "Starting audio bar display (Ctrl+C to stop)...\n";

  while (!g_stop) {
    int idx_in = flipCount % 3;
    int idx_proc = (flipCount + 1) % 3;

    // Capture audio
    if (pa_simple_read(pa_in, buffers[idx_in]->data(),
                       samples_per_buffer * sizeof(float), &error) < 0) {
      std::cerr << "Read error: " << pa_strerror(error) << "\n";
      break;
    }

    // Process buffer
    processingBlock(*buffers[idx_in]);
    *buffers[idx_proc] = *buffers[idx_in]; // copy to processing buffer

    // Copy to FFT input
    fft_in.head(nfft) = buffers[idx_proc]->head(nfft).cast<double>();

    // Execute FFT
    fftw_execute_dft_r2c(fft_plan, fft_in.data(), fft_out);

    // Convert FFT output to Eigen::ArrayXd and compute abs^2 (power)
    Eigen::ArrayXd power(n_bins);
    for (unsigned i = 0; i < n_bins; ++i) {
      double re = fft_out[i][0];
      double im = fft_out[i][1];
      power(i) = re * re + im * im; // vectorized power computation
    }

    // Human hearing range (20Hz-20kHz)
    double freq_res = SAMPLE_RATE / static_cast<double>(nfft);
    unsigned min_bin = static_cast<unsigned>(20.0 / freq_res);
    unsigned max_bin =
        std::min(static_cast<unsigned>(20000.0 / freq_res), n_bins - 1);
    Eigen::ArrayXd hrange = power.segment(min_bin, max_bin - min_bin + 1);

    // Mean power with Eigen
    double mean_power = hrange.mean();
    max_seen_power = std::max(max_seen_power * alpha, mean_power);

    // Compute bar length and build string efficiently
    int bar_len = static_cast<int>(std::round(
        BAR_WIDTH * std::log10(1 + 9 * mean_power / max_seen_power)));
    if (bar_len > BAR_WIDTH)
      bar_len = BAR_WIDTH;

    std::string bar;
    bar.reserve(bar_len * 3); // reserve for UTF-8
    for (int i = 0; i < bar_len; ++i)
      bar += block_utf8;

    // Single-line print
    std::cout << bar << std::endl; // flush;
    fflush(stdout);
    flipCount++;
  }

  std::cout << "\nStopping.\n";

  fftw_destroy_plan(fft_plan);
  fftw_free(fft_out);
  pa_simple_free(pa_in);

  return 0;
}
