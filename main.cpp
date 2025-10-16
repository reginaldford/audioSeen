#include <Eigen/Dense>
#include <algorithm>
#include <alsa/asoundlib.h>
#include <cmath>
#include <csignal>
#include <fftw3.h>
#include <iostream>
#include <vector>

volatile std::sig_atomic_t g_stop = 0;
void sigint_handler(int) { g_stop = 1; }

static const unsigned SAMPLE_RATE = 44100;
static const unsigned CHANNELS = 2;
static const unsigned FRAMES_PER_BUFFER = 512 * 2;
static const int BAR_WIDTH = 256;

// Simple processing: elementwise gain
inline void processingBlock(Eigen::ArrayXf &buffer) { buffer *= 0.8f; }

int main() {
  std::signal(SIGINT, sigint_handler);

  snd_pcm_t *pcm_handle;
  snd_pcm_hw_params_t *params;
  snd_pcm_uframes_t frames;
  int dir;
  unsigned int rate = SAMPLE_RATE;
  int err;

  // Open PCM device for recording (capture)
  if ((err = snd_pcm_open(&pcm_handle, "default", SND_PCM_STREAM_CAPTURE, 0)) <
      0) {
    std::cerr << "Failed to open PCM device: " << snd_strerror(err) << "\n";
    return 1;
  }

  snd_pcm_hw_params_alloca(&params);
  if ((err = snd_pcm_hw_params_any(pcm_handle, params)) < 0) {
    std::cerr << "Failed to initialize hardware parameters: "
              << snd_strerror(err) << "\n";
    return 1;
  }

  // Set sample format
  if ((err = snd_pcm_hw_params_set_format(pcm_handle, params,
                                          SND_PCM_FORMAT_FLOAT)) < 0) {
    std::cerr << "Failed to set format: " << snd_strerror(err) << "\n";
    return 1;
  }

  // Set sample rate
  if ((err = snd_pcm_hw_params_set_rate_near(pcm_handle, params, &rate, &dir)) <
      0) {
    std::cerr << "Failed to set rate: " << snd_strerror(err) << "\n";
    return 1;
  }

  // Set number of channels
  if ((err = snd_pcm_hw_params_set_channels(pcm_handle, params, CHANNELS)) <
      0) {
    std::cerr << "Failed to set channels: " << snd_strerror(err) << "\n";
    return 1;
  }

  // Set period size (buffer size)
  frames = FRAMES_PER_BUFFER;
  if ((err = snd_pcm_hw_params_set_period_size_near(pcm_handle, params, &frames,
                                                    &dir)) < 0) {
    std::cerr << "Failed to set period size: " << snd_strerror(err) << "\n";
    return 1;
  }

  // Apply hardware parameters
  if ((err = snd_pcm_hw_params(pcm_handle, params)) < 0) {
    std::cerr << "Failed to apply hardware parameters: " << snd_strerror(err)
              << "\n";
    return 1;
  }

  // Allocate buffer for audio data
  const size_t samples_per_buffer = FRAMES_PER_BUFFER * CHANNELS;
  std::vector<float> buffer(samples_per_buffer);

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
    // Capture audio
    if ((err = snd_pcm_readi(pcm_handle, buffer.data(), FRAMES_PER_BUFFER)) !=
        FRAMES_PER_BUFFER) {
      std::cerr << "Read error: " << snd_strerror(err) << "\n";
      break;
    }

    // Convert to Eigen array
    Eigen::ArrayXf eigen_buffer =
        Eigen::Map<Eigen::ArrayXf>(buffer.data(), samples_per_buffer);

    // Process buffer
    processingBlock(eigen_buffer);

    // Copy to FFT input
    fft_in.head(nfft) = eigen_buffer.head(nfft).cast<double>();

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
  snd_pcm_close(pcm_handle);

  return 0;
}
