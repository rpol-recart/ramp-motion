#include "ramp_motion_preproc.h"

#include <mutex>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc.hpp>

namespace ramp_motion {

struct PerSource {
    cv::Ptr<cv::cuda::CLAHE> clahe;
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> mog2;
    cv::cuda::Stream stream;
};

static std::mutex g_mu;
static std::unordered_map<uint64_t, PerSource> g_state;   // key: (source<<32)|roi

static inline uint64_t key(uint32_t source_id, uint32_t roi_id) {
    return (uint64_t)source_id << 32 | (uint64_t)roi_id;
}

PerSource& get_state(uint32_t source_id, uint32_t roi_id) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto k = key(source_id, roi_id);
    auto it = g_state.find(k);
    if (it != g_state.end()) return it->second;

    PerSource ps;
    ps.clahe = cv::cuda::createCLAHE(2.0, cv::Size(8, 8));
    ps.mog2 = cv::cuda::createBackgroundSubtractorMOG2(500, 16.0, true);
    auto [ins, _] = g_state.emplace(k, std::move(ps));
    return ins->second;
}

void reset_source(uint32_t source_id) {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto it = g_state.begin(); it != g_state.end();) {
        if (static_cast<uint32_t>(it->first >> 32) == source_id) {
            it = g_state.erase(it);
        } else {
            ++it;
        }
    }
}

int process_bgr(const uint8_t* bgr, int w, int h, int stride,
                uint32_t source_id, uint32_t roi_id, uint64_t ts_ns,
                MotionMeta* out)
{
    if (!out) return -1;
    out->source_id = source_id;
    out->roi_id = roi_id;
    out->ts_ns = ts_ns;
    out->frame_pts = 0;

    cv::Mat host_bgr(h, w, CV_8UC3, const_cast<uint8_t*>(bgr), stride);
    cv::cuda::GpuMat bgr_gpu;
    bgr_gpu.upload(host_bgr);

    cv::cuda::GpuMat gray;
    cv::cuda::cvtColor(bgr_gpu, gray, cv::COLOR_BGR2GRAY);

    PerSource& s = get_state(source_id, roi_id);

    cv::cuda::GpuMat clahe_out;
    s.clahe->apply(gray, clahe_out, s.stream);

    cv::cuda::GpuMat fg_mask;
    s.mog2->apply(clahe_out, fg_mask, -1.0, s.stream);

    cv::cuda::GpuMat fg_binary;
    cv::cuda::threshold(fg_mask, fg_binary, 200, 255, cv::THRESH_BINARY, s.stream);
    s.stream.waitForCompletion();

    int fg = cv::cuda::countNonZero(fg_binary);
    out->fg_pixel_count = static_cast<uint32_t>(fg);
    out->motion_fraction = (w * h > 0) ? static_cast<float>(fg) / (float)(w * h) : 0.f;

    cv::Mat fg_cpu;
    fg_binary.download(fg_cpu);
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(fg_cpu, labels, stats, centroids, 8, CV_32S);
    int max_blob = 0;
    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > max_blob) max_blob = area;
    }
    out->max_blob_area_px = static_cast<uint32_t>(max_blob);

    return 0;
}

} // namespace ramp_motion

extern "C" int ramp_motion_preproc_reset_source(uint32_t source_id) {
    ramp_motion::reset_source(source_id);
    return 0;
}

extern "C" int ramp_motion_preproc_process_image(
    const uint8_t* bgr, int w, int h, int stride,
    uint32_t source_id, uint32_t roi_id, uint64_t ts_ns,
    MotionMeta* out)
{
    return ramp_motion::process_bgr(bgr, w, h, stride, source_id, roi_id, ts_ns, out);
}
