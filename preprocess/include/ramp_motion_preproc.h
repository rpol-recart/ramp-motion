#pragma once

#include <cstdint>

extern "C" {

/// Motion metric emitted by the custom library per frame per ROI.
/// Kept ABI-stable — ctypes tests and the Python probe read raw bytes.
struct MotionMeta {
    uint32_t source_id;
    uint32_t roi_id;
    uint64_t ts_ns;
    uint64_t frame_pts;
    uint32_t fg_pixel_count;
    float    motion_fraction;
    uint32_t max_blob_area_px;
};

/// Used-meta type id — project-specific (NVDS_USER_META_BASE + 0x4D4F == "MO").
constexpr int kMotionUserMetaType = 0x1000 + 0x4D4F;

/// ctypes-callable smoke probe: returns library build tag.
const char* ramp_motion_preproc_version();

/// ctypes-callable synchronous "process image" for unit tests.
int ramp_motion_preproc_process_image(
    const uint8_t* bgr,
    int width, int height, int stride_bytes,
    uint32_t source_id, uint32_t roi_id,
    uint64_t ts_ns,
    MotionMeta* out);

/// Reset MOG2/CLAHE per-source state (called on disconnect + EOS).
int ramp_motion_preproc_reset_source(uint32_t source_id);

} // extern "C"
