#include "ramp_motion_preproc.h"

#include <cstdio>
#include <cstring>
#include <string>        // NVIDIA nvdspreprocess headers use std::string without including it
#include <unordered_map> // same — they use std::unordered_map without including it
#include <vector>

#include <gst/gst.h>

#include "nvbufsurface.h"
#include "nvdspreprocess_lib.h"
#include "nvdspreprocess_meta.h"
#include "gstnvdsmeta.h"

// Forward declarations from motion_meta.cpp:
extern "C" NvDsUserMeta*
ramp_motion_attach_user_meta(NvDsFrameMeta*, const MotionMeta&);

// Forward declaration from clahe_mog2.cu:
extern "C" int ramp_motion_preproc_process_image(
    const uint8_t*, int, int, int,
    uint32_t, uint32_t, uint64_t, MotionMeta*);

// ---------- nvdspreprocess API (DS 8.0 contract) ----------
// CustomCtx is an opaque type declared in nvdspreprocess_interface.h.
// We define its contents here.
struct CustomCtx {
    // No custom state required — per-source CLAHE/MOG2 state lives in clahe_mog2.cu.
};

extern "C" {

const char* ramp_motion_preproc_version() {
    return "ramp_motion_preproc 0.1.0 (clahe+mog2+nvdspreprocess DS8)";
}

NvDsPreProcessStatus CustomTensorPreparation(
    CustomCtx* /*ctx*/,
    NvDsPreProcessBatch* batch,
    NvDsPreProcessCustomBuf*& /*buf*/,
    CustomTensorParams& /*tensor_params*/,
    NvDsPreProcessAcquirer* /*acquirer*/)
{
    static int call_count = 0;
    ++call_count;
    if (call_count <= 3 || call_count % 100 == 0) {
        fprintf(stderr, "[custom_lib] CustomTensorPreparation call=%d batch=%p inbuf=%p units=%zu\n",
                call_count, (void*)batch,
                batch ? (void*)batch->inbuf : nullptr,
                batch ? batch->units.size() : 0);
        fflush(stderr);
    }

    if (!batch || !batch->inbuf) return NVDSPREPROCESS_INVALID_PARAMS;

    // Map the input GstBuffer to access the batched NvBufSurface.
    GstMapInfo gmap;
    if (!gst_buffer_map(batch->inbuf, &gmap, GST_MAP_READ)) {
        return NVDSPREPROCESS_INVALID_PARAMS;
    }
    NvBufSurface* src_surf = reinterpret_cast<NvBufSurface*>(gmap.data);
    if (!src_surf) {
        gst_buffer_unmap(batch->inbuf, &gmap);
        return NVDSPREPROCESS_INVALID_PARAMS;
    }

    for (const auto& unit : batch->units) {
        NvDsFrameMeta* frame_meta = unit.frame_meta;
        if (!frame_meta) continue;

        const int batch_idx = static_cast<int>(unit.batch_index);
        if (batch_idx < 0 ||
            static_cast<unsigned>(batch_idx) >= src_surf->numFilled) {
            continue;
        }

        // ROI rectangle in source-frame pixel coordinates.
        const NvDsRoiMeta& roi = unit.roi_meta;
        const int roi_x = static_cast<int>(roi.roi.left);
        const int roi_y = static_cast<int>(roi.roi.top);
        const int w     = static_cast<int>(roi.roi.width);
        const int h     = static_cast<int>(roi.roi.height);
        if (w <= 0 || h <= 0) continue;

        // Map this surface index to CPU-accessible memory (pinned).
        if (NvBufSurfaceMap(src_surf, batch_idx, 0, NVBUF_MAP_READ) != 0) {
            continue;
        }
        NvBufSurfaceSyncForCpu(src_surf, batch_idx, 0);

        const NvBufSurfaceParams* p = &src_surf->surfaceList[batch_idx];
        const uint8_t* mapped =
            static_cast<const uint8_t*>(p->mappedAddr.addr[0]);
        if (mapped == nullptr) {
            NvBufSurfaceUnMap(src_surf, batch_idx, 0);
            continue;
        }
        const int pitch = static_cast<int>(p->pitch);

        // Nvstreammux output is RGBA (NVBUF_COLOR_FORMAT_RGBA) by default.
        // Convert to tightly-packed BGR as we copy the ROI sub-rect out.
        std::vector<uint8_t> bgr_host(
            static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
        uint8_t* dst = bgr_host.data();
        for (int row = 0; row < h; ++row) {
            const uint8_t* src_row =
                mapped + (roi_y + row) * pitch + roi_x * 4;  // RGBA source
            uint8_t*       dst_row =
                dst + static_cast<size_t>(row) * static_cast<size_t>(w) * 3;
            for (int col = 0; col < w; ++col) {
                dst_row[col * 3 + 0] = src_row[col * 4 + 2];  // B ← R position
                dst_row[col * 3 + 1] = src_row[col * 4 + 1];  // G
                dst_row[col * 3 + 2] = src_row[col * 4 + 0];  // R ← B position
            }
        }

        NvBufSurfaceUnMap(src_surf, batch_idx, 0);

        MotionMeta meta{};
        const uint64_t ts_ns = frame_meta->buf_pts;
        ramp_motion_preproc_process_image(
            bgr_host.data(), w, h, w * 3,
            frame_meta->source_id, 0 /* roi_id */, ts_ns, &meta);
        meta.frame_pts = frame_meta->buf_pts;

        ramp_motion_attach_user_meta(frame_meta, meta);
    }

    gst_buffer_unmap(batch->inbuf, &gmap);
    // We don't actually produce a tensor — signal "skip downstream inference".
    return NVDSPREPROCESS_TENSOR_NOT_READY;
}

CustomCtx* initLib(CustomInitParams /*params*/) {
    return new CustomCtx();
}

void deInitLib(CustomCtx* ctx) {
    delete ctx;
}

// Pass-through ROI transformation: defer to nvbufsurftransform with the
// caller-supplied params. nvdspreprocess requires this symbol when the config
// specifies `custom-input-transformation-function`.
NvDsPreProcessStatus CustomTransformation(
    NvBufSurface* in_surf,
    NvBufSurface* out_surf,
    CustomTransformParams& params)
{
    NvBufSurfTransform_Error err =
        NvBufSurfTransform(in_surf, out_surf, &params.transform_params);
    return (err == NvBufSurfTransformError_Success)
        ? NVDSPREPROCESS_SUCCESS
        : NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
}

} // extern "C"
