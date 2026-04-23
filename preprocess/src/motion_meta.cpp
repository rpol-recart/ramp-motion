#include "ramp_motion_preproc.h"

#include "nvdsmeta.h"

extern "C" {

static void motion_meta_release(void* /*gst_meta*/, void* user_data) {
    auto* m = static_cast<MotionMeta*>(user_data);
    delete m;
}

NvDsUserMeta* ramp_motion_attach_user_meta(NvDsFrameMeta* frame_meta,
                                           const MotionMeta& src)
{
    if (!frame_meta) return nullptr;
    NvDsBatchMeta* batch_meta = frame_meta->base_meta.batch_meta;
    NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    if (!user_meta) return nullptr;

    auto* payload = new MotionMeta(src);
    user_meta->user_meta_data = payload;
    // kMotionUserMetaType is already in the NVDS_START_USER_META (0x1000+) range.
    // Do NOT add NVDS_USER_META on top — that was a bug that produced a
    // garbage meta_type that the probe could never match.
    user_meta->base_meta.meta_type =
        static_cast<NvDsMetaType>(kMotionUserMetaType);
    user_meta->base_meta.copy_func = nullptr;
    user_meta->base_meta.release_func = motion_meta_release;

    nvds_add_user_meta_to_frame(frame_meta, user_meta);
    return user_meta;
}

} // extern "C"
