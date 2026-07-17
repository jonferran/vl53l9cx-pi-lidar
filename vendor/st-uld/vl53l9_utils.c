/**
 ******************************************************************************
 * @file    vl53l9_utils.c
 * @author  IMD Software Team
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#include "vl53l9_utils.h"
#include <string.h>

#define USEC_PER_SEC           (1000000U)
#define FPS_TO_FRAME_PERIOD(x) ((uint32_t)((1 / (float)x) * USEC_PER_SEC))

typedef struct {
    uint8_t _binning;
    uint8_t _width;
    uint8_t _height;
} _resolution_lut_t;

vl53l9_profile_t g_ranging_profiles[VL53L9_NB_USECASES] = {
    {
        .id = VL53L9_USECASE_AR_RANGE,
        .sync = VL53L9_SYNC_AUTONOMOUS,
        .power = VL53L9_POWER_REGULAR,
        .context = VL53L9_CONTEXT_LONG,
        .frame_period_us = FPS_TO_FRAME_PERIOD(30),
        .binning = 2,
        .exposure_ms = 8,
    },
    {
        .id = VL53L9_USECASE_AR_PRECISION,
        .sync = VL53L9_SYNC_AUTONOMOUS,
        .power = VL53L9_POWER_ULTRA_LOW,
        .context = VL53L9_CONTEXT_SHORT,
        .frame_period_us = FPS_TO_FRAME_PERIOD(30),
        .binning = 2,
        .exposure_ms = 10,
    },
    {
        .id = VL53L9_USECASE_AF_RANGE,
        .sync = VL53L9_SYNC_AUTONOMOUS,
        .power = VL53L9_POWER_LOW,
        .context = VL53L9_CONTEXT_LONG,
        .frame_period_us = FPS_TO_FRAME_PERIOD(30),
        .binning = 4,
        .exposure_ms = 4,
    },
    {
        .id = VL53L9_USECASE_AF,
        .sync = VL53L9_SYNC_AUTONOMOUS,
        .power = VL53L9_POWER_ULTRA_LOW,
        .context = VL53L9_CONTEXT_SHORT,
        .frame_period_us = FPS_TO_FRAME_PERIOD(30),
        .binning = 4,
        .exposure_ms = 5,
    },
};

int vl53l9_utils_set_profile(vl53l9_device_t *p_dev, vl53l9_profile_t *p_profile) {
    int ret;

    if ((p_dev == NULL) || (p_profile == NULL)) {
        return -1;
    }

    ret = vl53l9_set_sync_mode(p_dev, p_profile->sync);
    if (ret) {
        return ret;
    }

    ret = vl53l9_set_power_mode(p_dev, p_profile->power);
    if (ret) {
        return ret;
    }

    ret = vl53l9_set_frame_period(p_dev, p_profile->frame_period_us);
    if (ret) {
        return ret;
    }

    ret = vl53l9_set_context(p_dev, p_profile->context);
    if (ret) {
        return ret;
    }

    ret = vl53l9_set_binning(p_dev, p_profile->context, p_profile->binning);
    if (ret) {
        return ret;
    }

    ret = vl53l9_set_exposure(p_dev, p_profile->context, p_profile->exposure_ms);
    if (ret) {
        return ret;
    }

    return 0;
}

/**
 * @brief Get the raw resolution of a frame given the binning factor
 * @param[in] binning Binning factor
 * @note The resolution is expressed in pixels and doesn't take into account the cropping
 * @return The raw resolution or 0 if the binning factor is not supported
 */
static size_t _get_raw_resolution(uint8_t binning) {

    size_t resolution = 0;

    static const _resolution_lut_t lut[] = { { 2, 54, 42 }, { 4, 24, 24 }, { 6, 18, 14 },
                                             { 8, 12, 10 }, { 12, 8, 8 },  { 24, 4, 4 } };

    for (uint8_t i = 0; i < (sizeof(lut) / sizeof(lut[0])); i++) {
        if (lut[i]._binning == binning) {
            resolution = (size_t)lut[i]._width * (size_t)lut[i]._height;
            break;
        }
    }
    return resolution;
}

int vl53l9_utils_get_frame_resolution(uint8_t binning, uint8_t *p_width, uint8_t *p_height) {

    static const _resolution_lut_t lut[] = { { 2, 54, 42 }, { 4, 24, 20 }, { 6, 18, 14 },
                                             { 8, 12, 10 }, { 12, 8, 6 },  { 24, 4, 4 } };

    if ((p_width == NULL) || (p_height == NULL)) {
        return -1;
    }

    for (uint8_t i = 0; i < (sizeof(lut) / sizeof(lut[0])); i++) {
        if (lut[i]._binning == binning) {
            *p_width = lut[i]._width;
            *p_height = lut[i]._height;
            return 0;
        }
    }
    return -1;
}

int vl53l9_utils_get_csi_resolution(uint8_t binning, uint8_t *p_width, uint8_t *p_height) {

    static const _resolution_lut_t lut[] = { { 2, 100, 149 }, { 4, 100, 39 }, { 6, 100, 18 },
                                             { 8, 100, 9 },   { 12, 100, 6 }, { 24, 100, 3 } };

    if ((p_width == NULL) || (p_height == NULL)) {
        return -1;
    }

    for (uint8_t i = 0; i < (sizeof(lut) / sizeof(lut[0])); i++) {
        if (lut[i]._binning == binning) {
            *p_width = lut[i]._width;
            *p_height = lut[i]._height;
            return 0;
        }
    }
    return -1;
}

int vl53l9_utils_parse_frame(uint8_t *p_buffer, size_t buffer_size, vl53l9_frame_t *p_frame) {

    if ((p_buffer == NULL) || (p_frame == NULL)) {
        return -1;
    }

    p_frame->p_metadata = (vl53l9_meta_t *)&p_buffer[buffer_size - sizeof(vl53l9_meta_t)];

    size_t resolution = _get_raw_resolution((uint8_t)p_frame->p_metadata->binning);

    size_t dss_size = resolution / 2u;
    if (buffer_size < ((resolution * 3u * 2u) + dss_size)) {
        return -1;
    }

    // set pointers to the right location in the raw data buffer
    size_t offset = 0;
    p_frame->p_distance = (vl53l9_distance_t *)&p_buffer[offset];
    offset += (uint16_t)(resolution * 2u);

    p_frame->p_amplitude = (uint16_t *)&p_buffer[offset];
    offset += (uint16_t)(resolution * 2u);

    p_frame->p_ambient = (uint16_t *)&p_buffer[offset];
    offset += (uint16_t)(resolution * 2u);

    p_frame->p_dss_lut = &p_buffer[offset]; // NOTE: no check since dss is always enabled

    return 0;
}

/* dump frame form csi to avoid overwrite and drop csi padding */
int vl53l9_utils_dump_csi_frame(vl53l9_frame_t *p_frame, uint8_t *p_buffer, size_t buffer_size) {

    if ((p_buffer == NULL) || (p_frame == NULL)) {
        return -1;
    }

    size_t resolution = _get_raw_resolution((uint8_t)p_frame->p_metadata->binning);
    size_t frame_size =
        resolution * (sizeof(uint16_t) + sizeof(uint16_t) + sizeof(uint16_t) + 0.5) + sizeof(vl53l9_meta_t);

    if (frame_size > buffer_size) {
        return -1; /* output buffer too small */
    }

    void *ptr_cursor = p_buffer;

    memcpy(ptr_cursor, p_frame->p_distance, sizeof(uint16_t) * resolution);
    ptr_cursor += resolution * sizeof(uint16_t);
    memcpy(ptr_cursor, p_frame->p_amplitude, sizeof(uint16_t) * resolution);
    ptr_cursor += resolution * sizeof(uint16_t);
    memcpy(ptr_cursor, p_frame->p_ambient, sizeof(uint16_t) * resolution);
    ptr_cursor += resolution * sizeof(uint16_t);
    memcpy(ptr_cursor, p_frame->p_dss_lut, 0.5 * resolution);
    ptr_cursor += (uint32_t)(0.5 * resolution);
    memcpy(ptr_cursor, p_frame->p_metadata, sizeof(vl53l9_meta_t));
    ptr_cursor += sizeof(vl53l9_meta_t);

    if ((ptr_cursor - (void *)p_buffer) != frame_size) {
        return -1;
    }

    return 0;
}
