/**
 ******************************************************************************
 * @file    vl53l9_utils.h
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

#ifndef VL53L9_UTILS_H
#define VL53L9_UTILS_H

#include <stddef.h>
#include <stdint.h>

#include "vl53l9_interface.h"

typedef enum {
    VL53L9_USECASE_AR_RANGE = 0U,
    VL53L9_USECASE_AR_PRECISION,
    VL53L9_USECASE_AF_RANGE,
    VL53L9_USECASE_AF,
    VL53L9_NB_USECASES,
} vl53l9_usecase_t;

typedef struct {
    uint32_t frame_counter;
    uint16_t temperature;
    uint16_t reserved_ldd[15];

    uint16_t ref_amplitude_ch1_long;
    uint16_t ref_distance_ch1_long;
    uint16_t ref_amplitude_ch2_long;
    uint16_t ref_distance_ch2_long;
    uint16_t ref_amplitude_ch1_short;
    uint16_t ref_distance_ch1_short;
    uint16_t ref_amplitude_ch2_short;
    uint16_t ref_distance_ch2_short;

    uint16_t frame_width;
    uint16_t frame_height;

    // static settings
    uint8_t sync_mode : 2;
    uint8_t power_mode : 2;
    uint8_t format : 2;
    uint8_t acquisition_mode : 2;

    uint8_t ambient_attenuation;

    // dynamic settings
    uint16_t reserved_dyn : 4;
    uint16_t dss_mode : 2;
    uint16_t binning : 5;
    uint16_t context : 1;
    uint16_t nb_step : 4;

    uint16_t error_code;
    uint8_t error_status;

    uint8_t reserved_ldd_error[5];
    uint32_t frame_period;

    uint32_t crop_x_size : 6;
    uint32_t crop_y_size : 6;
    uint32_t crop_x_offset : 6;
    uint32_t crop_y_offset : 6;
    uint32_t crop_enable : 1;

    uint8_t nb_shot_step1_lsb;
    uint8_t nb_shot_step1_mid;
    uint8_t nb_shot_step1_msb;

    uint8_t nb_shot_step4_5_lsb;
    uint8_t nb_shot_step4_5_mid;
    uint8_t nb_shot_step4_5_msb;

    uint8_t nb_shot_step6_lsb;
    uint8_t nb_shot_step6_mid;
    uint8_t nb_shot_step6_msb;

    uint8_t nb_shot_step7_lsb;
    uint8_t nb_shot_step7_mid;
    uint8_t nb_shot_step7_msb;

    uint32_t sest_reserved[3];

} vl53l9_meta_t;

typedef struct {
    uint16_t value : 15;
    uint16_t flag : 1;
} vl53l9_distance_t;

typedef struct {
    vl53l9_distance_t *p_distance;
    uint16_t *p_amplitude;
    uint16_t *p_ambient;
    uint8_t *p_dss_lut;
    vl53l9_meta_t *p_metadata;
} vl53l9_frame_t;

typedef struct {
    uint8_t id;
    vl53l9_sync_mode_t sync;
    vl53l9_power_mode_t power;
    vl53l9_context_t context;
    uint32_t frame_period_us;
    uint8_t binning;
    uint16_t exposure_ms;
} vl53l9_profile_t;

/**
 * @brief Set the ranging profile (sync mode, power mode, frame period, binning, etc.)
 * @param[in] p_dev Pointer to the device structure
 * @param[in] p_profile Pointer to the profile structure
 * @return 0 in case of success, -1 otherwise
 *
 * @note This function calls the vl53l9 driver, so the device must be in standby mode
 */
int vl53l9_utils_set_profile(vl53l9_device_t *p_dev, vl53l9_profile_t *p_profile);

/**
 * @brief Get the output frame resolution depending on the binning value
 * @param[in] binning The binning value
 * @param[out] p_width The width of the frame
 * @param[out] p_height The height of the frame
 * @return 0 in case of success, -1 otherwise
 *
 * @note The width and height values don't include the cropped area
 */
int vl53l9_utils_get_frame_resolution(uint8_t binning, uint8_t *p_width, uint8_t *p_height);

/**
 * @brief Get the CSI resolution depending on the binning value
 * @param[in] binning The binning value
 * @param[out] p_width The width of the CSI frame
 * @param[out] p_height The height of the CSI frame
 * @return 0 in case of success, -1 otherwise
 *
 * @note The width and height values include the cropped area
 */
int vl53l9_utils_get_csi_resolution(uint8_t binning, uint8_t *p_width, uint8_t *p_height);

/**
 * @brief Parse the raw data buffer and fill the frame structure
 * @param[in] p_buffer Pointer to the buffer containing the frame
 * @param[in] buffer_size Size of the buffer
 * @param[out] p_frame Pointer to the frame structure
 *
 * @return 0 in case of success, -1 otherwise
 */
int vl53l9_utils_parse_frame(uint8_t *p_buffer, size_t buffer_size, vl53l9_frame_t *p_frame);

/**
 * @brief Dump the CSI frame into a buffer and drop csi padding if any
 * @param[in] p_frame Pointer to the frame structure
 * @param[in] p_buffer Pointer to the buffer where the frame will be dumped
 * @param[in] buffer_size Size of the buffer
 *
 * @return 0 in case of success, -1 otherwise
 */
int vl53l9_utils_dump_csi_frame(vl53l9_frame_t *p_frame, uint8_t *p_buffer, size_t buffer_size);

extern vl53l9_profile_t g_ranging_profiles[VL53L9_NB_USECASES];

#endif // VL53L9_UTILS_H
