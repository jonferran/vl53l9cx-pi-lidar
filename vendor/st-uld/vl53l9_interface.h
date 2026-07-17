/**
 ******************************************************************************
 * @file    vl53l9_interface.h
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

#ifndef VL53L9_INTERFACE_H
#define VL53L9_INTERFACE_H

#include "vl53l9.h"
#include <stdint.h>

/* exported symbols */

/* when updating use semantic versioning (https://semver.org/) */
#define INTERFACE_MAJOR (1U)
#define INTERFACE_MINOR (2U)
#define INTERFACE_PATCH (0U)

#define BOARD_NAME_STR_SIZE (20U)

/* exported types */
typedef struct {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
} _version_t;

typedef struct {
    _version_t interface;
    _version_t firmware;
    _version_t driver;
    char board_name[BOARD_NAME_STR_SIZE];
} platform_version_t;

typedef enum {
    PLATFORM_GPIO_STATE_SET = 0,
    PLATFORM_GPIO_STATE_RESET = 1,
    PLATFORM_GPIO_STATE_TOGGLE = 2
} platform_gpio_state_t;

typedef struct {
    uint16_t pin;
    void *port;
} platform_gpio_t;

typedef enum {
    PLATFORM_NONE_EVT = 0,
    PLATFORM_GPIO_IT_EVT = 1,
    PLATFORM_I3C_IBI_EVT = 2,
    PLATFORM_I3C_DMA_RX_EVT = 4,
    PLATFORM_I3C_DAA_EVT = 8,
    PLATFORM_CAM_PIPE_FRAME_EVT = 16,
} platform_event_t;

typedef enum {
    PLATFORM_BUS_I2C = 1,
    PLATFORM_BUS_I3C = 2,
    PLATFORM_BUS_CSI = 4,
} platform_bus_type_t;

typedef enum {
    PLATFORM_BUS_PROPERTY_NONE = 0,
    PLATFORM_BUS_PROPERTY_I3C_LEGACY = 1,
    PLATFORM_BUS_PROPERTY_I3C_IBI = 2,
} platform_bus_property_t;

typedef struct {
    void *bus;
    platform_bus_type_t bus_type;
    platform_bus_property_t bus_property;
    uint8_t address;
    vl53l9_vdda_t vdda;
    vl53l9_vddio_t vddio;
    uint32_t ext_clock;
    uint8_t instance_id;
    platform_gpio_t xshut;
    platform_gpio_t intr;
} vl53l9_device_t;

typedef struct {
    uint32_t x_size;
    uint32_t y_size;
    uint32_t x_margin;
    uint32_t y_margin;
    uint32_t scaler;
    uint32_t *frame_buffer;
} platform_display_layer_config_t;

/* definition of external variables */

extern platform_gpio_t g_debug_gpio_1;
extern platform_gpio_t g_debug_gpio_2;

/* exported functions */

/**
 * @brief Delay execution for a specified number of milliseconds
 * @param delay_ms Number of milliseconds to delay
 * @return 0 in case of success, negative value otherwise
 */
int platform_delay(uint32_t delay_ms);

/**
 * @brief Get firmware version
 * @param version Structure filled with version details
 * @return 0 in case of success, negative value otherwise
 */
int platform_get_version(platform_version_t *version);

/**
 * @brief Reset a device
 * @param[in] id Instance identifier of the device
 * @return 0 in case of success, negative value otherwise
 */
int platform_power_reset(uint8_t id);

/**
 * @brief Power-up a device
 * @param[in] id Instance identifier of the device
 * @return 0 in case of success, negative value otherwise
 */
int platform_power_enable(uint8_t id);

/**
 * @brief Power-down a device
 * @param[in] id Instance identifier of the device
 * @return 0 in case of success, negative value otherwise
 */
int platform_power_disable(uint8_t id);

/**
 * @brief Update the I2C static address stored in the device descriptor
 *
 * This method is meant to be called after requesting the device to change its I2C static address.
 * In order to finalize the change on the platform and ensure coherency, the address must be updated in the device
 * descriptor as well.
 *
 * @param[in] id Instance identifier of the device
 * @param[in] address New address to be used (7-bit format)
 * @return 0 in case of success, negative value otherwise
 */
int platform_set_device_address(uint8_t id, uint8_t address);

/**
 * @brief Assign a dynamic address to a single I3C device
 * @return 0 in case of success, negative value otherwise
 */
int platform_assign_dynamic_address(void);

/**
 * @brief Assign dynamic addresses to multiple I3C devices
 * @return 0 in case of success, negative value otherwise
 */
int platform_assign_dynamic_address_multisensor(void);

/**
 * @brief Enable the cycle counter used for profiling
 * @return 0 in case of success, negative value otherwise
 */
int platform_profiler_enable(void);

/**
 * @brief Disable the cycle counter used for profiling
 * @return 0 in case of success, negative value otherwise
 */
int platform_profiler_disable(void);

/**
 * @brief Get the current profiler timestamp
 * @return Current cycle counter value
 */
uint32_t platform_profiler_get_timestamp(void);

/**
 * @brief Convert a profiler timestamp into microseconds
 * @param[in] timestamp Timestamp in CPU cycles
 * @return Timestamp converted to microseconds
 */
uint32_t platform_profiler_convert_to_us(uint32_t timestamp);

/**
 * @brief Start the CSI capture pipe
 * @param[in] buff_csi Destination buffer for CSI frames
 * @return 0 in case of success, negative value otherwise
 */
int platform_start_csi_pipe(uint8_t *buff_csi);

/**
 * @brief Stop the CSI capture pipe
 * @return 0 in case of success, negative value otherwise
 */
int platform_stop_csi_pipe(void);

/**
 * @brief Enable a platform event source
 * @param[in] event Event to enable
 * @return 0 in case of success, negative value otherwise
 */
int platform_enable_event(platform_event_t event);

/**
 * @brief Disable a platform event source
 * @param[in] event Event to disable
 * @return 0 in case of success, negative value otherwise
 */
int platform_disable_event(platform_event_t event);

/**
 * @brief Acknowledge a platform event
 * @param[in] event Event to acknowledge
 * @return 0 in case of success, negative value otherwise
 */
int platform_acknowledge_event(platform_event_t event);

/**
 * @brief Wait until a platform event is raised or a timeout expires
 * @param[in] event Event to wait for
 * @param[in] timeout_ms Timeout in milliseconds
 * @return 0 in case of success, negative value otherwise
 */
int platform_wait_for_event(platform_event_t event, uint32_t timeout_ms);

/**
 * @brief Get the status of a platform event
 * @param[in] event Event to query
 * @param[out] active Set to true when the event is active
 * @return 0 in case of success, negative value otherwise
 */
int platform_get_event_status(platform_event_t event, bool *active);

/**
 * @brief Control the state of a GPIO
 * @param[in] gpio GPIO descriptor
 * @param[in] state Requested GPIO state
 * @return 0 in case of success, negative value otherwise
 */
int platform_ctrl_gpio(platform_gpio_t gpio, platform_gpio_state_t state);

/**
 * @brief Enable the display
 * @return 0 in case of success, negative value otherwise
 */
int platform_display_enable(void);

/**
 * @brief Disable the display
 * @return 0 in case of success, negative value otherwise
 */
int platform_display_disable(void);

/**
 * @brief Configure a display layer
 * @param[in] config Layer configuration
 * @param[in] layer_id Layer identifier
 * @return 0 in case of success, negative value otherwise
 */
int platform_display_configure_layer(platform_display_layer_config_t *config, uint8_t layer_id);

/**
 * @brief Set the color lookup table (LUT) for a display layer
 * @param[in] layer_id Layer identifier
 * @return 0 in case of success, negative value otherwise
 */
int platform_display_set_color_lut(uint8_t layer_id);

#endif /* VL53L9_INTERFACE_H */
