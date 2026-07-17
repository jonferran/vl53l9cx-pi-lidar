/**
 * ============================================================================
 * @file        vl53l9_platform.c
 * @brief       Silent Linux User-Space I2C Bridge for Production Streaming
 * ============================================================================
 */

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <stdint.h>
#include <string.h>

#include "vl53l9_interface.h"

#define I2C_CHUNK_SIZE 256 

int vl53l9_write(void *const p_dev, uint16_t index, uint8_t *p_data, uint32_t count) {
    vl53l9_device_t *dev = (vl53l9_device_t *)p_dev;
    int fd = (int)(intptr_t)dev->bus;
    
    uint32_t position = 0;
    while (position < count) {
        uint32_t data_size = (count - position > I2C_CHUNK_SIZE) ? I2C_CHUNK_SIZE : (count - position);
        uint16_t current_addr = index + position;
        
        uint8_t buffer[data_size + 2];
        buffer[0] = (current_addr >> 8) & 0xFF; 
        buffer[1] = current_addr & 0xFF;        
        memcpy(&buffer[2], &p_data[position], data_size);
        
        if (write(fd, buffer, data_size + 2) != (data_size + 2)) return -1; 
        position += data_size;
    }
    return 0;
}

int vl53l9_read(void *const p_dev, uint16_t index, uint8_t *p_data, uint32_t count) {
    vl53l9_device_t *dev = (vl53l9_device_t *)p_dev;
    int fd = (int)(intptr_t)dev->bus;
    uint8_t linux_addr = dev->address >> 1;
    
    uint32_t position = 0;
    while (position < count) {
        uint32_t data_size = (count - position > I2C_CHUNK_SIZE) ? I2C_CHUNK_SIZE : (count - position);
        uint16_t current_addr = index + position;
        
        struct i2c_msg msgs[2];
        struct i2c_rdwr_ioctl_data msgset[1];
        
        uint8_t reg_addr[2] = {(current_addr >> 8) & 0xFF, current_addr & 0xFF};
        
        msgs[0].addr = linux_addr;
        msgs[0].flags = 0; 
        msgs[0].len = 2;
        msgs[0].buf = reg_addr;
        
        msgs[1].addr = linux_addr;
        msgs[1].flags = I2C_M_RD; 
        msgs[1].len = data_size;
        msgs[1].buf = &p_data[position];
        
        msgset[0].msgs = msgs;
        msgset[0].nmsgs = 2;
        
        if (ioctl(fd, I2C_RDWR, &msgset) < 0) return -1; 
        position += data_size;
    }
    return 0;
}

int vl53l9_read_async(void *const p_dev, uint16_t index, volatile uint8_t *p_data, uint32_t count) { 
    return vl53l9_read(p_dev, index, (uint8_t *)p_data, count); 
}

int vl53l9_write8(void *const p_dev, uint16_t index, uint8_t data) { 
    return vl53l9_write(p_dev, index, &data, 1); 
}

int vl53l9_read8(void *const p_dev, uint16_t index, uint8_t *p_data) { 
    return vl53l9_read(p_dev, index, p_data, 1); 
}

int vl53l9_write16(void *const p_dev, uint16_t index, uint16_t data) {
    uint8_t buf[2];
    buf[0] = data & 0xFF;
    buf[1] = (data >> 8) & 0xFF;
    return vl53l9_write(p_dev, index, buf, 2);
}

int vl53l9_read16(void *const p_dev, uint16_t index, uint16_t *p_data) {
    uint8_t buf[2];
    int status = vl53l9_read(p_dev, index, buf, 2);
    if (status == 0) *p_data = (uint16_t)(buf[0] | (buf[1] << 8));
    return status;
}

int vl53l9_write32(void *const p_dev, uint16_t index, uint32_t data) {
    uint8_t buf[4];
    buf[0] = data & 0xFF;
    buf[1] = (data >> 8) & 0xFF;
    buf[2] = (data >> 16) & 0xFF;
    buf[3] = (data >> 24) & 0xFF;
    return vl53l9_write(p_dev, index, buf, 4);
}

int vl53l9_read32(void *const p_dev, uint16_t index, uint32_t *p_data) {
    uint8_t buf[4];
    int status = vl53l9_read(p_dev, index, buf, 4);
    if (status == 0) *p_data = (uint32_t)(buf[0] | (buf[1] << 8) | (buf[2] << 16) | (buf[3] << 24));
    return status;
}

int vl53l9_get_config_ext_clock(void *const p_dev, uint32_t *p_ext_clk) { *p_ext_clk = ((vl53l9_device_t*)p_dev)->ext_clock; return 0; }
int vl53l9_get_config_vddio(void *const p_dev, vl53l9_vddio_t *p_vddio) { *p_vddio = ((vl53l9_device_t*)p_dev)->vddio; return 0; }
int vl53l9_get_config_vdda(void *const p_dev, vl53l9_vdda_t *p_vdda) { *p_vdda = ((vl53l9_device_t*)p_dev)->vdda; return 0; }
int vl53l9_wait_ms(void *const p_dev, uint32_t delay_ms) { usleep(delay_ms * 1000); return 0; }