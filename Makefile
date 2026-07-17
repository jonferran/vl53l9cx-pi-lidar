# Build the VL53L9CX bring-up / streaming app.
APP     := vl53l9_bringup
VENDOR  := vendor/st-uld
CFLAGS  += -O2 -Wall -I$(VENDOR)
LDLIBS  += -lrt -lpthread -lm
SRC     := app/vl53l9_bringup.c \
           $(VENDOR)/vl53l9.c \
           $(VENDOR)/vl53l9_platform.c \
           $(VENDOR)/vl53l9_utils.c

$(APP): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDLIBS)

.PHONY: module clean
module:                 ## build the dummy_cam kernel module
	$(MAKE) -C driver
clean:
	rm -f $(APP)
	-$(MAKE) -C driver clean
