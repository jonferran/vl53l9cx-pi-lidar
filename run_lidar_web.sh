#!/bin/bash
# VL53L9CX LiDAR Web Studio launcher.
# Boots the sensor (same bring-up as the particle viewer) but feeds the depth
# stream to web/lidar_web.py, which serves a live dashboard on the LAN.
# Open http://<pi-ip>:8080 from any device on the network.
#
# Alternative app to the particle viewer -- they both consume the sensor, so
# run one or the other. Pass a port as $1 to override 8080.
REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$REPO" || exit 1

BIN=2
EXP=4
USECASE=1; HEIGHT=148
POWER=0; DSS=0; PERIOD=10000
PIPE=/tmp/tof_pipe_web
PORT=${1:-8080}

echo "========================================================="
echo "  VL53L9CX LiDAR Web Studio  (binning $BIN, ${EXP}ms, 100 Hz)"
echo "========================================================="

[ -x ./vl53l9_bringup ] || make -s || exit 1
[ -p "$PIPE" ] || { rm -f "$PIPE"; mkfifo "$PIPE"; }

echo "[+] Clearing previous processes..."
sudo killall -9 python3 v4l2-ctl vl53l9_bringup 2>/dev/null
sleep 1

echo "[+] Setting dummy subdev pad format to 100x${HEIGHT}..."
v4l2-ctl -d /dev/v4l-subdev0 --set-subdev-fmt pad=0,width=100,height=${HEIGHT},code=0x2001 >/dev/null 2>&1

echo "[+] Launching web studio (port $PORT)..."
VL_BINNING=$BIN VL_PIPE=$PIPE VL_PORT=$PORT nohup python3 web/lidar_web.py >/tmp/lidar_web.log 2>&1 &
sleep 2

echo "[+] Arming V4L2 receiver (RAW8/GREY 100x${HEIGHT} -> $PIPE)..."
nohup v4l2-ctl -d /dev/video0 --set-fmt-video=width=100,height=${HEIGHT},pixelformat=GREY --stream-mmap --stream-to=$PIPE >/tmp/v4l2_web.log 2>&1 &
sleep 1

echo "[+] Booting sensor + starting MIPI stream..."
sudo ./vl53l9_bringup $USECASE $EXP $PERIOD $POWER $DSS 2>&1 \
  | grep -E 'vl53l9_init|profile|exposure|power|dss|readback|STREAMING|TRANSMITTER|did not'

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "========================================================="
echo "  Open the studio in a browser on any LAN device:"
echo "      http://${IP}:${PORT}"
echo "  Stop everything: sudo killall -9 python3 v4l2-ctl"
echo "========================================================="
