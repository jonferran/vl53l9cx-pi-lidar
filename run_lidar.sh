#!/bin/bash
# VL53L9CX live depth pipeline @ 100 fps.
# Usage: run_lidar.sh [binning] [exposure_ms]   (defaults: binning 2 -> 54x42 @ 100 fps)
#
# 100 fps recipe (datasheet "Gaming" profile): precision (SHORT context),
# REGULAR power, DSS disabled, 4 ms exposure, 10 ms frame period.
REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$REPO" || exit 1

BIN=${1:-2}
EXP=${2:-4}
if [ "$BIN" = 2 ]; then USECASE=1; HEIGHT=148; else USECASE=3; HEIGHT=38; fi
POWER=0; DSS=0; PERIOD=10000
PIPE=/tmp/tof_pipe

echo "========================================================="
echo "  VL53L9CX live depth  (binning $BIN, ${EXP}ms, 100 Hz)"
echo "========================================================="

# Build the app if needed.
[ -x ./vl53l9_bringup ] || make -s || exit 1
[ -p "$PIPE" ] || { rm -f "$PIPE"; mkfifo "$PIPE"; }

echo "[+] Clearing previous processes..."
sudo killall -9 python3 v4l2-ctl vl53l9_bringup 2>/dev/null
sleep 1
export DISPLAY=:0

echo "[+] Setting dummy subdev pad format to 100x${HEIGHT}..."
v4l2-ctl -d /dev/v4l-subdev0 --set-subdev-fmt pad=0,width=100,height=${HEIGHT},code=0x2001 >/dev/null 2>&1

echo "[+] Launching depth viewer..."
VL_BINNING=$BIN VL_ORIENT=0 VL_PIPE=$PIPE nohup python3 viewer/visualize.py >/tmp/viz.log 2>&1 &
sleep 2

echo "[+] Arming V4L2 receiver (RAW8/GREY 100x${HEIGHT} -> $PIPE)..."
nohup v4l2-ctl -d /dev/video0 --set-fmt-video=width=100,height=${HEIGHT},pixelformat=GREY --stream-mmap --stream-to=$PIPE >/tmp/v4l2.log 2>&1 &
sleep 1

echo "[+] Booting sensor + starting MIPI stream..."
sudo ./vl53l9_bringup $USECASE $EXP $PERIOD $POWER $DSS 2>&1 \
  | grep -E 'vl53l9_init|profile|exposure|power|dss|readback|STREAMING|TRANSMITTER|did not'

echo "[+] Live depth on the local monitor (DISPLAY=:0)."
echo "    Keys: r=rotate c=channel m=colormap v=3D f=front 3=3/4view z=fullscreen l=logcolor q=quit"
echo "    Stop everything: sudo killall -9 python3 v4l2-ctl"
