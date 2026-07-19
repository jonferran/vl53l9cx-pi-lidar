#!/bin/bash
# VL53L9CX GPU-shader particle viewer @ 100 fps capture.
# Separate from run_lidar.sh so the CPU viewer stays untouched -- just runs
# viewer/visualize_gpu.py instead of viewer/visualize.py.
REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$REPO" || exit 1

BIN=2
EXP=${1:-4}
USECASE=1; HEIGHT=148
POWER=0; DSS=0; PERIOD=10000
PIPE=/tmp/tof_pipe_gpu

echo "========================================================="
echo "  VL53L9CX GPU particle viewer  (binning $BIN, ${EXP}ms, 100 Hz)"
echo "========================================================="

[ -x ./vl53l9_bringup ] || make -s || exit 1
[ -p "$PIPE" ] || { rm -f "$PIPE"; mkfifo "$PIPE"; }

echo "[+] Clearing previous processes..."
sudo killall -9 python3 v4l2-ctl vl53l9_bringup 2>/dev/null
sleep 1
export DISPLAY=:0

echo "[+] Setting dummy subdev pad format to 100x${HEIGHT}..."
v4l2-ctl -d /dev/v4l-subdev0 --set-subdev-fmt pad=0,width=100,height=${HEIGHT},code=0x2001 >/dev/null 2>&1

echo "[+] Launching GPU particle viewer..."
VL_BINNING=$BIN VL_PIPE=$PIPE nohup python3 viewer/visualize_gpu.py >/tmp/viz_gpu.log 2>&1 &
sleep 2

echo "[+] Arming V4L2 receiver (RAW8/GREY 100x${HEIGHT} -> $PIPE)..."
nohup v4l2-ctl -d /dev/video0 --set-fmt-video=width=100,height=${HEIGHT},pixelformat=GREY --stream-mmap --stream-to=$PIPE >/tmp/v4l2_gpu.log 2>&1 &
sleep 1

echo "[+] Booting sensor + starting MIPI stream..."
sudo ./vl53l9_bringup $USECASE $EXP $PERIOD $POWER $DSS 2>&1 \
  | grep -E 'vl53l9_init|profile|exposure|power|dss|readback|STREAMING|TRANSMITTER|did not'

echo "[+] GPU particle viewer on the local monitor (DISPLAY=:0)."
echo "    Mouse: left-drag orbits, scroll zooms.  Keys: r=orient f=reset-view q=quit"
echo "    Stop everything: sudo killall -9 python3 v4l2-ctl"
