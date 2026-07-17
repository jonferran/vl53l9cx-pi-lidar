from gpiozero import OutputDevice
from time import sleep
import sys

print("Initializing hardware reset sequence on GPIO 5 and 16...", flush=True)

# Define the common XSHUT / Enable GPIO pins
xshut5 = OutputDevice(5, active_high=True, initial_value=True)
xshut16 = OutputDevice(16, active_high=True, initial_value=True)

print("[-] Pulling shutdown pins LOW (Powering off sensor core)...")
xshut5.off()
xshut16.off()
sleep(1.5)  # Hold in reset to completely drain volatile registers

print("[+] Pulling pins HIGH (Booting sensor MCU with clean state)...")
xshut5.on()
xshut16.on()
sleep(0.5)

print("[+] Hardware reset complete!")
