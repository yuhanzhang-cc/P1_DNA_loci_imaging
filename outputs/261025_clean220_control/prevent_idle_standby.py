"""Temporary AC-only display/system request, scoped to a training process.

An ES_SYSTEM_REQUIRED-only request did not prevent observed Modern Standby.
No power configuration or user-initiated sleep behavior is modified.
"""
import ctypes
from ctypes import wintypes
import sys
import time

import psutil

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.WaitForSingleObject.restype = wintypes.DWORD
kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
kernel.SetThreadExecutionState.restype = wintypes.DWORD
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL

pid = int(sys.argv[1])
handle = kernel.OpenProcess(0x00100000, False, pid)
if not handle:
    raise ctypes.WinError(ctypes.get_last_error())
try:
    active = None
    started = time.monotonic()
    while time.monotonic() - started < 12 * 3600:
        battery = psutil.sensors_battery()
        on_ac = battery is None or battery.power_plugged
        # CONTINUOUS | SYSTEM_REQUIRED | DISPLAY_REQUIRED on AC only.
        request = 0x80000003 if on_ac else 0x80000000
        if not kernel.SetThreadExecutionState(request):
            raise ctypes.WinError(ctypes.get_last_error())
        if active != on_ac:
            print(f"Training PID {pid}: temporary display/system request active={on_ac}; AC-only.", flush=True)
            active = on_ac
        result = kernel.WaitForSingleObject(handle, 15000)
        if result == 0:
            print("Training process exited.", flush=True)
            break
        if result != 258:
            raise ctypes.WinError(ctypes.get_last_error())
finally:
    kernel.SetThreadExecutionState(0x80000000)
    kernel.CloseHandle(handle)
    print("Temporary display/system request released.", flush=True)
