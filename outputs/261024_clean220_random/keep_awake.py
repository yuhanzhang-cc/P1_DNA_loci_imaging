"""Keep the system awake only while the specified training process exists.

No persistent power settings are changed; display sleep is still allowed.
"""

import ctypes
from ctypes import wintypes
import sys
import time

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
    if not kernel.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError(ctypes.get_last_error())
    print(f"Temporary system-awake request active for training PID {pid}.", flush=True)
    started = time.monotonic()
    while time.monotonic()-started < 12*3600:
        result = kernel.WaitForSingleObject(handle, 15000)
        if result == 0:
            print("Training process exited.", flush=True)
            break
        if result != 258:
            raise ctypes.WinError(ctypes.get_last_error())
finally:
    kernel.SetThreadExecutionState(0x80000000)
    kernel.CloseHandle(handle)
    print("Temporary system-awake request released.", flush=True)
