"""Temporary AC-only Windows display/system request, released on exit."""
import ctypes
from ctypes import wintypes
import os
import threading


class TrainingWakeGuard:
    def __enter__(self):
        self.stop = threading.Event()
        self.errors = []
        if os.name == "nt":
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def _run(self):
        class PowerStatus(ctypes.Structure):
            _fields_ = [("ac", wintypes.BYTE), ("flag", wintypes.BYTE),
                        ("percent", wintypes.BYTE), ("reserved", wintypes.BYTE),
                        ("life", wintypes.DWORD), ("full_life", wintypes.DWORD)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        kernel.SetThreadExecutionState.restype = wintypes.DWORD
        kernel.GetSystemPowerStatus.argtypes = [ctypes.POINTER(PowerStatus)]
        kernel.GetSystemPowerStatus.restype = wintypes.BOOL
        previous = None
        try:
            while not self.stop.is_set():
                status = PowerStatus()
                known = bool(kernel.GetSystemPowerStatus(ctypes.byref(status)))
                on_ac = known and status.ac == 1
                if not kernel.SetThreadExecutionState(0x80000003 if on_ac else 0x80000000):
                    raise ctypes.WinError(ctypes.get_last_error())
                if previous != on_ac:
                    print(f"AC-only temporary display/system request active={on_ac}", flush=True)
                    previous = on_ac
                self.stop.wait(15)
        except Exception as error:
            self.errors.append(repr(error))
            print(f"Wake guard failed: {error!r}", flush=True)
        finally:
            kernel.SetThreadExecutionState(0x80000000)
            print("Temporary display/system request released.", flush=True)

    def __exit__(self, *args):
        self.stop.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=5)
