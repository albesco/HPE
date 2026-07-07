#!/usr/bin/env python3
import argparse
import ctypes
import ctypes.util
from pathlib import Path

import torch


def load_cudart() -> ctypes.CDLL:
    candidates = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    candidates.extend(
        [
            "libcudart.so",
            "libcudart.so.11.0",
            str(Path(torch.__file__).resolve().parents[1] / "lib" / "libcudart.so"),
        ]
    )
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise RuntimeError("Could not load CUDA runtime library")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cudart = load_cudart()
    set_device = cudart.cudaSetDevice
    set_device.argtypes = [ctypes.c_int]
    set_device.restype = ctypes.c_int
    if set_device(args.device) != 0:
        raise RuntimeError(f"cudaSetDevice failed for device {args.device}")

    free_bytes = ctypes.c_size_t()
    total_bytes = ctypes.c_size_t()
    mem_get_info = cudart.cudaMemGetInfo
    mem_get_info.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    mem_get_info.restype = ctypes.c_int
    if mem_get_info(ctypes.byref(free_bytes), ctypes.byref(total_bytes)) != 0:
        raise RuntimeError("cudaMemGetInfo failed")

    print(f"{free_bytes.value / (1024**3):.2f}")


if __name__ == "__main__":
    main()
