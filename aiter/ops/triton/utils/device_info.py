import ctypes
import functools


@functools.lru_cache(maxsize=1)
def get_num_sms():
    # Returns the Compute Unit count of the device.
    #
    # Prefer chip_info.get_cu_num(): it honors the CU_NUM env override and is the
    # same value the tuning dispatch keys (gfx, cu_num, M, N, K) are built from,
    # so grid/segment sizing stays consistent with the selected tuned configs.
    # Fall back to torch's multi_processor_count when get_cu_num() is unavailable
    # (e.g. rocminfo missing/unparseable).
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return get_cu_num()
    except Exception:  # noqa: BLE001
        import torch

        current_device_index = torch.cuda.current_device()
        current_device = torch.cuda.get_device_properties(current_device_index)
        return current_device.multi_processor_count


_HIP_DEVICE_ATTRIBUTE_NUMBER_OF_XCCS = 10018
_HIP_ERROR_INVALID_VALUE = 1
_NUM_XCDS_COMPATIBILITY_FALLBACK = 8


def _query_num_xcds(device_id, libhip):
    value = ctypes.c_int(0)
    status = libhip.hipDeviceGetAttribute(
        ctypes.byref(value),
        _HIP_DEVICE_ATTRIBUTE_NUMBER_OF_XCCS,
        device_id,
    )
    if status == _HIP_ERROR_INVALID_VALUE:
        return _NUM_XCDS_COMPATIBILITY_FALLBACK
    if status != 0:
        raise RuntimeError(
            f"hipDeviceGetAttribute(NumberOfXccs) failed with error {status} "
            f"for device {device_id}"
        )
    if value.value <= 0:
        raise RuntimeError(
            f"hipDeviceGetAttribute(NumberOfXccs) returned {value.value} "
            f"for device {device_id}"
        )
    return value.value


@functools.lru_cache(maxsize=None)
def _get_num_xcds(device_id):
    libhip = ctypes.CDLL("libamdhip64.so")
    return _query_num_xcds(device_id, libhip)


def get_num_xcds(device_id=None):
    """Return the XCD count for a visible HIP device.

    ``device_id`` uses the same visible-device ordinal as torch and HIP. The
    compatibility fallback preserves the historical count when a runtime does
    not recognize ``hipDeviceAttributeNumberOfXccs``.
    """
    import torch

    if device_id is None:
        device_id = torch.cuda.current_device()
    elif (
        not isinstance(device_id, int)
        or isinstance(device_id, bool)
        or device_id < 0
        or device_id >= torch.cuda.device_count()
    ):
        raise ValueError(f"Invalid visible device ID: {device_id!r}")
    return _get_num_xcds(device_id)
