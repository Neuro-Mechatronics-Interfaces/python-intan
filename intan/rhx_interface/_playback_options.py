def execute_jump_to_start(device, run_after_jump=False):
    """Jump to the beginning of the loaded playback file."""
    set_run_after_jump(device, run_after_jump)
    execute_jump_to_time(device, 0.0)


def execute_jump_to_time(device, seconds):
    """Jump to a specific time in seconds (float or int)."""
    device.send(f"execute JumpToTime {seconds}")
    if device.verbose:
        print(f"[PLAYBACK] Jumped to time: {seconds:.2f} sec")


def execute_jump_to_timestamp(device, timestamp_index):
    """Jump to a specific timestamp index (int)."""
    device.send(f"execute JumpToTimestamp {timestamp_index}")
    if device.verbose:
        print(f"[PLAYBACK] Jumped to timestamp index: {timestamp_index}")


def set_run_after_jump(device, enable=True):
    """Set whether playback automatically starts after jumping to a new position."""
    value = "true" if enable else "false"
    device.send(f"set RunAfterJumpToPosition {value}")
    if device.verbose:
        print(f"[PLAYBACK] Run after jump set to: {enable}")
