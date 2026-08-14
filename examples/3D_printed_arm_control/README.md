# 3D-printed arm control firmware

This directory is an optional, project-specific CircuitPython example for a servo-driven arm. It is not part of the `python-intan` desktop runtime.

Contents include:

- `code.py`: CircuitPython arm controller.
- `Pico-CircuitPython-8.0.4/*.uf2`: a pinned firmware image retained for reproducibility.
- `lib/`: the pinned CircuitPython libraries used by the firmware.
- `pico_tests/`: hardware diagnostics that require the relevant Pico and peripherals.

Copy the appropriate firmware and files to a compatible microcontroller following the official CircuitPython installation process. Review pin assignments and servo limits in `code.py` before powering hardware.

The controller expects `usbserialreader.py`, supplied in the bundled `lib/` directory. These files compile as Python source but rely on CircuitPython-only modules such as `board` and `busio`; they are not expected to import on CPython.

No trained gesture model, mechanical design, wiring diagram, or host-side gesture controller is bundled here. Treat this directory as a firmware reference, not a complete arm-control product.
