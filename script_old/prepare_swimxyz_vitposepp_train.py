from __future__ import annotations

# Thin CLI wrapper kept in the root on purpose: this is the entrypoint we use
# to launch the consolidated SwimXYZ -> VitPose++ training dataset pipeline.
# The actual logic lives in the standard module so the command line remains
# short and stable while the implementation can evolve behind it.
from prepare_swimxyz_vitposepp import main as _prepare_main


if __name__ == "__main__":
    _prepare_main()
    raise SystemExit
