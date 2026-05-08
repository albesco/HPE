from __future__ import annotations

# Thin CLI wrapper kept in the root on purpose: this is the entrypoint we use
# to launch the consolidated SwimXYZ -> VitPose++ training dataset pipeline.
# The actual logic lives in the single-head module so the command line remains
# short and stable while the implementation can evolve behind it.
from prepare_swimxyz_vitposepp_single_head import main as _single_head_main


if __name__ == "__main__":
    _single_head_main()
    raise SystemExit
