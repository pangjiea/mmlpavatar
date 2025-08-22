import importlib.util
import os


def _load_sss_sghmc(sss_root: str = "3D-student-splatting-and-scooping"):
    """Load SSS's AdamSGHMC from its source file to avoid package name conflicts."""
    src = os.path.join(sss_root, "utils", "sghmc.py")
    spec = importlib.util.spec_from_file_location("sss_ext_sghmc", src)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot locate SSS sghmc.py at {src}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdamSGHMC


AdamSGHMC = _load_sss_sghmc()
