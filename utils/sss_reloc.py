import importlib.util
import os


def _load_sss_reloc(sss_root: str = "3D-student-splatting-and-scooping"):
    src = os.path.join(sss_root, "utils", "reloc_utils.py")
    spec = importlib.util.spec_from_file_location("sss_ext_reloc", src)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot locate SSS reloc_utils.py at {src}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reloc = _load_sss_reloc()
compute_relocation_student_t_cuda = _reloc.compute_relocation_student_t_cuda
