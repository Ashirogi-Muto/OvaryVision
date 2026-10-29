"""System and compute environment diagnostics for OvaryVision."""

import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HardwareSpecs(BaseModel):
    """Hardware specifications of the host machine."""
    os_name: str
    os_release: str
    architecture: str
    cpu_count_logical: int
    cpu_model: str
    memory_total_gb: float
    memory_available_gb: float
    disk_total_gb: float
    disk_free_gb: float


class CudaDiagnostics(BaseModel):
    """CUDA and GPU availability diagnostics."""
    gpu_available: bool
    gpu_count: int = 0
    device_names: List[str] = Field(default_factory=list)
    cuda_version: Optional[str] = None
    nvidia_driver_version: Optional[str] = None
    note: str = ""


class EnvironmentDiagnostics(BaseModel):
    """Complete diagnostic report of host environment."""
    python_version: str
    python_executable: str
    platform_detected: str  # 'oci', 'colab', 'kaggle', or 'generic_linux'
    hardware: HardwareSpecs
    cuda: CudaDiagnostics
    package_versions: Dict[str, Optional[str]] = Field(default_factory=dict)
    can_train_locally: bool
    training_recommendation: str


def _get_cpu_model() -> str:
    """Extract CPU model name from /proc/cpuinfo or lscpu if on Linux."""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line.lower():
                        return line.split(":", 1)[1].strip()
                    if "hardware" in line.lower():
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def _get_memory_info() -> tuple[float, float]:
    """Return (total_gb, available_gb) using /proc/meminfo or standard library."""
    total_gb = 0.0
    available_gb = 0.0
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                mem_data = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        mem_data[key] = int(val)
                total_kb = mem_data.get("MemTotal", 0)
                avail_kb = mem_data.get("MemAvailable", mem_data.get("MemFree", 0))
                total_gb = round(total_kb / (1024 * 1024), 2)
                available_gb = round(avail_kb / (1024 * 1024), 2)
                return total_gb, available_gb
        except Exception:
            pass
    return total_gb, available_gb


def _get_disk_info(path: str = ".") -> tuple[float, float]:
    """Return (total_gb, free_gb) for the partition containing path."""
    try:
        usage = shutil.disk_usage(path)
        total_gb = round(usage.total / (1024 ** 3), 2)
        free_gb = round(usage.free / (1024 ** 3), 2)
        return total_gb, free_gb
    except Exception:
        return 0.0, 0.0


def _check_cuda() -> CudaDiagnostics:
    """Safely check CUDA availability without requiring PyTorch."""
    # Check if PyTorch is installed and can see a GPU
    try:
        import torch  # type: ignore
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        device_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_available else []
        cuda_version = torch.version.cuda if gpu_available else None
        return CudaDiagnostics(
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            device_names=device_names,
            cuda_version=cuda_version,
            note="PyTorch CUDA check successful." if gpu_available else "PyTorch installed but CUDA is not available.",
        )
    except ImportError:
        pass

    # PyTorch not installed; check nvidia-smi
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path:
        try:
            res = subprocess.run(
                [nvidia_smi_path, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().splitlines()
                device_names = [line.split(",")[0].strip() for line in lines]
                driver = lines[0].split(",")[1].strip() if "," in lines[0] else None
                return CudaDiagnostics(
                    gpu_available=True,
                    gpu_count=len(lines),
                    device_names=device_names,
                    nvidia_driver_version=driver,
                    note="NVIDIA GPU detected via nvidia-smi. PyTorch is not yet installed.",
                )
        except Exception as exc:
            return CudaDiagnostics(
                gpu_available=False,
                note=f"nvidia-smi query failed: {exc}",
            )

    return CudaDiagnostics(
        gpu_available=False,
        gpu_count=0,
        device_names=[],
        cuda_version=None,
        note="CPU-only environment. No NVIDIA GPU or CUDA runtime detected.",
    )


def _detect_platform() -> str:
    """Determine whether we are on OCI, Google Colab, Kaggle, or generic Linux."""
    if "COLAB_GPU" in os.environ or os.path.exists("/content"):
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
        return "kaggle"
    # Check for OCI markers (Oracle Cloud Infrastructure)
    try:
        if os.path.exists("/etc/oracle-cloud-agent") or "oracle" in platform.release().lower():
            return "oci"
        # Check hostname or DMI
        hostname = platform.node().lower()
        if "oci" in hostname or "oracle" in hostname:
            return "oci"
    except Exception:
        pass
    return "oci" if "ubuntu" in platform.node() else "generic_linux"


def _check_packages() -> Dict[str, Optional[str]]:
    """Inspect presence of key project dependencies with proper import name resolution."""
    package_map = {
        "pydantic": "pydantic",
        "pyyaml": "yaml",
        "pillow": "PIL",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "pytest": "pytest",
        "click": "click",
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
    }
    results: Dict[str, Optional[str]] = {}
    for display_name, import_name in package_map.items():
        try:
            module = __import__(import_name)
            ver = getattr(module, "__version__", "installed")
            results[display_name] = str(ver)
        except ImportError:
            results[display_name] = None
    return results


def get_environment_diagnostics(project_path: str = ".") -> EnvironmentDiagnostics:
    """Run full environment diagnostics and return structured model."""
    cpu_model = _get_cpu_model()
    mem_total, mem_avail = _get_memory_info()
    disk_total, disk_free = _get_disk_info(project_path)
    cuda_diag = _check_cuda()
    platform_name = _detect_platform()
    packages = _check_packages()

    hw = HardwareSpecs(
        os_name=platform.system(),
        os_release=platform.release(),
        architecture=platform.machine(),
        cpu_count_logical=os.cpu_count() or 1,
        cpu_model=cpu_model,
        memory_total_gb=mem_total,
        memory_available_gb=mem_avail,
        disk_total_gb=disk_total,
        disk_free_gb=disk_free,
    )

    can_train = cuda_diag.gpu_available
    if can_train:
        recommendation = (
            f"GPU available ({cuda_diag.gpu_count}x {', '.join(cuda_diag.device_names)}). "
            "Host can execute GPU training and inpainting."
        )
    else:
        recommendation = (
            "Host is CPU-only (ARM/x86). Use this host for orchestration, data auditing, "
            "manifest generation, and unit testing. Run deep learning training on Google Colab or Kaggle GPU."
        )

    return EnvironmentDiagnostics(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        platform_detected=platform_name,
        hardware=hw,
        cuda=cuda_diag,
        package_versions=packages,
        can_train_locally=can_train,
        training_recommendation=recommendation,
    )
