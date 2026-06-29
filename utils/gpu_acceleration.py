"""
GPU Acceleration for Quant Operations

Provides GPU acceleration for:
- Large covariance matrices
- Deep learning inference
- Monte Carlo simulations
- Transformers

Supported backends:
- CUDA (NVIDIA)
- JAX
- PyTorch
"""

from typing import Any

import numpy as np

from utils.logger import get_logger

logger = get_logger("gpu_acceleration")


class GPUBackend:
    """GPU backend type."""

    CUDA = "cuda"
    JAX = "jax"
    PYTORCH = "pytorch"
    NONE = "none"


class GPUAccelerator:
    """GPU accelerator for quant operations."""

    def __init__(self, backend: str = "auto"):
        """
        Initialize GPU accelerator.

        Args:
            backend: GPU backend ("auto", "cuda", "jax", "pytorch", "none")
        """
        self.backend = self._detect_backend(backend)
        self.device = None
        self.logger = logger

        if self.backend != GPUBackend.NONE:
            self._initialize_backend()

    def _detect_backend(self, backend: str) -> str:
        """
        Detect available GPU backend.

        Args:
            backend: Requested backend

        Returns:
            Detected backend
        """
        if backend == "none":
            return GPUBackend.NONE

        if backend == "auto":
            # Try CUDA first
            if self._check_cuda():
                return GPUBackend.CUDA
            # Try JAX
            if self._check_jax():
                return GPUBackend.JAX
            # Try PyTorch
            if self._check_pytorch():
                return GPUBackend.PYTORCH
            return GPUBackend.NONE

        return backend

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import cupy as cp

            _ = cp.cuda.Device(0).compute_capability
            return True
        except Exception:
            return False

    def _check_jax(self) -> bool:
        """Check if JAX with GPU is available."""
        try:
            import jax

            return jax.devices("gpu")[0] is not None
        except Exception:
            return False

    def _check_pytorch(self) -> bool:
        """Check if PyTorch with CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _initialize_backend(self):
        """Initialize selected backend."""
        if self.backend == GPUBackend.CUDA:
            try:
                import cupy as cp

                self.device = cp.cuda.Device(0)
                self.logger.info(f"GPU accelerator initialized: CUDA (Device: {self.device})")
            except Exception as e:
                self.logger.error(f"Failed to initialize CUDA: {e}")
                self.backend = GPUBackend.NONE

        elif self.backend == GPUBackend.JAX:
            try:
                import jax

                self.device = jax.devices("gpu")[0]
                self.logger.info(f"GPU accelerator initialized: JAX (Device: {self.device})")
            except Exception as e:
                self.logger.error(f"Failed to initialize JAX: {e}")
                self.backend = GPUBackend.NONE

        elif self.backend == GPUBackend.PYTORCH:
            try:
                import torch

                self.device = torch.device("cuda:0")
                self.logger.info(f"GPU accelerator initialized: PyTorch (Device: {self.device})")
            except Exception as e:
                self.logger.error(f"Failed to initialize PyTorch: {e}")
                self.backend = GPUBackend.NONE

    def to_gpu(self, array: np.ndarray) -> Any:
        """
        Transfer array to GPU.

        Args:
            array: NumPy array

        Returns:
            GPU array
        """
        if self.backend == GPUBackend.NONE:
            return array

        if self.backend == GPUBackend.CUDA:
            import cupy as cp

            return cp.asarray(array)

        elif self.backend == GPUBackend.JAX:
            import jax.numpy as jnp

            return jnp.array(array)

        elif self.backend == GPUBackend.PYTORCH:
            import torch

            return torch.from_numpy(array).to(self.device)

        return array

    def to_cpu(self, array: Any) -> np.ndarray:
        """
        Transfer array from GPU to CPU.

        Args:
            array: GPU array

        Returns:
            NumPy array
        """
        if self.backend == GPUBackend.NONE:
            return array

        if self.backend == GPUBackend.CUDA:
            import cupy as cp

            return cp.asnumpy(array)

        elif self.backend == GPUBackend.JAX:
            import numpy as np

            return np.array(array)

        elif self.backend == GPUBackend.PYTORCH:
            return array.cpu().numpy()

        return array

    def compute_covariance_matrix(self, returns: np.ndarray, method: str = "sample") -> np.ndarray:
        """
        Compute covariance matrix on GPU.

        Args:
            returns: Returns matrix (n_samples, n_features)
            method: Covariance method ("sample", "shrinkage")

        Returns:
            Covariance matrix
        """
        if self.backend == GPUBackend.NONE:
            # CPU fallback
            if method == "sample":
                return np.cov(returns, rowvar=False)
            else:
                # Simple shrinkage
                sample_cov = np.cov(returns, rowvar=False)
                n = returns.shape[0]
                shrinkage = 1.0 / (n + 1)
                identity = np.eye(sample_cov.shape[0])
                return (1 - shrinkage) * sample_cov + shrinkage * identity * np.trace(sample_cov)

        # GPU computation
        gpu_returns = self.to_gpu(returns)

        if self.backend == GPUBackend.CUDA:
            import cupy as cp

            if method == "sample":
                gpu_cov = cp.cov(gpu_returns, rowvar=False)
            else:
                gpu_cov = cp.cov(gpu_returns, rowvar=False)
                n = gpu_returns.shape[0]
                shrinkage = 1.0 / (n + 1)
                identity = cp.eye(gpu_cov.shape[0])
                gpu_cov = (1 - shrinkage) * gpu_cov + shrinkage * identity * cp.trace(gpu_cov)
            return self.to_cpu(gpu_cov)

        elif self.backend == GPUBackend.JAX:
            import jax.numpy as jnp

            if method == "sample":
                gpu_cov = jnp.cov(gpu_returns, rowvar=False)
            else:
                gpu_cov = jnp.cov(gpu_returns, rowvar=False)
                n = gpu_returns.shape[0]
                shrinkage = 1.0 / (n + 1)
                identity = jnp.eye(gpu_cov.shape[0])
                gpu_cov = (1 - shrinkage) * gpu_cov + shrinkage * identity * jnp.trace(gpu_cov)
            return self.to_cpu(gpu_cov)

        elif self.backend == GPUBackend.PYTORCH:
            import torch

            if method == "sample":
                gpu_cov = torch.cov(gpu_returns.T)
            else:
                gpu_cov = torch.cov(gpu_returns.T)
                n = gpu_returns.shape[0]
                shrinkage = 1.0 / (n + 1)
                identity = torch.eye(gpu_cov.shape[0]).to(self.device)
                gpu_cov = (1 - shrinkage) * gpu_cov + shrinkage * identity * torch.trace(gpu_cov)
            return self.to_cpu(gpu_cov)

        return np.cov(returns, rowvar=False)

    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication on GPU.

        Args:
            a: First matrix
            b: Second matrix

        Returns:
            Result matrix
        """
        if self.backend == GPUBackend.NONE:
            return np.dot(a, b)

        gpu_a = self.to_gpu(a)
        gpu_b = self.to_gpu(b)

        if self.backend == GPUBackend.CUDA:
            import cupy as cp

            gpu_result = cp.dot(gpu_a, gpu_b)
            return self.to_cpu(gpu_result)

        elif self.backend == GPUBackend.JAX:
            import jax.numpy as jnp

            gpu_result = jnp.dot(gpu_a, gpu_b)
            return self.to_cpu(gpu_result)

        elif self.backend == GPUBackend.PYTORCH:
            import torch

            gpu_result = torch.mm(gpu_a, gpu_b)
            return self.to_cpu(gpu_result)

        return np.dot(a, b)

    def monte_carlo_simulation(
        self,
        n_simulations: int,
        n_steps: int,
        initial_price: float,
        drift: float,
        volatility: float,
        dt: float = 1.0 / 252.0,
    ) -> np.ndarray:
        """
        Monte Carlo simulation on GPU.

        Args:
            n_simulations: Number of simulations
            n_steps: Number of time steps
            initial_price: Initial price
            drift: Drift rate
            volatility: Volatility
            dt: Time step

        Returns:
            Simulation results (n_simulations, n_steps)
        """
        if self.backend == GPUBackend.NONE:
            # CPU fallback
            np.random.seed(42)
            shocks = np.random.normal(0, 1, (n_simulations, n_steps))
            paths = initial_price * np.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
            )
            return paths

        # GPU computation
        if self.backend == GPUBackend.CUDA:
            import cupy as cp

            cp.random.seed(42)
            gpu_shocks = cp.random.normal(0, 1, (n_simulations, n_steps))
            gpu_paths = initial_price * cp.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * cp.sqrt(dt) * gpu_shocks
            )
            return self.to_cpu(gpu_paths)

        elif self.backend == GPUBackend.JAX:
            import jax
            import jax.numpy as jnp

            key = jax.random.PRNGKey(42)
            gpu_shocks = jax.random.normal(key, (n_simulations, n_steps))
            gpu_paths = initial_price * jnp.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * jnp.sqrt(dt) * gpu_shocks
            )
            return self.to_cpu(gpu_paths)

        elif self.backend == GPUBackend.PYTORCH:
            import torch

            torch.manual_seed(42)
            gpu_shocks = torch.randn(n_simulations, n_steps).to(self.device)
            gpu_paths = initial_price * torch.exp(
                (drift - 0.5 * volatility**2) * dt
                + volatility * torch.sqrt(torch.tensor(dt)) * gpu_shocks
            )
            return self.to_cpu(gpu_paths)

        # CPU fallback
        np.random.seed(42)
        shocks = np.random.normal(0, 1, (n_simulations, n_steps))
        paths = initial_price * np.exp(
            (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
        )
        return paths

    def get_gpu_info(self) -> dict[str, Any]:
        """
        Get GPU information.

        Returns:
            GPU information dictionary
        """
        info = {"backend": self.backend, "available": self.backend != GPUBackend.NONE}

        if self.backend == GPUBackend.CUDA:
            try:
                import cupy as cp

                info["device_name"] = cp.cuda.Device(0).name
                info["compute_capability"] = cp.cuda.Device(0).compute_capability
                info["memory_total"] = cp.cuda.Device(0).mem_info[1]
                info["memory_free"] = cp.cuda.Device(0).mem_info[0]
            except Exception:
                pass

        elif self.backend == GPUBackend.JAX:
            try:
                import jax

                info["devices"] = [str(d) for d in jax.devices("gpu")]
            except Exception:
                pass

        elif self.backend == GPUBackend.PYTORCH:
            try:
                import torch

                info["device_name"] = torch.cuda.get_device_name(0)
                info["device_count"] = torch.cuda.device_count()
                info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
            except Exception:
                pass

        return info


# Global GPU accelerator instance
_gpu_accelerator: GPUAccelerator | None = None


def get_gpu_accelerator(backend: str = "auto") -> GPUAccelerator:
    """
    Get global GPU accelerator instance.

    Args:
        backend: GPU backend

    Returns:
        GPUAccelerator instance
    """
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator(backend)
    return _gpu_accelerator


def enable_gpu_acceleration(backend: str = "auto"):
    """
    Enable GPU acceleration globally.

    Args:
        backend: GPU backend
    """
    global _gpu_accelerator
    _gpu_accelerator = GPUAccelerator(backend)


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    accelerator = get_gpu_accelerator()
    return accelerator.backend != GPUBackend.NONE
