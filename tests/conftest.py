import os

# two CPU devices, for the multi-device tests (must be set before jax initializes its backend)
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
