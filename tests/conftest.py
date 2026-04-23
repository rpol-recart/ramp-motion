import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when FORCE_GPU_TESTS env not set."""
    import os
    if os.environ.get("FORCE_GPU_TESTS") == "1":
        return
    skip_gpu = pytest.mark.skip(reason="GPU test; set FORCE_GPU_TESTS=1 to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
