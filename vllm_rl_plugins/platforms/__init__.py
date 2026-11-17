"""Platform-specific code shared by all models (platforms/<name>/). current() names the platform vLLM runs on."""


def current() -> str:
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        return "rocm"
    if current_platform.is_cuda():
        return "cuda"
    return "other"
