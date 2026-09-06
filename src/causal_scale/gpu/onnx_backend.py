def status() -> dict[str, str]:
    return {
        "status": "SKIPPED_RESOURCE_LIMIT",
        "reason": "optional ONNX mirror deferred until all P0 gates pass",
    }
