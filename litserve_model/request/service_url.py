from config.read_config import get_config


def build_litserve_predict_url(service_name: str) -> str:
    endpoint = get_config(f"services.litserve.endpoints.{service_name}")
    if endpoint:
        return str(endpoint).rstrip("/")

    scheme = str(get_config("services.litserve.scheme", "http")).strip() or "http"
    public_host = get_config("services.litserve.public_host")
    host = public_host or get_config("services.litserve.host", "127.0.0.1")
    host = str(host).strip() or "127.0.0.1"

    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"

    port = int(get_config(f"services.litserve.port_{service_name}", 0) or 0)
    if port <= 0:
        defaults = {"sam3": 8022, "rmbg": 8023, "upscale": 8024}
        port = defaults.get(service_name, 8022)

    return f"{scheme}://{host}:{port}/predict"
