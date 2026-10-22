from .taps import ColdRates, FieldTap, never, openai_probe_body, ttft_inference


class LmStudioBackend:
    name = "lmstudio"
    version_path = "/api/v0/models"
    probe_path = "/v1/chat/completions"

    def __init__(self) -> None:
        self._rates = ColdRates()

    def version(self, payload: object) -> str | None:  # noqa: ARG002 - the models endpoint carries no version
        return None

    def tap(self, model: str) -> FieldTap:
        return FieldTap(never, ttft_inference(self._rates, model))

    def probe_body(self, model: str, messages: list[dict[str, str]]) -> dict[str, object]:
        return openai_probe_body(model, messages)
