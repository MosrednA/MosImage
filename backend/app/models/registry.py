from __future__ import annotations

from app.models.base import ImageModelProvider
from app.jobs import JobControl
from app.models.flux2_klein import Flux2KleinProvider
from app.models.flux2_klein_nvfp4 import Flux2Klein9bNvfp4Provider, Flux2KleinNvfp4Provider
from app.schemas import LoadModelRequest, ModelInfo


class ModelRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ImageModelProvider] = {
            "flux2-klein-9b-nvfp4": Flux2Klein9bNvfp4Provider(),
            "flux2-klein-4b-nvfp4": Flux2KleinNvfp4Provider(),
            "flux2-klein-4b": Flux2KleinProvider(),
        }

    def list(self) -> list[ModelInfo]:
        return [provider.info() for provider in self._providers.values()]

    def get(self, name: str) -> ImageModelProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers))
            raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    def unload(self, name: str) -> None:
        self.get(name).unload()

    def activate(self, name: str) -> ImageModelProvider:
        provider = self.get(name)
        for provider_name, other in self._providers.items():
            if provider_name != name and other.info().loaded:
                other.unload()
        return provider

    def preload(self, request: LoadModelRequest, control: JobControl) -> None:
        self.activate(request.model_name).preload(request, control)

    def unload_all(self) -> None:
        for provider in self._providers.values():
            provider.unload()


registry = ModelRegistry()
