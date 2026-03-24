"""Voice transcription provider using OpenAI-compatible Whisper API."""

import os
from pathlib import Path

import httpx
from loguru import logger


class TranscriptionProvider:
    """
    Voice transcription provider using any OpenAI-compatible Whisper API.

    Works with Groq, OpenRouter, OpenAI, or any endpoint that supports
    the /audio/transcriptions format.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        model: str = "whisper-large-v3",
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.api_url = api_url or "https://api.groq.com/openai/v1/audio/transcriptions"
        self.model = model

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Transcription API key not configured")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, self.model),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0,
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Transcription error ({}): {}", self.api_url, e)
            return ""


# Backward-compatible alias
GroqTranscriptionProvider = TranscriptionProvider
