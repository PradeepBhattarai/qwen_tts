from __future__ import annotations

import uvicorn

from qwen_tts_services.config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        ssl_certfile=settings.ssl_certfile,
        ssl_keyfile=settings.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
