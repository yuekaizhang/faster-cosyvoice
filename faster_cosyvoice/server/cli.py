"""Command-line entry point for the Faster CosyVoice HTTP server."""

import os

import uvicorn

from faster_cosyvoice.server.app import build_app
from faster_cosyvoice.server.arguments import (
    build_argument_parser,
    build_server_configs,
    validate_server_args,
)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, construct the application, and start Uvicorn."""
    args = build_argument_parser().parse_args(argv)
    validate_server_args(args)

    # Avoid an OpenMP crash after vLLM forks its EngineCore process.  setdefault
    # preserves an explicit deployment-level override.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    llm_config, token2wav_config, server_config = build_server_configs(args)
    app = build_app(llm_config, token2wav_config, server_config)
    uvicorn.run(app, host=server_config.host, port=server_config.port, log_level="info")


if __name__ == "__main__":
    main()
