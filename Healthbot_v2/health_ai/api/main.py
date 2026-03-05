"""
main.py — entry point for the Health AI backend server.

Run with:
    python -m health_ai.api.main                 # default port 8000
    python -m health_ai.api.main --port 9000      # custom port
    FORCE_REINDEX_IMAGES=1 python -m health_ai.api.main  # re-OCR all images

The server binds to 0.0.0.0, so it is reachable from:
    - Your laptop:  http://localhost:<port>
    - Your phone (same WiFi): http://<laptop-LAN-IP>:<port>
The LAN IP is printed at startup.
"""
import argparse
import os
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Health AI Backend Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)),
                        help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload on code changes (dev only)")
    parser.add_argument("--force-reindex-images", action="store_true",
                        help="Re-OCR all prescription images on startup")
    args = parser.parse_args()

    if args.force_reindex_images:
        os.environ["FORCE_REINDEX_IMAGES"] = "1"

    os.environ["PORT"] = str(args.port)

    uvicorn.run(
        "health_ai.api.server:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
