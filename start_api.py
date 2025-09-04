#!/usr/bin/env python3
"""
Startup script for the Knowledge Graph RAG API

This script provides an easy way to start the FastAPI application
with different configurations and environment checks.
"""

import os
import sys
import argparse
from pathlib import Path
import uvicorn

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        "GOOGLE_API_KEY",
        "NEO4J_URI", 
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    print("✅ All required environment variables are set")
    return True


def main():
    parser = argparse.ArgumentParser(description="Start the Knowledge Graph RAG API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    
    args = parser.parse_args()
    
    print("🚀 Starting Knowledge Graph RAG API")
    print("=" * 50)
    
    print("\n📊 Starting API server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Reload: {args.reload}")
    print(f"   Log Level: {args.log_level}")
    
    # Run the API from root directory
    try:
        uvicorn.run(
            "core.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 