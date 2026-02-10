from pyngrok import ngrok
import uvicorn
import os

# OPTIONAL: set token here if CLI command failed
# ngrok.set_auth_token("YOUR_TOKEN_HERE")

PORT = 8000

# Start ngrok tunnel
public_url = ngrok.connect(PORT, "http")
print(f"\n🚀 ngrok public URL: {public_url}\n")

# Start FastAPI
uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=PORT,
    reload=False
)
