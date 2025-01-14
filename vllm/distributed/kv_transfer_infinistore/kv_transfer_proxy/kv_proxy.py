import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx

# Initialize the FastAPI app
app = FastAPI()

# Base URLs for the two vLLM processes (set to the root of the API)
VLLM_1_BASE_URL = "http://localhost:8000/v1"
VLLM_2_BASE_URL = "http://localhost:8001/v1"

# Initialize variables to hold the persistent clients
app.state.vllm1_client = None
app.state.vllm2_client = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize persistent HTTPX clients for vLLM services on startup.
    """
    app.state.vllm2_client = httpx.AsyncClient(timeout=None,
                                               base_url=VLLM_2_BASE_URL)
    app.state.vllm1_client = httpx.AsyncClient(timeout=None,
                                               base_url=VLLM_1_BASE_URL)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Close the persistent HTTPX clients on shutdown.
    """
    await app.state.vllm1_client.aclose()
    await app.state.vllm2_client.aclose()


async def send_request_to_vllm(client: httpx.AsyncClient, req_data: dict):
    """
    Send a request to a vLLM process using a persistent client.
    """
    response = await client.post("/completions",
                                 json=req_data)  # Correct endpoint path
    response.raise_for_status()
    return response


async def stream_vllm_response(client: httpx.AsyncClient, req_data: dict):
    """
    Asynchronously stream the response from a vLLM process using a persistent client.

    Args:
        client (httpx.AsyncClient): The persistent HTTPX client.
        req_data (dict): The JSON payload to send.

    Yields:
        bytes: Chunks of the response data.
    """
    async with client.stream(
            "POST", "/completions",
            json=req_data) as response:  # Correct endpoint path
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


@app.post("/v1/completions")
async def proxy_request(request: Request):
    """
    Proxy endpoint that forwards requests to two vLLM services.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        StreamingResponse: The streamed response from the second vLLM service.
    """
    req_data = await request.json()
    try:
        # Send request to prefill worker, ignore the response
        await send_request_to_vllm(app.state.vllm1_client, req_data)

        # Stream response from decode worker
        async def generate_stream():
            async for chunk in stream_vllm_response(app.state.vllm2_client, req_data):
                yield chunk

        return StreamingResponse(generate_stream(),
                                        media_type="application/json")
    except Exception as e:
        print(f"Error streaming response from vLLM-2: {e}")
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
