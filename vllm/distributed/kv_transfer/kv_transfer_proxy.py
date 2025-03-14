import asyncio
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

# Initialize the FastAPI app
app = FastAPI()

# Single prefill vLLM worker
PREFILL_BASE_URL = "http://localhost:7080/v1"

# Single decode vLLM worker
DECODE_BASE_URL = "http://localhost:7090/v1"

# Persistent HTTP clients
app.state.prefill_client = None
app.state.decode_client = None

# Store request_id → (request_data, sender_info)
app.state.request_store = {}


@app.on_event("startup")
async def startup_event():
    """Initialize persistent HTTPX clients."""
    app.state.prefill_client = httpx.AsyncClient(timeout=None, base_url=PREFILL_BASE_URL)
    app.state.decode_client = httpx.AsyncClient(timeout=None, base_url=DECODE_BASE_URL)


@app.on_event("shutdown")
async def shutdown_event():
    """Close the persistent HTTPX clients on shutdown."""
    await app.state.prefill_client.aclose()
    await app.state.decode_client.aclose()


async def send_request_to_vllm(client: httpx.AsyncClient, req_data: dict):
    """Send a request to a vLLM process using a persistent client."""
    response = await client.post("/chat/completions", json=req_data)
    response.raise_for_status()
    return response


async def stream_vllm_response(client: httpx.AsyncClient, req_data: dict):
    """Stream the response from a vLLM process using a persistent client."""
    async with client.stream("POST", "/chat/completions", json=req_data) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


@app.post("/v1/chat/completions")
async def proxy_request(request: Request):
    """Handle initial request and send it to the prefill vLLM worker."""
    req_data = await request.json()

    # Generate a unique conduit UUID
    request_id = str(uuid.uuid4())

    # Extract sender information (IP address, headers, etc.)
    sender_info = {"ip": request.client.host, "headers": dict(request.headers)}

    # Modify request to include request_id
    req_data["request_id"] = request_id

    try:
        # Send request to the prefill worker
        await send_request_to_vllm(app.state.prefill_client, req_data)

        # Store request and sender info
        app.state.request_store[request_id] = (req_data, sender_info)

        # Acknowledge the request was received and being processed
        return JSONResponse({"message": "Request sent to prefill worker", "request_id": request_id})

    except Exception as e:
        print(f"Error sending request to prefill vLLM: {e}")
        raise HTTPException(status_code=500, detail="Failed to process request")


@app.post("/v1/kv_cache_ready")
async def kv_cache_ready(request: Request):
    """Handle notification that a vLLM has completed KV cache loading."""
    req_data = await request.json()

    # Extract request_id from the incoming request
    request_id = req_data.get("request_id")
    
    if not request_id:
        return JSONResponse(status_code=400, content={"error": "Missing request_id"})

    # Retrieve the original request and sender information
    if request_id not in app.state.request_store:
        return JSONResponse(status_code=404, content={"error": "Request not found for this request_id"})

    original_request, sender_info = app.state.request_store[request_id]  # Don't remove yet

    try:
        # Stream response from the decode worker
        async def generate_stream():
            try:
                async for chunk in stream_vllm_response(app.state.decode_client, original_request):
                    yield chunk
            except Exception as e:
                print(f"Error streaming decode response: {e}")
                # Notify the original user of the internal error
                yield b'{"error": "Internal Server Error"}'
            finally:
                # Once response is fully sent, remove entry from request_store
                app.state.request_store.pop(request_id, None)

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        print(f"Error processing decode response: {e}")
        # Ensure the original user gets an Internal Server Error
        app.state.request_store.pop(request_id, None)  # Cleanup request entry
        return JSONResponse(status_code=500, content={"error": "Failed to process decode response"})
