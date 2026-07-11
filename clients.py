import os
import time
import httpx

REQUEST_TIMEOUT = 25.0

class FireworksClient:
    def __init__(self):
        self.base_url = os.environ["FIREWORKS_BASE_URL"].rstrip("/")
        self.api_key = os.environ["FIREWORKS_API_KEY"]

    async def complete(self, model, system_prompt, user_prompt, max_tokens, reasoning_effort=None, temperature=0.0):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            if reasoning_effort is not None and 400 <= resp.status_code < 500:
                payload.pop("reasoning_effort")
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
            resp.raise_for_status()
            data = resp.json()
        latency_ms = int((time.perf_counter() - start) * 1000)

        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            content = (message.get("reasoning_content") or "").strip()
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return content, tokens, latency_ms
