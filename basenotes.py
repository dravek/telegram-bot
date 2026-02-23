"""Basenotes API client (async wrapper around urllib)."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class BasenotesError(RuntimeError):
    """Base error for Basenotes API failures."""


class BasenotesAuthError(PermissionError):
    """Raised on 401/403 responses from Basenotes."""


@dataclass(frozen=True)
class BasenotesClient:
    base_url: str
    timeout: float = 10.0

    def _build_url(self, path: str, query: dict[str, Any] | None = None) -> str:
        base = self.base_url.rstrip("/")
        url = f"{base}{path}"
        if query:
            qs = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
            if qs:
                url = f"{url}?{qs}"
        return url

    async def list_notes(
        self, token: str, *, cursor: str | None = None, per_page: int | None = 20
    ) -> dict[str, Any]:
        return await self._request_json(
            "GET",
            "/api/v1/notes",
            token,
            query={"cursor": cursor, "per_page": per_page},
        )

    async def create_note(self, token: str, *, title: str, content: str) -> dict[str, Any]:
        body = {
            "title": title,
            "content": content,
            "body": content,
            "content_md": content,
        }
        return await self._request_json(
            "POST",
            "/api/v1/notes",
            token,
            body=body,
        )

    async def update_note(
        self,
        token: str,
        note_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        if content is not None:
            body["content"] = content
            body["body"] = content
            body["content_md"] = content
        return await self._request_json(
            "PATCH",
            f"/api/v1/notes/{urllib.parse.quote(note_id)}",
            token,
            body=body,
        )

    async def get_note(self, token: str, note_id: str) -> dict[str, Any]:
        return await self._request_json(
            "GET",
            f"/api/v1/notes/{urllib.parse.quote(note_id)}",
            token,
        )

    async def delete_note(self, token: str, note_id: str) -> dict[str, Any]:
        return await self._request_json(
            "DELETE",
            f"/api/v1/notes/{urllib.parse.quote(note_id)}",
            token,
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        token: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._build_url(path, query=query)
        return await asyncio.to_thread(
            self._request_json_sync, method, url, token, body
        )

    def _request_json_sync(
        self,
        method: str,
        url: str,
        token: str,
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        data = None
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw = exc.read().decode("utf-8") if exc.fp else ""
            detail = raw.strip() or f"HTTP {status}"
            if status in (401, 403):
                raise BasenotesAuthError(detail)
            raise BasenotesError(detail)
        except urllib.error.URLError as exc:
            raise BasenotesError(str(exc)) from exc
