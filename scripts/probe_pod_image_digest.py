#!/usr/bin/env python3
"""Print only normalized OCI image digests visible to the current DLC pod."""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DIGEST = re.compile(r"sha256:[0-9a-fA-F]{64}")
SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")


def _normalized(value: str) -> str | None:
    match = DIGEST.search(value)
    return None if match is None else match.group(0).lower()


def _downward_api_candidates() -> list[dict[str, str]]:
    records = []
    for name in ("FASTWAM_POD_IMAGE_ID", "PAI_IMAGE_ID", "K8S_IMAGE_ID"):
        digest = _normalized(os.environ.get(name, ""))
        if digest is not None:
            records.append({"container": name, "digest": digest, "source": "environment"})
    return records


def _query_pod_status() -> list[dict[str, str]]:
    token_path = SERVICE_ACCOUNT / "token"
    namespace_path = SERVICE_ACCOUNT / "namespace"
    ca_path = SERVICE_ACCOUNT / "ca.crt"
    pod = os.environ.get("POD_NAME", os.environ.get("HOSTNAME", "")).strip()
    if not pod or not token_path.is_file() or not namespace_path.is_file():
        return []
    namespace = namespace_path.read_text(encoding="utf-8").strip()
    host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    url = (
        f"https://{host}:{port}/api/v1/namespaces/"
        f"{urllib.parse.quote(namespace, safe='')}/pods/"
        f"{urllib.parse.quote(pod, safe='')}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            # The token is used only in this request and is never printed.
            "Authorization": f"Bearer {token_path.read_text(encoding='utf-8').strip()}",
        },
    )
    context = ssl.create_default_context(cafile=str(ca_path) if ca_path.is_file() else None)
    try:
        with urllib.request.urlopen(request, context=context, timeout=10) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        # Do not print the response body because platforms may echo request
        # context. A status-only error is sufficient to diagnose RBAC denial.
        raise RuntimeError(f"Kubernetes pod-status query returned HTTP {error.code}") from None
    statuses = []
    status = payload.get("status", {}) if isinstance(payload, dict) else {}
    for family in ("initContainerStatuses", "containerStatuses", "ephemeralContainerStatuses"):
        entries = status.get(family, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            digest = _normalized(str(entry.get("imageID", "")))
            if digest is not None:
                statuses.append(
                    {
                        "container": str(entry.get("name", "unknown")),
                        "digest": digest,
                        "source": "pod_status.imageID",
                    }
                )
    return statuses


def main() -> None:
    records = _downward_api_candidates()
    if not records:
        records = _query_pod_status()
    unique = {
        (record["container"], record["digest"], record["source"]): record
        for record in records
    }
    normalized = [unique[key] for key in sorted(unique)]
    if not normalized:
        print(
            "Error: no OCI digest was exposed by environment or readable pod status",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(json.dumps(normalized, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
