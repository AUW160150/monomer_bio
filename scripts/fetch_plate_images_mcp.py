#!/usr/bin/env python3
"""
Fetch observation images for all wells of a plate at the latest timestamp
using the monomer-cloud MCP.

This script:
  - Reads all documents (resources) the MCP provides
  - Uses all MCP tools that are relevant for discovery and image fetch
  - Downloads observation images for every well at the plate's last observation time

Usage:
  # When run by Cursor agent (recommended): the agent has MCP access and will
  # call the MCP tools; pass plate identifier as argument.
  python scripts/fetch_plate_images_mcp.py [--plate-id ID | --plate-name BARCODE] [--output-dir DIR]

  # Standalone with MCP server: set MONOMER_MCP_COMMAND to start the server, e.g.:
  #   MONOMER_MCP_COMMAND="python -m monomer_cloud_mcp" python scripts/fetch_plate_images_mcp.py --plate-name MY_PLATE
  # (Requires: pip install mcp requests)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Optional: MCP SDK for standalone mode (script spawns the MCP server)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAS_MCP_SDK = True
except ImportError:
    HAS_MCP_SDK = False

import requests

# --- Resource URIs from monomer-cloud MCP (read all docs) ---
MCP_RESOURCE_URIS = [
    "schema://cultures-and-plates/models",
    "guide://integration/monomer-desktop",
    "guide://cultures-and-plates/images",
    "guide://cultures-and-plates/concepts",
    "doc://cultures-and-plates/api-usage",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch observation images for all wells of a plate at latest timestamp via monomer-cloud MCP."
    )
    p.add_argument("--plate-id", type=str, help="Plate external ID or desktop_plate_id")
    p.add_argument("--plate-name", type=str, help="Plate barcode/name (used if --plate-id not set)")
    p.add_argument("--output-dir", type=str, default="plate_images", help="Directory to save images (default: plate_images)")
    p.add_argument("--skip-docs", action="store_true", help="Skip reading all MCP resources (faster run)")
    p.add_argument("--skip-other-tools", action="store_true", help="Skip calling extra MCP tools (only plates + images)")
    return p.parse_args()


# --- Standalone MCP client (when MONOMER_MCP_COMMAND is set) ---

async def _run_with_mcp_sdk(plate_id: str | None, plate_name: str | None, output_dir: str, skip_docs: bool, skip_other_tools: bool) -> int:
    """Use MCP Python SDK: spawn server from MONOMER_MCP_COMMAND and call tools."""
    cmd = os.environ.get("MONOMER_MCP_COMMAND", "").strip()
    if not cmd:
        print("MONOMER_MCP_COMMAND is not set. Set it to the command that starts the monomer-cloud MCP server.", file=sys.stderr)
        print("Example: MONOMER_MCP_COMMAND='python -m monomer_cloud_mcp'", file=sys.stderr)
        return 1
    parts = cmd.split()
    command = parts[0]
    args = parts[1:] if len(parts) > 1 else []
    server_params = StdioServerParameters(command=command, args=args)

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            if not skip_docs:
                await _read_all_resources(session)
            if not skip_other_tools:
                await _call_other_tools(session, plate_id, plate_name)
            saved = await _fetch_plate_images(session, plate_id, plate_name, output_dir)

    print(f"Saved {saved} image(s) to {output_dir}")
    return 0 if saved >= 0 else 1


async def _read_all_resources(session: "ClientSession") -> None:
    """Read all documents (resources) the MCP provides."""
    # List resources (tool)
    list_result = await session.call_tool("list_resources", {})
    _log_tool_result("list_resources", list_result)

    for uri in MCP_RESOURCE_URIS:
        result = await session.call_tool("read_resource", {"uri": uri})
        _log_tool_result(f"read_resource({uri})", result)
        if not _is_error(result) and result.get("content"):
            for block in result.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    print(f"[Doc {uri}] length={len(text)} chars")


def _get_content_list(tool_result: dict | object) -> list:
    """Normalize MCP call_tool result to list of content parts (dicts with type/text)."""
    if tool_result is None:
        return []
    if isinstance(tool_result, dict):
        content = tool_result.get("content") or []
    else:
        content = getattr(tool_result, "content", None) or []
    if not isinstance(content, list):
        return []
    out = []
    for c in content:
        if isinstance(c, dict):
            out.append(c)
        else:
            # ContentPart object with .type and .text
            out.append({"type": getattr(c, "type", ""), "text": getattr(c, "text", "")})
    return out


def _is_error(tool_result: dict | object) -> bool:
    if tool_result is None:
        return True
    for c in _get_content_list(tool_result):
        if c.get("type") == "text":
            text = (c.get("text") or "").strip()
            if text.startswith("Error") or "error" in text.lower():
                return True
    return False


def _log_tool_result(name: str, result: dict | object) -> None:
    """Optionally log tool result (avoid huge prints)."""
    if os.environ.get("MCP_DEBUG"):
        try:
            out = result if isinstance(result, dict) else {"content": _get_content_list(result)}
            print(f"[{name}] -> {json.dumps(out, default=str)[:500]}...")
        except Exception:
            print(f"[{name}] -> (result logged)")


async def _call_other_tools(
    session: "ClientSession",
    plate_id: str | None,
    plate_name: str | None,
) -> None:
    """Call other MCP tools: get_server_info, list_plates, list_culture_statuses, list_comments, etc."""
    # get_server_info
    r = await session.call_tool("get_server_info", {})
    _log_tool_result("get_server_info", r)

    # list_plates
    r = await session.call_tool("list_plates", {"limit": 5})
    _log_tool_result("list_plates", r)

    # export_plate_observations (if we have a plate)
    if plate_id or plate_name:
        r = await session.call_tool(
            "export_plate_observations",
            {"plate_id": plate_id, "plate_name": plate_name, "dataset_limit": 1},
        )
        _log_tool_result("export_plate_observations", r)

    # list_culture_statuses
    r = await session.call_tool("list_culture_statuses", {})
    _log_tool_result("list_culture_statuses", r)

    # list_comments (for plate if we have one)
    if plate_id:
        r = await session.call_tool("list_comments", {"entity_type": "plate", "entity_id": plate_id, "limit": 5})
        _log_tool_result("list_comments(plate)", r)


def _parse_tool_result(result: dict | object) -> dict | list | None:
    """Extract JSON from MCP call_tool result (FastMCP often wraps in result)."""
    if result is None or _is_error(result):
        return None
    for c in _get_content_list(result):
        if c.get("type") == "text":
            text = c.get("text") or ""
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
    return None


async def _fetch_plate_images(
    session: "ClientSession",
    plate_id: str | None,
    plate_name: str | None,
    output_dir: str,
) -> int:
    """
    Fetch images for all wells at the latest observation timestamp.
    Returns number of images saved, or -1 on error.
    """
    # 1) Resolve plate: get_plate_details
    plate_selector = []
    if plate_id:
        plate_selector.append({"by": "id", "value": plate_id})
    elif plate_name:
        plate_selector.append({"by": "name", "value": plate_name})
    else:
        # No plate given: use first plate from list_plates
        r = await session.call_tool("list_plates", {"limit": 1})
        data = _parse_tool_result(r)
        if not data or not data.get("items"):
            print("No plate specified and list_plates returned no plates. Use --plate-id or --plate-name.", file=sys.stderr)
            return -1
        first = data["items"][0]
        plate_selector = [{"by": "id", "value": first["id"]}]
        plate_id = first["id"]
        plate_name = first.get("barcode", "")

    r = await session.call_tool("get_plate_details", {"plate_queries": plate_selector})
    details = _parse_tool_result(r)
    if not details or "result" not in details:
        print("get_plate_details failed or plate not found.", file=sys.stderr)
        return -1
    plate_result = details["result"][0]
    if plate_result.get("error"):
        print(f"Plate error: {plate_result['error']}", file=sys.stderr)
        return -1
    plate = plate_result.get("plate")
    if not plate or not plate.get("cultures"):
        print("Plate has no cultures.", file=sys.stderr)
        return 0

    cultures = plate["cultures"]
    plate_id_resolved = plate["id"]
    plate_barcode = plate.get("barcode", plate_name or "")

    # 2) Get latest dataset (timestamp) via get_plate_observations
    r = await session.call_tool(
        "get_plate_observations",
        {"plate_id": plate_id_resolved, "dataset_limit": 1, "cursor": None},
    )
    obs_data = _parse_tool_result(r)
    if not obs_data or "result" not in obs_data:
        print("get_plate_observations failed.", file=sys.stderr)
        return -1
    result = obs_data["result"]
    if not result or "items" not in result:
        print("No observation data for plate.", file=sys.stderr)
        return 0
    first_item = result["items"][0]
    datasets = first_item.get("datasets") or []
    if not datasets:
        print("No datasets (timepoints) for plate.", file=sys.stderr)
        return 0
    latest_dataset = datasets[0]
    latest_dataset_id = latest_dataset.get("dataset_id")
    latest_timestamp = latest_dataset.get("timestamp", "")

    print(f"Plate: {plate_barcode} ({plate_id_resolved}), latest timestamp: {latest_timestamp}, dataset_id: {latest_dataset_id}")

    # 3) For each culture, get_observation_image_access and download
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    subdir = out / f"{plate_barcode or plate_id_resolved}_{latest_timestamp.replace(':', '-')}"
    subdir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for culture in cultures:
        cid = culture.get("id")
        well = culture.get("well", "unknown")
        if not cid:
            continue
        r = await session.call_tool(
            "get_observation_image_access",
            {"culture_id": cid, "dataset_id": latest_dataset_id},
        )
        acc = _parse_tool_result(r)
        if not acc:
            # Try without dataset_id (latest)
            r = await session.call_tool("get_observation_image_access", {"culture_id": cid})
            acc = _parse_tool_result(r)
        if not acc:
            continue
        urls = acc.get("download_urls") or {}
        url = urls.get("standard_url") or urls.get("large_url")
        if not url:
            continue
        path = subdir / f"{well}.jpg"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            saved += 1
            print(f"  Saved {well} -> {path}")
        except Exception as e:
            print(f"  Failed {well}: {e}", file=sys.stderr)

    return saved


# --- Agent-driven mode: load JSON from file (agent writes MCP results to file) ---

def run_agent_mode(args: argparse.Namespace) -> int:
    """
    When Cursor agent runs this script, it can pass a JSON file path with
    pre-fetched MCP data (e.g. plate details, observations, image access).
    Env: FETCH_PLATE_IMAGES_DATA=/path/to/data.json
    """
    data_path = os.environ.get("FETCH_PLATE_IMAGES_DATA")
    if not data_path or not Path(data_path).exists():
        return -2  # Not agent mode
    with open(data_path) as f:
        data = json.load(f)
    # Expected: { "plate": {...}, "cultures": [...], "latest_dataset_id": "...", "image_access": { "culture_id": { "download_urls": {...} } } }
    plate = data.get("plate") or {}
    cultures = data.get("cultures") or []
    latest_dataset_id = data.get("latest_dataset_id") or ""
    image_access = data.get("image_access") or {}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    subdir = out / f"{plate.get('barcode', 'plate')}_{latest_dataset_id}"
    subdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for c in cultures:
        cid = c.get("id")
        well = c.get("well", "unknown")
        acc = image_access.get(cid) or {}
        urls = acc.get("download_urls") or {}
        url = urls.get("standard_url") or urls.get("large_url")
        if not url:
            continue
        path = subdir / f"{well}.jpg"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            saved += 1
        except Exception:
            pass
    print(f"Saved {saved} images to {subdir}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.plate_id and not args.plate_name:
        # Try agent-mode: data file provided by agent
        code = run_agent_mode(args)
        if code != -2:
            return code

    if HAS_MCP_SDK and os.environ.get("MONOMER_MCP_COMMAND"):
        return asyncio.run(
            _run_with_mcp_sdk(
                args.plate_id,
                args.plate_name,
                args.output_dir,
                args.skip_docs,
                args.skip_other_tools,
            )
        )

    # No MCP SDK or no server command: print instructions for Cursor agent
    print(__doc__, file=sys.stderr)
    print("\nTo fetch images using Cursor's MCP:", file=sys.stderr)
    print("  1. Ensure monomer-cloud MCP is enabled in Cursor.", file=sys.stderr)
    print("  2. Ask the Cursor agent to run this workflow:", file=sys.stderr)
    print("     - list_resources, then read_resource for each URI in MCP_RESOURCE_URIs", file=sys.stderr)
    print("     - get_server_info, list_plates, get_plate_details, get_plate_observations", file=sys.stderr)
    print("     - For the chosen plate (--plate-id or --plate-name), get cultures and latest dataset_id", file=sys.stderr)
    print("     - For each culture: get_observation_image_access(culture_id, dataset_id), then download from download_urls.standard_url", file=sys.stderr)
    print("     - Save images to --output-dir.", file=sys.stderr)
    if not HAS_MCP_SDK:
        print("\nFor standalone mode: pip install mcp requests", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
