#!/usr/bin/env python3
"""
Download observation images for all wells of a plate using monomer-cloud MCP tools.

This script uses the MCP Python SDK to:
1. Connect to the monomer-cloud MCP server
2. Get plate details and cultures
3. Get observation data for the latest timestamp
4. Download images for each well

Usage:
    python scripts/download_plate_images.py --plate-name PLATE_BARCODE [--output-dir DIR]
    python scripts/download_plate_images.py --plate-id PLATE_ID [--output-dir DIR]

Requirements:
    pip install mcp requests
    
Environment:
    MONOMER_MCP_URL: HTTP URL for the MCP server (default: https://backend-staging.monomerbio.com/mcp)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except ImportError:
    print("Error: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests package not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)


async def fetch_plate_images(plate_id: str | None, plate_name: str | None, output_dir: str) -> int:
    """Fetch and download plate images using MCP tools."""
    
    # Connect to MCP server
    mcp_url = "https://backend-staging.monomerbio.com/mcp"
    
    async with sse_client(mcp_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            print(f"Connected to MCP server: {mcp_url}")
            
            # Step 1: Get plate details
            plate_queries = []
            if plate_id:
                plate_queries.append({"by": "id", "value": plate_id})
            elif plate_name:
                plate_queries.append({"by": "name", "value": plate_name})
            else:
                print("Error: Must provide --plate-id or --plate-name", file=sys.stderr)
                return 1
            
            print(f"Fetching plate details...")
            result = await session.call_tool("get_plate_details", {"plate_queries": plate_queries})
            
            # Parse result
            plate_data = _parse_mcp_result(result)
            if not plate_data or "result" not in plate_data:
                print("Error: Failed to get plate details", file=sys.stderr)
                return 1
            
            plate_result = plate_data["result"][0]
            if plate_result.get("error"):
                print(f"Error: {plate_result['error']}", file=sys.stderr)
                return 1
            
            plate = plate_result.get("plate")
            if not plate:
                print("Error: Plate not found", file=sys.stderr)
                return 1
            
            plate_id_resolved = plate["id"]
            plate_barcode = plate.get("barcode", plate_name or plate_id or "unknown")
            cultures = plate.get("cultures", [])
            
            print(f"Plate: {plate_barcode} (ID: {plate_id_resolved})")
            print(f"Found {len(cultures)} cultures")
            
            if not cultures:
                print("No cultures found on plate")
                return 0
            
            # Step 2: Get latest observation dataset
            print("Fetching observation data...")
            result = await session.call_tool(
                "get_plate_observations",
                {
                    "plate_id": plate_id_resolved,
                    "dataset_limit": 1,
                    "cursor": None
                }
            )
            
            obs_data = _parse_mcp_result(result)
            if not obs_data or "result" not in obs_data:
                print("Error: Failed to get observations", file=sys.stderr)
                return 1
            
            obs_result = obs_data["result"]
            if not obs_result.get("items"):
                print("No observation data found")
                return 0
            
            datasets = obs_result["items"][0].get("datasets", [])
            if not datasets:
                print("No datasets found")
                return 0
            
            latest_dataset = datasets[0]
            dataset_id = latest_dataset.get("dataset_id")
            timestamp = latest_dataset.get("timestamp", "")
            
            print(f"Latest observation: {timestamp} (dataset: {dataset_id})")
            
            # Step 3: Create output directory
            output_path = Path(output_dir)
            safe_timestamp = timestamp.replace(":", "-").replace(" ", "_")
            plate_dir = output_path / f"{plate_barcode}_{safe_timestamp}"
            plate_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving images to: {plate_dir}")
            
            # Step 4: Download images for each culture
            saved_count = 0
            for culture in cultures:
                culture_id = culture.get("id")
                well = culture.get("well", "unknown")
                
                if not culture_id:
                    continue
                
                # Get image access URL
                result = await session.call_tool(
                    "get_observation_image_access",
                    {
                        "culture_id": culture_id,
                        "dataset_id": dataset_id
                    }
                )
                
                access_data = _parse_mcp_result(result)
                if not access_data:
                    print(f"  {well}: No image access data")
                    continue
                
                download_urls = access_data.get("download_urls", {})
                image_url = download_urls.get("standard_url") or download_urls.get("large_url")
                
                if not image_url:
                    print(f"  {well}: No download URL")
                    continue
                
                # Download image
                try:
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()
                    
                    image_path = plate_dir / f"{well}.jpg"
                    image_path.write_bytes(response.content)
                    
                    saved_count += 1
                    print(f"  {well}: ✓")
                    
                except Exception as e:
                    print(f"  {well}: Failed - {e}")
            
            print(f"\nDownloaded {saved_count}/{len(cultures)} images")
            return 0


def _parse_mcp_result(result) -> dict | None:
    """Parse MCP tool result and extract JSON data."""
    if result is None:
        return None
    
    # Get content list
    content = getattr(result, "content", None) or []
    if not isinstance(content, list):
        return None
    
    # Extract text from content
    for item in content:
        if hasattr(item, "type") and item.type == "text":
            text = getattr(item, "text", "")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download plate images using monomer-cloud MCP"
    )
    parser.add_argument("--plate-id", help="Plate ID")
    parser.add_argument("--plate-name", help="Plate barcode/name")
    parser.add_argument("--output-dir", default="scripts/plate_images", help="Output directory")
    
    args = parser.parse_args()
    
    if not args.plate_id and not args.plate_name:
        parser.error("Must provide either --plate-id or --plate-name")
    
    return asyncio.run(fetch_plate_images(args.plate_id, args.plate_name, args.output_dir))


if __name__ == "__main__":
    sys.exit(main())
