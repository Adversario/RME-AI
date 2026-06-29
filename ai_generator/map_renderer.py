"""
Lightweight synthetic renderer for generated map debugging.

It does not try to replicate real Tibia sprites. It produces a block-color
image with stable colors so the server can inspect composition, borders,
walls, and interactive objects before writing the final OTBM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RENDER_PATH = BASE_DIR / "../template/debug_render.png"

GRASS_GROUNDS = {4526}
ICE_GROUNDS = {101, *range(351, 368)}
DIRT_GROUNDS = {103, 350, *range(368, 381)}
STONE_GROUNDS = {424, 426, 444, 445, 724, 919}
WATER_GROUNDS = {460}

WALL_IDS = {
    903,
    904,
    905,
    907,
    909,
    911,
    913,
    1025,
    1026,
    1027,
    1029,
    1031,
    1033,
    1035,
    1049,
    1050,
    1051,
    1053,
    1055,
    1057,
    1059,
    3458,
    3460,
    3461,
    3462,
}
ROCK_IDS = {1285, 1292, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1356, 1358, 1359}
INTERACTIVE_IDS = {
    411,
    1385,
    1480,
    1526,
    1617,
    1618,
    1621,
    1622,
    1623,
    1662,
    1663,
    1738,
    1810,
    1815,
    2591,
    2592,
    2593,
}
TENT_OR_CAMP_IDS = {1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283}


def tile_color(ground_id: int, item_ids: list[int]) -> tuple[int, int, int]:
    """Return the base color for a materialized tile."""

    item_set = set(item_ids)
    if item_set & INTERACTIVE_IDS:
        return (238, 202, 71)
    if item_set & TENT_OR_CAMP_IDS:
        return (218, 143, 58)
    if item_set & WALL_IDS:
        if item_set & {3458, 3460, 3461, 3462}:
            return (150, 64, 51)
        return (110, 112, 112)
    if item_set & ROCK_IDS:
        return (92, 92, 88)
    if ground_id in GRASS_GROUNDS:
        return (74, 143, 65)
    if ground_id in ICE_GROUNDS:
        return (145, 205, 218)
    if ground_id in DIRT_GROUNDS:
        return (122, 86, 51)
    if ground_id in STONE_GROUNDS:
        return (136, 132, 120)
    if ground_id in WATER_GROUNDS:
        return (46, 94, 166)
    return (44, 44, 44)


def normalize_tiles(tiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, int, int]:
    """Normalize tiles and return the relative bounding box."""

    normalized: list[dict[str, Any]] = []
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        try:
            x = int(tile.get("x", 0))
            y = int(tile.get("y", 0))
            ground_id = int(tile.get("ground_id", 0))
        except (TypeError, ValueError):
            continue
        raw_items = tile.get("item_ids", [])
        item_ids: list[int] = []
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                try:
                    item_ids.append(int(raw_item))
                except (TypeError, ValueError):
                    continue
        normalized.append({"x": x, "y": y, "ground_id": ground_id, "item_ids": item_ids})

    if not normalized:
        return [], 0, 0, 0, 0
    min_x = min(tile["x"] for tile in normalized)
    max_x = max(tile["x"] for tile in normalized)
    min_y = min(tile["y"] for tile in normalized)
    max_y = max(tile["y"] for tile in normalized)
    return normalized, min_x, max_x, min_y, max_y


def render_debug_map(
    tiles: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_RENDER_PATH,
    cell_size: int = 12,
) -> Path:
    """Draw a debug PNG from materialized tiles."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is not installed. Install the 'Pillow' dependency to use render_debug_map.") from exc

    normalized, min_x, max_x, min_y, max_y = normalize_tiles(tiles)
    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)
    image = Image.new("RGB", (width * cell_size, height * cell_size), (18, 18, 18))
    draw = ImageDraw.Draw(image)

    for tile in normalized:
        x = (tile["x"] - min_x) * cell_size
        y = (tile["y"] - min_y) * cell_size
        color = tile_color(tile["ground_id"], tile["item_ids"])
        draw.rectangle((x, y, x + cell_size - 1, y + cell_size - 1), fill=color)
        if tile["item_ids"]:
            inset = max(2, cell_size // 4)
            draw.rectangle((x + inset, y + inset, x + cell_size - inset, y + cell_size - inset), outline=(255, 232, 128))

    for grid_x in range(0, width * cell_size, cell_size):
        draw.line((grid_x, 0, grid_x, height * cell_size), fill=(36, 36, 36))
    for grid_y in range(0, height * cell_size, cell_size):
        draw.line((0, grid_y, width * cell_size, grid_y), fill=(36, 36, 36))

    resolved = Path(output_path)
    if not resolved.is_absolute():
        resolved = (BASE_DIR / output_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    image.save(resolved)
    return resolved


def build_visual_feedback_payload(debug_render_path: Path) -> dict[str, Any]:
    """Future payload structure for the multimodal Gemini loop.

    In the next phase this payload can attach debug_render.png and real
    reference screenshots as multimodal parts for visual correction.
    """

    return {
        "debug_render_path": str(debug_render_path),
        "reference_images": [],
        "next_step": "Send synthetic PNG + real references to Gemini for composition correction.",
    }
