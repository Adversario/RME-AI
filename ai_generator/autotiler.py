"""
Semantic autotiling engine for materializing roles into Tibia 7.60 IDs.

The LLM no longer decides geometric IDs. It only emits roles; this module
resolves blob/Wang-style neighborhoods using native RME rules and 7.60 server
conventions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


SEMANTIC_ROLES = {
    "wall",
    "wall_ruins",
    "door",
    "floor_interior",
    "floor_exterior",
    "tent_north",
    "tent_south",
    "tent_roof",
    "depot_walkway",
    "depot_locker_north",
    "depot_locker_east",
    "depot_railing",
    "mailbox",
    "npc_counter",
}

CHUNK_SIZE = 8
MACRO_ROLES = {
    "spawn_hub_dense",
    "defensive_perimeter",
    "wild_surroundings",
    "camp_amenities",
}
SEMANTIC_ROLES.update(MACRO_ROLES)

WALL_FAMILY_ITEM_IDS: dict[str, set[int]] = {
    "rock wall": {903, 904, 905, 907, 909, 911, 913, 1267, 1268},
    "brick wall": {1025, 1026, 1027, 1029, 1031, 1033, 1035, 1267, 1268},
    "stone wall": {1049, 1050, 1051, 1053, 1055, 1057, 1059, 1267, 1268},
    "ruin wall": {3361, 3362, 3363, 3365, 3367, 3369, 1267, 1268},
    "sandstone railing": {1562, 1564, 1566, 1568, 1267, 1268},
    "jungle stone wall": {3458, 3460, 3461, 3462, 1267, 1268},
}

ALL_WALL_ITEM_IDS = set().union(*WALL_FAMILY_ITEM_IDS.values())

TENT_ROLE_ITEMS = {
    "tent_north": [2767],
    "tent_south": [2768],
    "tent_roof": [2701, 2702, 2703],
}

SURFACE_NATURE_GROUNDS = {4526}
UNDERGROUND_NATURE_GROUND = 103
UPPER_WOOD_PLATFORM_GROUND = 424
TREE_CANOPY_IDS = {2701, 2702, 2703, 2767, 2768}
STATIC_CORPSE_IDS = {3070}
DENSE_CAVE_BLOCKER_IDS = {1285}
RUSTIC_CAVE_DETAIL_IDS = [1307, 1308, 1309, 1310, 1311, 1312, 3648, 3649, 3650, 3651]
CAVE_GRAVEL_DETAIL_IDS = [1307, 1308, 1309, 1310, 1311, 1312]
CAVE_MUD_GROUND_IDS = [354, 355]
CAVE_BIOME_TAGS = {"dirt_cave", "ice_cave"}
CAVE_FLOOR_GROUNDS = {
    "dirt_cave": 103,
    "ice_cave": 351,
}


@dataclass
class SemanticTile:
    rel_x: int
    rel_y: int
    role: str


@dataclass
class MaterializedTile:
    rel_x: int
    rel_y: int
    ground_id: int
    item_ids: list[int] = field(default_factory=list)


def unique_items(item_ids: list[int]) -> list[int]:
    """Deduplicate item_ids while preserving order."""

    seen: set[int] = set()
    output: list[int] = []
    for item_id in item_ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        output.append(item_id)
    return output


def get_wall_rule(rme_rules: dict[str, Any], preferred_name: str = "stone wall") -> dict[str, Any]:
    """Get the wall palette from RME with a 7.60 fallback."""

    walls = rme_rules.get("walls", {}) if isinstance(rme_rules, dict) else {}
    rule = walls.get(preferred_name)
    if isinstance(rule, dict):
        return rule
    fallbacks = {
        "rock wall": {
            "horizontal": 904,
            "vertical": 903,
            "corner_variants": [907, 909, 911, 913],
            "pole": 905,
            "doors_horizontal": [{"id": 1267}],
            "doors_vertical": [{"id": 1268}],
        },
        "brick wall": {
            "horizontal": 1026,
            "vertical": 1025,
            "corner_variants": [1029, 1031, 1033, 1035],
            "pole": 1027,
            "doors_horizontal": [{"id": 1267}],
            "doors_vertical": [{"id": 1268}],
        },
        "ruin wall": {
            "horizontal": 3362,
            "vertical": 3361,
            "corner_variants": [3365, 3367, 3369],
            "pole": 3363,
            "doors_horizontal": [{"id": 1267}],
            "doors_vertical": [{"id": 1268}],
        },
        "sandstone railing": {
            "horizontal": 1562,
            "vertical": 1564,
            "corner_variants": [1568],
            "pole": 1566,
            "doors_horizontal": [{"id": 1267}],
            "doors_vertical": [{"id": 1268}],
        },
        "jungle stone wall": {
            "horizontal": 3458,
            "vertical": 3460,
            "corner_variants": [3461],
            "pole": 3462,
            "doors_horizontal": [{"id": 1267}],
            "doors_vertical": [{"id": 1268}],
        },
    }
    if preferred_name in fallbacks:
        return fallbacks[preferred_name]
    return {
        "horizontal": 1050,
        "vertical": 1049,
        "corner_variants": [1053, 1055, 1057, 1059],
        "pole": 1051,
        "doors_horizontal": [{"id": 1267}],
        "doors_vertical": [{"id": 1268}],
    }


def wall_family_for_tag(tag: str | None, role: str = "wall") -> str:
    """Select the RME wall family by biome and semantic role."""

    if role == "wall_ruins":
        return "ruin wall"
    if tag in {"dirt_cave", "ice_cave", "stone_mountain"}:
        return "rock wall"
    if tag in {"urban_floor", "depot"}:
        return "brick wall"
    if tag == "desert":
        return "sandstone railing"
    if tag in {"nature_surface", "swamp"}:
        return "jungle stone wall"
    return "stone wall"


def primary_int(value: Any, fallback: int) -> int:
    """Convert to int when possible."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if 1 <= parsed <= 65535 else fallback


def plain_int(value: Any) -> int | None:
    """Convert relative coordinates, allowing negatives."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def wall_ids_from_rule(rule: dict[str, Any]) -> dict[str, int]:
    """Normalize wall IDs into a small table."""

    corners = rule.get("corner_variants") or []
    corner = primary_int(corners[0], 1053) if isinstance(corners, list) and corners else 1053
    return {
        "horizontal": primary_int(rule.get("horizontal"), 1050),
        "vertical": primary_int(rule.get("vertical"), 1049),
        "corner": corner,
        "pole": primary_int(rule.get("pole"), 1051),
        "door_h": primary_int((rule.get("doors_horizontal") or [{}])[0].get("id"), 1267),
        "door_v": primary_int((rule.get("doors_vertical") or [{}])[0].get("id"), 1268),
    }


def semantic_index(tiles: list[SemanticTile]) -> dict[tuple[int, int], str]:
    """Index roles by coordinate."""

    return {(tile.rel_x, tile.rel_y): tile.role for tile in tiles}


def role_at(index: dict[tuple[int, int], str], x: int, y: int) -> str:
    """Return the role or empty."""

    return index.get((x, y), "empty")


def is_wall_like(role: str) -> bool:
    """Roles that connect to walls for blob autotiling."""

    return role in {"wall", "wall_ruins", "door"}


def resolve_wall_item(
    index: dict[tuple[int, int], str],
    x: int,
    y: int,
    wall_ids: dict[str, int],
    ruins_wall_ids: dict[str, int] | None = None,
) -> int:
    """Resolve wall/door IDs from four-direction neighbors."""

    role = role_at(index, x, y)
    if role == "wall_ruins" and ruins_wall_ids is not None:
        wall_ids = ruins_wall_ids

    west = is_wall_like(role_at(index, x - 1, y))
    east = is_wall_like(role_at(index, x + 1, y))
    north = is_wall_like(role_at(index, x, y - 1))
    south = is_wall_like(role_at(index, x, y + 1))
    horizontal = west or east
    vertical = north or south

    if role == "door":
        return wall_ids["door_h"] if horizontal and not vertical else wall_ids["door_v"]
    if horizontal and not vertical:
        return wall_ids["horizontal"]
    if vertical and not horizontal:
        return wall_ids["vertical"]
    if horizontal and vertical:
        return wall_ids["corner"]
    return wall_ids["pole"]


def counter_item_for_row(x: int, row_counter_positions: set[int]) -> int:
    """Autotile horizontal counter: 1617 al inicio, 1618 al continuar."""

    return 1617 if (x - 1) not in row_counter_positions else 1618


def vertical_depot_counter_item(_y: int, _column_counter_positions: set[int]) -> int:
    """Autotile vertical depot counter.

    El set actual de assets 7.60 confirmado para el flujo este usa 1622 como
    stable vertical counter. It remains a function so it can be extended to
    variantes top/middle/bottom si walls.xml/doodads.xml las expone luego.
    """

    return 1622


def add_item(tile: MaterializedTile, item_id: int) -> None:
    """Agrega item sin duplicar."""

    tile.item_ids = unique_items(tile.item_ids + [item_id])


def remove_items(tile: MaterializedTile, item_ids: set[int]) -> None:
    """Elimina IDs concretos de un tile materializado."""

    tile.item_ids = [item_id for item_id in tile.item_ids if item_id not in item_ids]


def tile_all_ids(tile: dict[str, Any]) -> set[int]:
    """Return ground_id and item_ids as a reference ID set."""

    ids: set[int] = set()
    for value in [tile.get("ground_id"), *(tile.get("item_ids") or [])]:
        parsed = primary_int(value, 0)
        if parsed:
            ids.add(parsed)
    return ids


def normalized_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    """Compute a coordinate bounding box or None."""

    if not coords:
        return None
    return (
        min(x for x, _y in coords),
        max(x for x, _y in coords),
        min(y for _x, y in coords),
        max(y for _x, y in coords),
    )


def map_archetype_point(
    rel_x: int,
    rel_y: int,
    source_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    """Map an archetype point into the generated semantic bounding box."""

    src_min_x, src_max_x, src_min_y, src_max_y = source_bbox
    dst_min_x, dst_max_x, dst_min_y, dst_max_y = target_bbox
    src_w = max(1, src_max_x - src_min_x)
    src_h = max(1, src_max_y - src_min_y)
    dst_w = max(1, dst_max_x - dst_min_x)
    dst_h = max(1, dst_max_y - dst_min_y)
    x_ratio = (rel_x - src_min_x) / src_w
    y_ratio = (rel_y - src_min_y) / src_h
    return (
        round(dst_min_x + x_ratio * dst_w),
        round(dst_min_y + y_ratio * dst_h),
    )


def has_depot_intent(semantic_tiles: list[SemanticTile], archetype: dict[str, Any] | None) -> bool:
    """Detect whether depot pattern tracing should be applied."""

    roles = {tile.role for tile in semantic_tiles}
    if roles & {"depot_locker_north", "depot_locker_east", "depot_walkway", "depot_railing"}:
        return True
    tags = set(str(tag) for tag in (archetype or {}).get("tags", []))
    return bool(tags & {"depot", "lockers", "mail"})


def archetype_tiles(archetype: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return archetype tiles when present."""

    tiles = (archetype or {}).get("tiles", [])
    return [tile for tile in tiles if isinstance(tile, dict)]


def apply_depot_archetype_pattern(
    output: dict[tuple[int, int], MaterializedTile],
    semantic_tiles: list[SemanticTile],
    width: int,
    height: int,
    archetype: dict[str, Any] | None,
) -> None:
    """Adjust micro-geometry by copying structural vectors from a real depot.

    The archetype is used as an offset field, not as an absolute replacement.
    Only confirmed Tibia 7.60 depot IDs are copied: counters, lockers, railings,
    mailbox, and walkways. This preserves the LLM intent while correcting fine
    structural rhythm.
    """

    if not has_depot_intent(semantic_tiles, archetype):
        return
    source_tiles = archetype_tiles(archetype)
    if not source_tiles:
        return

    source_coords: list[tuple[int, int]] = []
    source_specials: list[tuple[int, int, set[int]]] = []
    special_ids = {426, 1526, 1617, 1618, 1621, 1622, 1623, 2591, 2592, 2593}
    for tile in source_tiles:
        rel_x = plain_int(tile.get("rel_x"))
        rel_y = plain_int(tile.get("rel_y"))
        if rel_x is None or rel_y is None:
            continue
        source_coords.append((rel_x, rel_y))
        ids = tile_all_ids(tile)
        if ids & special_ids:
            source_specials.append((rel_x, rel_y, ids))

    source_bbox = normalized_bbox(source_coords)
    target_roles = {
        "wall",
        "door",
        "floor_interior",
        "depot_locker_north",
        "depot_locker_east",
        "depot_walkway",
        "depot_railing",
        "mailbox",
        "npc_counter",
    }
    target_coords = [(tile.rel_x, tile.rel_y) for tile in semantic_tiles if tile.role in target_roles]
    target_bbox = normalized_bbox(target_coords)
    if source_bbox is None or target_bbox is None or not source_specials:
        return

    def ensure(x: int, y: int, ground_id: int) -> MaterializedTile | None:
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        key = (x, y)
        if key not in output:
            output[key] = MaterializedTile(x, y, ground_id, [])
        else:
            output[key].ground_id = ground_id
        return output[key]

    counter_like = {1617, 1618, 1621, 1622, 1623, 2591, 2592}
    applied = 0
    for rel_x, rel_y, ids in source_specials:
        x, y = map_archetype_point(rel_x, rel_y, source_bbox, target_bbox)
        tile = ensure(x, y, 426 if 426 in ids and not ids & counter_like else 424)
        if tile is None:
            continue

        if 426 in ids and not ids & counter_like:
            tile.ground_id = 426
            applied += 1
            continue

        if 1526 in ids:
            tile.ground_id = 424
            remove_items(tile, counter_like)
            add_item(tile, 1526)
            applied += 1
            continue

        if 2593 in ids:
            tile.ground_id = 424
            add_item(tile, 2593)
            applied += 1
            continue

        if ids & {1617, 1618, 1621, 1623, 2591}:
            tile.ground_id = 424
            remove_items(tile, counter_like)
            if 1623 in ids:
                add_item(tile, 1623)
            elif 1617 in ids:
                add_item(tile, 1617)
            else:
                add_item(tile, 1618)
            if 2591 in ids:
                add_item(tile, 2591)
            applied += 1
            continue

        if ids & {1622, 2592}:
            tile.ground_id = 424
            remove_items(tile, counter_like)
            add_item(tile, 1622)
            if 2592 in ids:
                add_item(tile, 2592)
            applied += 1

    if applied:
        print(f"[autotiler] Applied depot archetype trace: tiles={applied}")


def apply_real_slice_pattern(
    output: dict[tuple[int, int], MaterializedTile],
    semantic_tiles: list[SemanticTile],
    width: int,
    height: int,
    pattern_slice: dict[str, Any] | None,
) -> None:
    """Overlay a real fragment by aligning walkability masks."""

    if not pattern_slice:
        return
    slice_tiles = pattern_slice.get("tiles", [])
    if not isinstance(slice_tiles, list) or not slice_tiles:
        return

    source_w = primary_int(pattern_slice.get("width"), width)
    source_h = primary_int(pattern_slice.get("height"), height)
    source_w = max(1, source_w)
    source_h = max(1, source_h)
    walkable_path = pattern_slice.get("walkable_path", [])
    if not isinstance(walkable_path, list) or len(walkable_path) != source_w * source_h:
        walkable_path = []
    applied = 0
    intent_mask = build_intention_walkable_mask(semantic_tiles, width, height)
    offset_x, offset_y, score = find_best_walkable_offset(intent_mask, width, height, walkable_path, source_w, source_h)
    source_by_coord: dict[tuple[int, int], dict[str, Any]] = {}
    role_index = semantic_index(semantic_tiles)

    for raw_tile in slice_tiles:
        if not isinstance(raw_tile, dict):
            continue
        src_x = plain_int(raw_tile.get("x"))
        src_y = plain_int(raw_tile.get("y"))
        if src_x is None or src_y is None:
            continue
        source_by_coord[(src_x, src_y)] = raw_tile

    for dst_y in range(height):
        for dst_x in range(width):
            src_x = dst_x + offset_x
            src_y = dst_y + offset_y
            if src_x < 0 or src_y < 0 or src_x >= source_w or src_y >= source_h:
                continue
            raw_tile = source_by_coord.get((src_x, src_y))
            if raw_tile is None:
                continue
            ground_id = primary_int(raw_tile.get("g"), 0)
            if not ground_id:
                continue
            item_ids = [
                item_id
                for item_id in (primary_int(value, 0) for value in (raw_tile.get("i") or []))
                if item_id
            ]
            source_walkable = bool(walkable_path[src_y * source_w + src_x]) if walkable_path else True
            target_walkable = bool(intent_mask[dst_y * width + dst_x])
            if target_walkable and not source_walkable and not item_ids:
                continue
            if (
                not target_walkable
                and source_walkable
                and role_at(role_index, dst_x, dst_y) in {"wall", "door"}
                and not item_ids
            ):
                continue

            target_role = role_at(role_index, dst_x, dst_y)
            if should_feather_pattern_edge(dst_x, dst_y, width, height, target_walkable, item_ids, target_role):
                continue
            output[(dst_x, dst_y)] = MaterializedTile(dst_x, dst_y, ground_id, unique_items(item_ids))
            applied += 1

    if applied:
        tag = pattern_slice.get("tag", "unknown")
        print(
            "[autotiler] Slice real alineado por mascara: "
            f"tag={tag}, offset=({offset_x},{offset_y}), score={score:.4f}, tiles={applied}"
        )


def build_intention_walkable_mask(semantic_tiles: list[SemanticTile], width: int, height: int) -> list[int]:
    """Create a binary walkability-intent mask from Gemini roles."""

    walkable_roles = {"floor_interior", "floor_exterior", "depot_walkway", "door", "mailbox"}
    mask = [0] * (width * height)
    for tile in semantic_tiles:
        if tile.rel_x < 0 or tile.rel_y < 0 or tile.rel_x >= width or tile.rel_y >= height:
            continue
        if tile.role in walkable_roles:
            mask[tile.rel_y * width + tile.rel_x] = 1
    if any(mask):
        return mask
    for index in range(width * height):
        mask[index] = 1
    return mask


def find_best_walkable_offset(
    intent_mask: list[int],
    width: int,
    height: int,
    source_mask: list[Any],
    source_w: int,
    source_h: int,
) -> tuple[int, int, float]:
    """Busca el desfase que maximiza IoU y similitud Hamming de caminabilidad."""

    if not source_mask:
        return 0, 0, 0.0

    best_offset = (0, 0)
    best_score = -1.0
    min_offset_x = min(0, source_w - width)
    max_offset_x = max(0, source_w - width)
    min_offset_y = min(0, source_h - height)
    max_offset_y = max(0, source_h - height)

    for offset_y in range(min_offset_y, max_offset_y + 1):
        for offset_x in range(min_offset_x, max_offset_x + 1):
            intersection = 0
            union = 0
            matches = 0
            compared = 0
            for y in range(height):
                src_y = y + offset_y
                if src_y < 0 or src_y >= source_h:
                    continue
                for x in range(width):
                    src_x = x + offset_x
                    if src_x < 0 or src_x >= source_w:
                        continue
                    intent = bool(intent_mask[y * width + x])
                    source = bool(source_mask[src_y * source_w + src_x])
                    if intent and source:
                        intersection += 1
                    if intent or source:
                        union += 1
                    if intent == source:
                        matches += 1
                    compared += 1
            if compared == 0:
                continue
            iou = intersection / union if union else 0.0
            hamming = matches / compared
            score = (iou * 0.75) + (hamming * 0.25)
            if score > best_score:
                best_score = score
                best_offset = (offset_x, offset_y)

    return best_offset[0], best_offset[1], max(0.0, best_score)


def should_feather_pattern_edge(
    dst_x: int,
    dst_y: int,
    width: int,
    height: int,
    target_walkable: bool,
    item_ids: list[int],
    role: str,
) -> bool:
    """Evita cortes cuadrados en el borde del slice real.

    Only skip filler without items. Real paths, walls, doors, and decorations
    se conservan para no romper arquitectura del patron.
    """

    if target_walkable or item_ids or role in {"wall", "wall_ruins", "door", "depot_locker_north", "depot_locker_east"}:
        return False
    edge_distance = min(dst_x, dst_y, width - 1 - dst_x, height - 1 - dst_y)
    if edge_distance < 0:
        return False
    if edge_distance == 0:
        return True
    if edge_distance == 1:
        return (dst_x + dst_y) % 2 == 0
    return False


def has_adjacent_family_wall(
    output: dict[tuple[int, int], MaterializedTile],
    x: int,
    y: int,
    family_ids: set[int],
) -> bool:
    """Detect a wall family in N/S/E/W neighbors."""

    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        neighbor = output.get((x + dx, y + dy))
        if neighbor is not None and any(item_id in family_ids for item_id in neighbor.item_ids):
            return True
    return False


def enforce_wall_family_contiguity(
    output: dict[tuple[int, int], MaterializedTile],
    semantic_tiles: list[SemanticTile],
    wall_family: str,
    wall_ids: dict[str, int],
) -> None:
    """Unify discordant walls with the dominant pattern_slice family.

    This pass acts as a WFC-style adjacency constraint: cave walls cannot touch
    or mix with urban walls unless the user
    haya marcado explicitamente wall_ruins.
    """

    dominant_ids = WALL_FAMILY_ITEM_IDS.get(wall_family, set())
    if not dominant_ids:
        return
    discordant_ids = ALL_WALL_ITEM_IDS - dominant_ids - WALL_FAMILY_ITEM_IDS.get("ruin wall", set())
    if not discordant_ids:
        return

    role_index = semantic_index(semantic_tiles)
    repair_index = dict(role_index)
    repairs = 0
    for (x, y), tile in list(output.items()):
        role = role_at(role_index, x, y)
        if role == "wall_ruins":
            continue
        item_set = set(tile.item_ids)
        if not item_set & discordant_ids:
            continue
        adjacent_dominant = has_adjacent_family_wall(output, x, y, dominant_ids)
        dominant_biome = wall_family in {"rock wall", "brick wall", "stone wall", "jungle stone wall"}
        if not adjacent_dominant and not dominant_biome:
            continue
        repair_index.setdefault((x, y), "wall")
        replacement = resolve_wall_item(repair_index, x, y, wall_ids)
        tile.item_ids = unique_items([replacement if item_id in discordant_ids else item_id for item_id in tile.item_ids])
        repairs += 1

    if repairs:
        print(f"[autotiler] Applied WFC wall-contiguity pass: family={wall_family}, repairs={repairs}")


def composite_variants_from_rules(rme_rules: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract multi-tile composites from RME doodad rules."""

    variants: list[dict[str, Any]] = []
    doodads = rme_rules.get("doodads", {}) if isinstance(rme_rules, dict) else {}
    if not isinstance(doodads, dict):
        return variants

    for name, rule in doodads.items():
        if not isinstance(rule, dict):
            continue
        composites = rule.get("composites", [])
        if not isinstance(composites, list):
            continue
        for composite in composites:
            if not isinstance(composite, dict):
                continue
            tiles = composite.get("tiles", [])
            if not isinstance(tiles, list) or len(tiles) < 2:
                continue
            parts: list[dict[str, Any]] = []
            item_ids: set[int] = set()
            for tile in tiles:
                if not isinstance(tile, dict):
                    continue
                x = plain_int(tile.get("x"))
                y = plain_int(tile.get("y"))
                if x is None or y is None:
                    continue
                ids = [
                    item_id
                    for item_id in (primary_int(value, 0) for value in (tile.get("item_ids") or []))
                    if item_id
                ]
                if not ids:
                    continue
                item_ids.update(ids)
                parts.append({"x": x, "y": y, "item_ids": ids})
            if len(parts) < 2:
                continue
            variants.append(
                {
                    "name": str(name),
                    "width": primary_int(composite.get("width"), 1),
                    "height": primary_int(composite.get("height"), 1),
                    "parts": parts,
                    "item_ids": item_ids,
                }
            )
    return variants


def composite_index_by_item(composites: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Indexa composites por cada item_id que participa."""

    index: dict[int, list[dict[str, Any]]] = {}
    for composite in composites:
        for item_id in composite.get("item_ids", set()):
            if isinstance(item_id, int):
                index.setdefault(item_id, []).append(composite)
    return index


def is_near_materialization_border(x: int, y: int, width: int, height: int, margin: int) -> bool:
    """Detect whether a tile is near or outside the requested area."""

    return x < margin or y < margin or x >= width - margin or y >= height - margin


def apply_composite_structure_linter(
    output: dict[tuple[int, int], MaterializedTile],
    width: int,
    height: int,
    rme_rules: dict[str, Any],
    dominant_ground_id: int,
    margin: int = 2,
) -> None:
    """Expande piezas hermanas de doodads multitile cerca del borde.

    Evita que rocas 2x2, carpas u otros composites declarados en doodads.xml
    from being cut when the real fragment lands on the injection boundary.
    """

    composites = composite_variants_from_rules(rme_rules)
    if not composites:
        return
    by_item = composite_index_by_item(composites)
    if not by_item:
        return

    min_x = -margin
    min_y = -margin
    max_x = width + margin - 1
    max_y = height + margin - 1
    repairs = 0
    inspected: set[tuple[int, int, int]] = set()

    for (x, y), tile in list(output.items()):
        if not is_near_materialization_border(x, y, width, height, margin):
            continue
        for item_id in list(tile.item_ids):
            for composite in by_item.get(item_id, []):
                composite_key = (x, y, item_id)
                if composite_key in inspected:
                    continue
                inspected.add(composite_key)
                parts = composite.get("parts", [])
                if not isinstance(parts, list):
                    continue
                anchors = [
                    part
                    for part in parts
                    if isinstance(part, dict) and item_id in set(part.get("item_ids", []))
                ]
                for anchor in anchors:
                    anchor_x = plain_int(anchor.get("x"))
                    anchor_y = plain_int(anchor.get("y"))
                    if anchor_x is None or anchor_y is None:
                        continue
                    origin_x = x - anchor_x
                    origin_y = y - anchor_y
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        part_x = plain_int(part.get("x"))
                        part_y = plain_int(part.get("y"))
                        if part_x is None or part_y is None:
                            continue
                        dst_x = origin_x + part_x
                        dst_y = origin_y + part_y
                        if dst_x < min_x or dst_y < min_y or dst_x > max_x or dst_y > max_y:
                            continue
                        key = (dst_x, dst_y)
                        if key not in output:
                            output[key] = MaterializedTile(dst_x, dst_y, dominant_ground_id, [])
                        elif dst_x < 0 or dst_y < 0 or dst_x >= width or dst_y >= height:
                            output[key].ground_id = dominant_ground_id
                        before = len(output[key].item_ids)
                        for raw_part_id in part.get("item_ids", []):
                            part_id = primary_int(raw_part_id, 0)
                            if part_id:
                                add_item(output[key], part_id)
                        if len(output[key].item_ids) != before:
                            repairs += 1

    if repairs:
        print(f"[autotiler] Linter de composites expandio frontera: margin={margin}, repairs={repairs}")


def dominant_ground_from_pattern_slice(pattern_slice: dict[str, Any] | None, fallback: int = 424) -> int:
    """Get the dominant ground from the real slice for margin filling."""

    if not isinstance(pattern_slice, dict):
        return fallback
    stats = pattern_slice.get("stats", {})
    ground_top = stats.get("ground_top", []) if isinstance(stats, dict) else []
    if isinstance(ground_top, list) and ground_top:
        first = ground_top[0]
        if isinstance(first, (list, tuple)) and first:
            parsed = primary_int(first[0], 0)
            if parsed:
                return parsed

    counter: dict[int, int] = {}
    tiles = pattern_slice.get("tiles", [])
    if isinstance(tiles, list):
        for tile in tiles:
            if not isinstance(tile, dict):
                continue
            ground_id = primary_int(tile.get("g"), 0)
            if ground_id:
                counter[ground_id] = counter.get(ground_id, 0) + 1
    if counter:
        return max(counter.items(), key=lambda entry: entry[1])[0]

    tag = str(pattern_slice.get("tag") or "")
    fallbacks = {
        "nature_surface": 4526,
        "ice_cave": 101,
        "dirt_cave": 103,
        "stone_mountain": 444,
        "desert": 231,
        "swamp": 4691,
        "urban_floor": 424,
        "depot": 424,
    }
    return fallbacks.get(tag, fallback)


def count_cellular_wall_neighbors(grid: list[list[int]], x: int, y: int) -> int:
    """Count wall neighbors in 8 directions; out-of-bounds counts as wall."""

    height = len(grid)
    width = len(grid[0]) if height else 0
    total = 0
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            if offset_x == 0 and offset_y == 0:
                continue
            nx = x + offset_x
            ny = y + offset_y
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                total += 1
            elif grid[ny][nx] == 1:
                total += 1
    return total


def generate_cellular_cave(
    width: int,
    height: int,
    iterations: int = 4,
    birth_limit: int = 4,
    death_limit: int = 4,
) -> list[list[int]]:
    """Generate a cave mask: 1 wall, 0 walkable floor."""

    rng = random.Random((width * 73856093) ^ (height * 19349663) ^ (iterations * 83492791))
    grid: list[list[int]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            is_border = x == 0 or y == 0 or x == width - 1 or y == height - 1
            row.append(1 if is_border or rng.random() < 0.45 else 0)
        grid.append(row)

    for _ in range(iterations):
        next_grid: list[list[int]] = []
        for y in range(height):
            next_row: list[int] = []
            for x in range(width):
                is_border = x == 0 or y == 0 or x == width - 1 or y == height - 1
                if is_border:
                    next_row.append(1)
                    continue
                wall_neighbors = count_cellular_wall_neighbors(grid, x, y)
                if grid[y][x] == 1:
                    next_row.append(0 if wall_neighbors < death_limit else 1)
                else:
                    next_row.append(1 if wall_neighbors > birth_limit else 0)
            next_grid.append(next_row)
        grid = next_grid

    if width >= 5 and height >= 5:
        mid_x = width // 2
        mid_y = height // 2
        for x in range(1, width - 1):
            grid[mid_y][x] = 0
        for y in range(1, height - 1):
            grid[y][mid_x] = 0
        for x, y in (
            (0, mid_y),
            (1, mid_y),
            (width - 2, mid_y),
            (width - 1, mid_y),
            (mid_x, 0),
            (mid_x, 1),
            (mid_x, height - 2),
            (mid_x, height - 1),
        ):
            grid[y][x] = 0
    return grid


def is_macro_plan(semantic_tiles: list[SemanticTile]) -> bool:
    """Detect whether the response uses macro zones instead of individual tiles."""

    return bool(semantic_tiles) and all(tile.role in MACRO_ROLES for tile in semantic_tiles)


def pattern_tile_index(pattern_slice: dict[str, Any] | None) -> dict[tuple[int, int], dict[str, Any]]:
    """Index compact real-slice tiles by local coordinate."""

    index: dict[tuple[int, int], dict[str, Any]] = {}
    tiles = (pattern_slice or {}).get("tiles", [])
    if not isinstance(tiles, list):
        return index
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        x = plain_int(tile.get("x"))
        y = plain_int(tile.get("y"))
        if x is None or y is None:
            continue
        index[(x, y)] = tile
    return index


def tile_item_ids_from_raw(raw_tile: dict[str, Any] | None) -> list[int]:
    """Extract compact item_ids from a slices_pool tile."""

    if not isinstance(raw_tile, dict):
        return []
    return [
        item_id
        for item_id in (primary_int(value, 0) for value in (raw_tile.get("i") or []))
        if item_id
    ]


def raw_tile_ground(raw_tile: dict[str, Any] | None, fallback: int) -> int:
    """Get compact ground from a slices_pool tile."""

    if not isinstance(raw_tile, dict):
        return fallback
    return primary_int(raw_tile.get("g"), fallback)


def score_source_chunk(
    source_index: dict[tuple[int, int], dict[str, Any]],
    source_w: int,
    source_h: int,
    origin_x: int,
    origin_y: int,
    macro_role: str,
) -> int:
    """Puntua una ventana 8x8 del slice real para un macro-rol."""

    item_total = 0
    unique_items: set[int] = set()
    wall_like = 0
    walkable = 0
    grass = 0
    for y in range(origin_y, min(source_h, origin_y + CHUNK_SIZE)):
        for x in range(origin_x, min(source_w, origin_x + CHUNK_SIZE)):
            raw = source_index.get((x, y))
            if raw is None:
                continue
            ground = raw_tile_ground(raw, 0)
            if ground == 4526:
                grass += 1
            if raw.get("w"):
                walkable += 1
            ids = tile_item_ids_from_raw(raw)
            item_total += len(ids)
            unique_items.update(ids)
            if set(ids) & (ALL_WALL_ITEM_IDS | {1285, 1296, 1297, 1298, 1299, 2767, 2768, 2701, 2702, 2703}):
                wall_like += 1

    entropy = item_total * 6 + len(unique_items) * 10 + wall_like * 8
    if macro_role == "spawn_hub_dense":
        return entropy + item_total * 10 - grass
    if macro_role == "camp_amenities":
        return entropy + len(unique_items) * 12
    if macro_role == "defensive_perimeter":
        return wall_like * 30 + item_total * 4
    if macro_role == "wild_surroundings":
        return walkable * 8 + grass * 3 - item_total * 2
    return entropy


def select_source_chunk(
    pattern_slice: dict[str, Any] | None,
    macro_role: str,
) -> tuple[int, int]:
    """Elige el origen 8x8 mas apropiado dentro del pattern_slice."""

    if not isinstance(pattern_slice, dict):
        return 0, 0
    source_w = max(1, primary_int(pattern_slice.get("width"), CHUNK_SIZE))
    source_h = max(1, primary_int(pattern_slice.get("height"), CHUNK_SIZE))
    source_index = pattern_tile_index(pattern_slice)
    if not source_index:
        return 0, 0

    best_origin = (0, 0)
    best_score = -10**9
    max_x = max(0, source_w - CHUNK_SIZE)
    max_y = max(0, source_h - CHUNK_SIZE)
    for origin_y in range(0, max_y + 1):
        for origin_x in range(0, max_x + 1):
            score = score_source_chunk(source_index, source_w, source_h, origin_x, origin_y, macro_role)
            if score > best_score:
                best_score = score
                best_origin = (origin_x, origin_y)
    return best_origin


def cave_wall_item_from_mask(mask: list[list[int]], x: int, y: int, wall_ids: dict[str, int]) -> int:
    """Resolve a cave wall by cardinal mask continuity."""

    height = len(mask)
    width = len(mask[0]) if height else 0

    def is_wall(nx: int, ny: int) -> bool:
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            return True
        return mask[ny][nx] == 1

    north = is_wall(x, y - 1)
    south = is_wall(x, y + 1)
    west = is_wall(x - 1, y)
    east = is_wall(x + 1, y)
    horizontal = west or east
    vertical = north or south
    if horizontal and vertical:
        return wall_ids["corner"]
    if horizontal:
        return wall_ids["horizontal"]
    if vertical:
        return wall_ids["vertical"]
    return wall_ids["pole"]


def cellular_floor_touches_wall(mask: list[list[int]], x: int, y: int) -> bool:
    """Return whether a floor tile touches a cardinal wall."""

    height = len(mask)
    width = len(mask[0]) if height else 0
    for offset_x, offset_y in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx = x + offset_x
        ny = y + offset_y
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            return True
        if mask[ny][nx] == 1:
            return True
    return False


def cellular_cave_floor_detail(
    pattern_tag: str,
    touches_wall: bool,
    seed: int,
) -> tuple[int | None, int | None]:
    """Choose cave floor detail, prioritizing rock-adjacent edges."""

    if touches_wall:
        if seed % 5 == 0:
            return None, CAVE_GRAVEL_DETAIL_IDS[seed % len(CAVE_GRAVEL_DETAIL_IDS)]
        if pattern_tag == "dirt_cave" and seed % 13 == 0:
            return CAVE_MUD_GROUND_IDS[seed % len(CAVE_MUD_GROUND_IDS)], None
        return None, None
    if seed % 29 == 0:
        return None, CAVE_GRAVEL_DETAIL_IDS[seed % len(CAVE_GRAVEL_DETAIL_IDS)]
    return None, None


def stamp_cellular_cave_chunk(
    output: dict[tuple[int, int, int], MaterializedTile],
    pattern_tag: str,
    dst_origin_x: int,
    dst_origin_y: int,
    width: int,
    height: int,
    rme_rules: dict[str, Any],
    fallback_ground: int,
) -> int:
    """Carve a cave chunk with Cellular Automata instead of copying a flat slice."""

    mask = generate_cellular_cave(CHUNK_SIZE, CHUNK_SIZE)
    chunk_x = dst_origin_x // CHUNK_SIZE
    chunk_y = dst_origin_y // CHUNK_SIZE
    if (chunk_x + chunk_y) % 2:
        mask = [list(reversed(row)) for row in mask]
    if chunk_y % 2:
        mask = list(reversed(mask))

    ground_id = CAVE_FLOOR_GROUNDS.get(pattern_tag, fallback_ground)
    wall_family = wall_family_for_tag(pattern_tag)
    wall_ids = wall_ids_from_rule(get_wall_rule(rme_rules, wall_family))
    applied = 0
    for local_y in range(CHUNK_SIZE):
        for local_x in range(CHUNK_SIZE):
            dst_x = dst_origin_x + local_x
            dst_y = dst_origin_y + local_y
            if dst_x < 0 or dst_y < 0 or dst_x >= width or dst_y >= height:
                continue
            item_ids: list[int] = []
            tile_ground_id = ground_id
            if mask[local_y][local_x] == 1:
                item_ids.append(cave_wall_item_from_mask(mask, local_x, local_y, wall_ids))
            else:
                touches_wall = cellular_floor_touches_wall(mask, local_x, local_y)
                seed = (local_x * 17) + (local_y * 31) + (chunk_x * 7) + (chunk_y * 11)
                detail_ground_id, detail_item_id = cellular_cave_floor_detail(pattern_tag, touches_wall, seed)
                if detail_ground_id is not None:
                    tile_ground_id = detail_ground_id
                if detail_item_id is not None:
                    item_ids.append(detail_item_id)
            output[(0, dst_x, dst_y)] = MaterializedTile(dst_x, dst_y, tile_ground_id, unique_items(item_ids))
            applied += 1
    return applied


def stamp_source_chunk(
    output: dict[tuple[int, int, int], MaterializedTile],
    pattern_slice: dict[str, Any] | None,
    macro_role: str,
    dst_origin_x: int,
    dst_origin_y: int,
    width: int,
    height: int,
    fallback_ground: int,
) -> int:
    """Stamp a real 8x8 block at the destination position."""

    source_index = pattern_tile_index(pattern_slice)
    source_x, source_y = select_source_chunk(pattern_slice, macro_role)
    applied = 0
    for local_y in range(CHUNK_SIZE):
        for local_x in range(CHUNK_SIZE):
            dst_x = dst_origin_x + local_x
            dst_y = dst_origin_y + local_y
            if dst_x < 0 or dst_y < 0 or dst_x >= width or dst_y >= height:
                continue

            raw = source_index.get((source_x + local_x, source_y + local_y))
            ground = raw_tile_ground(raw, fallback_ground)
            item_ids = tile_item_ids_from_raw(raw)
            key = (0, dst_x, dst_y)
            edge = local_x in {0, CHUNK_SIZE - 1} or local_y in {0, CHUNK_SIZE - 1}
            if edge and not item_ids and key in output:
                continue
            output[key] = MaterializedTile(dst_x, dst_y, ground, unique_items(item_ids))
            applied += 1
    return applied


def stamp_source_chunk_layers(
    output: dict[tuple[int, int, int], MaterializedTile],
    pattern_slice: dict[str, Any] | None,
    macro_role: str,
    source_x: int,
    source_y: int,
    dst_origin_x: int,
    dst_origin_y: int,
    width: int,
    height: int,
    fallback_ground: int,
) -> int:
    """Stamp z_layers using the same XY offset as the base chunk."""

    if not isinstance(pattern_slice, dict) or not pattern_slice.get("multilayer"):
        return 0
    pattern_tag = str(pattern_slice.get("tag") or "") or None
    z_layers = pattern_slice.get("z_layers", {})
    if not isinstance(z_layers, dict):
        return 0
    base_z = primary_int(pattern_slice.get("z"), 7)
    applied = 0

    for raw_layer_z, layer in z_layers.items():
        if not isinstance(layer, dict):
            continue
        layer_z = primary_int(raw_layer_z, base_z)
        z_offset = layer_z - base_z
        if z_offset == 0:
            continue
        layer_index = pattern_tile_index(layer)
        if not layer_index:
            continue
        for local_y in range(CHUNK_SIZE):
            for local_x in range(CHUNK_SIZE):
                dst_x = dst_origin_x + local_x
                dst_y = dst_origin_y + local_y
                if dst_x < 0 or dst_y < 0 or dst_x >= width or dst_y >= height:
                    continue
                raw = layer_index.get((source_x + local_x, source_y + local_y))
                if raw is None:
                    continue
                ground = raw_tile_ground(raw, fallback_ground)
                item_ids = tile_item_ids_from_raw(raw)
                ground, item_ids = apply_z_biome_rule(
                    ground,
                    item_ids,
                    z_offset,
                    pattern_tag,
                    local_x,
                    local_y,
                    macro_role,
                )
                if not ground and not item_ids:
                    continue
                output[(z_offset, dst_x, dst_y)] = MaterializedTile(dst_x, dst_y, ground, unique_items(item_ids))
                applied += 1
    return applied


def apply_z_biome_rule(
    ground_id: int,
    item_ids: list[int],
    z_offset: int,
    pattern_tag: str | None,
    local_x: int,
    local_y: int,
    macro_role: str,
) -> tuple[int, list[int]]:
    """Translate vertical biome data to avoid invalid floor inheritance."""

    if pattern_tag != "nature_surface":
        return ground_id, item_ids

    fixed_items = list(item_ids)
    if z_offset < 0:
        ground_id = UPPER_WOOD_PLATFORM_GROUND
        is_edge = local_x in {0, CHUNK_SIZE - 1} or local_y in {0, CHUNK_SIZE - 1}
        if macro_role in {"spawn_hub_dense", "camp_amenities"}:
            if not (set(fixed_items) & TREE_CANOPY_IDS):
                fixed_items.append(2701 + ((local_x + local_y) % 3))
        elif is_edge and macro_role in {"wild_surroundings", "defensive_perimeter"}:
            fixed_items.append(2703)
        return ground_id, unique_items(fixed_items)

    if z_offset > 0:
        ground_id = UNDERGROUND_NATURE_GROUND
        is_edge = local_x in {0, CHUNK_SIZE - 1} or local_y in {0, CHUNK_SIZE - 1}
        seed = (local_x * 17) + (local_y * 31) + (abs(z_offset) * 7)
        filtered_items: list[int] = []
        for item_id in fixed_items:
            if item_id in TREE_CANOPY_IDS or item_id in {2767, 2768} or item_id in STATIC_CORPSE_IDS:
                continue
            if item_id in DENSE_CAVE_BLOCKER_IDS:
                if macro_role == "defensive_perimeter" and is_edge and seed % 20 == 0:
                    filtered_items.append(item_id)
                continue
            filtered_items.append(item_id)
        fixed_items = filtered_items
        if not fixed_items:
            if macro_role == "defensive_perimeter" and is_edge and seed % 20 == 0:
                fixed_items.append(1285)
            elif seed % 7 == 0:
                fixed_items.append(RUSTIC_CAVE_DETAIL_IDS[seed % len(RUSTIC_CAVE_DETAIL_IDS)])
        return ground_id, unique_items(fixed_items)

    return ground_id, unique_items(fixed_items)


def fallback_macro_chunk(
    output: dict[tuple[int, int, int], MaterializedTile],
    macro_role: str,
    dst_origin_x: int,
    dst_origin_y: int,
    width: int,
    height: int,
    ground_id: int,
) -> int:
    """Fallback cuando no hay slice real disponible."""

    applied = 0
    for local_y in range(CHUNK_SIZE):
        for local_x in range(CHUNK_SIZE):
            dst_x = dst_origin_x + local_x
            dst_y = dst_origin_y + local_y
            if dst_x < 0 or dst_y < 0 or dst_x >= width or dst_y >= height:
                continue
            item_ids: list[int] = []
            if macro_role == "defensive_perimeter" and (local_x in {0, CHUNK_SIZE - 1} or local_y in {0, CHUNK_SIZE - 1}):
                item_ids = [3460]
            elif macro_role == "spawn_hub_dense" and 2 <= local_x <= 5 and 2 <= local_y <= 5:
                item_ids = [TENT_ROLE_ITEMS["tent_roof"][(local_x + local_y) % 3]]
            elif macro_role == "camp_amenities" and (local_x, local_y) in {(2, 2), (5, 4)}:
                item_ids = [1738]
            output[(0, dst_x, dst_y)] = MaterializedTile(dst_x, dst_y, ground_id, item_ids)
            applied += 1
    return applied


def structural_edge_score(tile: MaterializedTile | None) -> int:
    """Puntua si un tile de borde contiene estructura que debe continuar."""

    if tile is None:
        return 0
    ids = set(tile.item_ids)
    if ids & (ALL_WALL_ITEM_IDS | {1285, 1296, 1297, 1298, 1299, 3460, 3461, 3462}):
        return 3
    if ids:
        return 2
    if tile.ground_id in {101, 103, 444, 445, 919}:
        return 1
    return 0


def stitch_edge_pair(
    output: dict[tuple[int, int, int], MaterializedTile],
    left_key: tuple[int, int, int],
    right_key: tuple[int, int, int],
) -> int:
    """Stitch two adjacent tiles when one has structure and the other is sparse."""

    left = output.get(left_key)
    right = output.get(right_key)
    left_score = structural_edge_score(left)
    right_score = structural_edge_score(right)
    if left_score == right_score:
        return 0
    source = left if left_score > right_score else right
    target_key = right_key if left_score > right_score else left_key
    target = output.get(target_key)
    if source is None:
        return 0
    if target is None:
        _z, x, y = target_key
        output[target_key] = MaterializedTile(x, y, source.ground_id, [])
        target = output[target_key]
    if target.ground_id in {0, 4526, 724} or source.ground_id in {101, 103, 444, 445, 919}:
        target.ground_id = source.ground_id
    bridge_ids = [
        item_id
        for item_id in source.item_ids
        if item_id in (ALL_WALL_ITEM_IDS | {1285, 1296, 1297, 1298, 1299, 3460, 3461, 3462, 2767, 2768, 2701, 2702, 2703})
    ]
    before = list(target.item_ids)
    for item_id in bridge_ids[:2]:
        add_item(target, item_id)
    return 1 if before != target.item_ids or target.ground_id == source.ground_id else 0


def stitch_macro_chunk_edges(
    output: dict[tuple[int, int, int], MaterializedTile],
    width: int,
    height: int,
) -> None:
    """Cose fronteras compartidas entre chunks 8x8 en todos los planos Z."""

    repairs = 0
    z_offsets = sorted({z_offset for z_offset, _x, _y in output})
    for z_offset in z_offsets:
        for seam_x in range(CHUNK_SIZE, width, CHUNK_SIZE):
            for y in range(height):
                repairs += stitch_edge_pair(
                    output,
                    (z_offset, seam_x - 1, y),
                    (z_offset, seam_x, y),
                )
        for seam_y in range(CHUNK_SIZE, height, CHUNK_SIZE):
            for x in range(width):
                repairs += stitch_edge_pair(
                    output,
                    (z_offset, x, seam_y - 1),
                    (z_offset, x, seam_y),
                )
    if repairs:
        print(f"[autotiler] Edge Stitcher macro aplicado: repairs={repairs}, planes={z_offsets}")


def materialize_macro_chunk_map(
    semantic_tiles: list[SemanticTile],
    width: int,
    height: int,
    rme_rules: dict[str, Any],
    pattern_slice: dict[str, Any] | None,
) -> list[dict[str, int | list[int]]]:
    """Ensambla macro-zonas 8x8 usando chunks reales del slices_pool."""

    output: dict[tuple[int, int, int], MaterializedTile] = {}
    fallback_ground = dominant_ground_from_pattern_slice(pattern_slice, fallback=4526)
    pattern_tag = str((pattern_slice or {}).get("tag") or "")
    total = 0
    for tile in semantic_tiles:
        dst_x = tile.rel_x * CHUNK_SIZE
        dst_y = tile.rel_y * CHUNK_SIZE
        if tile.role == "wild_surroundings" and pattern_tag in CAVE_BIOME_TAGS:
            total += stamp_cellular_cave_chunk(
                output,
                pattern_tag,
                dst_x,
                dst_y,
                width,
                height,
                rme_rules,
                fallback_ground,
            )
            continue
        if pattern_slice:
            source_x, source_y = select_source_chunk(pattern_slice, tile.role)
            total += stamp_source_chunk(output, pattern_slice, tile.role, dst_x, dst_y, width, height, fallback_ground)
            total += stamp_source_chunk_layers(
                output,
                pattern_slice,
                tile.role,
                source_x,
                source_y,
                dst_x,
                dst_y,
                width,
                height,
                fallback_ground,
            )
        else:
            total += fallback_macro_chunk(output, tile.role, dst_x, dst_y, width, height, fallback_ground)

    stitch_macro_chunk_edges(output, width, height)
    base_output = {
        (x, y): tile
        for (z_offset, x, y), tile in output.items()
        if z_offset == 0
    }
    apply_composite_structure_linter(base_output, width, height, rme_rules, fallback_ground, margin=2)
    for (x, y), tile in base_output.items():
        output[(0, x, y)] = tile
    print(f"[autotiler] Macro-zone assembly applied: chunks={len(semantic_tiles)}, tiles={total}")
    return [
        {
            "x": tile.rel_x,
            "y": tile.rel_y,
            "ground_id": tile.ground_id,
            "item_ids": tile.item_ids,
            **({"z_offset": z_offset} if z_offset else {}),
        }
        for (z_offset, _x, _y), tile in sorted(output.items(), key=lambda entry: (entry[0][0], entry[1].rel_y, entry[1].rel_x))
    ]


def materialize_semantic_map(
    semantic_tiles: list[SemanticTile],
    width: int,
    height: int,
    rme_rules: dict[str, Any],
    archetype: dict[str, Any] | None = None,
    pattern_slice: dict[str, Any] | None = None,
) -> list[dict[str, int | list[int]]]:
    """Convert pure roles into tiles ready for inject_tiles."""

    if is_macro_plan(semantic_tiles):
        return materialize_macro_chunk_map(semantic_tiles, width, height, rme_rules, pattern_slice)

    index = semantic_index(semantic_tiles)
    pattern_tag = str((pattern_slice or {}).get("tag", "")) or None
    wall_family = wall_family_for_tag(pattern_tag)
    wall_ids = wall_ids_from_rule(get_wall_rule(rme_rules, wall_family))
    ruins_wall_ids = wall_ids_from_rule(get_wall_rule(rme_rules, wall_family_for_tag(pattern_tag, "wall_ruins")))
    output: dict[tuple[int, int], MaterializedTile] = {}
    if pattern_tag:
        print(f"[autotiler] Selected wall family: tag={pattern_tag}, wall_family={wall_family}")

    def ensure(x: int, y: int, ground_id: int) -> MaterializedTile | None:
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        key = (x, y)
        if key not in output:
            output[key] = MaterializedTile(x, y, ground_id, [])
        else:
            output[key].ground_id = ground_id
        return output[key]

    # First pass: base grounds and walls.
    for y in range(height):
        for x in range(width):
            role = role_at(index, x, y)
            if role == "empty":
                continue
            ground_id = 724 if role == "floor_exterior" else 424
            if role in TENT_ROLE_ITEMS:
                ground_id = 4526 if pattern_tag in {"nature_surface", "swamp"} else 424
            if role == "depot_walkway":
                ground_id = 426
            tile = ensure(x, y, ground_id)
            if tile is None:
                continue
            if role in {"wall", "wall_ruins", "door"}:
                add_item(tile, resolve_wall_item(index, x, y, wall_ids, ruins_wall_ids))
            elif role in TENT_ROLE_ITEMS:
                options = TENT_ROLE_ITEMS[role]
                add_item(tile, options[(x + y) % len(options)])
            elif role == "mailbox":
                add_item(tile, 2593)

    # Second pass: directional depot lockers, railings, and counters.
    north_locker_positions = sorted(
        (tile.rel_x, tile.rel_y) for tile in semantic_tiles if tile.role == "depot_locker_north"
    )
    east_locker_positions = sorted(
        (tile.rel_x, tile.rel_y) for tile in semantic_tiles if tile.role == "depot_locker_east"
    )
    railing_positions = sorted(
        (tile.rel_x, tile.rel_y) for tile in semantic_tiles if tile.role == "depot_railing"
    )
    counter_positions = {(tile.rel_x, tile.rel_y) for tile in semantic_tiles if tile.role == "npc_counter"}
    counter_positions.update(north_locker_positions)

    north_counter_rows: dict[int, set[int]] = {}
    for locker_x, locker_y in north_locker_positions:
        north_counter_rows.setdefault(locker_y, set()).add(locker_x)
    for row_y, row_positions in north_counter_rows.items():
        xs = sorted(row_positions)
        for left, right in zip(xs, xs[1:]):
            if right - left == 2:
                row_positions.add(left + 1)

    east_counter_columns: dict[int, set[int]] = {}
    for locker_x, locker_y in east_locker_positions:
        east_counter_columns.setdefault(locker_x, set()).add(locker_y)

    for x, y in north_locker_positions:
        counter_tile = ensure(x, y, 424)
        if counter_tile is not None:
            row_positions = north_counter_rows.get(y) or {cx for cx, cy in counter_positions if cy == y}
            add_item(counter_tile, counter_item_for_row(x, row_positions))
            add_item(counter_tile, 2591)

        walkway = ensure(x, y + 1, 426)
        if walkway is not None:
            walkway.ground_id = 426

    for x, y in north_locker_positions:
        right_is_locker = (x + 2, y) in north_locker_positions
        if right_is_locker:
            bridge = ensure(x + 1, y, 424)
            if bridge is not None:
                row_positions = north_counter_rows.get(y) or {cx for cx, cy in counter_positions if cy == y}
                add_item(bridge, counter_item_for_row(x + 1, row_positions))
            divider = ensure(x + 1, y + 1, 424)
            if divider is not None:
                add_item(divider, 1526)
            divider_south = ensure(x + 1, y + 2, 424)
            if divider_south is not None:
                add_item(divider_south, 1526)

    for x, y in east_locker_positions:
        counter_tile = ensure(x, y, 424)
        if counter_tile is not None:
            column_positions = east_counter_columns.get(x) or {cy for cx, cy in east_locker_positions if cx == x}
            add_item(counter_tile, vertical_depot_counter_item(y, column_positions))
            add_item(counter_tile, 2592)

        walkway = ensure(x - 1, y, 426)
        if walkway is not None:
            walkway.ground_id = 426

    for x, y in east_locker_positions:
        next_is_locker = (x, y + 2) in east_locker_positions
        if next_is_locker:
            bridge = ensure(x, y + 1, 424)
            if bridge is not None:
                column_positions = east_counter_columns.get(x) or {cy for cx, cy in east_locker_positions if cx == x}
                add_item(bridge, vertical_depot_counter_item(y + 1, column_positions))
            divider = ensure(x - 1, y + 1, 424)
            if divider is not None:
                add_item(divider, 1526)
            divider_west = ensure(x - 2, y + 1, 424)
            if divider_west is not None:
                add_item(divider_west, 1526)

    for x, y in railing_positions:
        tile = ensure(x, y, 424)
        if tile is not None:
            add_item(tile, 1526)

    locker_positions = set(north_locker_positions) | set(east_locker_positions)
    for x, y in sorted(counter_positions - locker_positions):
        tile = ensure(x, y, 424)
        if tile is None:
            continue
        row_positions = {cx for cx, cy in counter_positions if cy == y}
        add_item(tile, counter_item_for_row(x, row_positions))

    apply_real_slice_pattern(output, semantic_tiles, width, height, pattern_slice)
    apply_depot_archetype_pattern(output, semantic_tiles, width, height, archetype)
    enforce_wall_family_contiguity(output, semantic_tiles, wall_family, wall_ids)
    margin_ground_id = dominant_ground_from_pattern_slice(pattern_slice, fallback=424)
    apply_composite_structure_linter(output, width, height, rme_rules, margin_ground_id, margin=2)

    return [
        {
            "x": tile.rel_x,
            "y": tile.rel_y,
            "ground_id": tile.ground_id,
            "item_ids": tile.item_ids,
        }
        for tile in sorted(output.values(), key=lambda entry: (entry.rel_y, entry.rel_x))
    ]
