"""
Mass slicer for real OTBM fragments used by pattern-based inference.

Scans a large map with mmap, groups tiles into sliding windows, and exports
a compact pool of classified fragments so the server can search real patterns
by similarity.

Usage from the RME root:
    python ai_generator/map_slicer.py --size 16
    python ai_generator/map_slicer.py --size 24 --stride 12 --z 7
"""

from __future__ import annotations

import argparse
import json
import mmap
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from extractor import (
    NODE_END,
    NODE_START,
    OTBM_ITEM,
    OTBM_TILE,
    OTBM_TILE_AREA,
    OTSYS_ROOT,
    NodeHeader,
    decode_item_id,
    decode_tile_area_props,
    decode_tile_ground_from_props,
    parse_node_header,
    resolve_relative,
    skip_node,
)


SOURCE_OTBM_PATH = "../../template/real map/world.otbm"
OUTPUT_PATH = BASE_DIR / "slices_pool.json"
OUTPUT_JSONL_PATH = BASE_DIR / "slices_pool.jsonl"
ITEMS_XML_CANDIDATES = (
    BASE_DIR / "../../data/760/items.xml",
    BASE_DIR / "../data/760/items.xml",
)
SPAWN_XML_CANDIDATES = (
    BASE_DIR / "../../world-spawn.xml",
    BASE_DIR / "../world-spawn.xml",
    BASE_DIR / "../../template/real map/world-spawn.xml",
    BASE_DIR / "../template/real map/world-spawn.xml",
)
CONNECTOR_IDS = {1385, 411, 369, 370, 408, 409, 427, 429, 461, 462, 924}

ICE_CAVE_GROUNDS = {101}
DEPOT_IDS = {426, 2591}
DIRT_CAVE_GROUNDS = {103, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366}
STRUCTURAL_IDS = {1526, 1617, 1618, 1621, 1622, 1623, 2591, 2592, 2593}
BLOCKING_ATTRIBUTE_KEYS = {"unpassable", "blockpathfinder", "has_elevation"}
TYPE_ATTRIBUTE_VALUES = {"door", "depot", "mailbox"}
MANUAL_BLOCKING_IDS = {
    101,  # solid filler used by ice caves in the real map
    460,  # agua profunda / oceano
    1526,  # stone railing
    1617,
    1618,
    1621,
    1622,
    1623,
    2591,
    2592,
}


@dataclass(slots=True, frozen=True)
class ItemProperties:
    """Properties relevant to collision and semantics."""

    item_id: int
    name: str = ""
    attributes: frozenset[str] = frozenset()
    item_type: str = ""
    unpassable: bool = False
    blockpathfinder: bool = False
    has_elevation: bool = False

    @property
    def blocks_movement(self) -> bool:
        """Return whether the item blocks slice walkability."""

        return self.unpassable or self.blockpathfinder or self.has_elevation


@dataclass(slots=True)
class TileRecord:
    """Compact tile from the real map."""

    x: int
    y: int
    z: int
    ground_id: int | None
    item_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class WindowAccumulator:
    """Accumulate tiles for a sliding window."""

    origin_x: int
    origin_y: int
    z: int
    tiles: list[TileRecord] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class SpawnPoint:
    """Exact spawn used for targeted eco-slicing."""

    creature: str
    x: int
    y: int
    z: int
    center_x: int
    center_y: int
    center_z: int
    radius: int


def resolve_items_xml() -> Path | None:
    """Find data/760/items.xml with relative fallbacks."""

    for candidate in ITEMS_XML_CANDIDATES:
        path = candidate.resolve()
        if path.is_file():
            return path
    return None


def resolve_spawn_xml() -> Path | None:
    """Find world-spawn.xml in the root or template/real map."""

    for candidate in SPAWN_XML_CANDIDATES:
        path = candidate.resolve()
        if path.is_file():
            return path
    return None


def normalize_spawn_z(raw_z: int) -> int:
    """Normalize legacy spawn XML Z values to map coordinates used by RME."""

    return 7 if raw_z == 1 else raw_z


def load_target_spawns() -> list[SpawnPoint]:
    """Load world-spawn.xml and return spawns for every creature."""

    spawn_xml = resolve_spawn_xml()
    if spawn_xml is None:
        print("[slicer] world-spawn.xml not found; targeted mode has no spawns")
        return []

    output: list[SpawnPoint] = []
    try:
        root = ElementTree.parse(spawn_xml).getroot()
        for spawn in root.findall("spawn"):
            try:
                center_x = int(spawn.attrib.get("centerx", "0"))
                center_y = int(spawn.attrib.get("centery", "0"))
                center_z_raw = int(spawn.attrib.get("centerz", "0"))
            except ValueError:
                continue
            center_z = normalize_spawn_z(center_z_raw)
            radius = int(spawn.attrib.get("radius", "0") or "0")
            for child in list(spawn):
                if child.tag != "monster":
                    continue
                creature = (child.attrib.get("name") or "").strip().lower()
                if not creature:
                    continue
                try:
                    offset_x = int(child.attrib.get("x", "0"))
                    offset_y = int(child.attrib.get("y", "0"))
                    child_z_raw = int(child.attrib.get("z", str(center_z_raw)))
                except ValueError:
                    offset_x = offset_y = 0
                    child_z_raw = center_z_raw
                output.append(
                    SpawnPoint(
                        creature=creature,
                        x=center_x + offset_x,
                        y=center_y + offset_y,
                        z=normalize_spawn_z(child_z_raw),
                        center_x=center_x,
                        center_y=center_y,
                        center_z=center_z,
                        radius=radius,
                    )
                )
    except ElementTree.ParseError as exc:
        print(f"[slicer] invalid world-spawn.xml: {spawn_xml}: {exc}")
        return []

    unique_creatures = {spawn.creature for spawn in output}
    print(
        "[slicer] Targeted spawns loaded: "
        f"{len(output)} instances, {len(unique_creatures)} creatures from {spawn_xml}"
    )
    return output


def truthy_attribute(value: str | None) -> bool:
    """Interpreta valores XML comunes como booleanos."""

    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "none"}


def load_item_properties() -> dict[int, ItemProperties]:
    """Load items.xml properties indexed by ID."""

    items_xml = resolve_items_xml()
    if items_xml is None:
        print("[slicer] items.xml not found; collision map will use minimal heuristics")
        return {}

    properties: dict[int, ItemProperties] = {}
    try:
        iterator = ElementTree.iterparse(items_xml, events=("end",))
        for _event, elem in iterator:
            if elem.tag != "item":
                elem.clear()
                continue

            raw_id = elem.attrib.get("id")
            if raw_id is None:
                elem.clear()
                continue
            try:
                item_id = int(raw_id)
            except ValueError:
                elem.clear()
                continue

            name = elem.attrib.get("name", "")
            attributes: set[str] = set()
            item_type = elem.attrib.get("type", "")
            unpassable = truthy_attribute(elem.attrib.get("unpassable")) if "unpassable" in elem.attrib else False
            blockpathfinder = (
                truthy_attribute(elem.attrib.get("blockpathfinder")) if "blockpathfinder" in elem.attrib else False
            )
            has_elevation = truthy_attribute(elem.attrib.get("has_elevation")) if "has_elevation" in elem.attrib else False

            for attr in elem.findall("attribute"):
                key = (attr.attrib.get("key") or "").strip().lower()
                value = (attr.attrib.get("value") or "").strip().lower()
                if not key:
                    continue
                attributes.add(key)
                if key == "type":
                    item_type = value
                elif key == "unpassable":
                    unpassable = truthy_attribute(value)
                elif key == "blockpathfinder":
                    blockpathfinder = truthy_attribute(value)
                elif key == "has_elevation":
                    has_elevation = truthy_attribute(value)

            lowered_name = name.lower()
            if any(token in lowered_name for token in ("wall", "rock", "mountain wall")):
                unpassable = True
                blockpathfinder = True
            if item_type in TYPE_ATTRIBUTE_VALUES:
                attributes.add(item_type)

            properties[item_id] = ItemProperties(
                item_id=item_id,
                name=name,
                attributes=frozenset(attributes),
                item_type=item_type,
                unpassable=unpassable,
                blockpathfinder=blockpathfinder,
                has_elevation=has_elevation,
            )
            elem.clear()
    except ElementTree.ParseError as exc:
        print(f"[slicer] invalid items.xml for collision map: {items_xml}: {exc}")
        return {}

    print(f"[slicer] Loaded item properties: {items_xml} ({len(properties)} IDs)")
    return properties


ITEM_PROPERTIES = load_item_properties()


def item_blocks_movement(item_id: int, item_properties: dict[int, ItemProperties] | None = None) -> bool:
    """Return whether an ID blocks movement according to items.xml."""

    if item_id in MANUAL_BLOCKING_IDS:
        return True
    properties = item_properties or ITEM_PROPERTIES
    props = properties.get(item_id)
    if props is None:
        return False
    return props.blocks_movement


def is_tile_walkable(tile: TileRecord, item_properties: dict[int, ItemProperties] | None = None) -> bool:
    """Un tile es caminable si ground e items no bloquean movimiento."""

    if tile.ground_id is None:
        return False
    if item_blocks_movement(tile.ground_id, item_properties):
        return False
    return not any(item_blocks_movement(item_id, item_properties) for item_id in tile.item_ids)


def parse_tile_items(data: bytes, tile_header: NodeHeader) -> tuple[list[int], int]:
    """Lee hijos OTBM_ITEM directos de un tile."""

    item_ids: list[int] = []
    cursor = tile_header.children_offset
    while cursor < len(data):
        control = data[cursor]
        if control == NODE_START:
            child_header = parse_node_header(data, cursor)
            if child_header.node_type == OTBM_ITEM:
                item_id = decode_item_id(child_header.props)
                if item_id is not None:
                    item_ids.append(item_id)
            cursor = skip_node(data, cursor)
        elif control == NODE_END:
            return item_ids, cursor + 1
        else:
            raise ValueError(f"Byte OTBM inesperado 0x{control:02X} en tile offset {cursor}")
    raise ValueError("Tile OTBM sin cierre NODE_END")


def iter_tiles_in_area(
    data: bytes,
    area_header: NodeHeader,
    area_base_x: int,
    area_base_y: int,
    area_z: int,
) -> tuple[list[TileRecord], int]:
    """Materialize compact tiles from a TILE_AREA."""

    records: list[TileRecord] = []
    cursor = area_header.children_offset
    while cursor < len(data):
        control = data[cursor]
        if control == NODE_START:
            child_header = parse_node_header(data, cursor)
            if child_header.node_type != OTBM_TILE or len(child_header.props) < 2:
                cursor = skip_node(data, cursor)
                continue

            abs_x = area_base_x + child_header.props[0]
            abs_y = area_base_y + child_header.props[1]
            child_item_ids, cursor = parse_tile_items(data, child_header)
            prop_ground_id = decode_tile_ground_from_props(child_header.props)
            if prop_ground_id is not None:
                ground_id = prop_ground_id
                stacked_items = child_item_ids
            else:
                ground_id = child_item_ids[0] if child_item_ids else None
                stacked_items = child_item_ids[1:] if child_item_ids else []
            records.append(TileRecord(abs_x, abs_y, area_z, ground_id, stacked_items))
        elif control == NODE_END:
            return records, cursor + 1
        else:
            raise ValueError(f"Byte OTBM inesperado 0x{control:02X} en area offset {cursor}")
    raise ValueError("TILE_AREA sin cierre NODE_END")


def scan_tiles(data: bytes, offset: int, z_filter: set[int] | None = None) -> tuple[list[TileRecord], int]:
    """Traverse the OTBM tree and return all compact tiles."""

    header = parse_node_header(data, offset)
    records: list[TileRecord] = []

    if header.node_type == OTBM_TILE_AREA:
        coords = decode_tile_area_props(header.props)
        if coords is not None:
            area_base_x, area_base_y, area_z = coords
            if z_filter is None or area_z in z_filter:
                area_records, end_offset = iter_tiles_in_area(data, header, area_base_x, area_base_y, area_z)
                records.extend(area_records)
                return records, end_offset
        return records, skip_node(data, offset)

    cursor = header.children_offset
    while cursor < len(data):
        control = data[cursor]
        if control == NODE_START:
            child_records, cursor = scan_tiles(data, cursor, z_filter)
            records.extend(child_records)
        elif control == NODE_END:
            return records, cursor + 1
        else:
            raise ValueError(f"Byte OTBM inesperado 0x{control:02X} en offset {cursor}")
    raise ValueError("Nodo OTBM sin cierre NODE_END")


def load_all_tiles(path: Path, z_filter: set[int] | None = None) -> list[TileRecord]:
    """Open the map with mmap and extract all compact tiles."""

    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if len(data) < 6:
                raise ValueError(f"Archivo OTBM demasiado pequeno: {path}")
            if data[4] != NODE_START or data[5] != OTSYS_ROOT:
                raise ValueError("El archivo no parece tener root OTBM valido en offset 4")
            records, _end_offset = scan_tiles(data, 4, z_filter)
    return records


def window_origins_for_tile(value: int, size: int, stride: int) -> Iterable[int]:
    """Return sliding-window origins containing a coordinate."""

    base = (value // stride) * stride
    for origin in range(base - size + stride, base + stride, stride):
        if origin <= value < origin + size:
            yield origin


def accumulate_windows(
    tiles: list[TileRecord],
    size: int,
    stride: int,
) -> dict[tuple[int, int, int], WindowAccumulator]:
    """Agrupa cada tile en las ventanas deslizantes que lo contienen."""

    windows: dict[tuple[int, int, int], WindowAccumulator] = {}
    for tile in tiles:
        for origin_x in window_origins_for_tile(tile.x, size, stride):
            for origin_y in window_origins_for_tile(tile.y, size, stride):
                key = (tile.z, origin_x, origin_y)
                if key not in windows:
                    windows[key] = WindowAccumulator(origin_x=origin_x, origin_y=origin_y, z=tile.z)
                windows[key].tiles.append(tile)
    return windows


def classify_window(
    tiles: list[TileRecord],
    min_tiles: int,
    item_properties: dict[int, ItemProperties] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Classify a fragment by dominant IDs."""

    if len(tiles) < min_tiles:
        return None, {}

    biome_signatures: dict[str, set[int]] = {
        "ice_cave": {101, *range(351, 368)},
        "dirt_cave": {103, 350, *range(368, 381)},
        "stone_mountain": {444, 445, 919},
        "desert": {231, 232, 233, 234},
        "swamp": {*range(4691, 4700), 4712},
        "nature_surface": {4526},
        "urban_floor": {424, 426, 724, 1049, 1050},
    }
    water_or_empty_ids = {0, 460}

    ground_counter: Counter[int] = Counter()
    item_counter: Counter[int] = Counter()
    walkable_count = 0
    for tile in tiles:
        ground_id = tile.ground_id if tile.ground_id is not None else 0
        ground_counter[ground_id] += 1
        item_counter.update(tile.item_ids)
        if is_tile_walkable(tile, item_properties):
            walkable_count += 1

    total_ground = sum(ground_counter.values())
    if total_ground <= 0:
        return None, {}

    if sum(ground_counter[item_id] for item_id in water_or_empty_ids) == total_ground:
        return None, {}
    if walkable_count == 0:
        return None, {}

    depot_score = sum(ground_counter[item_id] for item_id in DEPOT_IDS) + sum(
        item_counter[item_id] for item_id in DEPOT_IDS
    )

    if depot_score >= 2:
        label = "depot"
    else:
        signature_scores = {
            label: sum(ground_counter[item_id] for item_id in ground_ids)
            for label, ground_ids in biome_signatures.items()
        }
        label, best_score = max(signature_scores.items(), key=lambda entry: entry[1])
        if best_score <= 0:
            return None, {}

        dominant_ground, dominant_count = ground_counter.most_common(1)[0]
        dominant_label = next(
            (
                signature_label
                for signature_label, ground_ids in biome_signatures.items()
                if dominant_ground in ground_ids
            ),
            None,
        )
        if dominant_label is not None:
            label = dominant_label

        if max(best_score, dominant_count) / total_ground < 0.25:
            return None, {}

    if label is None:
        return None, {}

    signature_scores = {
        signature_label: sum(ground_counter[item_id] for item_id in ground_ids)
        for signature_label, ground_ids in biome_signatures.items()
    }
    stats = {
        "tile_count": len(tiles),
        "ground_top": ground_counter.most_common(8),
        "item_top": item_counter.most_common(8),
        "dominant_ground": ground_counter.most_common(1)[0],
        "walkable_count": walkable_count,
        "collision_density": round(1.0 - (walkable_count / max(1, len(tiles))), 4),
        "scores": {
            "depot": depot_score,
            **signature_scores,
            "total_ground": total_ground,
        },
    }
    return label, stats


def compact_window(
    acc: WindowAccumulator,
    size: int,
    label: str,
    stats: dict[str, Any],
    item_properties: dict[int, ItemProperties] | None = None,
) -> dict[str, Any]:
    """Convert a classified window into compact JSON."""

    compact_tiles: list[dict[str, Any]] = []
    walkable_path = [0] * (size * size)
    for tile in sorted(acc.tiles, key=lambda entry: (entry.y, entry.x)):
        rel_x = tile.x - acc.origin_x
        rel_y = tile.y - acc.origin_y
        if rel_x < 0 or rel_y < 0 or rel_x >= size or rel_y >= size:
            continue
        entry: dict[str, Any] = {"x": rel_x, "y": rel_y}
        if tile.ground_id is not None:
            entry["g"] = tile.ground_id
        if tile.item_ids:
            entry["i"] = tile.item_ids
        if is_tile_walkable(tile, item_properties):
            entry["w"] = 1
            walkable_path[rel_y * size + rel_x] = 1
        compact_tiles.append(entry)

    walkable_count = sum(walkable_path)
    total_cells = size * size
    collision_density = round(1.0 - (walkable_count / max(1, total_cells)), 4)

    return {
        "tag": label,
        "z": acc.z,
        "origin": {"x": acc.origin_x, "y": acc.origin_y},
        "width": size,
        "height": size,
        "tile_count": len(compact_tiles),
        "walkable_count": walkable_count,
        "collision_density": collision_density,
        "walkable_path": walkable_path,
        "stats": stats,
        "tiles": compact_tiles,
    }


def compact_layer(
    acc: WindowAccumulator,
    size: int,
    item_properties: dict[int, ItemProperties] | None = None,
) -> dict[str, Any]:
    """Compact a single Z layer for z_layers."""

    compact_tiles: list[dict[str, Any]] = []
    walkable_path = [0] * (size * size)
    connector_positions: list[dict[str, int]] = []
    for tile in sorted(acc.tiles, key=lambda entry: (entry.y, entry.x)):
        rel_x = tile.x - acc.origin_x
        rel_y = tile.y - acc.origin_y
        if rel_x < 0 or rel_y < 0 or rel_x >= size or rel_y >= size:
            continue
        entry: dict[str, Any] = {"x": rel_x, "y": rel_y}
        ids = set(tile.item_ids)
        if tile.ground_id is not None:
            entry["g"] = tile.ground_id
            ids.add(tile.ground_id)
        if tile.item_ids:
            entry["i"] = tile.item_ids
        if is_tile_walkable(tile, item_properties):
            entry["w"] = 1
            walkable_path[rel_y * size + rel_x] = 1
        if ids & CONNECTOR_IDS:
            connector_positions.append({"x": rel_x, "y": rel_y, "ids": sorted(ids & CONNECTOR_IDS)})
        compact_tiles.append(entry)

    walkable_count = sum(walkable_path)
    return {
        "z": acc.z,
        "origin": {"x": acc.origin_x, "y": acc.origin_y},
        "width": size,
        "height": size,
        "tile_count": len(compact_tiles),
        "walkable_count": walkable_count,
        "collision_density": round(1.0 - (walkable_count / max(1, size * size)), 4),
        "walkable_path": walkable_path,
        "connectors": connector_positions,
        "tiles": compact_tiles,
    }


def layer_has_structure(layer: dict[str, Any]) -> bool:
    """Decide whether a neighboring layer is worth saving."""

    if layer.get("walkable_count", 0) > 0:
        return True
    if layer.get("connectors"):
        return True
    tiles = layer.get("tiles", [])
    if not isinstance(tiles, list):
        return False
    return any(tile.get("i") for tile in tiles if isinstance(tile, dict))


def build_slices_pool(
    source_path: Path,
    size: int = 16,
    stride: int | None = None,
    min_fill_ratio: float = 0.12,
    z_filter: set[int] | None = None,
    max_slices_per_tag: int | None = None,
    item_properties: dict[int, ItemProperties] | None = None,
) -> dict[str, Any]:
    """Scan the map and generate classified slices."""

    if size <= 0:
        raise ValueError("size debe ser mayor que cero")
    if stride is None:
        stride = max(1, size // 2)
    if stride <= 0:
        raise ValueError("stride debe ser mayor que cero")

    min_tiles = max(1, int(size * size * min_fill_ratio))
    print(f"[slicer] Mapa fuente: {source_path}")
    print(f"[slicer] Configuration: size={size}, stride={stride}, min_tiles={min_tiles}, z_filter={z_filter}")
    properties = item_properties or ITEM_PROPERTIES

    tiles = load_all_tiles(source_path, z_filter)
    print(f"[slicer] Tiles read: {len(tiles)}")

    windows = accumulate_windows(tiles, size=size, stride=stride)
    print(f"[slicer] Ventanas candidatas: {len(windows)}")

    slices_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned = 0
    skipped_by_collision = 0
    for acc in windows.values():
        scanned += 1
        label, stats = classify_window(acc.tiles, min_tiles=min_tiles, item_properties=properties)
        if label is None:
            if len(acc.tiles) >= min_tiles and not any(is_tile_walkable(tile, properties) for tile in acc.tiles):
                skipped_by_collision += 1
            continue
        if max_slices_per_tag is not None and len(slices_by_tag[label]) >= max_slices_per_tag:
            continue
        slices_by_tag[label].append(
            compact_window(acc, size=size, label=label, stats=stats, item_properties=properties)
        )
        if scanned % 5000 == 0:
            counts = {tag: len(entries) for tag, entries in slices_by_tag.items()}
            print(
                "[slicer] Progreso "
                f"ventanas={scanned}, pools={counts}, skipped_collision={skipped_by_collision}"
            )

    counts = {tag: len(entries) for tag, entries in sorted(slices_by_tag.items())}
    print(f"[slicer] Fragmentos descubiertos: {counts}")
    print(f"[slicer] Fragments skipped due to full collision: {skipped_by_collision}")

    return {
        "source": str(source_path),
        "config": {
            "size": size,
            "stride": stride,
            "min_fill_ratio": min_fill_ratio,
            "z_filter": sorted(z_filter) if z_filter is not None else None,
            "max_slices_per_tag": max_slices_per_tag,
            "collision_map": "items.xml",
        },
        "summary": counts,
        "skipped_by_collision": skipped_by_collision,
        "slices": dict(slices_by_tag),
    }


def index_tiles_by_z(tiles: list[TileRecord]) -> dict[int, list[TileRecord]]:
    """Group tiles by Z floor for targeted cuts."""

    by_z: dict[int, list[TileRecord]] = defaultdict(list)
    for tile in tiles:
        by_z[tile.z].append(tile)
    return dict(by_z)


def tiles_in_centered_window(
    tiles_by_z: dict[int, list[TileRecord]],
    center_x: int,
    center_y: int,
    z: int,
    size: int,
) -> WindowAccumulator:
    """Recorta ventana cuadrada centrada en coordenada de spawn."""

    origin_x = center_x - size // 2
    origin_y = center_y - size // 2
    max_x = origin_x + size - 1
    max_y = origin_y + size - 1
    acc = WindowAccumulator(origin_x=origin_x, origin_y=origin_y, z=z)
    for tile in tiles_by_z.get(z, []):
        if origin_x <= tile.x <= max_x and origin_y <= tile.y <= max_y:
            acc.tiles.append(tile)
    return acc


def build_targeted_slices_pool(
    source_path: Path,
    output_path: Path,
    size: int = 16,
    min_fill_ratio: float = 0.12,
    z_filter: set[int] | None = None,
    max_slices_per_tag: int | None = None,
    item_properties: dict[int, ItemProperties] | None = None,
) -> dict[str, Any]:
    """Generate targeted slices and write them incrementally as JSONL."""

    if size <= 0:
        raise ValueError("size debe ser mayor que cero")

    target_spawns = load_target_spawns()
    if z_filter is not None:
        target_spawns = [spawn for spawn in target_spawns if spawn.z in z_filter]
    z_needed = {
        layer_z
        for spawn in target_spawns
        for layer_z in (spawn.z - 1, spawn.z, spawn.z + 1)
        if 0 <= layer_z <= 15
    }
    if not z_needed:
        print("[slicer] No hay spawns targeted para los filtros solicitados")

    min_tiles = max(1, int(size * size * min_fill_ratio))
    properties = item_properties or ITEM_PROPERTIES
    print(f"[slicer] Mapa fuente: {source_path}")
    print(f"[slicer] Targeted mode: size={size}, min_tiles={min_tiles}, z_filter={z_filter}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    print(f"[slicer] Escritura incremental JSONL: {output_path}")

    tiles = load_all_tiles(source_path, z_needed or z_filter)
    tiles_by_z = index_tiles_by_z(tiles)
    print(f"[slicer] Tiles read for targeted mode: {len(tiles)}")

    counts_by_tag: Counter[str] = Counter()
    skipped_by_collision = 0
    scanned = 0
    seen_windows: set[tuple[str, int, int, int]] = set()
    with output_path.open("a", encoding="utf-8") as writer:
        for spawn in target_spawns:
            scanned += 1
            base_acc = tiles_in_centered_window(tiles_by_z, spawn.x, spawn.y, spawn.z, size)
            label, stats = classify_window(base_acc.tiles, min_tiles=min_tiles, item_properties=properties)
            if label is None:
                if len(base_acc.tiles) >= min_tiles and not any(is_tile_walkable(tile, properties) for tile in base_acc.tiles):
                    skipped_by_collision += 1
                continue
            key = (label, base_acc.z, base_acc.origin_x, base_acc.origin_y)
            if key in seen_windows:
                continue
            seen_windows.add(key)
            if max_slices_per_tag is not None and counts_by_tag[label] >= max_slices_per_tag:
                continue

            z_layers: dict[str, Any] = {}
            for layer_z in (spawn.z - 1, spawn.z, spawn.z + 1):
                if not 0 <= layer_z <= 15:
                    continue
                layer_acc = tiles_in_centered_window(tiles_by_z, spawn.x, spawn.y, layer_z, size)
                layer = compact_layer(layer_acc, size=size, item_properties=properties)
                if layer_z == spawn.z or layer_has_structure(layer):
                    z_layers[str(layer_z)] = layer

            entry = compact_window(base_acc, size=size, label=label, stats=stats, item_properties=properties)
            entry["targeted"] = True
            entry["multilayer"] = True
            entry["z_layers"] = z_layers
            entry["layer_range"] = [spawn.z - 1, spawn.z, spawn.z + 1]
            entry["spawn"] = {
                "creature": spawn.creature,
                "x": spawn.x,
                "y": spawn.y,
                "z": spawn.z,
                "centerx": spawn.center_x,
                "centery": spawn.center_y,
                "centerz": spawn.center_z,
                "radius": spawn.radius,
            }
            entry["fidelity"] = "spawn_density_targeted"
            writer.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            writer.flush()
            counts_by_tag[label] += 1
            print(
                "[slicer] Hot-saved slice for creature: "
                f"{spawn.creature} at coordinates ({spawn.x},{spawn.y},{spawn.z})"
            )
            if scanned % 500 == 0:
                print(f"[slicer] Targeted progreso spawns={scanned}, pools={dict(counts_by_tag)}")

    counts = dict(sorted(counts_by_tag.items()))
    print(f"[slicer] Targeted fragments discovered: {counts}")
    print(f"[slicer] Targeted slices skipped due to full collision: {skipped_by_collision}")

    return {
        "source": str(source_path),
        "config": {
            "mode": "targeted",
            "size": size,
            "stride": None,
            "min_fill_ratio": min_fill_ratio,
            "z_filter": sorted(z_filter) if z_filter is not None else None,
            "max_slices_per_tag": max_slices_per_tag,
            "collision_map": "items.xml",
            "target_creatures": "all_monsters",
            "z_layers": "spawn_z_minus_1_to_plus_1",
        },
        "summary": counts,
        "skipped_by_collision": skipped_by_collision,
        "target_spawn_count": len(target_spawns),
        "output_jsonl": str(output_path),
    }


def parse_z_filter(values: list[str] | None) -> set[int] | None:
    """Parsea lista de pisos z."""

    if not values:
        return None
    result: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            result.add(int(part))
    return result


def parse_args() -> argparse.Namespace:
    """Argumentos CLI."""

    parser = argparse.ArgumentParser(description="Generate slices_pool.json or slices_pool.jsonl from world.otbm")
    parser.add_argument("--source", default=SOURCE_OTBM_PATH, help="Ruta relativa/absoluta del world.otbm")
    parser.add_argument("--output", default=None, help="Ruta de salida; targeted usa JSONL por defecto")
    parser.add_argument("--mode", choices=("sequential", "targeted"), default="sequential", help="Modo de slicing")
    parser.add_argument("--size", type=int, default=16, help="Tamano de ventana cuadrada")
    parser.add_argument("--stride", type=int, default=None, help="Paso de sliding window; default=size//2")
    parser.add_argument("--min-fill-ratio", type=float, default=0.12, help="Minimum ratio of non-empty tiles")
    parser.add_argument("--z", nargs="*", default=None, help="Filtrar pisos z, ejemplo: --z 7 8 9 o --z 7,8,9")
    parser.add_argument("--max-slices-per-tag", type=int, default=None, help="Optional limit per classification")
    return parser.parse_args()


def main() -> None:
    """Punto de entrada CLI."""

    args = parse_args()
    source_path = resolve_relative(args.source)
    if not source_path.is_file():
        fallback = resolve_relative("../template/real map/world.otbm")
        if fallback.is_file():
            source_path = fallback
        else:
            raise FileNotFoundError(f"No existe world.otbm: {source_path}")

    default_output = OUTPUT_JSONL_PATH if args.mode == "targeted" else OUTPUT_PATH
    output_path = Path(args.output) if args.output else default_output
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z_filter = parse_z_filter(args.z)
    if args.mode == "targeted":
        summary = build_targeted_slices_pool(
            source_path=source_path,
            output_path=output_path,
            size=args.size,
            min_fill_ratio=args.min_fill_ratio,
            z_filter=z_filter,
            max_slices_per_tag=args.max_slices_per_tag,
        )
        print(f"[slicer] slices_pool.jsonl escrito incrementalmente: {output_path}")
        print(f"[slicer] resumen targeted: {json.dumps(summary.get('summary', {}), ensure_ascii=True)}")
        return

    pool = build_slices_pool(
        source_path=source_path,
        size=args.size,
        stride=args.stride,
        min_fill_ratio=args.min_fill_ratio,
        z_filter=z_filter,
        max_slices_per_tag=args.max_slices_per_tag,
    )
    output_path.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[slicer] slices_pool.json escrito: {output_path}")


if __name__ == "__main__":
    main()
