"""
Local FastAPI API for Tibia 7.60 map generation with Google GenAI.

Run from the RME root:
    uvicorn ai_generator.server:app --reload --host 127.0.0.1 --port 8000

Required variable:
    GEMINI_API_KEY
"""

from __future__ import annotations

import json
import os
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, List
from xml.etree import ElementTree

from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from .autotiler import CHUNK_SIZE, MACRO_ROLES, SEMANTIC_ROLES, SemanticTile, materialize_semantic_map
from .injector import inject_tiles, resolve_relative
from .map_renderer import build_visual_feedback_payload, render_debug_map
from .rme_parser import (
    build_prompt_geometry_rules,
    format_geometry_rules_for_prompt,
    load_rme_geometry_rules,
)


BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "tibia_760_catalog.json"
ARCHETYPES_PATH = BASE_DIR / "archetypes.json"
SLICES_POOL_PATH = BASE_DIR / "slices_pool.json"
SLICES_POOL_JSONL_PATH = BASE_DIR / "slices_pool.jsonl"
SPAWN_XML_CANDIDATES = (
    BASE_DIR / "../../world-spawn.xml",
    BASE_DIR / "../world-spawn.xml",
    BASE_DIR / "../../template/real map/world-spawn.xml",
    BASE_DIR / "../template/real map/world-spawn.xml",
)
ITEMS_XML_CANDIDATES = (
    BASE_DIR / "../../data/760/items.xml",
    BASE_DIR / "../data/760/items.xml",
)
REFERENCE_IMAGES_DIR = BASE_DIR / "references"
TEMPLATE_PATH = "../../template/base_760.otbm"
OUTPUT_PATH = "../../template/generated_760.otbm"
GENERATED_SPAWN_PATH = "../../template/generated_760-spawn.xml"

START_X = 118
START_Y = 123
TARGET_Z = 7
MAX_DIMENSION = 30
MAX_TILES = MAX_DIMENSION * MAX_DIMENSION
MODEL_NAME = "gemini-3.5-flash"
AVAILABLE_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
DEFAULT_ARCHETYPE_CANDIDATES = ("thais_depot_p1", "thais_temple", "temple_stone")

TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "temple": ("templo", "temple", "santuario", "sacred", "sagrado", "altar", "capilla"),
    "shop": ("tienda", "shop", "mercado", "merchant", "vendedor", "taberna", "tavern", "bar"),
    "wood": ("madera", "wood", "wooden", "casa", "house", "taberna", "tavern"),
    "depot": ("depot", "deposito", "deposito", "banco", "locker", "casillero", "almacen"),
    "lockers": ("locker", "lockers", "casillero", "casilleros", "depot"),
    "mail": ("mail", "correo", "postal", "letter", "letterbox", "mailbox"),
    "city": ("ciudad", "city", "town", "pueblo", "urbano", "thais", "carlin", "edron"),
    "stone": ("piedra", "stone", "templo", "temple", "ruina", "ruins"),
    "clean": ("limpio", "clean", "ordenado", "classic", "clasico"),
    "sacred": ("sagrado", "sacred", "holy", "divino", "altar"),
    "cave": ("cueva", "cave", "cavern", "mazmorra", "dungeon"),
    "ice": ("hielo", "ice", "helado", "icy"),
    "snow": ("nieve", "snow", "nevado", "folda"),
    "folda": ("folda",),
    "mines": ("mina", "mines", "mine", "minas", "kazordoon"),
    "kazordoon": ("kazordoon", "dwarf", "enano"),
    "amazon": ("amazon", "amazona"),
    "camp": ("campamento", "camp"),
    "nature": ("naturaleza", "nature", "bosque", "forest", "jungle"),
    "graveyard": ("cementerio", "graveyard", "tumba", "tomb", "lapida"),
    "ghostland": ("ghostland", "fantasma", "ghost"),
    "undead": ("undead", "muerto", "muertos", "skeleton", "esqueleto", "ghoul"),
    "dirt": ("dirt", "tierra", "barro", "mud"),
    "magic": ("magic", "magia", "academy", "academia"),
    "edron": ("edron",),
    "carlin": ("carlin",),
    "thais": ("thais",),
}

SLICE_TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "depot": ("depot", "deposito", "locker", "lockers", "casillero", "casilleros", "correo", "mail"),
    "ice_cave": (
        "hielo",
        "ice",
        "congelado",
        "congelada",
        "frozen",
        "frost",
        "nieve",
        "snow",
        "folda",
        "helado",
        "helada",
    ),
    "dirt_cave": ("tierra", "dirt", "cueva", "cave", "ancient", "temple", "barro", "subterraneo"),
    "stone_mountain": ("montana", "mountain", "kazordoon", "roca", "stone", "mina", "mines"),
    "desert": ("desierto", "desert", "arena", "sand", "ankrahmun", "darashia"),
    "swamp": ("pantano", "swamp", "venore", "poison", "verde"),
    "nature_surface": ("naturaleza", "nature", "bosque", "forest", "amazon", "camp", "cesped", "grass"),
    "urban_floor": ("ciudad", "city", "urbano", "urban", "templo", "temple", "calle", "street"),
}

STRUCTURAL_INTERACTIVE_IDS = {
    411,
    424,
    426,
    724,
    904,
    903,
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
    3458,
    3460,
    3461,
    3462,
}

CREATURE_TAG_HINTS: dict[str, str] = {
    "amazon": "nature_surface",
    "valkyrie": "nature_surface",
    "witch": "nature_surface",
    "frost troll": "ice_cave",
    "polar bear": "ice_cave",
    "winter wolf": "ice_cave",
    "rotworm": "dirt_cave",
    "carrion worm": "dirt_cave",
    "minotaur": "dirt_cave",
    "orc": "dirt_cave",
    "troll": "dirt_cave",
    "dwarf": "stone_mountain",
    "dwarf soldier": "stone_mountain",
    "dwarf guard": "stone_mountain",
    "skeleton": "dirt_cave",
    "ghoul": "dirt_cave",
    "dragon": "stone_mountain",
}

SEMANTIC_ITEM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "walls_and_doors": (
        "wall",
        "door",
        "gate",
        "window",
        "pillar",
        "column",
        "framework",
        "archway",
        "fence",
        "railing",
    ),
    "urban_depot": (
        "locker",
        "depot",
        "mail",
        "letterbox",
        "box",
        "counter",
        "table",
        "chair",
        "bench",
        "lamp",
        "sign",
    ),
    "ice_and_snow": ("ice", "snow", "frozen", "glacier"),
    "cave_and_stone": ("stone", "rock", "dirt", "gravel", "earth", "stalagmite", "moss"),
    "furniture_and_decorations": (
        "table",
        "chair",
        "bench",
        "chest",
        "barrel",
        "crate",
        "lamp",
        "torch",
        "fire",
        "counter",
        "bed",
        "carpet",
    ),
    "danger_and_corpses": ("blood", "bone", "skull", "corpse", "dead", "skeleton", "grave", "tomb"),
}

SPECIAL_CONTEXT_ITEMS: dict[str, list[dict[str, Any]]] = {
    "depot_ground": [
        {"id": 424, "name": "common urban ground / normal stone tile"},
        {"id": 426, "name": "special depot ground / player standing tile"},
        {"id": 724, "name": "exterior paved ground"},
    ],
    "depot_items": [
        {"id": 1617, "name": "horizontal counter left/end"},
        {"id": 1618, "name": "horizontal counter continuation"},
        {"id": 1621, "name": "depot counter"},
        {"id": 1623, "name": "depot counter corner"},
        {"id": 2591, "name": "physical depot locker placed on counter"},
        {"id": 2593, "name": "blue depot mailbox"},
        {"id": 1480, "name": "street lamp / exterior only"},
        {"id": 1810, "name": "blackboard horizontal wall"},
        {"id": 1815, "name": "blackboard vertical wall"},
        {"id": 1385, "name": "stairs up"},
        {"id": 411, "name": "stairs down"},
    ],
}

DEPOT_ASSEMBLY_MANUAL: dict[str, Any] = {
    "building_ground": 424,
    "player_tile_in_front_of_locker": 426,
    "exterior_paved_ground": 724,
    "counter_horizontal_left": 1617,
    "counter_horizontal_continue": 1618,
    "counter_depot": 1621,
    "counter_corner": 1623,
    "physical_locker": 2591,
    "blue_mailbox": 2593,
    "street_lamp_exterior": 1480,
    "stairs_up": 1385,
    "stairs_down": 411,
    "rules": [
        "If the user asks for a Depot, the building base floor should mostly be ID 424.",
        "For the locker area, create horizontal or vertical paired rows.",
        "Gemini must emit semantic roles; these IDs are only references for the Python engine.",
        "Use role depot_locker_north for north-row lockers/counters.",
        "Use role depot_locker_east for east-side vertical lockers/counters.",
        "Use role depot_walkway for walkable tiles in front of lockers.",
        "Use role depot_railing for stone railing separators.",
        "The autotiler converts those roles into real counters, lockers, and railings.",
        "Use role mailbox for blue mailbox 2593.",
        "Use role floor_exterior for exterior perimeter equivalent to ground 724.",
        "Do not emit ground_id or item_ids in the JSON response.",
    ],
}


class GenerateMapRequest(BaseModel):
    """HTTP input for the /generate-map endpoint."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(..., min_length=3, max_length=4000)
    width: int = 10
    height: int = 10


class TileDTO(BaseModel):
    """Semantic tile generated by Gemini."""

    rel_x: int
    rel_y: int
    role: str


class MapGenerationResponse(BaseModel):
    """Respuesta estructurada obligatoria de Gemini."""

    width: int
    height: int
    tiles: List[TileDTO]


class GenerateMapResponse(BaseModel):
    """Respuesta HTTP del servidor local."""

    message: str
    output_path: str
    spawn_output_path: str | None = None
    debug_render_path: str | None = None
    tiles_modified: int
    main_ids_used: list[int]
    archetype_used: str
    cleanup_ground_id: int
    start_x: int
    start_y: int
    z: int


def load_json_file(path: Path, label: str, required: bool = True) -> dict[str, Any]:
    """Load a local JSON file with ai_generator-style path fallbacks."""

    resolved = path if path.is_absolute() else resolve_relative(path)
    if not resolved.is_file():
        if required:
            raise RuntimeError(f"No existe {label}: {resolved}")
        print(f"[server] {label} not found: {resolved}")
        return {}

    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} contains invalid JSON: {resolved}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"{label} debe ser un objeto JSON: {resolved}")

    print(f"[server] {label} loaded: {resolved}")
    return data


def load_catalog() -> dict[str, list[dict[str, Any]]]:
    """Load the Tibia 7.60 catalog generated by indexer.py."""

    data = load_json_file(CATALOG_PATH, "tibia_760_catalog.json", required=True)
    return data  # type: ignore[return-value]


def load_archetypes() -> dict[str, Any]:
    """Load archetypes.json when present."""

    return load_json_file(ARCHETYPES_PATH, "archetypes.json", required=False)


def compact_slice_metadata(entry: dict[str, Any], line_no: int) -> dict[str, Any]:
    """Extrae metadata liviana de una linea JSONL del pool de slices."""

    stats = entry.get("stats", {})
    ground_top = stats.get("ground_top", []) if isinstance(stats, dict) else []
    item_top = stats.get("item_top", []) if isinstance(stats, dict) else []
    metadata: dict[str, Any] = {
        "_jsonl_line": line_no,
        "tag": entry.get("tag"),
        "z": entry.get("z"),
        "origin": entry.get("origin", {}),
        "width": entry.get("width"),
        "height": entry.get("height"),
        "tile_count": entry.get("tile_count"),
        "walkable_count": entry.get("walkable_count"),
        "collision_density": entry.get("collision_density"),
        "stats": {
            "ground_top": ground_top,
            "item_top": item_top,
        },
    }
    for key in ("spawn", "targeted", "multilayer", "fidelity", "layer_range"):
        if key in entry:
            metadata[key] = entry[key]
    return metadata


def load_slices_pool_jsonl(path: Path) -> dict[str, Any]:
    """Indexa slices_pool.jsonl por tag sin mantener todos los tiles en RAM."""

    resolved = path if path.is_absolute() else resolve_relative(path)
    if not resolved.is_file():
        print(f"[server] slices_pool.jsonl not found: {resolved}")
        return {}

    index: dict[str, list[dict[str, Any]]] = {}
    summary: Counter[str] = Counter()
    invalid_lines = 0
    with resolved.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(entry, dict):
                invalid_lines += 1
                continue
            tag = str(entry.get("tag") or "unknown")
            index.setdefault(tag, []).append(compact_slice_metadata(entry, line_no))
            summary[tag] += 1

    print(
        "[server] slices_pool.jsonl indexado: "
        f"{resolved} ({sum(summary.values())} slices, tags={dict(summary)}, invalidas={invalid_lines})"
    )
    return {
        "format": "jsonl",
        "path": str(resolved),
        "summary": dict(summary),
        "index": index,
        "_line_cache": {},
    }


def load_slices_pool() -> dict[str, Any]:
    """Load slices_pool.jsonl in a lightweight way, with fallback to the old JSON file."""

    jsonl_pool = load_slices_pool_jsonl(SLICES_POOL_JSONL_PATH)
    if jsonl_pool:
        return jsonl_pool
    return load_json_file(SLICES_POOL_PATH, "slices_pool.json", required=False)


def load_slice_from_jsonl(slices_pool: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Lazily load the full slice associated with a JSONL line."""

    path_text = slices_pool.get("path")
    if not isinstance(path_text, str):
        return None
    try:
        line_no = int(metadata.get("_jsonl_line", 0))
    except (TypeError, ValueError):
        return None
    if line_no <= 0:
        return None

    cache = slices_pool.setdefault("_line_cache", {})
    if isinstance(cache, dict) and line_no in cache:
        cached = cache[line_no]
        return cached if isinstance(cached, dict) else None

    path = Path(path_text)
    if not path.is_file():
        print(f"[server] JSONL de slices no disponible para lectura: {path}")
        return None

    with path.open("r", encoding="utf-8") as handle:
        for current_line, raw_line in enumerate(handle, start=1):
            if current_line != line_no:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                return None
            if not isinstance(entry, dict):
                return None
            if isinstance(cache, dict):
                if len(cache) > 64:
                    cache.clear()
                cache[line_no] = entry
            return entry
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


def load_spawn_index() -> dict[str, list[dict[str, Any]]]:
    """Load world-spawn.xml and index creatures by normalized name."""

    spawn_xml = resolve_spawn_xml()
    if spawn_xml is None:
        print("[server] world-spawn.xml not found; creature geolocation disabled")
        return {}

    index: dict[str, list[dict[str, Any]]] = {}
    try:
        root = ElementTree.parse(spawn_xml).getroot()
        for elem in root.findall("spawn"):
            try:
                center_x = int(elem.attrib.get("centerx", "0"))
                center_y = int(elem.attrib.get("centery", "0"))
                center_z_raw = int(elem.attrib.get("centerz", "0"))
            except ValueError:
                continue

            center_z = normalize_spawn_z(center_z_raw)
            radius = safe_int(elem.attrib.get("radius")) or 0
            for child in list(elem):
                if child.tag not in {"monster", "npc"}:
                    continue
                raw_name = child.attrib.get("name", "")
                if not raw_name:
                    continue
                name_key = normalize_text(raw_name)
                try:
                    offset_x = int(child.attrib.get("x", "0"))
                    offset_y = int(child.attrib.get("y", "0"))
                    child_z_raw = int(child.attrib.get("z", str(center_z_raw)))
                except ValueError:
                    offset_x = offset_y = 0
                    child_z_raw = center_z_raw
                entry = {
                    "name": raw_name,
                    "kind": child.tag,
                    "centerx": center_x,
                    "centery": center_y,
                    "centerz": center_z,
                    "spawn_z_raw": center_z_raw,
                    "x": center_x + offset_x,
                    "y": center_y + offset_y,
                    "z": normalize_spawn_z(child_z_raw),
                    "radius": radius,
                    "inject_spawn": False,
                }
                index.setdefault(name_key, []).append(entry)
    except ElementTree.ParseError as exc:
        print(f"[server] invalid world-spawn.xml: {spawn_xml}: {exc}")
        return {}

    total = sum(len(entries) for entries in index.values())
    print(f"[server] world-spawn.xml loaded: {spawn_xml} ({len(index)} creatures/NPCs, {total} spawns)")
    return index


def resolve_items_xml() -> Path | None:
    """Find data/760/items.xml to expand the semantic vocabulary."""

    for candidate in ITEMS_XML_CANDIDATES:
        path = candidate.resolve()
        if path.is_file():
            return path
    return None


def load_all_items_from_xml() -> list[dict[str, Any]]:
    """Load id/name pairs from items.xml without depending on the summarized catalog."""

    items_xml = resolve_items_xml()
    if items_xml is None:
        print("[server] items.xml not found; only tibia_760_catalog.json will be used")
        return []

    items: list[dict[str, Any]] = []
    try:
        for _event, elem in ElementTree.iterparse(items_xml, events=("end",)):
            if elem.tag == "item":
                item_id = elem.attrib.get("id")
                name = elem.attrib.get("name")
                if item_id and name:
                    parsed_id = safe_int(item_id)
                    if parsed_id is not None:
                        items.append({"id": parsed_id, "name": name})
            elem.clear()
    except ElementTree.ParseError as exc:
        print(f"[server] Could not parse items.xml: {items_xml}: {exc}")
        return []

    print(f"[server] items.xml loaded for semantic context: {items_xml} ({len(items)} items)")
    return items


def dedupe_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deduplica manteniendo orden y recorta para no saturar el prompt."""

    seen: set[int] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        item_id = safe_int(item.get("id"))
        name = item.get("name")
        if item_id is None or not isinstance(name, str) or item_id in seen:
            continue
        seen.add(item_id)
        output.append({"id": item_id, "name": name})
        if len(output) >= limit:
            break
    return output


def items_matching_keywords(items: list[dict[str, Any]], keywords: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    """Filter items by normalized name using semantic keywords."""

    matches: list[dict[str, Any]] = []
    normalized_keywords = tuple(normalize_text(keyword) for keyword in keywords)
    for item in items:
        name = item.get("name")
        if not isinstance(name, str):
            continue
        normalized_name = normalize_text(name)
        if any(keyword in normalized_name for keyword in normalized_keywords):
            matches.append(item)
    return dedupe_items(matches, limit)


def build_semantic_catalog_context(
    catalog: dict[str, list[dict[str, Any]]],
    all_items: list[dict[str, Any]],
    palette: dict[str, Any],
) -> dict[str, Any]:
    """Expand the real palette with semantically compatible items from the 7.60 catalog."""

    tags = set(str(tag) for tag in palette.get("tags", []))
    context: dict[str, Any] = {
        "catalog_grounds": dedupe_items(catalog.get("floors_and_ground", []), 80),
        "walls_doors_windows_pillars": items_matching_keywords(
            all_items,
            SEMANTIC_ITEM_KEYWORDS["walls_and_doors"],
            120,
        ),
        "decorations_generales": dedupe_items(catalog.get("furniture_and_decorations", []), 80),
    }

    if {"city", "depot", "thais", "shop", "wood"} & tags:
        context["urbano_depot_muebles"] = items_matching_keywords(
            all_items,
            SEMANTIC_ITEM_KEYWORDS["urban_depot"],
            100,
        )
        if {"depot", "lockers", "mail"} & tags:
            context["depot_760_grounds"] = SPECIAL_CONTEXT_ITEMS["depot_ground"]
            context["urbano_depot_muebles"] = dedupe_items(
                SPECIAL_CONTEXT_ITEMS["depot_items"] + context["urbano_depot_muebles"],
                120,
            )
            context["manual_ensamblaje_depot_760"] = DEPOT_ASSEMBLY_MANUAL
    if {"ice", "snow", "folda"} & tags:
        context["hielo_y_nieve"] = items_matching_keywords(
            all_items,
            SEMANTIC_ITEM_KEYWORDS["ice_and_snow"],
            100,
        )
    if {"cave", "stone", "mines", "dirt"} & tags:
        context["cave_stone_dirt"] = items_matching_keywords(
            all_items,
            SEMANTIC_ITEM_KEYWORDS["cave_and_stone"],
            100,
        )
    if {"undead", "graveyard", "ghostland"} & tags:
        context["peligro_cadaveres_sangre"] = items_matching_keywords(
            all_items,
            SEMANTIC_ITEM_KEYWORDS["danger_and_corpses"],
            80,
        )

    allowed_ids = set(safe_int(item_id) for item_id in palette.get("ids_recomendados", []))
    allowed_ids.discard(None)
    for entries in context.values():
        if isinstance(entries, list):
            allowed_ids.update(safe_int(item.get("id")) for item in entries if isinstance(item, dict))
    if "manual_ensamblaje_depot_760" in context:
        manual = context["manual_ensamblaje_depot_760"]
        if isinstance(manual, dict):
            allowed_ids.update(safe_int(value) for value in manual.values() if isinstance(value, int))
    allowed_ids.discard(None)
    context["ids_contexto_ampliado"] = sorted(allowed_ids)
    return context


def collect_ints_from_structure(value: Any) -> set[int]:
    """Collect integer IDs from nested JSON structures."""

    found: set[int] = set()
    if isinstance(value, int):
        if 1 <= value <= 65535:
            found.add(value)
    elif isinstance(value, list):
        for entry in value:
            found.update(collect_ints_from_structure(entry))
    elif isinstance(value, dict):
        for entry in value.values():
            found.update(collect_ints_from_structure(entry))
    return found


def merge_geometry_ids_into_context(semantic_context: dict[str, Any], geometry_rules: dict[str, Any]) -> None:
    """Allow the validator to accept RME pieces selected for the prompt."""

    ids = set(safe_int(item_id) for item_id in semantic_context.get("ids_contexto_ampliado", []))
    ids.discard(None)
    ids.update(collect_ints_from_structure(geometry_rules))
    ids.discard(None)
    semantic_context["ids_contexto_ampliado"] = sorted(ids)


def format_catalog_for_prompt(catalog: dict[str, list[dict[str, Any]]]) -> str:
    """Compact the catalog for secondary LLM context."""

    lines: list[str] = []
    for category, items in catalog.items():
        entries = []
        for item in items[:80]:
            item_id = item.get("id")
            name = item.get("name")
            if isinstance(item_id, int) and isinstance(name, str):
                entries.append(f"{item_id}: {name}")
        if entries:
            lines.append(f"{category}: " + "; ".join(entries))
    return "\n".join(lines)


def compact_archetype_tiles(tiles: list[dict[str, Any]], max_tiles: int = 90) -> list[dict[str, Any]]:
    """Reduce tiles to relative coordinates, ground_id, and item_ids."""

    compact_tiles: list[dict[str, Any]] = []
    for tile in tiles[:max_tiles]:
        compact_tiles.append(
            {
                "x": tile.get("rel_x"),
                "y": tile.get("rel_y"),
                "g": tile.get("ground_id"),
                "i": tile.get("item_ids", []),
            }
        )
    return compact_tiles


def format_archetype_for_prompt(archetype_name: str, archetype: dict[str, Any]) -> str:
    """Convert a single archetype into a compact few-shot example for Gemini."""

    if not archetype or "tiles" not in archetype:
        return "No valid archetype loaded."

    tiles = archetype.get("tiles", [])
    if not isinstance(tiles, list):
        return "No valid tiles in the selected archetype."

    compact = {
        archetype_name: {
            "tags": archetype.get("tags", []),
            "size": archetype.get("size"),
            "center": archetype.get("center"),
            "tile_count": archetype.get("tile_count"),
            "tiles": compact_archetype_tiles(tiles),
        }
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def prompt_to_tags(prompt: str) -> set[str]:
    """Detect desired tags with simple keyword search."""

    normalized = normalize_text(prompt)
    matched: set[str] = set()
    for tag, keywords in TAG_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            matched.add(tag)
    return matched


def prompt_to_slice_tag(prompt: str) -> str | None:
    """Detect the most likely slices_pool tag for the prompt."""

    normalized = normalize_text(prompt)
    best_tag: str | None = None
    best_score = 0
    for tag, keywords in SLICE_TAG_KEYWORDS.items():
        score = sum(1 for keyword in keywords if normalize_text(keyword) in normalized)
        if score > best_score:
            best_tag = tag
            best_score = score
    return best_tag


def infer_slice_tag_from_creature(creature_name: str) -> str | None:
    """Map a creature to a biome when the prompt does not state it explicitly."""

    normalized_name = normalize_text(creature_name)
    best_tag: str | None = None
    best_len = -1
    for creature_key, tag in CREATURE_TAG_HINTS.items():
        normalized_key = normalize_text(creature_key)
        if normalized_key in normalized_name and len(normalized_key) > best_len:
            best_tag = tag
            best_len = len(normalized_key)
    return best_tag


def find_spawn_for_prompt(prompt: str, spawn_index: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """Check whether the prompt mentions a creature present in world-spawn.xml."""

    if not spawn_index:
        return None

    normalized_prompt = normalize_text(prompt)
    best_name: str | None = None
    best_score = -1
    for name_key in spawn_index:
        if name_key and name_key in normalized_prompt:
            score = len(name_key) * 2
        else:
            tokens = [token for token in name_key.split() if token]
            if not tokens or not all(token in normalized_prompt for token in tokens):
                continue
            score = sum(len(token) for token in tokens)
        if score > best_score:
            best_score = score
            best_name = name_key

    if best_name is None:
        return None

    entries = spawn_index.get(best_name, [])
    if not entries:
        return None

    def entry_score(entry: dict[str, Any]) -> tuple[int, int, int]:
        kind_score = 0 if entry.get("kind") == "monster" else 1
        z = safe_int(entry.get("z")) or 7
        surface_score = 0 if z == 7 else 1
        radius = safe_int(entry.get("radius")) or 0
        return (kind_score, surface_score, -radius)

    selected = sorted(entries, key=entry_score)[0]
    print(
        "[server] Spawn detectado por prompt: "
        f"creature={selected.get('name')}, coord=({selected.get('x')},{selected.get('y')},{selected.get('z')}), "
        f"center=({selected.get('centerx')},{selected.get('centery')},{selected.get('centerz')})"
    )
    return selected


def slice_center(slice_entry: dict[str, Any]) -> tuple[int, int, int]:
    """Centro aproximado de un slice del pool."""

    origin = slice_entry.get("origin", {}) if isinstance(slice_entry, dict) else {}
    x = safe_int(origin.get("x")) or 0
    y = safe_int(origin.get("y")) or 0
    z = safe_int(slice_entry.get("z")) or 7
    width = safe_int(slice_entry.get("width")) or 0
    height = safe_int(slice_entry.get("height")) or 0
    return x + width // 2, y + height // 2, z


def counter_pairs_from_stats(value: Any) -> list[tuple[int, int]]:
    """Normalize [id, count] pairs from JSON stats."""

    pairs: list[tuple[int, int]] = []
    if not isinstance(value, list):
        return pairs
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        item_id = safe_int(entry[0])
        try:
            count = int(entry[1])
        except (TypeError, ValueError):
            count = 0
        if item_id is not None and count > 0:
            pairs.append((item_id, count))
    return pairs


def slice_ground_ratio(entry: dict[str, Any], ground_id: int) -> float:
    """Calcula ratio aproximado de un ground dentro del slice."""

    stats = entry.get("stats", {}) if isinstance(entry, dict) else {}
    ground_pairs = counter_pairs_from_stats(stats.get("ground_top") if isinstance(stats, dict) else None)
    total_ground = sum(count for _item_id, count in ground_pairs)
    if total_ground <= 0:
        tiles = entry.get("tiles", [])
        if isinstance(tiles, list) and tiles:
            total_ground = len(tiles)
            ground_count = sum(1 for tile in tiles if isinstance(tile, dict) and safe_int(tile.get("g")) == ground_id)
            return ground_count / total_ground
        return 0.0
    return sum(count for item_id, count in ground_pairs if item_id == ground_id) / total_ground


def slice_item_pairs(entry: dict[str, Any]) -> list[tuple[int, int]]:
    """Get item counts from stats or from tiles when the slice is already loaded."""

    stats = entry.get("stats", {}) if isinstance(entry, dict) else {}
    pairs = counter_pairs_from_stats(stats.get("item_top") if isinstance(stats, dict) else None)
    if pairs:
        return pairs

    counter: Counter[int] = Counter()
    tiles = entry.get("tiles", [])
    if isinstance(tiles, list):
        for tile in tiles:
            if not isinstance(tile, dict):
                continue
            for raw_item_id in tile.get("i") or []:
                item_id = safe_int(raw_item_id)
                if item_id is not None:
                    counter[item_id] += 1
    return counter.most_common()


def design_entropy_score(entry: dict[str, Any]) -> int:
    """Puntua densidad estetica: variedad, objetos y structures interactivas."""

    item_pairs = slice_item_pairs(entry)
    unique_items = len(item_pairs)
    total_items = sum(count for _item_id, count in item_pairs)
    structural_hits = sum(count for item_id, count in item_pairs if item_id in STRUCTURAL_INTERACTIVE_IDS)
    rare_item_bonus = sum(1 for _item_id, count in item_pairs if count <= 3)
    walkable_count = safe_int(entry.get("walkable_count")) or 0
    tile_count = safe_int(entry.get("tile_count")) or 0
    targeted_bonus = 60 if entry.get("fidelity") == "spawn_density_targeted" or entry.get("targeted") else 0
    multilayer_bonus = 20 if entry.get("multilayer") else 0
    density_bonus = min(80, total_items * 2) + min(90, unique_items * 8) + min(120, structural_hits * 12)
    path_bonus = min(30, walkable_count // 4)
    occupancy_bonus = min(30, tile_count // 10)
    grass_ratio = slice_ground_ratio(entry, 4526)
    grass_penalty = 0
    if grass_ratio > 0.75 and structural_hits < 2 and unique_items < 4:
        grass_penalty = 220
    elif grass_ratio > 0.75:
        grass_penalty = 80
    return targeted_bonus + multilayer_bonus + density_bonus + path_bonus + occupancy_bonus + rare_item_bonus - grass_penalty


def select_slice_for_prompt(
    prompt: str,
    slices_pool: dict[str, Any],
    width: int,
    height: int,
    spawn_hint: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Select a real grid from the pool by tag and size compatibility."""

    target_tag = prompt_to_slice_tag(prompt)
    if target_tag is None and spawn_hint is not None:
        target_tag = infer_slice_tag_from_creature(str(spawn_hint.get("name", "")))
    elif spawn_hint is not None:
        creature_tag = infer_slice_tag_from_creature(str(spawn_hint.get("name", "")))
        if creature_tag is not None and target_tag == "dirt_cave":
            normalized_prompt = normalize_text(prompt)
            explicit_dirt = any(token in normalized_prompt for token in ("tierra", "dirt", "barro", "ancient"))
            if not explicit_dirt:
                target_tag = creature_tag
    if target_tag is None:
        return None, None

    is_jsonl_pool = isinstance(slices_pool, dict) and slices_pool.get("format") == "jsonl"
    pool_key = "index" if is_jsonl_pool else "slices"
    pool = slices_pool.get(pool_key, {}) if isinstance(slices_pool, dict) else {}
    candidates = pool.get(target_tag, []) if isinstance(pool, dict) else []
    if not isinstance(candidates, list) or not candidates:
        print(f"[server] No hay slices disponibles para tag={target_tag}")
        return target_tag, None

    def score_slice(entry: Any) -> tuple[int, int, int, int]:
        if not isinstance(entry, dict):
            return (10_000, 10_000, 0, 0)
        slice_w = safe_int(entry.get("width")) or width
        slice_h = safe_int(entry.get("height")) or height
        tile_count = safe_int(entry.get("tile_count")) or 0
        dimension_delta = abs(slice_w - width) + abs(slice_h - height)
        entropy_score = design_entropy_score(entry)
        targeted_rank = 0 if entry.get("fidelity") == "spawn_density_targeted" or entry.get("targeted") else 1
        if spawn_hint is None:
            return (dimension_delta, targeted_rank, -entropy_score, -tile_count)
        center_x, center_y, center_z = slice_center(entry)
        spawn_x = safe_int(spawn_hint.get("x")) or safe_int(spawn_hint.get("centerx")) or center_x
        spawn_y = safe_int(spawn_hint.get("y")) or safe_int(spawn_hint.get("centery")) or center_y
        spawn_z = safe_int(spawn_hint.get("z")) or safe_int(spawn_hint.get("centerz")) or center_z
        distance = abs(center_x - spawn_x) + abs(center_y - spawn_y)
        z_penalty = 5000 if center_z != spawn_z else 0
        proximity_bucket = distance // 32
        return (z_penalty + dimension_delta * 8 + proximity_bucket, targeted_rank, -entropy_score, -tile_count)

    selected = min(candidates, key=score_slice)
    if not isinstance(selected, dict):
        return target_tag, None

    if is_jsonl_pool:
        loaded = load_slice_from_jsonl(slices_pool, selected)
        if loaded is None:
            print(f"[server] Could not load JSONL slice for tag={target_tag}: {selected}")
            return target_tag, None
        selected = loaded

    origin = selected.get("origin", {})
    print(
        "[server] Selected real slice: "
        f"tag={target_tag}, origin={origin}, size={selected.get('width')}x{selected.get('height')}, "
        f"tiles={selected.get('tile_count')}, entropy={design_entropy_score(selected)}, "
        f"grass4526={slice_ground_ratio(selected, 4526):.2f}"
    )
    return target_tag, selected


def normalize_text(value: str) -> str:
    """Normalize lowercase text and remove diacritics for robust searches."""

    decomposed = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def fallback_archetype_name(archetypes: dict[str, Any]) -> str:
    """Select a default archetype compatible with old and new datasets."""

    for candidate in DEFAULT_ARCHETYPE_CANDIDATES:
        if candidate in archetypes:
            return candidate
    return next(iter(archetypes), "thais_temple")


def select_archetype_for_prompt(
    prompt: str,
    archetypes: dict[str, Any],
    spawn_hint: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Select the archetype that best matches the prompt."""

    if not archetypes:
        return fallback_archetype_name(archetypes), {}

    if spawn_hint is not None:
        spawn_x = safe_int(spawn_hint.get("x")) or safe_int(spawn_hint.get("centerx")) or 0
        spawn_y = safe_int(spawn_hint.get("y")) or safe_int(spawn_hint.get("centery")) or 0
        spawn_z = safe_int(spawn_hint.get("z")) or safe_int(spawn_hint.get("centerz")) or 7
        nearest_name: str | None = None
        nearest_score: int | None = None
        for name, archetype in archetypes.items():
            if not isinstance(archetype, dict):
                continue
            center = archetype.get("center", {})
            if not isinstance(center, dict):
                continue
            center_x = safe_int(center.get("x"))
            center_y = safe_int(center.get("y"))
            center_z = safe_int(center.get("z"))
            if center_x is None or center_y is None:
                continue
            distance = abs(center_x - spawn_x) + abs(center_y - spawn_y)
            z_penalty = 1000 if center_z is not None and center_z != spawn_z else 0
            score = distance + z_penalty
            if nearest_score is None or score < nearest_score:
                nearest_name = name
                nearest_score = score

        if nearest_name is not None and nearest_score is not None and nearest_score <= 256:
            selected = archetypes.get(nearest_name, {})
            print(
                "[server] Archetype selected by spawn proximity: "
                f"{nearest_name}, score={nearest_score}"
            )
            return nearest_name, selected if isinstance(selected, dict) else {}

        synthetic_name = f"spawn_{normalize_text(str(spawn_hint.get('name', 'creature'))).replace(' ', '_')}"
        synthetic = {
            "tags": ["spawn", "creature", infer_slice_tag_from_creature(str(spawn_hint.get("name", ""))) or "wild"],
            "center": {"x": spawn_x, "y": spawn_y, "z": spawn_z},
            "tile_count": 0,
            "tiles": [],
            "spawn_hint": spawn_hint,
        }
        print(
            "[server] Arquetipo sintetico por spawn: "
            f"{synthetic_name}, coord=({spawn_x},{spawn_y},{spawn_z})"
        )
        return synthetic_name, synthetic

    requested_tags = prompt_to_tags(prompt)
    prompt_normalized = normalize_text(prompt)
    best_name = fallback_archetype_name(archetypes)
    best_score = -1

    for name, archetype in archetypes.items():
        tags = set(archetype.get("tags", [])) if isinstance(archetype, dict) else set()
        score = len(requested_tags & tags) * 3
        if normalize_text(name) in prompt_normalized:
            score += 6
        for tag in tags:
            if normalize_text(tag) in prompt_normalized:
                score += 1
        if score > best_score:
            best_score = score
            best_name = name

    if best_score <= 0:
        best_name = fallback_archetype_name(archetypes)

    selected = archetypes.get(best_name, {})
    print(
        "[server] Selected archetype: "
        f"{best_name}, prompt_tags={sorted(requested_tags)}, "
        f"archetype_tags={selected.get('tags', []) if isinstance(selected, dict) else []}"
    )
    return best_name, selected if isinstance(selected, dict) else {}


def safe_int(value: Any) -> int | None:
    """Convert to int only when the value represents a valid ID."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= parsed <= 65535:
        return parsed
    return None


def plain_int(value: Any) -> int | None:
    """Convert relative coordinates, allowing negative values."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def item_role(tile: dict[str, Any], item_id: int) -> str:
    """Look up the semantic role extractor.py may have stored for an item."""

    for item in tile.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        if safe_int(item.get("id")) == item_id:
            return str(item.get("role", "")).lower()
    return ""


def extract_dynamic_palette(archetype_name: str, archetype: dict[str, Any]) -> dict[str, Any]:
    """Extract the visual DNA of a real archetype stored in archetypes.json."""

    tiles = archetype.get("tiles", []) if isinstance(archetype, dict) else []
    if not isinstance(tiles, list) or not tiles:
        print("[server] Archetype has no tiles; using minimal emergency palette.")
        return {
            "archetype": archetype_name,
            "tags": [],
            "main_ground": 405,
            "secondary_grounds": [],
            "structures_and_walls": [],
            "common_decorations": [],
            "recommended_ids": [405],
        }

    ground_counter: Counter[int] = Counter()
    item_counter: Counter[int] = Counter()
    border_item_counter: Counter[int] = Counter()
    role_structure_counter: Counter[int] = Counter()
    role_decoration_counter: Counter[int] = Counter()

    coords: list[tuple[int, int]] = []
    valid_tiles: list[dict[str, Any]] = []
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        rel_x = plain_int(tile.get("rel_x"))
        rel_y = plain_int(tile.get("rel_y"))
        ground_id = safe_int(tile.get("ground_id"))
        if rel_x is None or rel_y is None:
            continue
        coords.append((rel_x, rel_y))
        valid_tiles.append(tile)
        if ground_id is not None:
            ground_counter[ground_id] += 1
        for raw_item_id in tile.get("item_ids", []) or []:
            item_id = safe_int(raw_item_id)
            if item_id is None:
                continue
            item_counter[item_id] += 1
            role = item_role(tile, item_id)
            if any(marker in role for marker in ("wall", "door", "structure")):
                role_structure_counter[item_id] += 2
            elif role:
                role_decoration_counter[item_id] += 1

    if coords:
        min_x = min(x for x, _y in coords)
        max_x = max(x for x, _y in coords)
        min_y = min(y for _x, y in coords)
        max_y = max(y for _x, y in coords)
    else:
        min_x = max_x = min_y = max_y = 0

    for tile in valid_tiles:
        rel_x = plain_int(tile.get("rel_x"))
        rel_y = plain_int(tile.get("rel_y"))
        if rel_x is None or rel_y is None:
            continue
        is_border = rel_x in (min_x, max_x) or rel_y in (min_y, max_y)
        if not is_border:
            continue
        for raw_item_id in tile.get("item_ids", []) or []:
            item_id = safe_int(raw_item_id)
            if item_id is not None:
                border_item_counter[item_id] += 1

    main_ground = ground_counter.most_common(1)[0][0] if ground_counter else 405
    secondary_grounds = [
        item_id for item_id, _count in ground_counter.most_common(8) if item_id != main_ground
    ]

    structure_scores = Counter()
    structure_scores.update(border_item_counter)
    structure_scores.update(role_structure_counter)
    structure_ids = [item_id for item_id, _count in structure_scores.most_common(24)]

    decoration_scores = Counter()
    decoration_scores.update(item_counter)
    decoration_scores.update(role_decoration_counter)
    for item_id in role_structure_counter:
        decoration_scores.pop(item_id, None)
    decoration_ids = [item_id for item_id, _count in decoration_scores.most_common(32)]

    recommended_ids = sorted(set([main_ground, *secondary_grounds, *structure_ids, *decoration_ids]))
    palette = {
        "archetype": archetype_name,
        "tags": archetype.get("tags", []),
        "main_ground": main_ground,
        "secondary_grounds": secondary_grounds,
        "structures_and_walls": structure_ids,
        "common_decorations": decoration_ids,
        "recommended_ids": recommended_ids,
        "stats": {
            "tiles_analyzed": len(valid_tiles),
            "unique_grounds": len(ground_counter),
            "unique_items": len(item_counter),
            "relative_bbox": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
        },
    }
    print(
        "[server] Extracted dynamic palette: "
        f"archetype={archetype_name}, main_ground={main_ground}, "
        f"structures={structure_ids[:8]}, decorations={decoration_ids[:8]}"
    )
    return palette


def format_dynamic_palette_for_prompt(palette: dict[str, Any], semantic_context: dict[str, Any]) -> str:
    """Serialize archetype DNA and expanded vocabulary for Gemini."""

    compact = {
        "real_archetype": palette.get("archetype"),
        "tags": palette.get("tags", []),
        "main_ground": palette.get("main_ground"),
        "secondary_grounds": palette.get("secondary_grounds", []),
        "structures_and_walls": palette.get("structures_and_walls", []),
        "common_decorations": palette.get("common_decorations", []),
        "recommended_archetype_ids": palette.get("recommended_ids", []),
        "expanded_semantic_catalog": semantic_context,
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def build_system_prompt(
    catalog: dict[str, list[dict[str, Any]]],
    archetype_name: str,
    archetype: dict[str, Any],
    dynamic_palette: dict[str, Any],
    semantic_context: dict[str, Any],
    geometry_rules: dict[str, Any],
    width: int,
    height: int,
) -> str:
    """Construye el prompt de sistema con ADN real y vocabulario semantico ampliado."""

    catalog_text = format_catalog_for_prompt(catalog)
    palette_text = format_dynamic_palette_for_prompt(dynamic_palette, semantic_context)
    archetype_text = format_archetype_for_prompt(archetype_name, archetype)
    geometry_text = format_geometry_rules_for_prompt(geometry_rules)
    main_ground = dynamic_palette.get("main_ground")
    horseshoe_example = json.dumps(
        {
            "width": 9,
            "height": 8,
            "tiles": [
                {"rel_x": 1, "rel_y": 0, "role": "wall"},
                {"rel_x": 2, "rel_y": 0, "role": "wall"},
                {"rel_x": 3, "rel_y": 0, "role": "wall"},
                {"rel_x": 4, "rel_y": 0, "role": "wall"},
                {"rel_x": 5, "rel_y": 0, "role": "wall"},
                {"rel_x": 6, "rel_y": 0, "role": "wall"},
                {"rel_x": 7, "rel_y": 0, "role": "wall"},
                {"rel_x": 1, "rel_y": 1, "role": "wall"},
                {"rel_x": 2, "rel_y": 1, "role": "depot_locker_north"},
                {"rel_x": 3, "rel_y": 1, "role": "depot_railing"},
                {"rel_x": 4, "rel_y": 1, "role": "depot_locker_north"},
                {"rel_x": 5, "rel_y": 1, "role": "depot_railing"},
                {"rel_x": 6, "rel_y": 1, "role": "depot_locker_north"},
                {"rel_x": 7, "rel_y": 1, "role": "wall"},
                {"rel_x": 1, "rel_y": 2, "role": "wall"},
                {"rel_x": 2, "rel_y": 2, "role": "depot_walkway"},
                {"rel_x": 3, "rel_y": 2, "role": "depot_railing"},
                {"rel_x": 4, "rel_y": 2, "role": "depot_walkway"},
                {"rel_x": 5, "rel_y": 2, "role": "depot_railing"},
                {"rel_x": 6, "rel_y": 2, "role": "depot_walkway"},
                {"rel_x": 7, "rel_y": 2, "role": "depot_locker_east"},
                {"rel_x": 1, "rel_y": 3, "role": "wall"},
                {"rel_x": 2, "rel_y": 3, "role": "floor_interior"},
                {"rel_x": 3, "rel_y": 3, "role": "floor_interior"},
                {"rel_x": 4, "rel_y": 3, "role": "floor_interior"},
                {"rel_x": 5, "rel_y": 3, "role": "floor_interior"},
                {"rel_x": 6, "rel_y": 3, "role": "depot_walkway"},
                {"rel_x": 7, "rel_y": 3, "role": "depot_railing"},
                {"rel_x": 1, "rel_y": 4, "role": "wall"},
                {"rel_x": 2, "rel_y": 4, "role": "npc_counter"},
                {"rel_x": 3, "rel_y": 4, "role": "npc_counter"},
                {"rel_x": 4, "rel_y": 4, "role": "mailbox"},
                {"rel_x": 5, "rel_y": 4, "role": "floor_interior"},
                {"rel_x": 6, "rel_y": 4, "role": "depot_walkway"},
                {"rel_x": 7, "rel_y": 4, "role": "depot_locker_east"},
                {"rel_x": 1, "rel_y": 5, "role": "wall"},
                {"rel_x": 2, "rel_y": 5, "role": "floor_interior"},
                {"rel_x": 3, "rel_y": 5, "role": "floor_interior"},
                {"rel_x": 4, "rel_y": 5, "role": "door"},
                {"rel_x": 5, "rel_y": 5, "role": "floor_interior"},
                {"rel_x": 6, "rel_y": 5, "role": "depot_walkway"},
                {"rel_x": 7, "rel_y": 5, "role": "depot_railing"},
                {"rel_x": 1, "rel_y": 6, "role": "wall"},
                {"rel_x": 2, "rel_y": 6, "role": "wall"},
                {"rel_x": 3, "rel_y": 6, "role": "wall"},
                {"rel_x": 4, "rel_y": 6, "role": "door"},
                {"rel_x": 5, "rel_y": 6, "role": "wall"},
                {"rel_x": 6, "rel_y": 6, "role": "wall"},
                {"rel_x": 7, "rel_y": 6, "role": "wall"},
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"""
You act as the Dungeon Master and CipSoft map designer from 2003.
Your goal is to create immersive medieval roleplay environments for Tibia 7.60.
Do not design generic square rooms, symmetric arenas, or checkerboard patterns.
Design a rectangular {width}x{height} tile zone using native RME brush logic and spatial storytelling.
Your priority is to create cohesive maps with story: defense, rest, supplies, patrol paths, and roleplay corners.
Your job is to define the macro-zone layout, not individual tiles.
Python will assemble real CipSoft-style 8x8 chunks taken from slices_pool.jsonl.
Your only job is to decide which macro-zone type occupies each 8x8 quadrant.
The internal Python engine will stamp complete real fragments to preserve tents, palisades, rocks, props, and original geometry without AI deformation.
If the prompt mentions creatures, the server will use world-spawn.xml only as an ecological hint for selecting coordinates and real style. Do not generate physical creatures or spawns in the map yet.

MANDATORY MACRO-ZONE CONTRACT
- You must no longer choose numeric Tibia IDs. The Python autotiling engine resolves all IDs through RME data.
- Each tile must return only one role.
- The JSON width and height fields must remain the full map size in tiles: width={width}, height={height}.
- rel_x and rel_y are macro-chunk coordinates, not tile coordinates.
- Each macro-chunk is {CHUNK_SIZE}x{CHUNK_SIZE} tiles.
- For this map, emit a {(width + CHUNK_SIZE - 1) // CHUNK_SIZE}x{(height + CHUNK_SIZE - 1) // CHUNK_SIZE} macro-role matrix.
- Allowed roles: spawn_hub_dense, defensive_perimeter, wild_surroundings, camp_amenities.
- spawn_hub_dense: dense spawn core, tents/buildings, creature area, main structures.
- defensive_perimeter: palisades, walls, rocks, terrain breaks, or biome defenses.
- wild_surroundings: nature, forest, normal cave, walkable corridors, and organic filler.
- camp_amenities: rest areas, crates, campfires, chests, bunks, and small roleplay details.
- Do not use fine-grained roles such as wall, tent_roof, depot_locker_north, or floor_interior in this macro phase.

DYNAMIC PALETTE AND EXPANDED VOCABULARY
The recommended_archetype_ids section was statistically extracted from a real Tibia map.
The expanded_semantic_catalog section contains compatible Tibia 7.60 catalog IDs for doors, windows, pillars, corners, and logical decorations.
{palette_text}

CONTEXT USAGE RULES
- Archetype and catalog IDs are style context only; they are not part of your JSON output.
- Your output must use macro roles. The Python engine will choose IDs, orientation, corners, tents, counters, props, and doors.
- Do not try to place pixel-perfect cubicle offsets. Mark the general structure; the archetype optimizer copies real Thais Depot offsets when relevant.
- Do not draw organic cave noise tile by tile. Real slices and the Cellular Automata Micro-Construction layer provide rock, dirt, ice, walls, and organic variation.
- If the palette says the real dominant ground is ID {main_ground}, use it only as a biome hint for choosing between wild_surroundings, defensive_perimeter, spawn_hub_dense, or camp_amenities.
- Do not resolve walls, corners, or furniture by ID. Mark spatial intent with macro roles.
- Do not mix roles randomly: use a small number of chunks with clear intent and readable composition.

RME RAW PALETTE CONCEPT
- In addition to automated brushes, you conceptually have access to the RME RAW Palette.
- If fine ambience details do not have an automatic brush, you may plan semantic space for direct RAW Palette sprites to enrich mountain edges, camps, caves, or ruins.
- Do not write IDs in the final JSON. Use this information only to decide where semantic space for fine details is useful; Python chooses real sprites from the catalog, archetype, and mined slices.

CIPSOFT 2003 MANUAL: AMAZON CAMP AND TRIBAL CAMPS
- If you design an Amazon Camp, do not create a clean square. Place spawn_hub_dense in the tent core and camp_amenities near that core.
- Use defensive_perimeter on edges where palisades, rocks, or improvised defense should appear; leave wild_surroundings as natural entrances.
- Use camp_amenities for stacked supply crates, scattered weapons, campfires, rustic chests, and Valkyrie rest bunks.
- For fine details, the autotiler copies real micro-decoration from Venore/Amazon Camp slices using RAW Palette data.
- The silhouette should feel organic and tribal: broken paths, surrounding vegetation, grouped tents, not a perfect grid.

TIBIA 7.60 DEPOT ASSEMBLY MANUAL
- If the user asks for a Depot, create the building with wall, floor_interior, and an entry door conceptually, but emit only macro roles.
- For north-wall lockers, reserve a dense/service area that the autotiler can translate into depot_locker_north with depot_walkway south of it.
- For east-wall lockers, reserve a dense/service area that the autotiler can translate into depot_locker_east with depot_walkway west of it.
- For horseshoe layouts, combine a north service row with an east-side service row while keeping the center navigable.
- Use service/perimeter chunks to imply railings between cubicles or service areas; the autotiler converts them into stone railing 1526 when the fine-role path is active.
- For mail or NPC service, reserve camp_amenities or spawn_hub_dense space near counters; the autotiler chooses the actual counter IDs.
- Use an exterior perimeter around the building. Do not place lamps or decorations by ID.
- If vertical connectivity is needed, reserve interior floor space and let a future specialized rule add stairs.

RME GEOMETRIC CONSTRUCTION RULES
These rules come directly from data/760/walls.xml and data/760/doodads.xml. They take priority over item-name guesses from items.xml.
{geometry_text}

Mandatory geometric instructions:
- For walls, use horizontal for north/south segments and vertical for east/west segments according to the selected RME template.
- Do not build corners by repeating the same flat wall ID. Close corners and junctions with corner_variants or pole pieces from the same family.
- If you open a wall with a door or window, use doors_horizontal or doors_vertical from the same family and preserve closure with matching corners/pillars.
- For composite doodads, preserve the exact x/y offsets of every composite. For example, a two-piece horizontal bench must keep the left-to-right order defined by RME.
- For tables/counters, respect align values: horizontal, vertical, north, south, east, west, or alone. Do not mix furniture families in one composite set.

REAL DESIGN EXAMPLE (SELECTED ARCHETYPE)
Only the most relevant archetype for the current request is included to avoid context overload.
Compact tile format: x=rel_x, y=rel_y, g=ground_id, i=item_ids stacked on top.
{archetype_text}

SEMANTIC FEW-SHOT: HORSESHOE DEPOT
This historical example shows microstructure, but in the macro phase you must not copy its fine roles. Use it only to understand which dense/service areas should become spawn_hub_dense or camp_amenities.
{horseshoe_example}

Learning rules from archetypes:
- Analyze the attached real example to copy spatial relationships, density, borders, and item groups.
- Do not copy numeric IDs in the output. Translate what you see into macro roles: spawn_hub_dense, defensive_perimeter, wild_surroundings, or camp_amenities.
- If you see walls, palisades, or edge rocks, respond with defensive_perimeter.
- If you see tents, shops, shelters, or the spawn core, respond with spawn_hub_dense.
- If you see crates, campfires, beds, chests, or rest areas, respond with camp_amenities.
- If you see forest, normal cave, corridors, or natural filler, respond with wild_surroundings.

Mandatory architectural rules:
- Work as if using native RME brushes: continuous ground areas, clean borders, sparse and meaningful decoration.
- Macro-zones must structure the map coherently by creating paths, cores, defenses, and logical borders.
- Leave open spaces when the design needs entrances or connections.
- Place camp_amenities near spawn_hub_dense, never isolated without context.
- Avoid checkerboard macro-role alternation. Each macro-zone must have clear intent.

Mandatory output rules:
- Return only data that satisfies the structured schema.
- Ensure rel_x goes from 0 to macro_width-1 and rel_y goes from 0 to macro_height-1.
- For this request: width={width}, height={height}, macro_width={(width + CHUNK_SIZE - 1) // CHUNK_SIZE}, macro_height={(height + CHUNK_SIZE - 1) // CHUNK_SIZE}.
- Return width={width} and height={height} in the JSON.
- Do not generate more than {((width + CHUNK_SIZE - 1) // CHUNK_SIZE) * ((height + CHUNK_SIZE - 1) // CHUNK_SIZE)} macro tiles.
- Each tile must have exactly one role field.
- Do not include ground_id or item_ids.
- Do not use fine roles such as wall, wall_ruins, tent_roof, floor_interior, depot_locker_north, or npc_counter.
- Use only spawn_hub_dense, defensive_perimeter, wild_surroundings, and camp_amenities.

Summarized Tibia 7.60 catalog as a secondary name reference. ID selection must come from the recommended DNA or expanded vocabulary:
{catalog_text}
""".strip()


def create_genai_client() -> genai.Client:
    """Initialize the Google GenAI client using GEMINI_API_KEY from the environment."""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable")
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Google GenAI client: {exc}") from exc


def is_quota_or_rate_limit_error(exc: Exception) -> bool:
    """Detect Google GenAI quota/rate-limit errors without coupling to one exact class."""

    error_name = exc.__class__.__name__.lower()
    error_text = str(exc).lower()
    quota_markers = (
        "resourceexhausted",
        "resource exhausted",
        "quota",
        "rate limit",
        "ratelimit",
        "429",
        "too many requests",
    )
    return any(marker in error_name or marker in error_text for marker in quota_markers)


def generate_content_with_model_failover(
    client: genai.Client,
    contents: Any,
    config: types.GenerateContentConfig,
) -> Any:
    """Llama a Gemini rotando modelos cuando aparece quota/rate-limit."""

    last_error: Exception | None = None
    for model in AVAILABLE_MODELS:
        try:
            print(f"[server] Llamando a Gemini con modelo: {model}")
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_error = exc
            if is_quota_or_rate_limit_error(exc):
                print(f"[server] Modelo {model} sin cuota, alternando al fallback...")
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("No hay modelos disponibles para llamar a Gemini.")


def image_mime_type(path: Path) -> str:
    """Return a simple MIME type for local images."""

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def select_reference_image(prompt: str, selected_slice_tag: str | None) -> Path | None:
    """Find a real reference image for the visual feedback phase."""

    if not REFERENCE_IMAGES_DIR.is_dir():
        print(f"[server] Visual reference directory not found: {REFERENCE_IMAGES_DIR}")
        return None

    candidates = [
        path
        for path in REFERENCE_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not candidates:
        print(f"[server] No reference images found in: {REFERENCE_IMAGES_DIR}")
        return None

    prompt_text = normalize_text(" ".join([prompt, selected_slice_tag or ""]))

    def score(path: Path) -> tuple[int, str]:
        name = normalize_text(path.stem.replace("_", " "))
        tokens = [token for token in name.split() if len(token) >= 3]
        matched = sum(1 for token in tokens if token in prompt_text)
        if selected_slice_tag and normalize_text(selected_slice_tag.replace("_", " ")) in name:
            matched += 3
        if "amazon" in prompt_text and "amazon" in name:
            matched += 5
        if "camp" in prompt_text and "camp" in name:
            matched += 4
        return (-matched, name)

    selected = sorted(candidates, key=score)[0]
    if score(selected)[0] == 0 and len(candidates) > 1:
        print(f"[server] Referencia visual por fallback: {selected}")
    else:
        print(f"[server] Selected visual reference: {selected}")
    return selected


def image_part_from_path(path: Path) -> types.Part:
    """Load a local image as a Gemini multimodal part."""

    return types.Part.from_bytes(data=path.read_bytes(), mime_type=image_mime_type(path))


def run_visual_feedback_phase(
    client: genai.Client,
    initial_plan: MapGenerationResponse,
    debug_render_path: Path | None,
    prompt: str,
    selected_slice_tag: str | None,
    width: int,
    height: int,
) -> MapGenerationResponse:
    """Second multimodal pass: compare debug_render against the real reference."""

    if debug_render_path is None or not debug_render_path.is_file():
        print("[server] Critica visual omitida: debug_render.png no disponible")
        return initial_plan

    reference_path = select_reference_image(prompt, selected_slice_tag)
    if reference_path is None or not reference_path.is_file():
        print("[server] Visual critique skipped: no reference image")
        return initial_plan

    critique_prompt = f"""
Act as a CipSoft designer specialized in Tibia 7.60.
Compare the structural distribution of the original reference image with debug_render.png.
If the design lacks tent density, palisades, rocks, crates, or the maze-like flow of the real camp, rewrite the JSON by correcting the macro-role matrix.
Use debug_render.png to evaluate corridor flow between quadrants.
If you notice artificial dividing lines at chunk borders, reorder macro roles to force smooth transitions between cores, perimeters, and nature.
Avoid placing a dense chunk next to an empty-looking chunk if it creates a straight cut; use defensive_perimeter or camp_amenities as a buffer when needed.

ATTENTION: Do not send individual tiles.
You must respond ONLY by rewriting the {CHUNK_SIZE}x{CHUNK_SIZE} chunk macro-role matrix.
The only roles allowed in your JSON response are: ['spawn_hub_dense', 'defensive_perimeter', 'wild_surroundings', 'camp_amenities'].
Your response must keep the simplified grid format.

Strict rules:
- Return exactly the MapGenerationResponse schema.
- width must be {width} and height must be {height}; these are the final sizes in tiles.
- rel_x and rel_y are NOT tile coordinates; they are chunk coordinates.
- macro_width={((width + CHUNK_SIZE - 1) // CHUNK_SIZE)} and macro_height={((height + CHUNK_SIZE - 1) // CHUNK_SIZE)}.
- rel_x goes from 0 to {((width + CHUNK_SIZE - 1) // CHUNK_SIZE) - 1}; rel_y goes from 0 to {((height + CHUNK_SIZE - 1) // CHUNK_SIZE) - 1}.
- Use only allowed macro roles: {", ".join(sorted(MACRO_ROLES))}.
- Do not return ground_id, item_ids, or numeric IDs.
- Keep macro-zones coherent. Use spawn_hub_dense for the core, defensive_perimeter for borders/palisades, wild_surroundings for nature/transit, and camp_amenities for rest/supplies.
- Do not use old fine-grained roles such as wall, floor_interior, tent_roof, depot_locker_north, mailbox, or npc_counter.

Initial macro JSON:
{initial_plan.model_dump_json()}
""".strip()

    try:
        visual_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MapGenerationResponse,
            system_instruction=(
                "You are a multimodal visual critic for OpenTibia maps. "
                "Your only valid output is structured JSON compatible with MapGenerationResponse, "
                "using only 8x8 chunk macro roles."
            ),
        )
        contents = [
            types.Part.from_text(text=critique_prompt),
            types.Part.from_text(text="Original real reference image:"),
            image_part_from_path(reference_path),
            types.Part.from_text(text="debug_render.png generated by the engine:"),
            image_part_from_path(debug_render_path),
        ]
        response = generate_content_with_model_failover(client, contents, visual_config)
    except Exception as exc:
        print(f"[server] Visual critique failed; keeping initial plan: {exc}")
        return initial_plan

    if not response.text:
        print("[server] Critica visual sin JSON; se conserva plan inicial")
        return initial_plan

    try:
        refined = MapGenerationResponse.model_validate_json(response.text)
        validate_generated_map(refined, width=width, height=height)
    except Exception as exc:
        print(f"[server] Critica visual invalida; se conserva plan inicial: {exc}")
        return initial_plan

    print(
        "[server] Critica visual aplicada: "
        f"reference={reference_path.name}, tiles={len(refined.tiles)}"
    )
    return refined


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


def tile_key(tile: TileDTO) -> tuple[int, int]:
    """Relative coordinate key."""

    return (tile.rel_x, tile.rel_y)


def build_tile_index(plan: MapGenerationResponse) -> dict[tuple[int, int], TileDTO]:
    """Merge duplicate tiles generated by the AI."""

    index: dict[tuple[int, int], TileDTO] = {}
    for tile in plan.tiles:
        key = tile_key(tile)
        if key not in index:
            tile.item_ids = unique_items(list(tile.item_ids))
            index[key] = tile
            continue
        existing = index[key]
        existing.ground_id = tile.ground_id or existing.ground_id
        existing.item_ids = unique_items(existing.item_ids + list(tile.item_ids))
    return index


def get_or_create_tile(
    plan: MapGenerationResponse,
    index: dict[tuple[int, int], TileDTO],
    rel_x: int,
    rel_y: int,
    ground_id: int,
) -> TileDTO | None:
    """Get or create a tile within plan bounds."""

    if rel_x < 0 or rel_y < 0 or rel_x >= plan.width or rel_y >= plan.height:
        return None
    key = (rel_x, rel_y)
    if key in index:
        return index[key]
    tile = TileDTO(rel_x=rel_x, rel_y=rel_y, ground_id=ground_id, item_ids=[])
    index[key] = tile
    return tile


def remove_items(tile: TileDTO, item_ids: set[int]) -> None:
    """Elimina items especificos de un tile."""

    tile.item_ids = [item_id for item_id in tile.item_ids if item_id not in item_ids]


def ensure_items(tile: TileDTO, item_ids: list[int], replace: bool = False) -> None:
    """Asegura item_ids y conserva orden."""

    if replace:
        tile.item_ids = list(item_ids)
    else:
        tile.item_ids = unique_items(tile.item_ids + item_ids)


def collect_wall_item_ids() -> set[int]:
    """Collect wall/pillar/door IDs parsed from RME."""

    wall_ids: set[int] = set()
    for rule in RME_GEOMETRY_RULES.get("walls", {}).values():
        if not isinstance(rule, dict):
            continue
        wall_ids.update(collect_ints_from_structure(rule.get("all_items_by_role", {})))
        wall_ids.update(collect_ints_from_structure(rule.get("doors_horizontal", [])))
        wall_ids.update(collect_ints_from_structure(rule.get("doors_vertical", [])))
    return wall_ids


def wall_orientation_sets() -> tuple[set[int], set[int], set[int]]:
    """Return horizontal, vertical, and structural wall IDs."""

    horizontal_ids: set[int] = set()
    vertical_ids: set[int] = set()
    structural_ids: set[int] = set()
    for rule in RME_GEOMETRY_RULES.get("walls", {}).values():
        if not isinstance(rule, dict):
            continue
        roles = rule.get("all_items_by_role", {})
        if isinstance(roles, dict):
            horizontal_ids.update(collect_ints_from_structure(roles.get("horizontal", [])))
            vertical_ids.update(collect_ints_from_structure(roles.get("vertical", [])))
            structural_ids.update(collect_ints_from_structure(roles))
        structural_ids.update(collect_ints_from_structure(rule.get("doors_horizontal", [])))
        structural_ids.update(collect_ints_from_structure(rule.get("doors_vertical", [])))
    return horizontal_ids, vertical_ids, structural_ids


def classify_wall_orientation(tile: TileDTO) -> str | None:
    """Clasifica un tile de muro como horizontal, vertical o mixto."""

    horizontal_ids, vertical_ids, wall_ids = wall_orientation_sets()
    item_set = set(tile.item_ids)
    if not (item_set & wall_ids):
        return None
    has_horizontal = bool(item_set & horizontal_ids)
    has_vertical = bool(item_set & vertical_ids)
    if has_horizontal and not has_vertical:
        return "horizontal"
    if has_vertical and not has_horizontal:
        return "vertical"
    if has_horizontal and has_vertical:
        return "mixed"
    return "structural"


def building_wall_tiles(index: dict[tuple[int, int], TileDTO]) -> list[TileDTO]:
    """Return tiles containing RME walls or structural pieces."""

    _horizontal_ids, _vertical_ids, wall_ids = wall_orientation_sets()
    return [tile for tile in index.values() if set(tile.item_ids) & wall_ids]


def building_bbox(index: dict[tuple[int, int], TileDTO]) -> tuple[int, int, int, int] | None:
    """Calcula bounding box de las murallas del edificio."""

    walls = building_wall_tiles(index)
    if not walls:
        return None
    return (
        min(tile.rel_x for tile in walls),
        max(tile.rel_x for tile in walls),
        min(tile.rel_y for tile in walls),
        max(tile.rel_y for tile in walls),
    )


def force_building_floor(
    plan: MapGenerationResponse,
    index: dict[tuple[int, int], TileDTO],
    ground_id: int,
) -> None:
    """Pave the wall bounding-box interior and a south threshold."""

    bbox = building_bbox(index)
    if bbox is None:
        return

    min_x, max_x, min_y, max_y = bbox
    changed = 0
    for rel_x in range(max(0, min_x + 1), min(plan.width, max_x)):
        for rel_y in range(max(0, min_y + 1), min(plan.height, max_y + 1)):
            tile = get_or_create_tile(plan, index, rel_x, rel_y, ground_id)
            if tile is None:
                continue
            if tile.ground_id != ground_id:
                tile.ground_id = ground_id
                changed += 1
    if changed:
        print(f"[postprocess] Forced interior floor: ground={ground_id}, tiles={changed}")


def stone_wall_corner_id() -> int:
    """Return the main stone-wall corner with a safe fallback."""

    rule = RME_GEOMETRY_RULES.get("walls", {}).get("stone wall")
    if isinstance(rule, dict):
        corners = rule.get("corner_variants", [])
        if isinstance(corners, list) and corners:
            parsed = safe_int(corners[0])
            if parsed is not None:
                return parsed
    return 1053


def force_south_wall_end_caps(index: dict[tuple[int, int], TileDTO]) -> None:
    """Cap south wall ends with a stone-wall pillar/corner."""

    bbox = building_bbox(index)
    if bbox is None:
        return
    min_x, max_x, _north_y, south_y = bbox
    corner_id = stone_wall_corner_id()
    changed = 0
    for rel_x in (min_x, max_x):
        tile = index.get((rel_x, south_y))
        if tile is None:
            continue
        inward_tile = index.get((rel_x, south_y - 1))
        if inward_tile is None or classify_wall_orientation(inward_tile) != "vertical":
            continue
        orientation = classify_wall_orientation(tile)
        if orientation in {"horizontal", "structural", "mixed"}:
            remove_items(tile, collect_wall_item_ids())
            ensure_items(tile, [corner_id], replace=True)
            changed += 1
    if changed:
        print(f"[postprocess] Remates sur aplicados: corner={corner_id}, count={changed}")


def normalize_horizontal_counter_row(
    index: dict[tuple[int, int], TileDTO],
    row_y: int,
    start_x: int,
    end_x: int,
) -> None:
    """Autotilea counters horizontales como 1617 + 1618..."""

    changed = 0
    for rel_x in range(start_x, end_x + 1):
        tile = index.get((rel_x, row_y))
        if tile is None:
            continue
        if not ({1617, 1618, 1621, 1623} & set(tile.item_ids)):
            continue
        has_locker = 2591 in tile.item_ids
        non_counter_items = [
            item_id
            for item_id in tile.item_ids
            if item_id not in {1617, 1618, 1621, 1623, 2591}
        ]
        counter_id = 1617 if rel_x == start_x else 1618
        tile.item_ids = unique_items([counter_id] + ([2591] if has_locker else []) + non_counter_items)
        changed += 1
    if changed:
        print(
            "[postprocess] Counters horizontales autotileados: "
            f"row={row_y}, x={start_x}..{end_x}, tiles={changed}"
        )


def add_mail_counter_horseshoe(
    plan: MapGenerationResponse,
    index: dict[tuple[int, int], TileDTO],
    bbox: tuple[int, int, int, int] | None,
    locker_counter_y: int,
) -> None:
    """Construye mostrador postal en L y coloca mailbox azul."""

    if bbox is not None:
        min_x, max_x, _north_y, south_y = bbox
        base_x = max(0, min(max_x - 2, plan.width - 1))
        start_y = min(plan.height - 1, locker_counter_y + 2)
        end_y = max(start_y, min(plan.height - 2, south_y - 1))
        horizontal_y = end_y
        horizontal_start_x = max(min_x + 2, base_x - 2)
    else:
        base_x = min(plan.width - 2, max(1, plan.width // 2 + 2))
        start_y = min(plan.height - 1, locker_counter_y + 2)
        end_y = min(plan.height - 2, start_y + 2)
        horizontal_y = end_y
        horizontal_start_x = max(1, base_x - 2)

    min_base_x = max(1, (bbox[0] + 2) if bbox is not None else 1)
    while base_x > min_base_x and any(
        1526 in index.get((base_x, rel_y), TileDTO(rel_x=base_x, rel_y=rel_y, ground_id=424, item_ids=[])).item_ids
        for rel_y in range(start_y, end_y + 1)
    ):
        base_x -= 1
    horizontal_start_x = min(horizontal_start_x, base_x)

    placed = 0
    for rel_y in range(start_y, end_y + 1):
        tile = get_or_create_tile(plan, index, base_x, rel_y, 424)
        if tile is None:
            continue
        tile.ground_id = 424
        ensure_items(tile, [1621], replace=True)
        placed += 1

    for rel_x in range(horizontal_start_x, base_x + 1):
        tile = get_or_create_tile(plan, index, rel_x, horizontal_y, 424)
        if tile is None:
            continue
        tile.ground_id = 424
        ensure_items(tile, [1621], replace=True)
        placed += 1

    corner_tile = get_or_create_tile(plan, index, base_x, horizontal_y, 424)
    if corner_tile is not None:
        corner_tile.ground_id = 424
        ensure_items(corner_tile, [1623], replace=True)

    mailbox_positions = [
        (horizontal_start_x - 1, horizontal_y),
        (horizontal_start_x, horizontal_y + 1),
        (base_x - 1, start_y),
    ]
    for rel_x, rel_y in mailbox_positions:
        tile = get_or_create_tile(plan, index, rel_x, rel_y, 424)
        if tile is None:
            continue
        tile.ground_id = 424
        ensure_items(tile, [2593])
        print(
            "[postprocess] Mail counter agregado: "
            f"counters={placed}, mailbox=({rel_x},{rel_y})"
        )
        return


def item_name(item_id: int) -> str:
    """Look up a local item name from items.xml."""

    for item in ALL_ITEMS:
        if item.get("id") == item_id:
            return str(item.get("name", "")).lower()
    return ""


def is_hanging_or_orphan_risk_item(item_id: int) -> bool:
    """Detect signs and wall lamps/torches that must not float."""

    name = item_name(item_id)
    if item_id == 1480:
        return False
    if item_id in {1810, 1815, 2038, 2040}:
        return True
    return any(keyword in name for keyword in ("sign", "wall lamp", "wall torch", "sconce"))


def find_nearest_wall_tile(
    source: TileDTO,
    index: dict[tuple[int, int], TileDTO],
    wall_ids: set[int],
) -> TileDTO | None:
    """Find the nearest wall tile for relocating a hanging item."""

    best_tile: TileDTO | None = None
    best_distance: int | None = None
    for tile in index.values():
        if not (set(tile.item_ids) & wall_ids):
            continue
        distance = abs(tile.rel_x - source.rel_x) + abs(tile.rel_y - source.rel_y)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_tile = tile
    return best_tile


def repair_orphan_hanging_items(index: dict[tuple[int, int], TileDTO]) -> None:
    """Move or remove signs/wall lamps left on free floor."""

    wall_ids = collect_wall_item_ids()
    if not wall_ids:
        return

    moves: list[tuple[int, TileDTO, TileDTO]] = []
    removals: list[tuple[int, TileDTO]] = []
    for tile in list(index.values()):
        if set(tile.item_ids) & wall_ids:
            continue
        for item_id in list(tile.item_ids):
            if not is_hanging_or_orphan_risk_item(item_id):
                continue
            target = find_nearest_wall_tile(tile, index, wall_ids)
            if target is None:
                removals.append((item_id, tile))
            else:
                moves.append((item_id, tile, target))

    for item_id, source, target in moves:
        remove_items(source, {item_id})
        fixed_id = item_id
        if item_id in {1810, 1815, 2038, 2040}:
            orientation = classify_wall_orientation(target)
            if item_id in {2038, 2040} and orientation == "horizontal":
                fixed_id = 2040
            elif item_id in {2038, 2040} and orientation == "vertical":
                fixed_id = 2038
            elif item_id in {1810, 1815} and orientation == "horizontal":
                fixed_id = 1810
            elif item_id in {1810, 1815} and orientation == "vertical":
                fixed_id = 1815
        ensure_items(target, [fixed_id])
        print(
            "[postprocess] Item colgante reubicado: "
            f"id={item_id}->{fixed_id}, from=({source.rel_x},{source.rel_y}), to=({target.rel_x},{target.rel_y})"
        )
    for item_id, source in removals:
        remove_items(source, {item_id})
        print(
            "[postprocess] Hanging item removed because no wall was found: "
            f"id={item_id}, at=({source.rel_x},{source.rel_y})"
        )


def repair_wall_fixture_orientation(index: dict[tuple[int, int], TileDTO]) -> None:
    """Force lamps and blackboards according to their shared wall orientation."""

    for tile in index.values():
        if not ({1810, 1815, 2038, 2040} & set(tile.item_ids)):
            continue
        orientation = classify_wall_orientation(tile)
        if orientation not in {"horizontal", "vertical"}:
            continue
        old_items = list(tile.item_ids)
        next_items = [item_id for item_id in tile.item_ids if item_id not in {1810, 1815, 2038, 2040}]
        if {2038, 2040} & set(old_items):
            next_items.append(2040 if orientation == "horizontal" else 2038)
        if {1810, 1815} & set(old_items):
            next_items.append(1810 if orientation == "horizontal" else 1815)
        tile.item_ids = unique_items(next_items)
        if old_items != tile.item_ids:
            print(
                "[postprocess] Fixture de muro orientado: "
                f"tile=({tile.rel_x},{tile.rel_y}), orientation={orientation}, items={tile.item_ids}"
            )


def find_nearest_ground_tile(
    source: TileDTO,
    index: dict[tuple[int, int], TileDTO],
    ground_id: int,
) -> TileDTO | None:
    """Find the nearest tile with a specific ground."""

    best_tile: TileDTO | None = None
    best_distance: int | None = None
    for tile in index.values():
        if tile.ground_id != ground_id:
            continue
        distance = abs(tile.rel_x - source.rel_x) + abs(tile.rel_y - source.rel_y)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_tile = tile
    return best_tile


def repair_exterior_street_lamps(index: dict[tuple[int, int], TileDTO]) -> None:
    """Mueve o elimina faroles 1480 que quedaron sobre piso interior."""

    moves: list[tuple[TileDTO, TileDTO]] = []
    removals: list[TileDTO] = []
    for tile in list(index.values()):
        if 1480 not in tile.item_ids or tile.ground_id == 724:
            continue
        target = find_nearest_ground_tile(tile, index, 724)
        if target is None:
            removals.append(tile)
        else:
            moves.append((tile, target))

    for source, target in moves:
        remove_items(source, {1480})
        ensure_items(target, [1480])
        print(
            "[postprocess] Farol exterior reubicado: "
            f"from=({source.rel_x},{source.rel_y}), to=({target.rel_x},{target.rel_y})"
        )
    for source in removals:
        remove_items(source, {1480})
        print(
            "[postprocess] Exterior lamp removed because no 724 ground was found: "
            f"at=({source.rel_x},{source.rel_y})"
        )


def repair_incomplete_benches(
    plan: MapGenerationResponse,
    index: dict[tuple[int, int], TileDTO],
) -> None:
    """Completa bancos compuestos definidos por RME."""

    bench_pairs = {
        1662: (1, 0, 1663),
        1663: (-1, 0, 1662),
        1664: (0, 1, 1665),
        1665: (0, -1, 1664),
    }
    for tile in list(index.values()):
        for item_id, (dx, dy, mate_id) in bench_pairs.items():
            if item_id not in tile.item_ids:
                continue
            mate_key = (tile.rel_x + dx, tile.rel_y + dy)
            mate_tile = index.get(mate_key)
            if mate_tile is not None and mate_id in mate_tile.item_ids:
                continue
            created = get_or_create_tile(
                plan,
                index,
                tile.rel_x + dx,
                tile.rel_y + dy,
                tile.ground_id,
            )
            if created is None:
                continue
            ensure_items(created, [mate_id])
            print(
                "[postprocess] Banco compuesto reparado: "
                f"{item_id}->{mate_id} at=({created.rel_x},{created.rel_y})"
            )


def normalize_depot_cubicles(
    plan: MapGenerationResponse,
    index: dict[tuple[int, int], TileDTO],
) -> None:
    """Ordena cubiculos depot pegados bajo la muralla norte."""

    depot_tiles = [
        tile
        for tile in index.values()
        if 1621 in tile.item_ids or 2591 in tile.item_ids
    ]
    if not depot_tiles:
        return

    bbox = building_bbox(index)
    ordered = sorted(depot_tiles, key=lambda tile: (tile.rel_y, tile.rel_x))
    if bbox is not None:
        min_x, max_x, north_y, _south_y = bbox
        max_cubicles = max(1, ((max_x - 1) - (min_x + 1)) // 2 + 1)
        if len(ordered) > max_cubicles:
            print(
                "[postprocess] Cubiculos depot recortados por limite este: "
                f"original={len(ordered)}, max={max_cubicles}"
            )
            for tile in ordered[max_cubicles:]:
                remove_items(tile, {1617, 1618, 1621, 1623, 2591})
            ordered = ordered[:max_cubicles]
        cubicle_count = len(ordered)
        counter_y = min(plan.height - 1, max(0, north_y + 1))
        start_x = max(0, max_x - 1 - (cubicle_count - 1) * 2)
        start_x = max(start_x, min_x + 1)
    else:
        cubicle_count = len(ordered)
        row_counts = Counter(tile.rel_y for tile in depot_tiles)
        counter_y = row_counts.most_common(1)[0][0]
        start_x = min(tile.rel_x for tile in depot_tiles)
        if start_x + (cubicle_count - 1) * 2 >= plan.width:
            start_x = max(0, plan.width - (cubicle_count * 2 - 1))

    for tile in depot_tiles:
        remove_items(tile, {1617, 1618, 1621, 1623, 2591})

    placed = 0
    for idx, _tile in enumerate(ordered):
        counter_x = start_x + idx * 2
        counter_tile = get_or_create_tile(plan, index, counter_x, counter_y, 424)
        if counter_tile is None:
            continue
        counter_tile.ground_id = 424
        ensure_items(counter_tile, [1621, 2591], replace=True)
        placed += 1

        player_y = counter_y + 1 if counter_y + 1 < plan.height else counter_y - 1
        player_tile = get_or_create_tile(plan, index, counter_x, player_y, 426)
        if player_tile is not None:
            player_tile.ground_id = 426

        if idx < cubicle_count - 1:
            counter_bridge = get_or_create_tile(plan, index, counter_x + 1, counter_y, 424)
            if counter_bridge is not None:
                counter_bridge.ground_id = 424
                ensure_items(counter_bridge, [1621], replace=True)

            divider_y = player_y
            divider_tile = get_or_create_tile(plan, index, counter_x + 1, divider_y, 424)
            if divider_tile is not None:
                divider_tile.ground_id = 424
                ensure_items(divider_tile, [1526], replace=True)

            divider_south = get_or_create_tile(plan, index, counter_x + 1, divider_y + 1, 424)
            if divider_south is not None:
                divider_south.ground_id = 424
                ensure_items(divider_south, [1526], replace=True)

    add_mail_counter_horseshoe(plan, index, bbox, counter_y)
    if placed:
        normalize_horizontal_counter_row(index, counter_y, start_x, start_x + (placed - 1) * 2)
    force_south_wall_end_caps(index)

    print(
        "[postprocess] Cubiculos depot normalizados: "
        f"count={placed}, orientation=horizontal, counter_y={counter_y}, start_x={start_x}"
    )


def apply_tibia_architect_rules(
    response_data: MapGenerationResponse,
    archetype_tags: list,
) -> MapGenerationResponse:
    """Apply deterministic architectural rules before injecting the OTBM."""

    tags = {str(tag) for tag in archetype_tags}
    index = build_tile_index(response_data)

    if "depot" in tags:
        force_building_floor(response_data, index, 424)
        repair_exterior_street_lamps(index)
        normalize_depot_cubicles(response_data, index)

    repair_orphan_hanging_items(index)
    repair_wall_fixture_orientation(index)
    repair_incomplete_benches(response_data, index)
    repair_exterior_street_lamps(index)

    response_data.tiles = sorted(index.values(), key=lambda tile: (tile.rel_y, tile.rel_x))
    for tile in response_data.tiles:
        tile.item_ids = unique_items(tile.item_ids)
    return response_data


def validate_generated_map(plan: MapGenerationResponse, width: int, height: int) -> None:
    """Valida limites dinamicos y roles semanticos."""

    if plan.width != width or plan.height != height:
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini returned dimensions different from the requested ones: "
                f"response={plan.width}x{plan.height}, request={width}x{height}"
            ),
        )
    if not plan.tiles:
        raise HTTPException(status_code=502, detail="Gemini returned no tiles.")
    has_macro_roles = any(tile.role in MACRO_ROLES for tile in plan.tiles)
    is_macro_plan = has_macro_roles and all(tile.role in MACRO_ROLES for tile in plan.tiles)
    if has_macro_roles and not is_macro_plan:
        raise HTTPException(status_code=502, detail="Gemini mezclo macro-roles con roles finos.")
    max_rel_x = (width + CHUNK_SIZE - 1) // CHUNK_SIZE if is_macro_plan else width
    max_rel_y = (height + CHUNK_SIZE - 1) // CHUNK_SIZE if is_macro_plan else height
    max_tiles_allowed = max_rel_x * max_rel_y if is_macro_plan else width * height

    if len(plan.tiles) > MAX_TILES or len(plan.tiles) > max_tiles_allowed:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned too many tiles: {len(plan.tiles)}",
        )

    seen: set[tuple[int, int]] = set()
    for tile in plan.tiles:
        if tile.rel_x < 0 or tile.rel_y < 0 or tile.rel_x >= max_rel_x or tile.rel_y >= max_rel_y:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Gemini returned a tile outside the requested area: "
                    f"rel=({tile.rel_x},{tile.rel_y}), area={max_rel_x}x{max_rel_y}"
                ),
            )
        coords = (tile.rel_x, tile.rel_y)
        if coords in seen:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini returned duplicate coordinates: {coords}",
            )
        seen.add(coords)
        if tile.role not in SEMANTIC_ROLES:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini returned a disallowed role: {tile.role}",
            )
        if is_macro_plan and tile.role not in MACRO_ROLES:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini mezclo macro-roles con roles finos: {tile.role}",
            )


def validate_palette_usage(
    plan: MapGenerationResponse,
    palette: dict[str, Any],
    semantic_context: dict[str, Any],
) -> None:
    """Impide que Gemini inyecte IDs fuera del ADN o vocabulario ampliado."""

    allowed_ids = {safe_int(item_id) for item_id in palette.get("ids_recomendados", [])}
    allowed_ids.update(safe_int(item_id) for item_id in semantic_context.get("ids_contexto_ampliado", []))
    allowed_ids.discard(None)
    if not allowed_ids:
        raise HTTPException(status_code=500, detail="El contexto dinamico no contiene IDs validos.")

    ground_ids = {safe_int(palette.get("main_ground"))}
    ground_ids.update(safe_int(item_id) for item_id in palette.get("secondary_grounds", []))
    ground_ids.update(
        safe_int(item.get("id"))
        for item in semantic_context.get("catalog_grounds", [])
        if isinstance(item, dict)
    )
    ground_ids.update(
        safe_int(item.get("id"))
        for item in semantic_context.get("depot_760_grounds", [])
        if isinstance(item, dict)
    )
    ground_ids.discard(None)
    if not ground_ids:
        ground_ids = allowed_ids

    invalid_ground = sorted({tile.ground_id for tile in plan.tiles if tile.ground_id not in ground_ids})
    invalid_items = sorted(
        {
            item_id
            for tile in plan.tiles
            for item_id in tile.item_ids
            if item_id not in allowed_ids
        }
    )
    if invalid_ground or invalid_items:
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini used IDs outside the archetype DNA and expanded vocabulary. "
                f"invalid_grounds={invalid_ground}, invalid_items={invalid_items}"
            ),
        )

    if "manual_ensamblaje_depot_760" in semantic_context:
        depot_ground_ids = {424, 426, 724}
        plan_index = build_tile_index(plan)
        ground_items_removed: set[int] = set()
        locker_repairs = 0
        for tile in plan.tiles:
            before_items = list(tile.item_ids)
            remove_items(tile, depot_ground_ids)
            ground_items_removed.update(set(before_items) - set(tile.item_ids))
            if 2591 in tile.item_ids:
                counter_ids = {1617, 1618, 1621, 1623}
                counter_positions = [
                    tile.item_ids.index(item_id)
                    for item_id in counter_ids
                    if item_id in tile.item_ids
                ]
                locker_position = tile.item_ids.index(2591)
                if not counter_positions:
                    tile.item_ids = [1618] + tile.item_ids
                    locker_repairs += 1
                elif min(counter_positions) > locker_position:
                    counters = [item_id for item_id in tile.item_ids if item_id in counter_ids]
                    rest = [item_id for item_id in tile.item_ids if item_id not in counter_ids]
                    tile.item_ids = counters + rest
                    locker_repairs += 1

        repair_exterior_street_lamps(plan_index)
        plan.tiles = sorted(plan_index.values(), key=lambda tile: (tile.rel_y, tile.rel_x))
        if ground_items_removed or locker_repairs:
            print(
                "[validator] Manual depot saneado: "
                f"grounds_como_items_removidos={sorted(ground_items_removed)}, "
                f"lockers_reparados={locker_repairs}"
            )


def to_injector_tiles(plan: MapGenerationResponse) -> list[dict[str, int | list[int]]]:
    """Adapt Gemini rel_x/rel_y to the current inject_tiles contract."""

    return [
        {
            "x": tile.rel_x,
            "y": tile.rel_y,
            "ground_id": tile.ground_id,
            "item_ids": tile.item_ids,
        }
        for tile in plan.tiles
    ]


def summarize_ids(plan: MapGenerationResponse, limit: int = 12) -> list[int]:
    """Return the most-used IDs, including ground and decoration."""

    counter: Counter[int] = Counter()
    for tile in plan.tiles:
        counter[tile.ground_id] += 1
        counter.update(tile.item_ids)
    return [item_id for item_id, _count in counter.most_common(limit)]


def summarize_materialized_ids(
    tiles: list[dict[str, int | list[int]]],
    limit: int = 12,
) -> list[int]:
    """Resume IDs reales despues del autotiling."""

    counter: Counter[int] = Counter()
    for tile in tiles:
        ground_id = tile.get("ground_id")
        if isinstance(ground_id, int):
            counter[ground_id] += 1
        item_ids = tile.get("item_ids", [])
        if isinstance(item_ids, list):
            counter.update(item_id for item_id in item_ids if isinstance(item_id, int))
    return [item_id for item_id, _count in counter.most_common(limit)]


def tile_z_offset(tile: dict[str, Any]) -> int:
    """Read optional z_offset emitted by the macro autotiler."""

    try:
        return int(tile.get("z_offset", 0))
    except (TypeError, ValueError):
        return 0


def group_tiles_by_z_offset(tiles: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Group materialized tiles by relative height plane."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for tile in tiles:
        z_offset = tile_z_offset(tile)
        stripped = dict(tile)
        stripped.pop("z_offset", None)
        grouped.setdefault(z_offset, []).append(stripped)
    return grouped


def ordered_z_offsets(grouped: dict[int, list[dict[str, Any]]]) -> list[int]:
    """Order planes so the surface is injected first, followed by adjacent layers."""

    return sorted(grouped, key=lambda value: (0 if value == 0 else 1, value))


def cleanup_ground_for_group(tiles: list[dict[str, Any]], fallback: int) -> int:
    """Select a coherent cleanup ground for a plane."""

    counter: Counter[int] = Counter()
    for tile in tiles:
        ground_id = safe_int(tile.get("ground_id"))
        if ground_id is not None:
            counter[ground_id] += 1
    return counter.most_common(1)[0][0] if counter else fallback


def enforce_vertical_biome_safety(tiles: list[dict[str, Any]], selected_slice_tag: str | None) -> None:
    """Sanitize vertical layers so they do not inherit surface biomes."""

    if selected_slice_tag != "nature_surface":
        return

    lower_repairs = 0
    upper_repairs = 0
    forbidden_surface_items = {2701, 2702, 2703, 2767, 2768}
    for tile in tiles:
        z_offset = tile_z_offset(tile)
        if z_offset > 0:
            if safe_int(tile.get("ground_id")) in {4526, 724, 0, None}:
                tile["ground_id"] = 103
                lower_repairs += 1
            raw_items = tile.get("item_ids", [])
            if isinstance(raw_items, list):
                cleaned = [item_id for item_id in raw_items if safe_int(item_id) not in forbidden_surface_items]
                if cleaned != raw_items:
                    tile["item_ids"] = cleaned
                    lower_repairs += 1
        elif z_offset < 0:
            if safe_int(tile.get("ground_id")) == 4526:
                tile["ground_id"] = 424
                upper_repairs += 1

    if lower_repairs or upper_repairs:
        print(
            "[server] Z-Biome safety aplicado: "
            f"tag={selected_slice_tag}, lower_repairs={lower_repairs}, upper_repairs={upper_repairs}"
        )


def render_layer_debugs(tiles: list[dict[str, Any]]) -> dict[int, Path]:
    """Hornea renders PNG por cada plano z_offset presente."""

    outputs: dict[int, Path] = {}
    grouped = group_tiles_by_z_offset(tiles)
    for z_offset, layer_tiles in grouped.items():
        if z_offset == 0:
            path = resolve_relative("../../template/debug_render.png")
        else:
            path = resolve_relative(f"../../template/debug_render_z{z_offset:+d}.png")
        try:
            outputs[z_offset] = render_debug_map(layer_tiles, path)
            print(f"[server] Debug render layer z_offset={z_offset}: {outputs[z_offset]}")
            if z_offset < 0:
                alias = resolve_relative("../../template/debug_render_p0.png")
                render_debug_map(layer_tiles, alias)
                print(f"[server] Upper debug render alias: {alias}")
        except Exception as exc:
            print(f"[server] Skipped debug render layer z_offset={z_offset}: {exc}")
    return outputs


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    """Limita un entero a un rango inclusivo."""

    return max(minimum, min(maximum, value))


def spawn_entries_from_slice(pattern_slice: dict[str, Any] | None, width: int, height: int) -> list[dict[str, Any]]:
    """Extract creatures from targeted metadata in the selected slice."""

    if not isinstance(pattern_slice, dict) or not pattern_slice.get("targeted"):
        return []

    raw_spawns: list[Any] = []
    if isinstance(pattern_slice.get("spawns"), list):
        raw_spawns.extend(pattern_slice["spawns"])
    elif isinstance(pattern_slice.get("spawn"), dict):
        raw_spawns.append(pattern_slice["spawn"])

    origin = pattern_slice.get("origin", {})
    origin_x = plain_int(origin.get("x")) if isinstance(origin, dict) else None
    origin_y = plain_int(origin.get("y")) if isinstance(origin, dict) else None
    source_w = safe_int(pattern_slice.get("width")) or width
    source_h = safe_int(pattern_slice.get("height")) or height
    scale_x = width / max(1, source_w)
    scale_y = height / max(1, source_h)
    max_offset_x = max(0, width // 2)
    max_offset_y = max(0, height // 2)

    entries: list[dict[str, Any]] = []
    for raw_spawn in raw_spawns:
        if not isinstance(raw_spawn, dict):
            continue
        name = str(raw_spawn.get("creature") or raw_spawn.get("name") or "").strip()
        if not name:
            continue

        rel_x = 0
        rel_y = 0
        spawn_x = plain_int(raw_spawn.get("x"))
        spawn_y = plain_int(raw_spawn.get("y"))
        if spawn_x is not None and spawn_y is not None and origin_x is not None and origin_y is not None:
            source_rel_x = spawn_x - origin_x
            source_rel_y = spawn_y - origin_y
            rel_x = round((source_rel_x - source_w // 2) * scale_x)
            rel_y = round((source_rel_y - source_h // 2) * scale_y)
            rel_x = clamp_int(rel_x, -max_offset_x, max_offset_x)
            rel_y = clamp_int(rel_y, -max_offset_y, max_offset_y)

        entries.append(
            {
                "name": name,
                "x": rel_x,
                "y": rel_y,
                "z": 0,
            }
        )
    return entries


def write_generated_spawn_xml(
    pattern_slice: dict[str, Any] | None,
    width: int,
    height: int,
    start_x: int,
    start_y: int,
    z: int,
) -> Path | None:
    """Generate template/generated-spawn.xml for the selected targeted slice."""

    entries = spawn_entries_from_slice(pattern_slice, width, height)
    if not entries:
        return None

    center_x = start_x + width // 2
    center_y = start_y + height // 2
    radius = max(1, min(width, height) // 2)
    root = ElementTree.Element("spawns")
    spawn_node = ElementTree.SubElement(
        root,
        "spawn",
        {
            "centerx": str(center_x),
            "centery": str(center_y),
            "centerz": str(z),
            "radius": str(radius),
        },
    )
    for entry in entries:
        ElementTree.SubElement(
            spawn_node,
            "monster",
            {
                "name": str(entry["name"]),
                "x": str(entry["x"]),
                "y": str(entry["y"]),
                "z": str(entry["z"]),
            },
        )

    output_path = resolve_relative(GENERATED_SPAWN_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.indent(root, space="  ")
    ElementTree.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"[server] generated-spawn.xml written: {output_path} ({len(entries)} creatures)")
    return output_path


catalog = load_catalog()
ALL_ITEMS = load_all_items_from_xml()
CONFIG_ARCHETYPES = load_archetypes()
CONFIG_SLICES_POOL = load_slices_pool()
CONFIG_SPAWN_INDEX = load_spawn_index()
RME_GEOMETRY_RULES = load_rme_geometry_rules()
GENAI_CLIENT = create_genai_client()
app = FastAPI(
    title="RME AI Map Generator",
    version="0.1.0",
    description="Local API for generating Tibia 7.60 OTBM zones with Google GenAI and injecting them into RME.",
)


@app.post("/generate-map", response_model=GenerateMapResponse)
def generate_map(request: GenerateMapRequest) -> GenerateMapResponse:
    """Generate a zone with Gemini and inject it into template/generated_760.otbm."""

    if request.width > MAX_DIMENSION or request.height > MAX_DIMENSION:
        raise HTTPException(status_code=400, detail="Maximum size is 30x30")
    if request.width < 1 or request.height < 1:
        raise HTTPException(status_code=400, detail="Minimum size is 1x1")

    spawn_hint = find_spawn_for_prompt(request.prompt, CONFIG_SPAWN_INDEX)
    selected_archetype_name, selected_archetype = select_archetype_for_prompt(
        request.prompt,
        CONFIG_ARCHETYPES,
        spawn_hint=spawn_hint,
    )
    selected_slice_tag, selected_slice = select_slice_for_prompt(
        request.prompt,
        CONFIG_SLICES_POOL,
        request.width,
        request.height,
        spawn_hint=spawn_hint,
    )
    dynamic_palette = extract_dynamic_palette(selected_archetype_name, selected_archetype)
    semantic_context = build_semantic_catalog_context(catalog, ALL_ITEMS, dynamic_palette)
    geometry_rules = build_prompt_geometry_rules(
        RME_GEOMETRY_RULES,
        request.prompt,
        [str(tag) for tag in dynamic_palette.get("tags", [])],
    )
    merge_geometry_ids_into_context(semantic_context, geometry_rules)

    client = GENAI_CLIENT
    system_prompt = build_system_prompt(
        catalog,
        selected_archetype_name,
        selected_archetype,
        dynamic_palette,
        semantic_context,
        geometry_rules,
        width=request.width,
        height=request.height,
    )
    user_content = (
        f"User prompt: {request.prompt}\n"
        f"Required dimensions in tiles: width={request.width}, height={request.height}.\n"
        f"Required macro-grid: {((request.width + CHUNK_SIZE - 1) // CHUNK_SIZE)}x{((request.height + CHUNK_SIZE - 1) // CHUNK_SIZE)} chunks of {CHUNK_SIZE}x{CHUNK_SIZE}.\n"
        f"Selected archetype reference: {selected_archetype_name}.\n"
        f"Selected real slice inference pattern: {selected_slice_tag or 'none'}.\n"
        f"Detected ecological spawn: {spawn_hint.get('name') if spawn_hint else 'none'}.\n"
        "Return only macro roles: spawn_hub_dense, defensive_perimeter, wild_surroundings, camp_amenities. Do not return ground_id, item_ids, or numeric IDs."
    )

    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MapGenerationResponse,
            system_instruction=system_prompt,
        )
        response = generate_content_with_model_failover(client, user_content, config)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error calling Gemini: {exc}") from exc

    if not response.text:
        raise HTTPException(status_code=502, detail="Gemini returned no structured JSON text.")

    try:
        parsed = MapGenerationResponse.model_validate_json(response.text)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned JSON incompatible with the schema: {exc}",
        ) from exc

    validate_generated_map(parsed, width=request.width, height=request.height)
    tiles_to_inject = materialize_semantic_map(
        [
            SemanticTile(rel_x=tile.rel_x, rel_y=tile.rel_y, role=tile.role)
            for tile in parsed.tiles
        ],
        width=request.width,
        height=request.height,
        rme_rules=RME_GEOMETRY_RULES,
        archetype=selected_archetype,
        pattern_slice=selected_slice,
    )
    enforce_vertical_biome_safety(tiles_to_inject, selected_slice_tag)
    cleanup_ground_id = int(dynamic_palette["main_ground"])
    debug_render_path: Path | None = None
    try:
        layer_renders = render_layer_debugs(tiles_to_inject)
        debug_render_path = layer_renders.get(0)
        visual_feedback_payload = build_visual_feedback_payload(debug_render_path)
        print(f"[server] debug_render.png generado: {debug_render_path}")
        print(f"[server] Visual Agentic Loop preparado: {visual_feedback_payload}")
    except Exception as exc:
        print(f"[server] Skipped debug render: {exc}")

    refined = run_visual_feedback_phase(
        client,
        parsed,
        debug_render_path,
        request.prompt,
        selected_slice_tag,
        request.width,
        request.height,
    )
    if refined is not parsed:
        parsed = refined
        tiles_to_inject = materialize_semantic_map(
            [
                SemanticTile(rel_x=tile.rel_x, rel_y=tile.rel_y, role=tile.role)
                for tile in parsed.tiles
            ],
            width=request.width,
            height=request.height,
            rme_rules=RME_GEOMETRY_RULES,
            archetype=selected_archetype,
            pattern_slice=selected_slice,
        )
        enforce_vertical_biome_safety(tiles_to_inject, selected_slice_tag)
        try:
            layer_renders = render_layer_debugs(tiles_to_inject)
            debug_render_path = layer_renders.get(0)
            print(f"[server] debug_render.png actualizado tras critica visual: {debug_render_path}")
        except Exception as exc:
            print(f"[server] Skipped refined debug render: {exc}")

    try:
        grouped_tiles = group_tiles_by_z_offset(tiles_to_inject)
        output_path: Path | None = None
        current_input: str | Path = TEMPLATE_PATH
        for z_offset in ordered_z_offsets(grouped_tiles):
            layer_tiles = grouped_tiles[z_offset]
            target_z = TARGET_Z + z_offset
            layer_cleanup_ground = cleanup_ground_id if z_offset == 0 else cleanup_ground_for_group(layer_tiles, cleanup_ground_id)
            output_path = inject_tiles(
                current_input,
                OUTPUT_PATH,
                layer_tiles,
                start_x=START_X,
                start_y=START_Y,
                z=target_z,
                cleanup_ground_id=layer_cleanup_ground,
            )
            current_input = output_path
            print(
                "[server] Capa OTBM inyectada: "
                f"z={target_z}, z_offset={z_offset}, tiles={len(layer_tiles)}, cleanup_ground={layer_cleanup_ground}"
            )
        if output_path is None:
            raise RuntimeError("There were no tiles to inject.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error inyectando OTBM: {exc}") from exc

    spawn_output_path = write_generated_spawn_xml(
        selected_slice,
        request.width,
        request.height,
        START_X,
        START_Y,
        TARGET_Z,
    )
    main_ids = summarize_materialized_ids(tiles_to_inject)
    return GenerateMapResponse(
        message="Mapa modificado exitosamente por IA.",
        output_path=str(output_path),
        spawn_output_path=str(spawn_output_path) if spawn_output_path else None,
        debug_render_path=str(debug_render_path) if debug_render_path else None,
        tiles_modified=len(parsed.tiles),
        main_ids_used=main_ids,
        archetype_used=selected_archetype_name,
        cleanup_ground_id=cleanup_ground_id,
        start_x=START_X,
        start_y=START_Y,
        z=TARGET_Z,
    )
