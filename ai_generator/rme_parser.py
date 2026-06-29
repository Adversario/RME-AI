"""
Parser liviano de reglas geometricas nativas de RME para Tibia 7.60.

Lee archivos como data/760/walls.xml y data/760/doodads.xml y devuelve
combinaciones atomicas aptas para alimentar a un LLM sin depender de nombres
genericos de items.xml.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


BASE_DIR = Path(__file__).resolve().parent
DATA_760_CANDIDATES = (
    BASE_DIR / "../../data/760",
    BASE_DIR / "../data/760",
)


def resolve_data_760() -> Path:
    """Encuentra la carpeta data/760 desde ai_generator."""

    for candidate in DATA_760_CANDIDATES:
        path = candidate.resolve()
        if path.is_dir():
            return path
    raise FileNotFoundError("No se encontro la carpeta data/760")


def read_xml_root(path: Path) -> ElementTree.Element:
    """Parsea XML de RME tolerando lineas comentadas estilo Lua."""

    text = path.read_text(encoding="utf-8")
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("--"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    return ElementTree.fromstring(cleaned)


def int_attr(element: ElementTree.Element, name: str) -> int | None:
    """Lee un atributo entero si existe."""

    value = element.attrib.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def item_ids_from(parent: ElementTree.Element) -> list[int]:
    """Extrae IDs de items hijos directos."""

    ids: list[int] = []
    for item in parent.findall("item"):
        item_id = int_attr(item, "id")
        if item_id is not None:
            ids.append(item_id)
    return ids


def primary_item_id(parent: ElementTree.Element) -> int | None:
    """Devuelve el item principal segun mayor chance y orden de aparicion."""

    best_id: int | None = None
    best_chance = -1
    for index, item in enumerate(parent.findall("item")):
        item_id = int_attr(item, "id")
        if item_id is None:
            continue
        chance = int_attr(item, "chance")
        if chance is None:
            chance = 0
        # El indice negativo mantiene estable el primer empate.
        score = chance * 10000 - index
        if score > best_chance:
            best_chance = score
            best_id = item_id
    return best_id


def parse_doors(parent: ElementTree.Element) -> list[dict[str, Any]]:
    """Extrae puertas/ventanas asociadas a un tramo de muro."""

    doors: list[dict[str, Any]] = []
    for door in parent.findall("door"):
        door_id = int_attr(door, "id")
        if door_id is None:
            continue
        doors.append(
            {
                "id": door_id,
                "type": door.attrib.get("type", ""),
                "open": door.attrib.get("open"),
            }
        )
    return doors


def parse_walls_xml(path: Path | None = None) -> dict[str, Any]:
    """Parsea walls.xml y devuelve brushes de muro con orientacion RME."""

    data_dir = resolve_data_760()
    xml_path = path or data_dir / "walls.xml"
    root = read_xml_root(xml_path)
    brushes: dict[str, Any] = {}

    for brush in root.findall("brush"):
        if brush.attrib.get("type") != "wall":
            continue
        name = brush.attrib.get("name", "").strip()
        if not name:
            continue

        record: dict[str, Any] = {
            "name": name,
            "server_lookid": int_attr(brush, "server_lookid"),
            "horizontal": None,
            "vertical": None,
            "corner_variants": [],
            "pole": None,
            "doors_horizontal": [],
            "doors_vertical": [],
            "all_items_by_role": {},
        }

        for wall in brush.findall("wall"):
            wall_type = wall.attrib.get("type", "")
            ids = item_ids_from(wall)
            primary = primary_item_id(wall)
            record["all_items_by_role"][wall_type] = ids

            if wall_type == "horizontal":
                record["horizontal"] = primary
                record["doors_horizontal"] = parse_doors(wall)
            elif wall_type == "vertical":
                record["vertical"] = primary
                record["doors_vertical"] = parse_doors(wall)
            elif wall_type == "corner":
                record["corner_variants"] = ids
            elif wall_type == "pole":
                record["pole"] = primary

        if record["horizontal"] or record["vertical"] or record["corner_variants"]:
            brushes[name] = record

    return {"source": str(xml_path), "walls": brushes}


def parse_composite(composite: ElementTree.Element) -> dict[str, Any]:
    """Extrae una composicion relativa de tiles."""

    tiles: list[dict[str, Any]] = []
    for tile in composite.findall("tile"):
        x = int_attr(tile, "x") or 0
        y = int_attr(tile, "y") or 0
        z = int_attr(tile, "z") or 0
        ids = item_ids_from(tile)
        tiles.append({"x": x, "y": y, "z": z, "item_ids": ids})

    max_x = max((tile["x"] for tile in tiles), default=0)
    max_y = max((tile["y"] for tile in tiles), default=0)
    orientation = "single"
    if max_x > 0 and max_y == 0:
        orientation = "horizontal"
    elif max_y > 0 and max_x == 0:
        orientation = "vertical"
    elif max_x > 0 and max_y > 0:
        orientation = "area"

    return {
        "orientation": orientation,
        "width": max_x + 1,
        "height": max_y + 1,
        "tiles": tiles,
    }


def parse_doodads_xml(path: Path | None = None) -> dict[str, Any]:
    """Parsea doodads.xml y preserva alternates, composites, tables y carpets."""

    data_dir = resolve_data_760()
    xml_path = path or data_dir / "doodads.xml"
    root = read_xml_root(xml_path)
    brushes: dict[str, Any] = {}

    for brush in root.findall("brush"):
        name = brush.attrib.get("name", "").strip()
        if not name:
            continue

        record: dict[str, Any] = {
            "name": name,
            "type": brush.attrib.get("type", ""),
            "server_lookid": int_attr(brush, "server_lookid"),
            "single_items": item_ids_from(brush),
            "tables": {},
            "carpets": {},
            "composites": [],
        }

        for table in brush.findall("table"):
            align = table.attrib.get("align", "unknown")
            record["tables"][align] = item_ids_from(table)

        for carpet in brush.findall("carpet"):
            align = carpet.attrib.get("align", "unknown")
            item_id = int_attr(carpet, "id")
            if item_id is not None:
                record["carpets"][align] = item_id

        for composite in brush.findall(".//composite"):
            parsed = parse_composite(composite)
            if parsed["tiles"]:
                record["composites"].append(parsed)

        if record["single_items"] or record["tables"] or record["carpets"] or record["composites"]:
            brushes[name] = record

    return {"source": str(xml_path), "doodads": brushes}


def load_rme_geometry_rules() -> dict[str, Any]:
    """Carga reglas geometricas desde walls.xml y doodads.xml."""

    walls = parse_walls_xml()
    doodads = parse_doodads_xml()
    return {
        "walls_source": walls["source"],
        "doodads_source": doodads["source"],
        "walls": walls["walls"],
        "doodads": doodads["doodads"],
    }


def normalize_text(value: str) -> str:
    """Normaliza texto para busquedas simples."""

    return re.sub(r"[^a-z0-9_ ]+", " ", value.lower())


def select_named_entries(entries: dict[str, Any], keywords: set[str], limit: int) -> dict[str, Any]:
    """Selecciona entradas por keywords con fallback a las primeras."""

    selected: dict[str, Any] = {}
    for name, value in entries.items():
        normalized = normalize_text(name)
        if any(keyword in normalized for keyword in keywords):
            selected[name] = value
        if len(selected) >= limit:
            return selected

    if selected:
        return selected

    for name, value in entries.items():
        selected[name] = value
        if len(selected) >= limit:
            break
    return selected


def keywords_for_prompt(prompt: str, tags: list[str]) -> set[str]:
    """Convierte prompt/tags a keywords utiles para brushes RME."""

    source = normalize_text(" ".join([prompt, *tags]))
    keywords: set[str] = set()
    mapping = {
        "depot": {"stone", "brick", "counter", "bench", "locker", "table"},
        "city": {"stone", "brick", "counter", "bench", "table"},
        "thais": {"stone", "brick", "counter", "bench"},
        "temple": {"stone", "marble", "altar"},
        "cave": {"stone", "rock", "rubble", "debris"},
        "ice": {"snow", "ice", "rock"},
        "snow": {"snow", "ice"},
        "wood": {"wood", "wooden", "bench", "table"},
        "shop": {"counter", "table", "bench", "wood"},
    }
    for token, expanded in mapping.items():
        if token in source:
            keywords.update(expanded)

    for token in source.split():
        if len(token) >= 4:
            keywords.add(token)

    if not keywords:
        keywords.update({"stone", "brick", "table", "bench"})
    return keywords


def compact_wall_rule(rule: dict[str, Any]) -> dict[str, Any]:
    """Reduce una regla de muro a la forma que necesita Gemini."""

    return {
        "horizontal": rule.get("horizontal"),
        "vertical": rule.get("vertical"),
        "corner_variants": rule.get("corner_variants", [])[:8],
        "pole": rule.get("pole"),
        "doors_horizontal": rule.get("doors_horizontal", [])[:6],
        "doors_vertical": rule.get("doors_vertical", [])[:6],
    }


def compact_doodad_rule(rule: dict[str, Any]) -> dict[str, Any]:
    """Reduce una regla de doodad/table a combinaciones atomicas."""

    return {
        "type": rule.get("type"),
        "single_items": rule.get("single_items", [])[:12],
        "tables": rule.get("tables", {}),
        "carpets": rule.get("carpets", {}),
        "composites": rule.get("composites", [])[:8],
    }


def build_prompt_geometry_rules(
    rules: dict[str, Any],
    prompt: str,
    tags: list[str],
    wall_limit: int = 8,
    doodad_limit: int = 10,
) -> dict[str, Any]:
    """Construye un bloque compacto de reglas geometricas relevantes."""

    keywords = keywords_for_prompt(prompt, tags)
    selected_walls = select_named_entries(rules.get("walls", {}), keywords, wall_limit)
    selected_doodads = select_named_entries(rules.get("doodads", {}), keywords, doodad_limit)

    return {
        "keywords": sorted(keywords),
        "muros": {name: compact_wall_rule(rule) for name, rule in selected_walls.items()},
        "doodads_y_muebles": {
            name: compact_doodad_rule(rule) for name, rule in selected_doodads.items()
        },
        "uso": [
            "Usa horizontal y vertical como tramos planos.",
            "Usa corner_variants y pole para cerrar esquinas, encuentros y terminaciones.",
            "Usa doors_horizontal/doors_vertical cuando abras un muro.",
            "Usa composites respetando x/y relativos exactos; no repitas una sola pieza.",
            "Usa tables por align para counters, mesas y muebles conectados.",
        ],
    }


def format_geometry_rules_for_prompt(geometry: dict[str, Any]) -> str:
    """Serializa reglas geometricas compactas para el prompt."""

    return json.dumps(geometry, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    parsed = load_rme_geometry_rules()
    sample = build_prompt_geometry_rules(parsed, "depot thais con benches y counters", ["city", "depot"])
    print(format_geometry_rules_for_prompt(sample))
