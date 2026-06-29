"""
Extractor de arquetipos esteticos desde mapas OTBM grandes.

El objetivo es minar fragmentos reales de un mapa original y convertirlos en
JSON legible por humanos y por LLMs. Este script evita construir el arbol OTBM
completo: recorre los bytes del archivo y solo materializa tiles dentro de las
cajas solicitadas.

Uso:
    python ai_generator/extractor.py

Edita ARCHETYPE_TARGETS para apuntar a tu mapa de 90 MB y a coordenadas reales.
"""

from __future__ import annotations

import json
import mmap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from injector import (
    ESCAPE_CHAR,
    NODE_END,
    NODE_START,
    OTBM_ITEM,
    OTBM_TILE,
    OTBM_TILE_AREA,
    OTSYS_ROOT,
    read_props_until_control,
    resolve_relative,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "archetypes.json"
SOURCE_OTBM_PATH = "../../template/real map/world.otbm"
ITEMS_XML_CANDIDATES = (
    BASE_DIR / "../../data/760/items.xml",
    BASE_DIR / "../data/760/items.xml",
)

ARCHETYPE_TARGETS: dict[str, dict[str, Any]] = {
    # --- URBANOS (CORREGIDOS) ---
    "thais_depot_p1": {
        "x": 32350,
        "y": 32220,
        "z": 7,
        "size": 16,
        "tags": ["city", "depot", "thais", "lockers", "mail"],
    },

    # --- SUBTERRANEOS DE FOLDA (SUBDIVIDIDOS POR CUARTOS) ---
    "folda_ice_hall": {
        "x": 32050,
        "y": 31650,
        "z": 8,
        "size": 18,
        "tags": ["cave", "folda", "ice", "snow", "room", "hall"],
    },
    "folda_ice_corridor": {
        "x": 32030,
        "y": 31630,
        "z": 8,
        "size": 14,
        "tags": ["cave", "folda", "ice", "snow", "corridor", "pasillo"],
    },
    "folda_ice_depths": {
        "x": 32070,
        "y": 31670,
        "z": 9,
        "size": 16,
        "tags": ["cave", "folda", "ice", "deep", "floor"],
    },
}


@dataclass(frozen=True)
class NodeHeader:
    """Cabecera liviana de un nodo OTBM."""

    node_type: int
    props: bytes
    children_offset: int


def resolve_items_xml() -> Path | None:
    """Devuelve items.xml si existe en la instalacion local."""

    for candidate in ITEMS_XML_CANDIDATES:
        path = candidate.resolve()
        if path.is_file():
            return path
    return None


def load_item_names() -> dict[int, str]:
    """Carga id -> name desde data/760/items.xml para clasificacion heuristica."""

    items_xml = resolve_items_xml()
    if items_xml is None:
        return {}

    names: dict[int, str] = {}
    for _event, elem in ElementTree.iterparse(items_xml, events=("end",)):
        if elem.tag == "item":
            item_id = elem.attrib.get("id")
            name = elem.attrib.get("name")
            if item_id and name:
                try:
                    names[int(item_id)] = name
                except ValueError:
                    pass
        elem.clear()
    return names


ITEM_NAMES = load_item_names()


def parse_node_header(data: bytes, offset: int) -> NodeHeader:
    """Lee cabecera y propiedades de un nodo sin consumir sus hijos."""

    if offset >= len(data) or data[offset] != NODE_START:
        raise ValueError(f"Se esperaba NODE_START en offset {offset}")
    if offset + 1 >= len(data):
        raise ValueError("Nodo incompleto: falta node_type")

    node_type = data[offset + 1]
    props, children_offset = read_props_until_control(data, offset + 2)
    return NodeHeader(node_type=node_type, props=props, children_offset=children_offset)


def skip_node(data: bytes, offset: int) -> int:
    """Salta un nodo completo sin materializarlo.

    Esta version esta optimizada para mapas grandes: no parsea cabeceras hijas,
    solo cuenta NODE_START/NODE_END sin olvidar ESCAPE_CHAR.
    """

    if offset >= len(data) or data[offset] != NODE_START:
        raise ValueError(f"Se esperaba NODE_START en offset {offset}")

    depth = 1
    cursor = offset + 2  # salta NODE_START + node_type
    while cursor < len(data) and depth > 0:
        byte = data[cursor]
        if byte == ESCAPE_CHAR:
            cursor += 2
        elif byte == NODE_START:
            depth += 1
            cursor += 2
        elif byte == NODE_END:
            depth -= 1
            cursor += 1
        else:
            cursor += 1

    if depth != 0:
        raise ValueError("Nodo OTBM sin cierre NODE_END")
    return cursor


def decode_tile_area_props(props: bytes) -> tuple[int, int, int] | None:
    """Decodifica props de OTBM_TILE_AREA: base_x, base_y, z."""

    if len(props) < 5:
        return None
    return (
        int.from_bytes(props[0:2], "little"),
        int.from_bytes(props[2:4], "little"),
        props[4],
    )


def decode_item_id(props: bytes) -> int | None:
    """Extrae el item id de un nodo OTBM_ITEM."""

    if len(props) < 2:
        return None
    return int.from_bytes(props[0:2], "little")


def decode_tile_ground_from_props(props: bytes) -> int | None:
    """
    Extrae ground_id cuando RME lo serializa en props del OTBM_TILE.

    En mapas guardados por RME 3.7.0 para 7.60 se observa el patron:
    rel_x, rel_y, 0x09, ground_id(uint16 little-endian).
    """

    cursor = 2
    while cursor + 2 < len(props):
        attr = props[cursor]
        if attr == 0x09:
            return int.from_bytes(props[cursor + 1 : cursor + 3], "little")
        # No conocemos la longitud generica de todos los atributos aqui; este
        # extractor solo necesita el ground observado en RME. Cortamos seguro.
        break
    return None


def classify_item(item_id: int, stack_index: int) -> str:
    """
    Clasifica de forma heuristica el rol visual del item.

    OTBM no guarda "pared norte/sur" como etiqueta semantica. Esa distincion
    depende del ID y de los nombres del items.xml/materials. Por eso aqui se
    preserva el ID exacto y se agrega una etiqueta util para LLMs.
    """

    name = ITEM_NAMES.get(item_id, "").lower()
    if stack_index == 0:
        return "ground"
    if "wall" in name:
        if item_id in {1025, 1027, 1030, 1032, 3361, 3366, 3459, 3460}:
            return "wall_north_south_or_vertical"
        if item_id in {1026, 1028, 1031, 1033, 3362, 3364, 3457, 3458}:
            return "wall_west_east_or_horizontal"
        return "wall_unknown_orientation"
    if "door" in name:
        return "door"
    if any(token in name for token in ("chair", "table", "bed", "chest", "drawer", "bookcase")):
        return "furniture"
    if any(token in name for token in ("tree", "bush", "flower", "mushroom")):
        return "nature"
    if any(token in name for token in ("corpse", "blood", "bone", "remains")):
        return "corpse_or_blood"
    if any(token in name for token in ("fire", "torch", "lamp", "candel")):
        return "light_or_fire"
    return "decoration_or_item"


def parse_tile_items(data: bytes, tile_header: NodeHeader) -> tuple[list[int], int]:
    """Lee hijos OTBM_ITEM directos de un tile y devuelve IDs + offset final."""

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


def tile_to_record(
    abs_x: int,
    abs_y: int,
    center_x: int,
    center_y: int,
    z: int,
    ground_id: int | None,
    stacked_items: list[int],
) -> dict[str, Any]:
    """Convierte un tile OTBM a formato arquetipo JSON."""

    return {
        "rel_x": abs_x - center_x,
        "rel_y": abs_y - center_y,
        "abs_x": abs_x,
        "abs_y": abs_y,
        "z": z,
        "ground_id": ground_id,
        "ground_name": ITEM_NAMES.get(ground_id, "") if ground_id is not None else "",
        "item_ids": stacked_items,
        "items": [
            {
                "id": item_id,
                "name": ITEM_NAMES.get(item_id, ""),
                "role": classify_item(item_id, stack_index + 1),
                "stack_index": stack_index + 1,
            }
            for stack_index, item_id in enumerate(stacked_items)
        ],
    }


def iter_relevant_tiles_in_area(
    data: bytes,
    area_header: NodeHeader,
    area_base_x: int,
    area_base_y: int,
    area_z: int,
    bounds: tuple[int, int, int, int],
    center_x: int,
    center_y: int,
) -> tuple[list[dict[str, Any]], int]:
    """Extrae tiles dentro de bounds desde un TILE_AREA ya seleccionado."""

    min_x, max_x, min_y, max_y = bounds
    records: list[dict[str, Any]] = []
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
            if min_x <= abs_x <= max_x and min_y <= abs_y <= max_y:
                child_item_ids, cursor = parse_tile_items(data, child_header)
                prop_ground_id = decode_tile_ground_from_props(child_header.props)
                if prop_ground_id is not None:
                    ground_id = prop_ground_id
                    stacked_items = child_item_ids
                else:
                    ground_id = child_item_ids[0] if child_item_ids else None
                    stacked_items = child_item_ids[1:] if child_item_ids else []
                records.append(
                    tile_to_record(abs_x, abs_y, center_x, center_y, area_z, ground_id, stacked_items)
                )
            else:
                cursor = skip_node(data, cursor)
        elif control == NODE_END:
            return records, cursor + 1
        else:
            raise ValueError(f"Byte OTBM inesperado 0x{control:02X} en area offset {cursor}")
    raise ValueError("TILE_AREA sin cierre NODE_END")


def area_intersects_bounds(
    area_base_x: int,
    area_base_y: int,
    area_z: int,
    target_z: int,
    bounds: tuple[int, int, int, int],
) -> bool:
    """Indica si un TILE_AREA de 256x256 cruza la caja solicitada."""

    if area_z != target_z:
        return False
    min_x, max_x, min_y, max_y = bounds
    area_max_x = area_base_x + 255
    area_max_y = area_base_y + 255
    return not (area_max_x < min_x or area_base_x > max_x or area_max_y < min_y or area_base_y > max_y)


def scan_for_submap_tiles(
    data: bytes,
    offset: int,
    bounds: tuple[int, int, int, int],
    center_x: int,
    center_y: int,
    z: int,
) -> tuple[list[dict[str, Any]], int]:
    """Recorre nodos y extrae tiles de TILE_AREA que intersecten la caja."""

    header = parse_node_header(data, offset)
    records: list[dict[str, Any]] = []

    if header.node_type == OTBM_TILE_AREA:
        coords = decode_tile_area_props(header.props)
        if coords is not None:
            area_base_x, area_base_y, area_z = coords
            if area_intersects_bounds(area_base_x, area_base_y, area_z, z, bounds):
                area_records, end_offset = iter_relevant_tiles_in_area(
                    data,
                    header,
                    area_base_x,
                    area_base_y,
                    area_z,
                    bounds,
                    center_x,
                    center_y,
                )
                records.extend(area_records)
                return records, end_offset
        return records, skip_node(data, offset)

    cursor = header.children_offset
    while cursor < len(data):
        control = data[cursor]
        if control == NODE_START:
            child_records, cursor = scan_for_submap_tiles(data, cursor, bounds, center_x, center_y, z)
            records.extend(child_records)
        elif control == NODE_END:
            return records, cursor + 1
        else:
            raise ValueError(f"Byte OTBM inesperado 0x{control:02X} en offset {cursor}")
    raise ValueError("Nodo OTBM sin cierre NODE_END")


def extract_submap(otbm_path: str | Path, center_x: int, center_y: int, z: int, size: int = 10) -> dict[str, Any]:
    """
    Extrae una caja alrededor de center_x/center_y/z.

    Las coordenadas rel_x/rel_y del JSON quedan relativas al centro solicitado,
    no a la esquina superior izquierda. Esto ayuda a que un LLM aprenda patron
    espacial: centro, bordes, esquinas y simetrias.
    """

    if size <= 0:
        raise ValueError("size debe ser mayor que cero")
    if not 0 <= z <= 15:
        raise ValueError("z debe estar entre 0 y 15")

    path = resolve_relative(otbm_path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el mapa OTBM: {path}")

    half = size // 2
    min_x = center_x - half
    min_y = center_y - half
    max_x = min_x + size - 1
    max_y = min_y + size - 1
    bounds = (min_x, max_x, min_y, max_y)

    print(
        "[extractor] Extrayendo submapa: "
        f"path={path}, center=({center_x},{center_y},{z}), bounds=({min_x},{min_y})..({max_x},{max_y})"
    )

    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if len(data) < 6:
                raise ValueError(f"Archivo OTBM demasiado pequeno: {path}")
            if data[4] != NODE_START or data[5] != OTSYS_ROOT:
                raise ValueError("El archivo no parece tener root OTBM valido en offset 4")

            tiles, _end_offset = scan_for_submap_tiles(data, 4, bounds, center_x, center_y, z)
    tiles.sort(key=lambda tile: (tile["rel_y"], tile["rel_x"]))

    return {
        "source": str(path),
        "center": {"x": center_x, "y": center_y, "z": z},
        "size": size,
        "bounds": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
        "tile_count": len(tiles),
        "tiles": tiles,
    }


def summarize_extracted_archetype(name: str, archetype: dict[str, Any]) -> None:
    """Imprime estadisticas utiles para validar el ADN visual extraido."""

    tiles = archetype.get("tiles", [])
    if not isinstance(tiles, list):
        print(f"[extractor] {name}: sin tiles validos para resumir")
        return

    ground_counter: Counter[int] = Counter()
    item_counter: Counter[int] = Counter()
    tiles_with_items = 0
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        ground_id = tile.get("ground_id")
        if isinstance(ground_id, int):
            ground_counter[ground_id] += 1
        item_ids = [item_id for item_id in tile.get("item_ids", []) if isinstance(item_id, int)]
        if item_ids:
            tiles_with_items += 1
            item_counter.update(item_ids)

    total_items = sum(item_counter.values())
    dominant_ground = ground_counter.most_common(1)[0] if ground_counter else None
    top_items = [
        {
            "id": item_id,
            "name": ITEM_NAMES.get(item_id, ""),
            "count": count,
        }
        for item_id, count in item_counter.most_common(12)
    ]
    ice_like_items = [
        {
            "id": item_id,
            "name": ITEM_NAMES.get(item_id, ""),
            "count": count,
        }
        for item_id, count in item_counter.items()
        if any(marker in ITEM_NAMES.get(item_id, "").lower() for marker in ("ice", "snow", "frozen"))
    ]

    print(
        f"[extractor] {name}: tiles={len(tiles)}, tiles_con_items={tiles_with_items}, "
        f"items_totales={total_items}, ground_dominante={dominant_ground}"
    )
    print(f"[extractor] {name}: depot_lockers_426={item_counter.get(426, 0)}")
    print(f"[extractor] {name}: top_items={top_items}")
    if ice_like_items:
        ice_like_items.sort(key=lambda entry: entry["count"], reverse=True)
        print(f"[extractor] {name}: ice_like_items={ice_like_items[:12]}")


def extract_configured_archetypes(
    targets: dict[str, dict[str, Any]] = ARCHETYPE_TARGETS,
    otbm_path: str | Path = SOURCE_OTBM_PATH,
) -> dict[str, Any]:
    """Extrae todos los arquetipos configurados, omitiendo mapas inexistentes."""

    output: dict[str, Any] = {}
    source_path = resolve_relative(otbm_path)
    print(f"[extractor] Mapa fuente: {source_path}")
    for name, target in targets.items():
        try:
            center_x = int(target.get("center_x", target.get("x")))
            center_y = int(target.get("center_y", target.get("y")))
            z = int(target["z"])
            size = int(target.get("size", 10))
            print(f"[extractor] Extrayendo {name}... center=({center_x},{center_y},{z}) size={size}")
            output[name] = extract_submap(
                source_path,
                center_x,
                center_y,
                z,
                size,
            )
            output[name]["tags"] = list(target.get("tags", []))
            print(f"[extractor] {name}: {output[name]['tile_count']} tiles procesados")
            summarize_extracted_archetype(name, output[name])
        except FileNotFoundError as exc:
            print(f"[extractor] Saltando '{name}': {exc}")
        except Exception as exc:
            print(f"[extractor] Error en '{name}': {exc}")
            output[name] = {"error": str(exc), "target": target}
    return output


def save_archetypes(archetypes: dict[str, Any], output_path: Path = OUTPUT_PATH) -> Path:
    """Guarda archetypes.json en formato estable y legible."""

    output_path.write_text(
        json.dumps(archetypes, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Punto de entrada CLI."""

    archetypes = extract_configured_archetypes()
    output_path = save_archetypes(archetypes)
    print(f"[extractor] Arquetipos guardados en: {output_path}")


if __name__ == "__main__":
    main()
