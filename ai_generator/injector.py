"""
Inyector OTBM minimo para mapas RME/OpenTibia.

Implementa lectura/escritura de nodos OTBM con bytes de control:
NODE_START=0xFE, NODE_END=0xFF, ESCAPE_CHAR=0xFD.

Nota de compatibilidad: en OTBM el ground de un tile se serializa como un
OTBM_ITEM hijo del tile. La API acepta `ground_id` por claridad, y el escritor
lo emite como primer item dentro del nodo tile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


NODE_START = 0xFE
NODE_END = 0xFF
ESCAPE_CHAR = 0xFD

OTSYS_ROOT = 0x00
OTBM_MAP_DATA = 0x02
OTBM_TILE_AREA = 0x04
OTBM_TILE = 0x05
OTBM_ITEM = 0x06

BASE_DIR = Path(__file__).resolve().parent


class OTBMError(RuntimeError):
    """Error especifico de parsing/escritura OTBM."""


@dataclass
class OTBMNode:
    """Nodo generico de un arbol OTBM."""

    node_type: int
    props: bytes = b""
    children: list["OTBMNode"] = field(default_factory=list)


def resolve_relative(path: str | Path) -> Path:
    """Resuelve una ruta relativa desde ai_generator/.

    Se intenta primero la ruta literal indicada. Si no existe y empieza con
    ../../, se prueba tambien ../ porque en esta instalacion ai_generator/
    cuelga directamente de la raiz de RME.
    """

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    resolved = (BASE_DIR / candidate).resolve()
    if resolved.exists():
        return resolved

    parts = candidate.parts
    if len(parts) >= 3 and parts[0] == ".." and parts[1] == "..":
        fallback = (BASE_DIR / Path(*parts[1:])).resolve()
        if fallback.exists() or fallback.parent.exists():
            return fallback

    return resolved


def escape_otbm(data: bytes) -> bytes:
    """Escapa bytes de propiedades que colisionan con controles OTBM."""

    output = bytearray()
    for byte in data:
        if byte in (NODE_START, NODE_END, ESCAPE_CHAR):
            output.append(ESCAPE_CHAR)
        output.append(byte)
    return bytes(output)


def unescape_next(data: bytes, offset: int) -> tuple[int, int]:
    """Lee un byte de propiedades aplicando escaping."""

    if offset >= len(data):
        raise OTBMError("Fin inesperado leyendo propiedades OTBM")

    byte = data[offset]
    if byte == ESCAPE_CHAR:
        if offset + 1 >= len(data):
            raise OTBMError("Escape OTBM incompleto al final del archivo")
        return data[offset + 1], offset + 2

    return byte, offset + 1


def read_props_until_control(data: bytes, offset: int) -> tuple[bytes, int]:
    """Lee propiedades hasta NODE_START o NODE_END sin consumir el control."""

    props = bytearray()
    while offset < len(data):
        byte = data[offset]
        if byte in (NODE_START, NODE_END):
            break
        value, offset = unescape_next(data, offset)
        props.append(value)
    return bytes(props), offset


def parse_node(data: bytes, offset: int = 0) -> tuple[OTBMNode, int]:
    """Parsea un nodo OTBM desde `offset`."""

    if offset >= len(data) or data[offset] != NODE_START:
        raise OTBMError(f"Se esperaba NODE_START en offset {offset}")
    if offset + 1 >= len(data):
        raise OTBMError("Nodo sin tipo")

    node_type = data[offset + 1]
    offset += 2
    props, offset = read_props_until_control(data, offset)
    node = OTBMNode(node_type=node_type, props=props)

    while offset < len(data):
        control = data[offset]
        if control == NODE_START:
            child, offset = parse_node(data, offset)
            node.children.append(child)
        elif control == NODE_END:
            return node, offset + 1
        else:
            raise OTBMError(f"Byte de control invalido 0x{control:02X} en offset {offset}")

    raise OTBMError("Nodo OTBM sin NODE_END")


def serialize_node(node: OTBMNode) -> bytes:
    """Serializa un nodo y sus hijos respetando escaping en propiedades."""

    output = bytearray((NODE_START, node.node_type))
    output.extend(escape_otbm(node.props))
    for child in node.children:
        output.extend(serialize_node(child))
    output.append(NODE_END)
    return bytes(output)


def read_otbm(path: Path) -> tuple[bytes, OTBMNode]:
    """Lee un archivo OTBM y devuelve firma magica + arbol root."""

    raw = path.read_bytes()
    if len(raw) < 6:
        raise OTBMError(f"Archivo demasiado pequeno para OTBM: {path}")

    magic = raw[:4]
    root, offset = parse_node(raw, 4)
    if root.node_type != OTSYS_ROOT:
        raise OTBMError(f"Root inesperado: {root.node_type}, esperado {OTSYS_ROOT}")
    if offset != len(raw):
        trailing = len(raw) - offset
        raise OTBMError(f"Bytes sobrantes tras root OTBM: {trailing}")

    return magic, root


def write_otbm(path: Path, magic: bytes, root: OTBMNode) -> None:
    """Escribe un archivo OTBM completo."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(magic + serialize_node(root))


def u16(value: int) -> bytes:
    """Serializa entero uint16 little-endian."""

    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"Valor fuera de uint16: {value}")
    return value.to_bytes(2, "little", signed=False)


def find_first_child(node: OTBMNode, node_type: int) -> OTBMNode | None:
    """Busca el primer hijo directo de un tipo dado."""

    return next((child for child in node.children if child.node_type == node_type), None)


def decode_tile_area_props(props: bytes) -> tuple[int, int, int] | None:
    """Decodifica propiedades de OTBM_TILE_AREA: base_x, base_y, z."""

    if len(props) < 5:
        return None
    return (
        int.from_bytes(props[0:2], "little"),
        int.from_bytes(props[2:4], "little"),
        props[4],
    )


def decode_tile_props(area: OTBMNode, tile: OTBMNode) -> tuple[int, int, int] | None:
    """Devuelve coordenadas absolutas de un tile si sus props son validas."""

    area_coords = decode_tile_area_props(area.props)
    if area_coords is None or len(tile.props) < 2:
        return None

    base_x, base_y, z = area_coords
    return base_x + tile.props[0], base_y + tile.props[1], z


def normalize_tile_spec(spec: dict) -> tuple[int, int, int, list[int]]:
    """Valida y normaliza una entrada de tiles_to_inject."""

    try:
        rel_x = int(spec["x"])
        rel_y = int(spec["y"])
        ground_id = int(spec["ground_id"])
    except KeyError as exc:
        raise ValueError(f"Tile incompleto, falta llave: {exc.args[0]}") from exc

    raw_items = spec.get("item_ids", [])
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, Iterable) or isinstance(raw_items, (str, bytes)):
        raise ValueError("item_ids debe ser una lista de enteros")

    item_ids = [int(item_id) for item_id in raw_items]
    if rel_x < 0 or rel_y < 0:
        raise ValueError("x/y relativos deben ser mayores o iguales a cero")
    if ground_id <= 0:
        raise ValueError("ground_id debe ser mayor que cero")
    if any(item_id <= 0 for item_id in item_ids):
        raise ValueError("item_ids debe contener IDs mayores que cero")

    return rel_x, rel_y, ground_id, item_ids


def make_item_node(item_id: int) -> OTBMNode:
    """Crea un nodo OTBM_ITEM con ID uint16."""

    return OTBMNode(node_type=OTBM_ITEM, props=u16(item_id))


def make_tile_node(local_x: int, local_y: int, ground_id: int, item_ids: list[int]) -> OTBMNode:
    """Crea un nodo OTBM_TILE con ground + items."""

    if not 0 <= local_x <= 255 or not 0 <= local_y <= 255:
        raise ValueError("local_x/local_y deben caber en uint8 dentro de TILE_AREA")

    children = [make_item_node(ground_id)]
    children.extend(make_item_node(item_id) for item_id in item_ids)
    return OTBMNode(node_type=OTBM_TILE, props=bytes((local_x, local_y)), children=children)


def list_tile_areas(map_data: OTBMNode) -> list[tuple[OTBMNode, tuple[int, int, int]]]:
    """Lista TILE_AREA validos junto con su coordenada base."""

    areas: list[tuple[OTBMNode, tuple[int, int, int]]] = []
    for child in map_data.children:
        if child.node_type != OTBM_TILE_AREA:
            continue
        coords = decode_tile_area_props(child.props)
        if coords is not None:
            areas.append((child, coords))
    return areas


def find_covering_area(
    map_data: OTBMNode,
    abs_x: int,
    abs_y: int,
    z: int,
) -> tuple[OTBMNode, tuple[int, int, int]] | None:
    """Busca el TILE_AREA existente que cubre la coordenada absoluta."""

    for area, (base_x, base_y, area_z) in list_tile_areas(map_data):
        if area_z != z:
            continue
        if base_x <= abs_x <= base_x + 255 and base_y <= abs_y <= base_y + 255:
            return area, (base_x, base_y, area_z)
    return None


def create_tile_area(map_data: OTBMNode, abs_x: int, abs_y: int, z: int) -> tuple[OTBMNode, tuple[int, int, int]]:
    """Crea un TILE_AREA alineado a bloque de 256x256 cuando no existe uno compatible."""

    base_x = abs_x & 0xFF00
    base_y = abs_y & 0xFF00
    area = OTBMNode(node_type=OTBM_TILE_AREA, props=u16(base_x) + u16(base_y) + bytes((z,)))
    map_data.children.append(area)
    print(f"[injector] TILE_AREA creado: base_x={base_x}, base_y={base_y}, z={z}")
    return area, (base_x, base_y, z)


def find_tile_in_area(area: OTBMNode, rel_x: int, rel_y: int) -> OTBMNode | None:
    """Busca un tile por coordenada relativa dentro de un TILE_AREA."""

    target = bytes((rel_x, rel_y))
    for child in area.children:
        if child.node_type == OTBM_TILE and child.props[:2] == target:
            return child
    return None


def overwrite_tile(tile: OTBMNode, rel_x: int, rel_y: int, ground_id: int, item_ids: list[int]) -> None:
    """Sobrescribe ground/decoracion manteniendo cualquier prop extra del tile."""

    extra_props = tile.props[2:]
    tile.props = bytes((rel_x, rel_y)) + extra_props
    tile.children = [make_item_node(ground_id)]
    tile.children.extend(make_item_node(item_id) for item_id in item_ids)


def write_tile_at_absolute_position(
    map_data: OTBMNode,
    abs_x: int,
    abs_y: int,
    z: int,
    ground_id: int,
    item_ids: list[int],
) -> str:
    """Crea o sobrescribe un tile usando coordenadas absolutas."""

    if not 0 <= abs_x <= 0xFFFF or not 0 <= abs_y <= 0xFFFF:
        raise ValueError(f"Coordenada absoluta fuera de rango uint16: {abs_x}, {abs_y}")

    area_match = find_covering_area(map_data, abs_x, abs_y, z)
    if area_match is None:
        area, (area_base_x, area_base_y, area_z) = create_tile_area(map_data, abs_x, abs_y, z)
    else:
        area, (area_base_x, area_base_y, area_z) = area_match

    rel_x = abs_x - area_base_x
    rel_y = abs_y - area_base_y
    if not 0 <= rel_x <= 255 or not 0 <= rel_y <= 255:
        raise OTBMError(
            f"Coordenada relativa invalida: abs=({abs_x},{abs_y},{z}) "
            f"base=({area_base_x},{area_base_y},{area_z}) rel=({rel_x},{rel_y})"
        )

    existing_tile = find_tile_in_area(area, rel_x, rel_y)
    if existing_tile is None:
        area.children.append(make_tile_node(rel_x, rel_y, ground_id, item_ids))
        return "creado"

    overwrite_tile(existing_tile, rel_x, rel_y, ground_id, item_ids)
    return "sobrescrito"


def cleanup_generation_area(
    map_data: OTBMNode,
    normalized_tiles: list[tuple[int, int, int, list[int]]],
    start_x: int,
    start_y: int,
    z: int,
    cleanup_ground_id: int,
    margin: int = 2,
) -> None:
    """Limpia la caja de generacion expandida con un ground coherente."""

    abs_positions = [(start_x + rel_x, start_y + rel_y) for rel_x, rel_y, _ground_id, _items in normalized_tiles]
    min_x = max(0, min(x for x, _y in abs_positions) - margin)
    max_x = min(0xFFFF, max(x for x, _y in abs_positions) + margin)
    min_y = max(0, min(y for _x, y in abs_positions) - margin)
    max_y = min(0xFFFF, max(y for _x, y in abs_positions) + margin)

    print(
        "[injector] Limpieza previa: "
        f"bbox=({min_x},{min_y},{z})..({max_x},{max_y},{z}) "
        f"margin={margin} cleanup_ground_id={cleanup_ground_id}"
    )

    cleaned = 0
    created = 0
    overwritten = 0
    for abs_y in range(min_y, max_y + 1):
        for abs_x in range(min_x, max_x + 1):
            action = write_tile_at_absolute_position(
                map_data,
                abs_x,
                abs_y,
                z,
                cleanup_ground_id,
                [],
            )
            cleaned += 1
            if action == "creado":
                created += 1
            else:
                overwritten += 1

    print(
        "[injector] Limpieza previa completada: "
        f"tiles={cleaned}, creados={created}, sobrescritos={overwritten}"
    )


def inject_tiles(
    template_relative_path: str | Path,
    output_relative_path: str | Path,
    tiles_to_inject: Iterable[dict],
    start_x: int = 1000,
    start_y: int = 1000,
    z: int = 7,
    cleanup_ground_id: int | None = None,
    cleanup_margin: int = 2,
) -> Path:
    """
    Inyecta tiles en una plantilla OTBM y guarda un nuevo archivo.

    Cada tile debe tener la forma:
    {"x": 0, "y": 0, "ground_id": 405, "item_ids": [371, 1740]}
    """

    template_path = resolve_relative(template_relative_path)
    output_path = resolve_relative(output_relative_path)
    if not template_path.is_file():
        raise FileNotFoundError(f"No existe la plantilla OTBM: {template_path}")

    normalized_tiles = [normalize_tile_spec(spec) for spec in tiles_to_inject]
    if not normalized_tiles:
        raise ValueError("tiles_to_inject esta vacio")
    if not 0 <= z <= 15:
        raise ValueError(f"z fuera de rango Tibia: {z}")
    if cleanup_margin < 0:
        raise ValueError("cleanup_margin debe ser mayor o igual que cero")
    if cleanup_ground_id is None:
        cleanup_ground_id = normalized_tiles[0][2]
    if not 1 <= cleanup_ground_id <= 0xFFFF:
        raise ValueError(f"cleanup_ground_id fuera de rango uint16: {cleanup_ground_id}")

    magic, root = read_otbm(template_path)
    map_data = find_first_child(root, OTBM_MAP_DATA)
    if map_data is None:
        raise OTBMError("La plantilla no contiene nodo OTBM_MAP_DATA")

    areas = list_tile_areas(map_data)
    if areas:
        for _area, (base_x, base_y, area_z) in areas:
            print(f"[injector] TILE_AREA encontrado: base_x={base_x}, base_y={base_y}, z={area_z}")
    else:
        print("[injector] No hay TILE_AREA en la plantilla; se crearan areas segun necesidad.")

    cleanup_generation_area(
        map_data,
        normalized_tiles,
        start_x=start_x,
        start_y=start_y,
        z=z,
        cleanup_ground_id=cleanup_ground_id,
        margin=cleanup_margin,
    )

    for tile_rel_x, tile_rel_y, ground_id, item_ids in normalized_tiles:
        abs_x = start_x + tile_rel_x
        abs_y = start_y + tile_rel_y
        area_match = find_covering_area(map_data, abs_x, abs_y, z)
        if area_match is None:
            raise OTBMError(f"No se encontro TILE_AREA para tile recien limpiado: ({abs_x},{abs_y},{z})")
        _area, (area_base_x, area_base_y, area_z) = area_match
        rel_x = abs_x - area_base_x
        rel_y = abs_y - area_base_y
        action = write_tile_at_absolute_position(map_data, abs_x, abs_y, z, ground_id, item_ids)

        print(
            "[injector] Tile "
            f"{action}: abs=({abs_x},{abs_y},{z}) "
            f"area_base=({area_base_x},{area_base_y},{area_z}) "
            f"rel=({rel_x},{rel_y}) ground_id={ground_id} item_ids={item_ids}"
        )

    write_otbm(output_path, magic, root)
    return output_path


if __name__ == "__main__":
    demo_tiles = [{"x": 0, "y": 0, "ground_id": 405, "item_ids": []}]
    created = inject_tiles("../../template/base_760.otbm", "../../template/generated_760.otbm", demo_tiles)
    print(f"OTBM generado: {created}")
