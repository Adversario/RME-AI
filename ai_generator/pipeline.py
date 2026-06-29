"""
Pipeline de prueba para generacion IA -> OTBM 7.60.

Orquesta:
1. Indexacion de items.xml.
2. Construccion de un mock 5x5 como si viniera de un LLM.
3. Inyeccion sobre template/base_760.otbm.
4. Validaciones basicas de integridad del archivo generado.
"""

from __future__ import annotations

import json
from pathlib import Path

import indexer
from injector import inject_tiles, resolve_relative


BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "tibia_760_catalog.json"
TEMPLATE_PATH = "../../template/base_760.otbm"
OUTPUT_PATH = "../../template/generated_760.otbm"
START_X = 118
START_Y = 123
TARGET_Z = 7


def ensure_catalog() -> dict[str, list[dict[str, int | str]]]:
    """Ejecuta el indexador si hace falta y devuelve el catalogo."""

    if not CATALOG_PATH.is_file():
        indexer.main()

    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Catalogo JSON invalido: {CATALOG_PATH}") from exc


def item_exists(item_id: int) -> bool:
    """Valida rapidamente que un ID exista en data/760/items.xml."""

    items_xml = indexer.resolve_items_xml()
    for item in indexer.iter_items(items_xml):
        if item.id == item_id:
            return True
    return False


def find_item_id_by_name(
    catalog: dict[str, list[dict[str, int | str]]],
    name_fragment: str,
    fallback_id: int,
) -> int:
    """Busca un ID por nombre en el catalogo y luego en items.xml."""

    needle = name_fragment.lower()
    for items in catalog.values():
        for item in items:
            if needle in str(item.get("name", "")).lower():
                return int(item["id"])

    items_xml = indexer.resolve_items_xml()
    for item in indexer.iter_items(items_xml):
        if needle in item.name.lower():
            return item.id

    if item_exists(fallback_id):
        return fallback_id
    raise ValueError(f"No se encontro item para '{name_fragment}' ni fallback valido {fallback_id}")


def build_mock_llm_tiles(catalog: dict[str, list[dict[str, int | str]]]) -> list[dict[str, int | list[int]]]:
    """
    Simula una salida LLM para una zona 5x5.

    El catalogo JSON se usa como primera fuente. Si el catalogo limitado a 50
    items no contiene algun asset semantico, se busca en items.xml y se valida
    el fallback conocido para Tibia 7.60.
    """

    wooden_floor = find_item_id_by_name(catalog, "wooden floor", fallback_id=405)
    stone_wall = find_item_id_by_name(catalog, "stone wall", fallback_id=371)
    chest = find_item_id_by_name(catalog, "chest", fallback_id=1423)

    for item_id in (wooden_floor, stone_wall, chest):
        assert item_exists(item_id), f"ID requerido no existe en items.xml: {item_id}"

    print(
        "[pipeline] Mock 5x5: "
        f"wooden_floor={wooden_floor}, stone_wall={stone_wall}, central_chest={chest}"
    )

    tiles: list[dict[str, int | list[int]]] = []
    size = 5
    for y in range(size):
        for x in range(size):
            item_ids: list[int] = []
            if x in (0, size - 1) or y in (0, size - 1):
                item_ids.append(stone_wall)
            if x == 2 and y == 2:
                item_ids.append(chest)

            tiles.append(
                {
                    "x": x,
                    "y": y,
                    "ground_id": wooden_floor,
                    "item_ids": item_ids,
                }
            )

    return tiles


def run_pipeline() -> Path:
    """Ejecuta el flujo completo y retorna la ruta del OTBM generado."""

    catalog = ensure_catalog()
    assert isinstance(catalog, dict), "El catalogo debe ser un objeto JSON"
    assert "floors_and_ground" in catalog, "El catalogo no contiene floors_and_ground"

    template_path = resolve_relative(TEMPLATE_PATH)
    output_path = resolve_relative(OUTPUT_PATH)
    assert template_path.is_file(), f"No existe la plantilla: {template_path}"

    original = template_path.read_bytes()
    assert len(original) >= 6, "La plantilla es demasiado pequena para OTBM"
    original_magic = original[:4]

    tiles = build_mock_llm_tiles(catalog)
    print(f"[pipeline] Inyectando en start_x={START_X}, start_y={START_Y}, z={TARGET_Z}")
    generated_path = inject_tiles(
        TEMPLATE_PATH,
        OUTPUT_PATH,
        tiles,
        start_x=START_X,
        start_y=START_Y,
        z=TARGET_Z,
    )

    assert generated_path.is_file(), "No se genero el archivo OTBM final"
    generated = generated_path.read_bytes()
    assert len(generated) >= len(original), "El archivo generado quedo mas pequeno que la plantilla"
    assert generated[:4] == original_magic, "La firma magica OTBM no fue preservada"
    assert generated_path == output_path

    return generated_path


def main() -> None:
    """Punto de entrada CLI."""

    generated = run_pipeline()
    print(f"Pipeline OK. Archivo generado: {generated}")


if __name__ == "__main__":
    main()
