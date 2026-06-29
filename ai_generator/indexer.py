"""
Indexador de assets Tibia 7.60 para RME.

Lee items.xml desde rutas relativas a este archivo y genera un catalogo JSON
pequeno, determinista y facil de consumir por el pipeline de generacion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree


BASE_DIR = Path(__file__).resolve().parent

# La ruta pedida por el usuario se intenta primero. Con la estructura real
# ai_generator/ dentro de la raiz de RME, ../data/760/items.xml es el fallback.
ITEMS_XML_CANDIDATES = (
    BASE_DIR / "../../data/760/items.xml",
    BASE_DIR / "../data/760/items.xml",
)
CATALOG_PATH = BASE_DIR / "tibia_760_catalog.json"

CATEGORIES = (
    "floors_and_ground",
    "walls_and_doors",
    "furniture_and_decorations",
    "nature_and_trees",
    "corpses_and_blood",
)


@dataclass(frozen=True)
class CatalogItem:
    """Representacion minima de un item de Tibia/RME."""

    id: int
    name: str


def resolve_items_xml() -> Path:
    """Devuelve la primera ruta valida para data/760/items.xml."""

    for candidate in ITEMS_XML_CANDIDATES:
        path = candidate.resolve()
        if path.is_file():
            return path

    searched = "\n".join(f"- {candidate.resolve()}" for candidate in ITEMS_XML_CANDIDATES)
    raise FileNotFoundError(f"No se encontro items.xml. Rutas revisadas:\n{searched}")


def classify_item(name: str) -> str:
    """Clasifica un item por nombre usando reglas simples y explicitas."""

    normalized = name.lower()

    if any(token in normalized for token in ("corpse", "blood", "dead human", "remains")):
        return "corpses_and_blood"
    if any(token in normalized for token in ("tree", "bush", "grass tuft", "fern", "flower", "mushroom")):
        return "nature_and_trees"
    if any(token in normalized for token in ("wall", "door", "window", "gate", "fence", "railing")):
        return "walls_and_doors"
    if any(
        token in normalized
        for token in (
            "chair",
            "table",
            "bed",
            "carpet",
            "chest",
            "dresser",
            "lamp",
            "candelabrum",
            "tapestry",
            "depot",
            "bookcase",
        )
    ):
        return "furniture_and_decorations"
    if any(
        token in normalized
        for token in (
            "floor",
            "grass",
            "earth",
            "dirt",
            "sand",
            "gravel",
            "stone",
            "snow",
            "ice",
            "water",
            "lava",
            "swamp",
            "mud",
        )
    ):
        return "floors_and_ground"

    return "furniture_and_decorations"


def iter_items(items_xml: Path) -> Iterable[CatalogItem]:
    """Itera items desde XML sin cargar estructuras auxiliares pesadas."""

    try:
        for _event, elem in ElementTree.iterparse(items_xml, events=("end",)):
            if elem.tag != "item":
                elem.clear()
                continue

            item_id = elem.attrib.get("id")
            name = elem.attrib.get("name")
            if item_id and name:
                try:
                    yield CatalogItem(id=int(item_id), name=name.strip())
                except ValueError:
                    pass
            elem.clear()
    except ElementTree.ParseError as exc:
        raise ValueError(f"items.xml no es XML valido: {items_xml}") from exc


def build_catalog(limit: int = 50) -> dict[str, list[dict[str, int | str]]]:
    """
    Construye un catalogo con los primeros `limit` items encontrados.

    La distribucion por categoria depende del orden real del items.xml, tal como
    se solicito, y queda serializada bajo llaves estables.
    """

    if limit <= 0:
        raise ValueError("limit debe ser mayor que cero")

    catalog: dict[str, list[dict[str, int | str]]] = {category: [] for category in CATEGORIES}
    items_xml = resolve_items_xml()

    for index, item in enumerate(iter_items(items_xml)):
        if index >= limit:
            break
        category = classify_item(item.name)
        catalog[category].append(asdict(item))

    return catalog


def save_catalog(catalog: dict[str, list[dict[str, int | str]]], output_path: Path = CATALOG_PATH) -> Path:
    """Guarda el catalogo JSON con encoding estable."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> Path:
    """Punto de entrada CLI."""

    catalog = build_catalog()
    output = save_catalog(catalog)
    print(f"Catalogo generado: {output}")
    return output


if __name__ == "__main__":
    main()
