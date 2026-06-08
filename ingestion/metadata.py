import re

from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode

ALLERGENS = [
    "nuts",
    "gluten",
    "dairy",
    "eggs",
    "shellfish",
    "soy",
    "sesame",
    "fish",
]

PRICE_RE = re.compile(r"₹\\s?(\\d+(?:\\.\\d{1,2})?)")


class RestaurantMetadataExtractor(BaseExtractor):

    async def aextract(
        self,
        nodes: list[BaseNode],
    ) -> list[dict]:

        metadata_list = []

        for node in nodes:

            text = node.get_content().lower()

            allergens = [
                a for a in ALLERGENS
                if a in text
            ]

            prices = [
                float(p)
                for p in PRICE_RE.findall(node.get_content())
            ]

            metadata_list.append({
                "allergens": allergens,
                "price": prices[0] if prices else None,
            })

        return metadata_list