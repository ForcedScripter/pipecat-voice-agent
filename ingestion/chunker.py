from typing import Any, Sequence

from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, TextNode


def make_node(text: str, source: BaseNode, extra_meta=None):

    extra_meta = extra_meta or {}

    return TextNode(
        text=text,
        metadata={
            **source.metadata,
            **extra_meta,
        },
    )


class SimpleChunker(NodeParser):

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:

        result = []

        for node in nodes:

            text = node.get_content().strip()

            if not text:
                continue

            result.append(
                make_node(
                    text,
                    node,
                    {"chunk_strategy": "simple"},
                )
            )

        return result