from frontmatter_format import yaml_util

from kash.exec import kash_action
from kash.exec.preconditions import is_docx_resource
from kash.kits.docs.doc_formats.docx_dump import dump_mammoth_docx
from kash.model import Format, Item, ItemType


@kash_action(precondition=is_docx_resource, mcp_tool=True)
def docx_dump_raw(item: Item) -> Item:
    """
    Dump the raw internal structure of a docx file (parsed by Mammoth)
    as a YAML string body.
    """
    doc_dict = dump_mammoth_docx(item.absolute_path())

    yaml_body = yaml_util.to_yaml_string(doc_dict)

    return item.derived_copy(
        type=ItemType.data,
        format=Format.yaml,
        title=f"{item.pick_title()} (raw dump)",
        body=yaml_body,
    )
