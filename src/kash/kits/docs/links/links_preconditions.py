from frontmatter_format.yaml_util import from_yaml_string
from ruamel.yaml.error import YAMLError

from kash.exec import kash_precondition
from kash.kits.docs.links.links_model import Link, LinkResults
from kash.model import Format, Item, ItemType


@kash_precondition
def is_links_data(item: Item) -> bool:
    """Check if an item is a data item containing a list of links."""
    if item.type != ItemType.data or item.format != Format.yaml or not item.body:
        return False

    try:
        data = from_yaml_string(item.body)
        if not isinstance(data, dict) or "links" not in data:
            return False

        # Validate that it has links and that each link has a url field
        links = data["links"]
        if not isinstance(links, list):
            return False
        for link in links:
            if not isinstance(link, dict) or "url" not in link:
                return False

        return True
    except (TypeError, KeyError, YAMLError):
        return False


## Tests


def test_has_links_data_precondition():
    """Test the has_links_data precondition with various item types."""
    from frontmatter_format import to_yaml_string

    # Test valid links data
    links = [Link(url="https://example.com", title="Example")]
    results = LinkResults(links=links)
    yaml_content = to_yaml_string(results.model_dump())

    valid_item = Item(
        type=ItemType.data,
        format=Format.yaml,
        body=yaml_content,
    )
    assert is_links_data(valid_item) is True

    # Test empty links data
    empty_results = LinkResults(links=[])
    empty_yaml = to_yaml_string(empty_results.model_dump())

    empty_item = Item(
        type=ItemType.data,
        format=Format.yaml,
        body=empty_yaml,
    )
    assert is_links_data(empty_item) is True

    # Test invalid item type
    invalid_type = Item(
        type=ItemType.doc,
        format=Format.yaml,
        body=yaml_content,
    )
    assert is_links_data(invalid_type) is False

    # Test invalid format
    invalid_format = Item(
        type=ItemType.data,
        format=Format.markdown,
        body=yaml_content,
    )
    assert is_links_data(invalid_format) is False

    # Test no body
    no_body = Item(
        type=ItemType.data,
        format=Format.yaml,
        body=None,
    )
    assert is_links_data(no_body) is False

    # Test invalid YAML
    invalid_yaml = Item(
        type=ItemType.data,
        format=Format.yaml,
        body="invalid: yaml: content:",
    )
    assert is_links_data(invalid_yaml) is False

    # Test missing links key
    missing_links = Item(
        type=ItemType.data,
        format=Format.yaml,
        body="data: []\nother: value",
    )
    assert is_links_data(missing_links) is False

    # Test invalid links structure
    invalid_links = Item(
        type=ItemType.data,
        format=Format.yaml,
        body="links: not_a_list",
    )
    assert is_links_data(invalid_links) is False
