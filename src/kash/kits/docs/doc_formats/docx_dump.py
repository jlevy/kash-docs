from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from mammoth.documents import Document


def read_mammoth_docx(docx_path: Path) -> Document:
    """
    Parses a .docx file using Mammoth and pretty-prints the internal document structure.
    """
    from mammoth.docx import read as read_docx

    with open(docx_path, "rb") as fileobj:
        result = read_docx(fileobj)

    return result.value


def dump_mammoth_docx(docx_path: Path) -> object:
    """
    Parses a .docx file using Mammoth and returns the internal document
    structure as a dictionary.
    """
    return cobble_to_dict(read_mammoth_docx(docx_path))


@runtime_checkable
class HasCobbleFields(Protocol):
    _cobble_fields: list[tuple[str, Any]]


def cobble_to_dict(obj: object, visited: set[int] | None = None) -> object:
    """
    Recursively converts a cobble object (esp. from mammoth.documents)
    or other data structure into native Python types suitable for serialization.

    Handles basic types, lists, tuples, dicts, cobble objects, the specific
    mammoth.documents.Notes class, and detects circular references.
    """
    from mammoth import documents

    if visited is None:
        visited = set()

    # Handle primitives immediately
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    obj_id = id(obj)
    if obj_id in visited:
        # Circular reference detected
        return f"<Circular Reference: {type(obj).__name__} id={obj_id}>"

    visited.add(obj_id)

    result: object
    if isinstance(obj, list):
        result = [cobble_to_dict(item, visited.copy()) for item in obj]
    elif isinstance(obj, tuple):
        result = tuple(cobble_to_dict(item, visited.copy()) for item in obj)
    elif isinstance(obj, dict):
        # Assumes keys are serializable (usually strings)
        result = {k: cobble_to_dict(v, visited.copy()) for k, v in obj.items()}
    elif hasattr(obj, "_cobble_fields"):  # Standard cobble object
        # Cast obj to the protocol type after the hasattr check
        cobble_obj = cast(HasCobbleFields, obj)
        data: dict[str, Any] = {"_type": type(cobble_obj).__name__}
        # Note: _cobble_fields is already sorted by definition order in cobble's data decorator
        # Access _cobble_fields via the casted variable
        for name, _field in cobble_obj._cobble_fields:
            value = getattr(cobble_obj, name)
            data[name] = cobble_to_dict(value, visited.copy())
        result = data
    elif isinstance(obj, documents.Notes):  # Specific handling for Notes container
        # Convert the internal dict of notes to a list for better serialization
        notes_list = [cobble_to_dict(note, visited.copy()) for note in obj._notes.values()]
        result = {"_type": "Notes", "notes": notes_list}
    elif isinstance(obj, documents.Tab):  # Handle singleton Tab
        result = {"_type": "Tab"}
    else:
        # Option 1: Raise error
        raise TypeError(
            f"Object of type {type(obj).__name__} is not directly serializable by cobble_to_dict"
        )
        # Option 2: Use string representation (might lose info)
        # result = repr(obj)

    return result
