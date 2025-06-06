import re
import json

DEFAULT_INDENT_SIZE = 4

def indent_string(input_string, indent_level=0, skip_first=True, indent_size=DEFAULT_INDENT_SIZE):
    lines = input_string.splitlines()
    indent = " " * indent_size * indent_level
    indented_lines = [indent + line for line in lines]
    if skip_first:
        indented_lines[0] = lines[0]
    return "\n".join(indented_lines)

def var_name_from_pascal_string(pascal_case_string):
    if not isinstance(pascal_case_string, str):
        raise TypeError("Must be string type")
    var = ""
    for c in pascal_case_string:
        if "A" <= c <= "Z":
            var += c.lower()
    return var

def raise_exception(message):
    raise ValueError(message)

def render_primitive_value(value, indent_level=0, skip_first=True):
    if isinstance(value, str):
        out = f"\"{value}\""
    elif isinstance(value, dict):
        out = json.dumps(value, indent=4)
    else:
        out = repr(value)
    if indent_level > 0:
        return indent_string(out, indent_level, skip_first)
    return out

def convert_pascal_to_snake_case(pascal_case_string):
    snake_case_string = re.sub(r'(?<!^)(?=[A-Z])', '_', pascal_case_string).lower()
    return snake_case_string