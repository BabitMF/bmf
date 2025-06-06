import sys
import os
import argparse

from jinja2 import Environment, FileSystemLoader

from utils import *

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "jinja")

env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    extensions=['jinja2.ext.do']
)

env.globals['raise_error'] = raise_exception

def indent_component(component_path, indent_level=0, skip_first=True, **kwargs):
    global env
    template = env.get_template(component_path)
    context = {
        "indent_component": indent_component,
        "var_name_from_pascal_string": var_name_from_pascal_string,
        "render_primitive_value": render_primitive_value,
        "convert_pascal_to_snake_case": convert_pascal_to_snake_case
    }
    component_content = template.render(kwargs | context)
    if indent_level > 0:
        return indent_string(component_content, indent_level, skip_first)
    return component_content

def render_module_template(template_name, output_path, **kwargs):
    global env

    template_rel_path = None
    for fn in os.listdir(os.path.join(TEMPLATE_DIR, "modules")):
        if fn.startswith(template_name):
            template_rel_path = f"modules/{fn}"

    if template_rel_path is None:
        raise ValueError(f"Template '{template_name}' not found")

    template = env.get_template(template_rel_path)

    # Render the template with data
    render_context = {
        "filename": os.path.basename(output_path),
        "indent_component": indent_component,
        "var_name_from_pascal_string": var_name_from_pascal_string,
        "render_primitive_value": render_primitive_value,
        "convert_pascal_to_snake_case": convert_pascal_to_snake_case
    }

    render_context.update(kwargs)
    rendered_class = template.render(render_context)

    # Write the rendered code to a file
    with open(output_path, 'w') as f:
        f.write(rendered_class)

def generate_all_modules():
    module_template_dir = os.path.join(os.path.dirname(__file__), "jinja/modules")
    for fn in os.listdir(module_template_dir):
        os.makedirs("output", exist_ok=True)
        template_name = f"modules/{fn}"
        output_path = f"output/{os.path.splitext(fn)[0]}.py"
        render_module_template(template_name, output_path)    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Generate BMF Module template classes.'
    )
    argparser.add_argument('-t', '--template_name', type=str, required=False, default="", help="Template name, options: 'base', 'default', 'n_streams'.")
    argparser.add_argument('-o', '--output_file', type=str, required=False, default="", help="Output file path.")
    args = argparser.parse_args()

    if args.template_name:
        render_module_template(args.template_name, args.output_file if args.output_file else args.template_name + ".py")
    else:
        generate_all_modules()
