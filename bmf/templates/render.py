import os
import argparse
from pathlib import Path
from importlib.resources import files
from jinja2 import Environment, FileSystemLoader, PackageLoader

try:
    from bmf.templates.utils import *
except ModuleNotFoundError:
    from utils import *

try:
    env = Environment(
        loader=PackageLoader("bmf.templates", "jinja_templates"),
        extensions=['jinja2.ext.do']
    )
except:
    # Fallback to FileSystemLoader if bmf is installed but templates not found
    script_dir = Path(__file__).parent
    templates_dir = script_dir / "jinja_templates"
    env = Environment(
        loader=FileSystemLoader(templates_dir),
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
    script_dir = Path(__file__).parent
    
    try:
        modules_dir = files("bmf.templates.jinja_templates").joinpath("modules")
    except:
        modules_dir = script_dir / "jinja_templates" / "modules"

    # template_name is the prefix you're searching for
    template_rel_path = None
    for entry in modules_dir.iterdir():
        if entry.name.startswith(template_name):
            template_rel_path = f"modules/{entry.name}"
            break

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
    script_dir = Path(__file__).parent
    try:
        modules_template_dir = files("bmf.templates.jinja_templates").joinpath("modules")
    except:
        modules_template_dir = script_dir / "jinja_templates" / "modules"
        
    for entry in modules_template_dir.iterdir():
        os.makedirs("output", exist_ok=True)
        output_path = f"output/{os.path.splitext(entry.name)[0]}.py"
        render_module_template(entry.name, output_path)    

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
