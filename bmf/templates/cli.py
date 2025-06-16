import json
import click
import os
import re
from InquirerPy import inquirer

try:
    from bmf.templates.render import render_module_template
except ModuleNotFoundError:
    from render import render_module_template

# Total number of stages
TOTAL_STAGES = 8

# Helper to style headers
def style_header(text):
    return click.style(text, fg='cyan', bold=True)

def to_pascal_case(s):
    return ''.join(word.capitalize() for word in re.split(r'[^a-zA-Z0-9]', s) if word)

@click.command()
@click.help_option('--help', '-h', help='Show this help message and exit.')
@click.option('--from-json', type=click.Path(exists=True), help='Import an existing BMF module definition JSON file.')
def create_module(from_json):
    """A CLI application to generate BMF module template files!"""
    # Initialize values
    module_definition = {}

    # Load from JSON if provided
    if from_json:
        with open(from_json, 'r') as f:
            try:
                module_definition = json.load(f)
            except json.JSONDecodeError:
                click.secho("Error: Failed to parse provided JSON file.", fg='red', err=True)
                raise click.Abort()
    else:
        # Welcome Banner
        click.echo()
        click.echo(style_header("🛠️ BMF Module Template Generator CLI 🛠️"))
        click.echo(style_header("=" * 36))

        # Stage 1: Module Name
        stage(1)
        module_definition["module_name"] = click.prompt(
            click.style("Enter the module name", fg='yellow', bold=True),
            default="my_module"
        ).strip()

        # Stage 2: Class Name
        stage(2)
        default_class_name = to_pascal_case(module_definition["module_name"])
        module_definition["class_name"] = click.prompt(
            click.style("Enter the class name", fg='yellow', bold=True),
            default=default_class_name
        ).strip()

        # Stage 3: Template Selection
        stage(3)
        module_definition["template_name"] = inquirer.select(
            message="Select a template:",
            choices=[
                "base", 
                "default", 
                # "n_streams"
            ],
            default="default",
            pointer="👉",
            instruction="Use ↑ ↓ to navigate"
        ).execute()

        # Stage 4: Passthrough module
        stage(4)
        module_definition["passthrough"] = click.confirm(
            click.style("Passthrough module?", fg='yellow', bold=True),
            default=True
        )

        # Stage 5: Default logging
        stage(5)
        module_definition["default_logging"] = click.confirm(
            click.style("Add default logging?", fg='yellow', bold=True),
            default=True
        )

        # Stage 6: JSON Settings
        stage(6)
        if click.confirm(
            click.style("Load default options from example module options JSON file?", fg='yellow', bold=True),
            default=False
        ):
            file_path = inquirer.filepath(
                message="Enter the path to example module options JSON file:",
                only_files=True,
                validate=lambda path: path.endswith(".json") and os.path.isfile(path),
                instruction="(use Tab to autocomplete)"
            ).execute()
            try:
                with open(file_path, 'r') as f:
                    example_data = json.load(f)
                if isinstance(example_data, dict):
                    module_definition["option_schema"] = [
                        {"name": k, "default": v, "requred": False}
                        for k, v in example_data.items()
                    ]
                else:
                    click.secho("Error: Invalid example module options JSON file.", fg='red', err=True)
                    raise click.Abort()
            except json.JSONDecodeError:
                click.secho("Error: Failed to parse JSON.", fg='red', err=True)
                raise click.Abort()
        else:
            click.echo(click.style("Enter custom settings (leave name blank to finish):", fg='green', bold=True))
            while True:
                name = click.prompt(click.style("  Setting name", fg='green'), default="", show_default=False).strip()
                if not name:
                    break

                while True:
                    default_str = click.prompt(click.style("  Default value (as JSON)", fg='green'), default='null')
                    try:
                        default_val = json.loads(default_str)
                        break
                    except json.JSONDecodeError:
                        click.secho("Error: Invalid JSON input. Please enter a valid JSON string.", fg='red', err=True)
                        continue

                required = click.confirm(click.style("  Is this setting required?", fg='green'), default=False)

                if "option_schema" not in module_definition:
                    module_definition["option_schema"] = []

                module_definition["option_schema"].append({
                    "name": name,
                    "default": default_val,
                    "required": required
                })

    # Stage 7: Do generate template
    stage(7)
    do_generate_template = click.confirm(
        click.style("Would you like to generate a BMF module template based on the collected information?", fg='yellow', bold=True),
        default=True
    )

    # Stage 8: Ask for output filename
    stage(8)
    default_filename = f"{module_definition['module_name']}.py" if do_generate_template else f"{module_definition['module_name']}_definition.json"
    module_definition["output_path"] = os.path.join(
        os.getcwd(),
        click.prompt(
            click.style("Enter the output path", fg='yellow', bold=True),
            default=default_filename
        )
    )
    
    # Summary Output
    click.echo()
    click.echo(style_header("🔍 BMF Module Template Definition Summary"))
    click.echo(
        click.style(
            json.dumps(module_definition, indent=2),
            fg='magenta'
        )
    )

    if do_generate_template:
        # Generate template
        click.echo()
        click.echo(style_header("🚀 Generating BMF Module Template..."))
        render_module_template(module_definition.pop("template_name"), module_definition.pop("output_path"), **module_definition)
    else:
        click.echo()
        click.echo(style_header("📝 Writing Module Definition JSON..."))
        with open(module_definition["output_path"], 'w') as f:
            json.dump(module_definition, f, indent=4)

    click.echo()
    click.echo(style_header("🎉 Module Template Generation Complete!"))

def stage(n):
    header = f"--- Stage {n} of {TOTAL_STAGES} ---"
    click.echo(click.style(header, fg='blue', bold=True))

def main():
    create_module()

if __name__ == '__main__':
    main()