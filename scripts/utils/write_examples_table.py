from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape


ROOT_DIR = Path(__file__).parent.parent.parent
EXAMPLES_DIR = ROOT_DIR / "examples"
TEMPLATE_DIR = ROOT_DIR / "templates"
HTML_FILE = ROOT_DIR / "audio_examples.html"
TEMPLATE_FILENAME = "examples_table_template.html"

experiments = sorted([d.name for d in EXAMPLES_DIR.iterdir() if d.is_dir()])

# Prepare data for template
all_sounds = set() # denotes the rows of the table
for exp in experiments:
    exp_dir = EXAMPLES_DIR / exp
    sounds = [f.name for f in exp_dir.iterdir() if f.suffix == '.wav']
    all_sounds.update(sounds)
sounds = sorted(all_sounds)

audio_exists = {exp: {} for exp in experiments}
for exp in experiments:
    for sound in sounds:
        path = EXAMPLES_DIR / exp / sound
        if path.exists():
            audio_exists[exp][sound] = str(path)
        else:
            audio_exists[exp][sound] = None

# Render HTML file
env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(['html'])
)
template = env.get_template(TEMPLATE_FILENAME)

html = template.render(
    experiments=experiments,
    sounds=sounds,
    audio_exists=audio_exists
)

with open(HTML_FILE, "w") as f:
    f.write(html)

print(f"HTML file written to {HTML_FILE}")
