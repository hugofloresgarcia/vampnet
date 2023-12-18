from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import textwrap


# Define the parameters
template_path = 'data/artifacts/salad_bowl/template.png'

font_path = "data/artifacts/salad_bowl/fonts/Laca Text SemiBold.otf"  
font_size = 110  

output_folder = 'data/artifacts/salad_bowl/cards'
output_folder = Path(output_folder)
output_folder.mkdir(exist_ok=True, parents=True)

# Define the prompts
import yaml
with open("scripts/utils/salad_bowl/prompts.yml", "r") as f:
    prompts = yaml.load(f, Loader=yaml.FullLoader)
    prompts = [p for promptlist in prompts.values() for p in promptlist]
# Load the template image
template_image = Image.open(template_path)

# Prepare the font
font = ImageFont.truetype(font_path, font_size)

# The path where the cards are saved
output_files = [output_folder / f'{p}.png' for p in prompts]

# Function to wrap text
def draw_text(draw, text, font, max_width):
    # Break the text into lines
    print(max_width)
    lines = textwrap.wrap(text, width=max_width)
    for idx, line in enumerate(lines):
        print(f"Drawing line {idx} of {len(lines)}")
        text_length = draw.textlength(line, font=font)
        height = font_size
        text_x = (template_image.width - text_length) / 2 
        text_y = (template_image.height - (610 - height * idx))
        draw.text((text_x, text_y),  line, font=font, fill="white")

for i, prompt in enumerate(prompts):
    # Make a copy of the template image
    card_image = template_image.copy()
    draw = ImageDraw.Draw(card_image)

    # Draw the text on the image
    # draw.text((text_x, text_y), prompt, font=font, fill="white")
    draw_text(draw, prompt,  font, 16)

    # Save the card
    card_image.save(output_files[i])
