import random
import vampnet
import audiotools as at

# load the default vampnet model
interface = vampnet.interface.Interface.default()

# list available finetuned models
finetuned_model_choices = interface.available_models()
print(f"available finetuned models: {finetuned_model_choices}")

# pick a random finetuned model
model_choice = random.choice(finetuned_model_choices)
print(f"choosing model: {model_choice}")

# load a finetuned model
interface.load_finetuned(model_choice)

# load an example audio file
signal = at.AudioSignal("assets/example.wav")

# get the tokens for the audio
codes = interface.encode(signal)

# build a mask for the audio
mask = interface.build_mask(
    codes, signal,
    periodic_prompt=7, 
    upper_codebook_mask=3,
)

# generate the output tokens
output_tokens = interface.vamp(
    codes, mask, return_mask=False,
    temperature=1.0, 
    typical_filtering=True, 
)

# convert them to a signal
output_signal = interface.decode(output_tokens)

# save the output signal
output_signal.write("scratch/output.wav")