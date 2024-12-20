import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import music21 as m21
import ast

# Load the Mistral 7B model from Hugging Face
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define Sampling Parameters
temperature = 0.36
min_p = 0.02
top_p = 0.88
max_length = 1024

# Inference Module
def inference(prompt, model, tokenizer, temperature, top_p, min_p, max_length):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        do_sample=True  # Enables sampling
    )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# Prompting and Dict Filtering
def generate_music_composition(prompt):
    structured_prompt = (
        f"Generate music21 Python code to create a music composition. Your output should contain only python code and should always begin by importing music21, along with any other requisite libraries."
        f"The code should define a stream and add notes to it, in accordance with the prompt defined below. Be sure to only output in music21."
        f"{prompt}"
    )
    composition_code = inference(structured_prompt, model, tokenizer, temperature=0.66, top_p=0.88, min_p=0.02, max_length=1024)

    write_to_markdown(composition_code, 'output.md')

    # Post-processing to clean and validate the output
    composition_code = composition_code.strip()
    if not composition_code.startswith("import music21"):
        raise ValueError("Model output is not valid music21 Python code.")

    # Strip non-code elements
    composition_code = strip_non_code_elements(composition_code)

    # Validate syntax
    validate_python_syntax(composition_code)

    return composition_code

# Strip Non-Code Elements
def strip_non_code_elements(code):
    # Remove any non-code elements (e.g., comments, docstrings)
    lines = code.split('\n')
    cleaned_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
    return '\n'.join(cleaned_lines)

# Validate Python Syntax
def validate_python_syntax(code):
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in generated code: {e}")

# Write model output to markdown
def write_to_markdown(code, file_path):
    with open(file_path, 'w') as file:
        file.write(f"# Generated music21 Python Code\n\n\n{code}\n\n")

# Execute the generated code and convert to MIDI and MusicXML
def execute_music21_code(composition_code):
    # Execute the generated code in a controlled environment
    local_vars = {}
    exec(composition_code, globals(), local_vars)

    # Extract the stream from the local variables
    stream = local_vars.get('stream')
    if stream is None:
        raise ValueError("Generated code does not define a 'stream' variable.")

    return stream

# Strip system prompt from output
def strip_prompt(output, prompt):
    # Remove the prompt from the output
    if output.startswith(prompt):
        output = output[len(prompt):].strip()
    return output

def write_to_midi(stream, file_path):
    stream.write('midi', fp=file_path)

def write_to_musicxml(stream, file_path):
    stream.write('musicxml', fp=file_path)

# Example Usage
prompt = "Create a simple melody with a tempo of 120 bpm."
composition_code = generate_music_composition(prompt)

# Execute the generated code to get the music21 stream
stream = execute_music21_code(composition_code)

# Write to MIDI
write_to_midi(stream, 'output.mid')

# Write to MusicXML
write_to_musicxml(stream, 'output.xml')