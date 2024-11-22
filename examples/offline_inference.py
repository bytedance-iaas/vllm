import os
from vllm import LLM, SamplingParams

# os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "True"

# Set CUDA_LAUNCH_BLOCKING to 1 for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Optionally, enable device-side assertions if needed
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
# llm = LLM(model="facebook/opt-125m", enforce_eager=True)
# llm = LLM(model="meta-llama/Llama-2-7b-hf", enforce_eager=True, block_size=32)
llm = LLM(model="meta-llama/Llama-2-7b-hf", enforce_eager=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
