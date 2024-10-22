import time
from transformers import pipeline, set_seed
from vllm import LLM, SamplingParams

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
os.environ['HF_HOME'] = '/data00/jack/as_tmp'
os.environ['TRANSFORMERS_CACHE'] = '/data00/jack/as_tmp'
os.environ['HF_DATASETS_CACHE'] = '/data00/jack/as_tmp'
os.environ['TMPDIR'] = '/data00/jack/as_tmp/tmp'

##os.environ['CUDACXX'] = '/usr/local/cuda-12.1/bin/nvcc' # T4
os.environ['CUDACXX'] = '/usr/local/cuda-12.4/bin/nvcc' # L4
#os.environ['CUDA_LAUNCH_BLOCKING']= '1'
os.environ['HF_TOKEN'] = 'YOUR KEY'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # CN mirror
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'

os.environ['http_proxy'] = 'http://sys-proxy-rd-relay.byted.org:8118'
os.environ['https_proxy']= 'http://sys-proxy-rd-relay.byted.org:8118'

MAX_TOKENS = 2048 

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", enforce_eager=True,
          trust_remote_code=True, enable_prefix_caching=False)
          #trust_remote_code=True, enable_prefix_caching=True)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_TOKENS)
sampling_params_truncate = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_TOKENS, truncate_prompt_tokens=1)

######################################################################################

def print_outputs(outputs, generation_time):
    for output in outputs:
        prompt = output.prompt
        # Calc tokens
        output_count = len(output.outputs)
        print(f"Number of generated outputs: {output_count}")

        prompt_count = len(prompt.split())
        generated_text = output.outputs[0].text
        decode_count = len(generated_text.split())
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        print(f"Prompt count: {prompt_count}, Decode count: {decode_count}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"MAX_TOKENS: {MAX_TOKENS}")
        print(f"url token count: https://quizgecko.com/tools/token-counter")
        print(f"url word count: https://wordcounter.net/")
    print("-" * 80)

#print("=" * 80)

# Initialize the conversation with the user's question
conversation = [
    {
        "role": "system",
        "content": "You are helpful travel guide."
    },
    {
        "role": "user",
        "content": (
            "I will write you my location, "
            "and you will suggest a place to visit nearby. I am currently in San Jose, "
            "USA, and I want to explore various attractions. However, I have some "
            "specific preferences that you should consider when making recommendations. "
            "I particularly enjoy visiting museums that focus on technology and art, "
            "as I find them both fascinating and enriching. In addition, I love spending "
            "time at the beach; the sound of the waves and the warmth of the sun always "
            "put me in a great mood.\n\nWhen it comes to dining, I prefer restaurants "
            "that offer sweet dishes, such as dessert cafes or places known for their "
            "delicious pastries. It’s important to note that I do not eat pork, so please "
            "suggest restaurants that offer a variety of options, particularly those with "
            "delectable seafood or vegetarian dishes. I also have a fondness for coffee "
            "shops that serve unique and flavorful drinks, but I don’t consume any sodas "
            "or sugary beverages, so I appreciate recommendations for places that offer "
            "healthy alternatives or artisanal coffee.\n\nAdditionally, I have a slight "
            "allergy to mosquitoes, so outdoor venues should ideally be equipped with "
            "adequate seating and insect repellents. I would love to hear about local "
            "parks that have pleasant walking paths and are family-friendly, as well as "
            "shopping malls with a variety of stores, particularly bookstores and bakeries. "
            "If there are any nearby pharmacies or fitness centers where I could pick up "
            "some essentials or get in a quick workout, that would be fantastic too."
        )
    }
]

# Generate the assistant's response
start_time = time.time()
response1 = llm.chat(conversation, sampling_params=sampling_params_truncate, use_tqdm=True)
#response1 = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=True)
end_time = time.time()
generation_time = end_time - start_time
print_outputs(response1, generation_time)


print(f"Number of elements in response1: {len(response1)}")
## Iterate through each output in response1 (assuming it's a list)
#for output in response1:
#    # Get the generated text from each output
#    response1_text = output.outputs[0].text if output.outputs else "No response"
#    
#    # Append the assistant's response to the conversation
#    conversation.append({
#        "role": "assistant",
#        "content": response1_text  # Use the extracted text here
#    })

# 相同prompt連續跑兩次.chat結果分別是:
# 1. [00:47<00:00, 47.08s/it, est. speed input: 6.97 toks/s, output: 16.29 toks/s]
# 2. [00:38<00:00, 38.96s/it, est. speed input: 8.42 toks/s, output: 16.30 toks/s]

# 我這範例才能確實用到APC (明顯prefill throughput大大提)
# 1. [00:47<00:00, 47.03s/it, est. speed input: 6.97 toks/s, output: 16.31 toks/s]
# 2. [00:13<00:00, 13.22s/it, est. speed input: 83.15 toks/s, output: 16.04 toks/s]

# 理論上truncate_prompt_tokens=500/100 應該要導致APC失效, 但是input throughput還是超高....表示還是有不清楚的步驟. (光看我打印出來的不準, 因為truncate發生在vllm內部, 所以只能看perf結果)
# truncate_prompt_tokens=1, 結果還是照常推理.....明顯這變數使用方式有誤 (使用enable_prefix_caching=False 也沒改善)

# Now you can continue the conversation with a new user prompt
#start_time = time.time()
#response2 = llm.chat(conversation, sampling_params=sampling_params_truncate, use_tqdm=True)
#end_time = time.time()
#generation_time = end_time - start_time
#print_outputs(response2, generation_time)

