import os
import torch._dynamo
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

def main():
    torch._dynamo.config.cache_size_limit = 64
    model_path = os.path.expanduser("../models/Qwen3.5-4B")

    prompts = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "assets/logo.png"},
                    {"type": "text", "text": "Please describe the picture"}
                    ]
            },
            {
                "role": "user",
                "content": "Please introduce Microsoft",
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "assets/example.mp4"},
                    {"type": "text", "text": "What scene does this video depict? Please provide a detailed explanation."}
                ]
            },
        ]

    llm = LLM(
        model_path,
        enforce_eager=False,
        tensor_parallel_size=2,
        max_model_len=65536,
        max_num_batched_tokens=32768,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=4096,
    )

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        # print(f"Completion: {output['text']!r}")
        print(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()
