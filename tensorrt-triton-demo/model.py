"""
Triton Python Backend for TensorRT-LLM.
"""

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from tensorrt_llm import LLM, SamplingParams, BuildConfig
from tensorrt_llm.plugin.plugin import PluginConfig
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_DIR = f"/persistent-storage/models/{MODEL_ID}"

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95


def _first_value(array):
    """First element of a Triton tensor as a Python scalar or string."""
    return array.item() if array.ndim == 0 else array.flat[0]


def _read_text(request):
    tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
    if tensor is None:
        raise ValueError("missing required input 'text_input'")
    text = _first_value(tensor.as_numpy())
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return str(text)


def _optional_value(request, name, cast, default):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return default
    return cast(_first_value(tensor.as_numpy()))


def _text_response(text):
    return pb_utils.InferenceResponse(
        output_tensors=[pb_utils.Tensor(
            "text_output",
            np.array([text.encode("utf-8")], dtype=object),
        )]
    )


def _error_response(message):
    return pb_utils.InferenceResponse(error=pb_utils.TritonError(message))


class TritonPythonModel:
    def initialize(self, args):
        """Initialize TensorRT-LLM with PyTorch backend."""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        print("Initializing TensorRT-LLM...")
        plugin_config = PluginConfig.from_dict({
            "paged_kv_cache": True,
        })

        build_config = BuildConfig(
            plugin_config=plugin_config,
            max_input_len=4096,
            max_batch_size=128,  # Matches Triton max_batch_size in config.pbtxt
        )

        self.llm = LLM(
            model=MODEL_DIR,
            build_config=build_config,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        print("✓ Model ready")

    def execute(self, requests):
        """
        Execute inference on a batch of requests.

        Triton returns exactly one response per request, in the same order.
        Requests that fail validation get an error response and are left out of
        the generation batch.
        """
        responses = [None] * len(requests)
        prompts = []
        sampling_params = []
        slots = []  # request index for each prompt, in the same order as prompts

        for i, request in enumerate(requests):
            try:
                text = _read_text(request)
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                params = SamplingParams(
                    temperature=_optional_value(request, "temperature", float, DEFAULT_TEMPERATURE),
                    top_p=_optional_value(request, "top_p", float, DEFAULT_TOP_P),
                    max_tokens=_optional_value(request, "max_tokens", int, DEFAULT_MAX_TOKENS),
                )
            except Exception as e:
                print(f"Error processing request {i}: {e}", flush=True)
                responses[i] = _error_response(f"Invalid request: {e}")
                continue

            prompts.append(prompt)
            sampling_params.append(params)
            slots.append(i)

        if prompts:
            try:
                outputs = self.llm.generate(prompts, sampling_params)
            except Exception as e:
                print(f"Error in generate: {e}", flush=True)
                for i in slots:
                    responses[i] = _error_response(f"Generation failed: {e}")
            else:
                for i, output in zip(slots, outputs):
                    responses[i] = _text_response(output.outputs[0].text)

        # Safety net: every request must get a response.
        return [
            r if r is not None else _error_response("no output produced for request")
            for r in responses
        ]

    def finalize(self):
        """Cleanup on shutdown."""
        if hasattr(self, "llm"):
            self.llm.shutdown()
            torch.cuda.empty_cache()
