# tensorrt-triton-demo

Llama 3.2 3B Instruct served by TensorRT-LLM (PyTorch backend) behind Triton Inference Server on Cerebrium.
Built from https://cerebrium.ai/docs/v4/examples/deploy-an-llm-with-tensorrtllm-tritonserver

## Files
- `Dockerfile` – Triton 25.10 trtllm image, copies the model repo
- `config.pbtxt` – Triton model config (python backend, dynamic batching, max_batch_size 128)
- `model.py` – Triton python backend wrapping `tensorrt_llm.LLM`
- `download_model.py` – one-off job that pulls the HF weights into `/persistent-storage/models`
- `cerebrium.toml` – A10 GPU, 4 CPU, 40 GB, custom runtime on port 8000
- `test_infer.sh` – curl helper for the KServe v2 infer endpoint

## Steps
1. `cerebrium login`
2. Accept the licence at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Add secret `HF_AUTH_TOKEN` (your HF token) in the Cerebrium dashboard, or `cerebrium secrets add HF_AUTH_TOKEN=hf_...`
4. `cerebrium run download_model.py`
5. `cerebrium deploy`
6. `./test_infer.sh https://api.cerebrium.ai/v4/p-xxxxxxxx/tensorrt-triton-demo`
