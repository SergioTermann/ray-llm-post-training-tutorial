# Ray LLM Post-Training Tutorial

An English-only Jupyter tutorial covering Ray Core, Ray Data, LoRA supervised fine-tuning with Ray Train, vLLM and SGLang inference, and the integration requirements for DPO and GRPO.

**[Open the English notebook](ray_llm_post_training_tutorial_en.ipynb)**

## Getting started

Use Linux or WSL2, Python 3.11, and a separate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name ray-posttrain --display-name "Ray Post-training"
jupyter lab ray_llm_post_training_tutorial_en.ipynb
```

Select the `Ray Post-training` kernel. Install a PyTorch build compatible with your GPU driver using the [official installation instructions](https://pytorch.org/get-started/locally/), then run the notebook's dependency installation cell and restart the kernel.

Native Windows is suitable for reading the notebook; use Linux or WSL2 for the GPU training workflow. The notebook recommends a single NVIDIA GPU with at least 24 GB of VRAM as a conservative starting point, not a measured minimum. Initial model downloads require access to Hugging Face.

## Learning path

1. Run CPU examples of Ray tasks, ObjectRefs, and actors.
2. Prepare instruction data with Ray Data and the model's chat template.
3. Inspect the Ray Train + Transformers + PEFT LoRA training loop.
4. Set `RUN_TRAINING=True` to run 10 optimizer steps as a smoke test.
5. Load the adapter and distinguish model exports from resumable training checkpoints.
6. Explore multi-GPU execution, DPO preference pairs, GRPO advantages, and troubleshooting.
7. Learn prefill/decode, KV cache, continuous batching, prefix caching, and tensor parallelism.
8. Start vLLM or SGLang in a separate environment and query either through the same OpenAI-compatible client.
9. Compare request latency and throughput, then orchestrate bounded rollout requests with Ray.
10. Study policy versions, token-level log probabilities, weight synchronization, cache invalidation, and GPU placement for online RL.

## Optional inference labs

Sections 12–18 include server launch instructions, a shared chat client, an offline vLLM example, a small concurrency benchmark, and Ray tasks that generate and score rollout answers. The notebook now contains 43 cells. All GPU and request examples are opt-in: `RUN_INFERENCE`, `RUN_OFFLINE_VLLM`, `RUN_BENCHMARK`, and `RUN_RAY_ROLLOUT` default to `False`.

Install the lightweight `openai` client in the notebook kernel if needed. Install each serving engine in its own environment using its current official installation guide, rather than adding both engines to the pinned SFT environment. Run only one GPU workload at a time unless resources are explicitly partitioned. The serving examples use the base model, not the SFT adapter.

The Ray HTTP tasks schedule client requests; they do not reserve or manage the external server's GPU. The toy reward and advantage examples do not implement a policy update. Full RL training requires a framework-supported integration with exact token data, policy-version tracking, and synchronized weights.

All explanations, comments, example prompts, dataset answers, and status messages in the English notebook are in English. The model is `Qwen/Qwen2.5-0.5B-Instruct`, chosen for a manageable learning example rather than recency. The tiny synthetic dataset is not suitable for measuring model quality. DPO and GRPO sections explain interfaces and concepts; they do not launch complete preference-optimization runs.

## Versions and validation

On November 17, 2026, PyPI and Ray's official latest documentation reported **Ray 2.60.0**. The Hugging Face integration was also checked against source code at the `ray-2.60.0` tag.

The notebook pins Ray and selected companion libraries to limit API drift. This is not a complete environment lock file, and the dependency combination has not been GPU integration-tested here.

Completed checks: notebook format validation, Python syntax parsing for every code cell, and a scan for untranslated Chinese text. Training dependencies have not been installed for validation, Ray workers and GPU training have not been executed, and model weights have not been downloaded. No successful training run, measured throughput, or quality improvement is claimed. After running successfully, record the resolved dependencies, GPU, driver, and CUDA versions.

The inference additions were checked against the official vLLM quickstart source and SGLang request tutorial. Serving, GPU benchmarks, rollout requests, and weight synchronization have not been executed. No inference performance comparison is claimed.

## References

- [Ray package metadata](https://pypi.org/pypi/ray/json)
- [Ray Train](https://docs.ray.io/en/latest/train/train.html)
- [Ray Hugging Face integration source](https://github.com/ray-project/ray/blob/ray-2.60.0/python/ray/train/huggingface/transformers/_transformers_utils.py)
- [TRL DPO](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)
- [vLLM quickstart source](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/quickstart.md)
- [SGLang request tutorial](https://docs.sglang.io/docs/basic_usage/send_request)
- [SGLang installation and documentation](https://docs.sglang.io/)
- [TRL vLLM integration](https://huggingface.co/docs/trl/vllm_integration)

The original Chinese edition remains available in [README.md](README.md) and [the original notebook](ray_llm_post_training_tutorial.ipynb).
