# Ray LLM Post-Training Tutorial

An English-only Jupyter tutorial covering Ray Core, Ray Data, and LoRA supervised fine-tuning with Ray Train, followed by a roadmap to DPO and GRPO.

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

All explanations, comments, example prompts, dataset answers, and status messages in the English notebook are in English. The model is `Qwen/Qwen2.5-0.5B-Instruct`, chosen for a manageable learning example rather than recency. The tiny synthetic dataset is not suitable for measuring model quality. DPO and GRPO sections explain interfaces and concepts; they do not launch complete preference-optimization runs.

## Versions and validation

On November 17, 2026, PyPI and Ray's official latest documentation reported **Ray 2.60.0**. The Hugging Face integration was also checked against source code at the `ray-2.60.0` tag.

The notebook pins Ray and selected companion libraries to limit API drift. This is not a complete environment lock file, and the dependency combination has not been GPU integration-tested here.

Completed checks: notebook format validation, Python syntax parsing for every code cell, and a scan for untranslated Chinese text. Training dependencies have not been installed for validation, Ray workers and GPU training have not been executed, and model weights have not been downloaded. No successful training run, measured throughput, or quality improvement is claimed. After running successfully, record the resolved dependencies, GPU, driver, and CUDA versions.

## References

- [Ray package metadata](https://pypi.org/pypi/ray/json)
- [Ray Train](https://docs.ray.io/en/latest/train/train.html)
- [Ray Hugging Face integration source](https://github.com/ray-project/ray/blob/ray-2.60.0/python/ray/train/huggingface/transformers/_transformers_utils.py)
- [TRL DPO](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)

The original Chinese edition remains available in [README.md](README.md) and [the original notebook](ray_llm_post_training_tutorial.ipynb).
