# Ray LLM 后训练 Notebook（中文）

[English README](README.en.md) · [English-only Notebook](ray_llm_post_training_tutorial_en.ipynb)

打开 `ray_llm_post_training_tutorial.ipynb`，从上往下阅读和运行。

中文版现包含 **58 个单元**，已与英文版同步 vLLM、SGLang 和 GRPO 集成章节；说明、代码注释、示例提示词和运行提示均提供中文。

## 启动

建议使用 Linux / WSL2，Python 3.11，以及独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name ray-posttrain --display-name "Ray Post-training"
jupyter lab ray_llm_post_training_tutorial.ipynb
```

在 Jupyter 中选择上述 kernel。先按 PyTorch 官网安装适合驱动/CUDA 的 PyTorch，再运行 Notebook 安装单元并重启 kernel。Windows 原生可阅读，GPU 训练建议移至 Linux / WSL2。项目路径在 WSL 中通常为 `/mnt/c/Users/kevin/Desktop/ray`。

## 学习顺序

1. Ray Task、ObjectRef、Actor 的 CPU 小实验。
2. 指令数据与聊天模板，Ray Data 预处理。
3. Ray Train + Transformers + PEFT 的 LoRA SFT（默认不开启 GPU 训练）。
4. 修改 `RUN_TRAINING=True`，用 10 个更新步检查训练链路。
5. 加载 adapter、理解模型导出与精确恢复的区别。
6. 多卡部署、DPO 偏好对、GRPO 组内奖励及排错。
7. 理解 prefill/decode、KV cache、连续批处理、前缀缓存与张量并行。
8. 在独立环境启动 vLLM 或 SGLang，用统一的 OpenAI-compatible 客户端调用。
9. 比较延迟与吞吐，用 Ray 编排有并发上限的 rollout 请求并计算规则奖励。
10. 理解策略版本、token log-prob、权重同步、缓存失效和 GPU 资源放置。

## 可选推理实验

第 19–27 节新增进阶学习：数据泄漏审计、SFT assistant-label 检查、DPO 损失与梯度、配对评估、分布式性能诊断、恢复验收、实验 manifest 和毕业项目。详见[教程审视记录](CURRICULUM_REVIEW.md)。

请克隆完整仓库并从根目录启动 Notebook；新增单元依赖 `labs/posttrain_checks.py`。运行 `python labs/posttrain_checks.py` 即可执行无需 GPU 或模型下载的 CPU 正确性检查。已实际运行通过，但不代表真实 LLM 训练成功。实验清单写入已忽略的本地 `artifacts/` 目录。

第 12–18 节提供服务启动命令、统一客户端、vLLM 离线生成、并发测量及 Ray rollout 示例。`RUN_INFERENCE`、`RUN_OFFLINE_VLLM`、`RUN_BENCHMARK`、`RUN_RAY_ROLLOUT` 默认均为 `False`，准备好环境后再开启。

Notebook kernel 只需按需安装轻量 `openai` 客户端。推理引擎按官方指南安装到各自独立环境，避免覆盖 SFT 的依赖。单 GPU 默认顺序运行训练与推理。服务示例使用基础模型，并不自动加载 SFT adapter。

Ray HTTP 任务仅编排客户端请求，不管理外部服务的 GPU。规则奖励和组内 advantage 计算也不构成完整 GRPO；实际训练还需要精确 token 数据、策略版本管理和框架支持的权重同步。

示例模型为 Qwen2.5-0.5B-Instruct，目的是降低学习成本，并非声称它是最新模型。SFT 使用极小教学数据，不能据此评估模型效果。首次下载需要能访问 Hugging Face。

## 版本与验证

2026-11-17 核对：PyPI 与 Ray 官方 latest 文档均显示 **2.60.0**；关键 Transformers 集成接口同时对照了 Ray 2.60.0 标签源码。

Notebook 锁定 Ray 2.60.0，配套库采用明确版本以限制接口漂移；不是完整环境锁文件，尚未验证 GPU 依赖组合。

已完成 Notebook JSON/nbformat 校验及全部 Python 单元语法解析。**未安装训练依赖、未执行 Ray worker 或 GPU 训练、未下载模型**，因此没有训练成功、吞吐或质量改善的实测结论。执行后请保存实际依赖锁文件并记录 GPU/CUDA 版本。

2026-11-18 同步了英文版推理章节的中文版本。推理接口依据 2026-11-17 核对的 vLLM 官方快速入门源码和 SGLang 请求教程；未启动推理服务、执行性能测量或测试分布式权重同步。

主要来源：

- https://pypi.org/pypi/ray/json
- https://docs.ray.io/en/latest/train/train.html
- https://github.com/ray-project/ray/blob/ray-2.60.0/python/ray/train/huggingface/transformers/_transformers_utils.py
- https://huggingface.co/docs/trl/dpo_trainer
- https://huggingface.co/docs/trl/grpo_trainer
- https://github.com/vllm-project/vllm/blob/main/docs/getting_started/quickstart.md
- https://docs.sglang.io/docs/basic_usage/send_request
- https://docs.sglang.io/
- https://huggingface.co/docs/trl/vllm_integration
