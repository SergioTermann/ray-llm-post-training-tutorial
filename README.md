# Ray LLM 后训练 Notebook（中文）

[English README](README.en.md) · [English-only Notebook](ray_llm_post_training_tutorial_en.ipynb)

打开 `ray_llm_post_training_tutorial.ipynb`，从上往下阅读和运行。

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

示例模型为 Qwen2.5-0.5B-Instruct，目的是降低学习成本，并非声称它是最新模型。SFT 使用极小教学数据，不能据此评估模型效果。首次下载需要能访问 Hugging Face。

## 版本与验证

2026-11-17 核对：PyPI 与 Ray 官方 latest 文档均显示 **2.60.0**；关键 Transformers 集成接口同时对照了 Ray 2.60.0 标签源码。

Notebook 锁定 Ray 2.60.0，配套库采用明确版本以限制接口漂移；不是完整环境锁文件，尚未验证 GPU 依赖组合。

已完成 Notebook JSON/nbformat 校验及全部 Python 单元语法解析。**未安装训练依赖、未执行 Ray worker 或 GPU 训练、未下载模型**，因此没有训练成功、吞吐或质量改善的实测结论。执行后请保存实际依赖锁文件并记录 GPU/CUDA 版本。

主要来源：

- https://pypi.org/pypi/ray/json
- https://docs.ray.io/en/latest/train/train.html
- https://github.com/ray-project/ray/blob/ray-2.60.0/python/ray/train/huggingface/transformers/_transformers_utils.py
- https://huggingface.co/docs/trl/dpo_trainer
- https://huggingface.co/docs/trl/grpo_trainer
