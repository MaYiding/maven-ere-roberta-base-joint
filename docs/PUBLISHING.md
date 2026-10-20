# 发布代码与权重

训练完成且状态为 `completed` 或 `early_stopped` 后，执行：

```bash
python scripts/package_release.py --run-dir runs/roberta-joint-seed42 \
  --output-dir artifacts/roberta-joint-seed42
```

本次服务器上的正式训练包装器会自动执行该步骤。重复打包请指定新目录。

- GitHub：发布当前代码仓库，或解开 `training-code.tar.gz`。保留 LICENSE、CITATION 和上游署名。
- Hugging Face：上传 `artifacts/roberta-joint-seed42/huggingface/` 中的全部文件。
- `release_manifest.json` 包含模型与代码归档的 SHA-256。
- `latest.pt` 含优化器及 RNG，是续训状态，不属于 HF 推理权重包。
- `test_predictions.jsonl` 留作官方评测提交，不表示已经获得测试成绩。

HF 目录提供 safetensors、完整四任务自定义源码、tokenizer、模型卡、验证指标、训练配方、来源清单与许可归属。
上传时可使用 `hf upload <你的HF模型仓库ID> <huggingface目录>`，或网页上传。
GitHub 远程仓库与HF仓库ID由发布者创建后配置；训练脚本不会登录账号、创建仓库或上传内容。

## English

After full training completes or early-stops, `scripts/package_release.py` creates
an upload-ready `huggingface/` directory, `training-code.tar.gz`, and a release
checksum manifest. Upload all files from `huggingface/` to your model repository;
publish the code repository or extracted code archive on GitHub. Retain licenses
and attribution. `latest.pt` is a private resume checkpoint, not the inference
release. Hidden-test predictions do not constitute test scores. No script logs
into accounts, creates remote repositories, or uploads files automatically.
