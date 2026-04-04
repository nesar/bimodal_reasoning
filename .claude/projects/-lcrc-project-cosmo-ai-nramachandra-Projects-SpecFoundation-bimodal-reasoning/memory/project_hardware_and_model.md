---
name: Hardware and model targets
description: GPU allocation (6xA100, 2 currently), model progression from gpt-oss-20b to 120b
type: project
---

Current model: openai/gpt-oss-20b on HuggingFace.
Target model: 120b variant (near-future plan).

Hardware: 6xA100 GPUs per node on LCRC cluster.
Current allocation: only 2xA100 in active runs.

**Why:** 20b is the validation/development model; 120b is the science-grade target once the pipeline is proven.
**How to apply:** All training configs, DeepSpeed settings, and batch sizes should work on 2xA100 now but be designed to scale to 6xA100 with 120b. The existing ds_config_zero3.json with CPU offloading is already set up for this scaling path.
