# ASC26 Embodied World Model 提交说明

## 提交材料

```text
chenyc_worldmodel/
├── code/
├── proposal_file/
└── results/
```

`code/` 包含优化后的完整推理源码、20 个官方 case 脚本和 AOTInductor
运行包；`proposal_file/` 包含项目报告；`results/` 包含 20 个最终 MP4、20 个 `output.log` 和 `summary.json`。

## 运行环境

最终测试环境为 Python 3.10.20、PyTorch 2.3.1+cu121、CUDA 12.1 和
NVIDIA A100 80GB（SM80）。AOTInductor 文件与该软件栈及 GPU 架构绑定；
其他环境应设置 `WMA_AOT_INDUCTOR=0`，使用较慢的 TorchInductor JIT 路径。

输入数据和 checkpoint 未包含在提交包中。运行前需按官方目录恢复：

```text
ckpts/unifolm_wma_dual.ckpt
examples/world_model_interaction_prompts/
unitree_*/case*/world_model_interaction_prompts/
```

如果共享归一化数据位于其他位置，应设置绝对路径：

```bash
export WMA_DATA_DIR=/absolute/path/to/world_model_interaction_prompts
```

## 运行方法

```bash
cd chenyc_worldmodel/code/unifolm-world-model-action
pip install -e .
source optimization_env.sh
bash unitree_g1_pack_camera/case1/run_world_model_interaction.sh
```

官方 case 脚本保持不变；优化开关也已写成提交源码的默认值。
`optimization_env.sh` 用于明确复现实测环境，并在 Conda 可用时选择对应的
`libstdc++.so.6`。模型默认使用 `HF_HUB_OFFLINE=1`，运行期间不访问网络。

## 最终结果

- 完成 case：20/20
- PSNR 达标：20/20，均不低于 25 dB
- 最低 PSNR：25.0862 dB
- 优化后总推理时间：3909.336 秒
- 平均每 case：195.467 秒
- 输出视频：H.264，512×320，8 FPS

提交副本已通过 Python、Shell、JSON、视频格式、文件数量、SHA-256 和解压回读
检查。迁移后的 AOT 路径也已完成 GPU smoke test，并确认实际从提交目录加载
全部 380 个 cubin 文件。

## 已知限制

现有记录只保留了 5 个可信的未优化 baseline 时间，其余 15 个 case 在
`summary.json` 中为 `null`、报告中为 `N/A`。官方说明要求所有 20 个 case
提供优化前后时间；若需完全满足该要求，必须使用未优化配置重新运行这 15 个
case。缺失数据不可推算或编造。
