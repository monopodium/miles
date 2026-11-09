# LZip-P2P (UCCL-P2P + 无损压缩) 在 miles / SGLang 中的集成与 rebuttal 实验指南

本指南对应 ICLR 2027 rebuttal 需要补的两个实验：

| Reviewer 问题 | 实验 | 脚本 |
|---|---|---|
| Q1：与 ZipNN / NeuZip / DFloat11 / ZipCCL 对比 | A1 编解码器对比（单 GPU，真实权重）；A2 整模型 weight-sync（2 节点） | `uccl/p2p/benchmarks/benchmark_codecs.py`、`benchmark_weight_sync.py` |
| Q2：RL 端到端收益 | B miles RL 训练端到端（2 节点 × 1 GPU，trainer 与 rollout 分节点） | `examples/infra_features/p2p_weight_transfer/run.py` + `summarize_lzip_e2e.py` |

## 0. 代码分支（都在 github.com/monopodium 下）

| 仓库 | 分支 | 内容 |
|---|---|---|
| `monopodium/uccl` | `lzip-rl-weight-sync` | `Endpoint.flush_compressed_writes()` 完成栅栏；子区间写不再误压缩；两个新 benchmark |
| `monopodium/sglang` | `lzip-uccl-weight-sync`（基于 `sglang-miles`） | `--remote-instance-weight-loader-start-seed-via-uccl`：rollout 端发布 UCCL endpoint + 每个权重的 RDMA 写描述符 |
| `monopodium/miles` | `lzip-uccl-weight-sync` | `--p2p-transport uccl`（`transports/uccl.py`）；`run.py` 支持 `--p2p-transport uccl`、`Qwen3-8B` 2 节点配置、`NUM_ROLLOUT`/`ROLLOUT_MAX_RESPONSE_LEN` |

### 数据通路（为什么这样集成才有收益）

```
trainer GPU (Megatron)            ── RDMA (IB/RoCE/EFA) ──>        rollout GPU (SGLang)
HF 权重 → miles 用 SGLang loader      split: 符号+尾数(~50%) 立即发      UCCL 接收线程：
装进 GPU transfer buffer (uint8)      encode: ANS(指数) 随后发          decompress → 直接写入 SGLang
每个权重 = 一个 RDMA WRITE            ↑ split-send 流水线                  参数地址，然后回 ACK
                                     flush_compressed_writes() 等全部 ACK → miles 才通知引擎更新完成
```

收益的前提（代码里都已处理）：

1. **transfer buffer 必须在 GPU 上**（`UcclTransport.transfer_buffer_device = cuda`）：DietGPU 在 GPU 上压缩，NIC 走 GPUDirect RDMA。Mooncake/NIXL 默认用 CPU buffer，多一次 D2H 拷贝。
2. **按权重精确注册压缩上下文**：UCCL 的压缩上下文绑定 `(addr, size, dtype)`。miles 端用 `reg_compress_only` 为每个权重槽注册一次并缓存（buffer 布局每次 sync 都一样）。整块 buffer 另有一个普通 `reg` 给小张量/非浮点张量直接写。
3. **完成栅栏**：压缩写的 `write()` 返回时，数据只是到了对端的 staging arena，解压是异步的。`flush_compressed_writes(conn)` 等到所有 ACK（解压 kernel 完成后才发 ACK），之后 miles 才去 finalize。这一点同时回答了 Reviewer 2 的 Q2（完整性如何保证）。
4. **每进程一个压缩器**：UCCL 的 DietGPU compressor 是进程级单例，miles 端用 `_compress_lock` 串行化压缩写。miles 保证每个 rollout rank 只有一个发送者（`assign_rollout_engine_ranks`），这满足 UCCL 接收端 arena 只服务一个压缩发送者的约束。**trainer rank 数 ≥ rollout rank 数时并行度最好**（下面的 1+1 配置正好 1:1）。
5. 只有 ≥ `UCCL_P2P_MIN_COMPRESS_BYTES`（默认 16 MiB）且 ≤ `UCCL_P2P_COMPRESS_BUFFER_BYTES` 的浮点权重会被压缩，其余原样写。

## 1. 机器要求

- 2 台机器，每台 **1×B200**，每 GPU 一张 RDMA NIC（IB 或 RoCEv2；AWS 用 EFA 加 `UCCL_P2P_TRANSPORT=efa`），GPUDirect RDMA 可用（`nvidia_peermem` 或 dmabuf）。租机时确认：两台在同一 RDMA fabric 上（不是只有 TCP 以太网），并记下每 GPU 的 NIC 带宽（论文里要写）。
- trainer 那台主机内存 ≥ 256 GB（Qwen3-8B 的 Adam 状态 offload 到 CPU 约 100 GB，加上 checkpoint 转换和 Ray）。
- 共享或各自本地盘 ≥ 300 GB（模型、Megatron torch_dist checkpoint、数据集）。
- 两台机器能互相 ssh / 访问 Ray 端口 6379、8265。

## 2. 环境（两台机器都做）

```bash
docker run --gpus all --network host --ipc host --privileged --ulimit memlock=-1 \
  -v /data:/root -it radixark/miles:latest bash

# 2.1 miles：换成 fork 分支
cd /root/miles && git remote add fork https://github.com/monopodium/miles.git
git fetch fork lzip-uccl-weight-sync && git checkout -B lzip-uccl-weight-sync FETCH_HEAD

# 2.2 sglang：把一个 commit cherry-pick 到镜像自带的 sglang-miles 上（避免版本漂移）
cd /sgl-workspace/sglang && git remote add fork https://github.com/monopodium/sglang.git
git fetch fork lzip-uccl-weight-sync && git cherry-pick FETCH_HEAD
#   （如果冲突：直接 git checkout FETCH_HEAD，它基于 2026-11-09 的 sglang-miles）

# 2.3 uccl：带 DietGPU 编译
cd /root && git clone -b lzip-rl-weight-sync https://github.com/monopodium/uccl.git && cd uccl
ARCH=sm_100  # B200/B300；H100/H200 用 sm_90
(cd thirdparty/dietgpu/dietgpu/float && make -j -f Makefile.cuda GPU_ARCH=$ARCH)
(cd thirdparty/dietgpu && python setup.py build)          # 仅实验 A1 需要（torch op，用于单 GPU codec 测试）
(cd p2p && USE_DIETGPU=1 make -j install)
export LD_LIBRARY_PATH=/root/uccl/thirdparty/dietgpu/dietgpu/float:$LD_LIBRARY_PATH
python -c "import torch; from uccl import p2p; print(p2p.backend_info())"

# 2.4 实验 A1 的对比库
pip install zipnn "nvidia-nvcomp-cu12>=4.2"
```

建议把 `LD_LIBRARY_PATH` 和下面的 UCCL 环境变量写进 `~/.bashrc`，因为 Ray worker 继承启动 shell 的环境，`run.py` 也会把所有 `UCCL_*` 变量转发到 runtime env。

常用 UCCL 环境变量：

| 变量 | 建议值 | 说明 |
|---|---|---|
| `UCCL_P2P_COMPRESS_STRATEGY` | `split_only`（LZip）/ `none`（基线） | 发送端和接收端必须一致 |
| `UCCL_P2P_COMPRESS_BUFFER_BYTES` | `1073741824` | 每进程 GPU 显存约占 **5×** 此值（压缩 buffer + 解压 arena + 3× DietGPU 临时区）。必须 ≥ 想被压缩的最大单个权重分片 |
| `UCCL_P2P_MIN_COMPRESS_BYTES` | 默认 16 MiB | 小于此值的张量不压缩 |
| `UCCL_P2P_RDMA_GID_INDEX` | RoCE 通常 3 | IB 一般不用设 |
| `UCCL_P2P_RDMA_DEV` | 可选 | 强制 NIC 列表 |
| `UCCL_P2P_LOG_LEVEL=INFO UCCL_DEBUG_SUBSYS=RDMA` | 调试用 | 能看到 `Encode done ... ratio` 即压缩生效 |

## 3. 冒烟测试（2 节点，先做）

```bash
cd /root/uccl/p2p
# 正确性 + 新的完成栅栏（不再 sleep）
UCCL_P2P_COMPRESS_STRATEGY=split_only torchrun --nnodes=2 --nproc_per_node=1 --node-rank=<0|1> \
  --master_addr=<HEAD_IP> tests/test_engine_write_compress.py --size $((64<<20)) --iters 32
# 单张量吞吐（论文 Fig. 2 在 write 路径上的复现）。write 是单边操作，--fence 决定计时是否包含接收端解压
# （论文原图是 send/recv 测的，recv 返回时解压已完成，本来就包含解压）
for S in none encode split_only; do for F in none end each; do
  UCCL_P2P_COMPRESS_STRATEGY=$S torchrun ... benchmarks/benchmark_uccl_compression.py --dtype bfloat16 --fence $F
done; done
```

`--fence none` 只计到压缩数据到达对端 staging arena；`end` 是包含解压的稳态吞吐；`each` 是每条消息都等解压完成。在 write 路径上报数字时用 `end`/`each`，口径才和论文原来的 send/recv 一致。

## 4. 实验 A：与 ZipNN / NeuZip / DFloat11 / ZipCCL 对比（回答 Q1）

### A1 编解码器对比（单 GPU，真实 BF16 权重）

```bash
hf download zai-org/GLM-4-9B-0414 --local-dir /root/models/GLM-4-9B-0414
hf download Qwen/Qwen3-30B-A3B   --local-dir /root/models/Qwen3-30B-A3B    # 或 Qwen3.5-35B-A3B（论文用的）
cd /root/uccl/p2p
for M in GLM-4-9B-0414 Qwen3-30B-A3B; do
  python benchmarks/benchmark_codecs.py --model /root/models/$M --max-tensors 32 \
    --link-gbps 400 --out /root/results/codecs_$M.jsonl | tee /root/results/codecs_$M.txt
done
```

结果的表述口径（写进论文时必须照实写）：

| 方法 | 可比性 | 我们怎么比 |
|---|---|---|
| NeuZip | 最可比：GPU 在线压缩/解压，指数位 ANS（nvCOMP） | 实测压缩率 + GPU 压缩/解压吞吐（复现其无损方案：nvCOMP ANS 编码指数字节，符号+尾数原样存） |
| DFloat11 | 解压在 GPU，但**压缩是离线 CPU**（Llama-3.1-8B 每个 block 约 191 s） | 压缩率按其 Huffman 方案精确计算；无法用于每步都变的 RL 权重 |
| ZipNN | **CPU 压缩/解压** | 实测压缩率 + CPU 吞吐（16 线程） |
| ZipCCL | GPU 在线、面向 collective，**无公开代码、无 P2P** | 按其固定 3-bit top-7 指数编码精确计算压缩率（oracle top-7，是其上界），吞吐引用其论文 |

脚本最后打印 `modeled xfer s` = 压缩 + 压缩后字节/链路 + 解压（无重叠，因为 4 个 baseline 都不做 compression–communication overlap）。LZip-P2P 的真实端到端数字用 A2/B 的实测值。

### A2 整模型 weight-sync（2 节点，trainer GPU → rollout GPU）

```bash
cd /root/uccl/p2p
for S in none split_only; do
  UCCL_P2P_COMPRESS_STRATEGY=$S UCCL_P2P_COMPRESS_BUFFER_BYTES=$((1<<30)) \
  torchrun --nnodes=2 --nproc_per_node=1 --node-rank=<0|1> --master_addr=<HEAD_IP> \
    benchmarks/benchmark_weight_sync.py --model /root/models/GLM-4-9B-0414 --iters 5 \
    --out /root/results/weight_sync.jsonl
done
```

每次迭代都做 bit-exact 校验；输出 `sync_s_median`（整模型同步 + 完成栅栏的时间）和 `effective_GBps`。注意 GLM-4-9B 的 `embed_tokens`/`lm_head`（约 1.2 GB）要能被压缩，就需要 `UCCL_P2P_COMPRESS_BUFFER_BYTES ≥ 2^31`；1 GiB 时它们会走不压缩路径，结果里 `compressed_path_fraction` 会体现。

## 5. 实验 B：miles RL 端到端（回答 Q2）

拓扑：node0 = 1 个 trainer GPU（Megatron，TP=1），node1 = 1 个 rollout GPU（一个单卡 SGLang engine）。每次权重更新都跨 RDMA 网络（一对 NIC）。模型用稠密的 Qwen3-4B 和 Qwen3-8B（8B 开 `--optimizer-cpu-offload`）；MoE 在单卡上训练放不下，不做。每个模型跑 3 组，**唯一变量是传输方式**：

| 组 | `--p2p-transport` | `UCCL_P2P_COMPRESS_STRATEGY` | 作用 |
|---|---|---|---|
| baseline | `uccl` | `none` | 同一套 GPU buffer + UCCL，不压缩（隔离出压缩本身的收益） |
| **LZip** | `uccl` | `split_only` | 我们的方法 |
| 参考 | `mooncake` | — | miles 默认 P2P 方案（CPU buffer），作为外部参考 |

```bash
cd /root/miles/examples/infra_features/p2p_weight_transfer
python run.py prepare Qwen3-8B            # 两台都做（worker 可加 --download-only）
python run.py prepare Qwen3-4B-2node

export UCCL_P2P_COMPRESS_BUFFER_BYTES=$((1<<30))
export NUM_ROLLOUT=20
# 第一次跑带校验（默认 --check-weight-update-equal，证明 bit-exact）；计时跑可 SKIP_VALIDATION=1
for S in none split_only; do
  export UCCL_P2P_COMPRESS_STRATEGY=$S
  # node1 先起 worker：
  #   python run.py run Qwen3-8B --p2p-transport uccl --node-rank 1 --head-ip <HEAD_IP>
  python run.py run Qwen3-8B --p2p-transport uccl --node-rank 0 --head-ip <HEAD_IP> 2>&1 \
    | tee /root/results/e2e_qwen3_8b_uccl_$S.log
  ray stop --force   # 两台都 stop，再跑下一组
done
unset UCCL_P2P_COMPRESS_STRATEGY
python run.py run Qwen3-8B --node-rank 0 --head-ip <HEAD_IP> 2>&1 | tee /root/results/e2e_qwen3_8b_mooncake.log
```

Qwen3-4B 同理，把 `Qwen3-8B` 换成 `Qwen3-4B-2node`。

显存预算（B200 180 GB）：Qwen3-4B 单卡全量训练约 75 GB（权重+fp32 梯度+Adam）；Qwen3-8B 的 Adam 状态 offload 到 CPU 后约 50 GB + 激活。再加 UCCL 约 5 GB、两个 transfer buffer 各 1 GB。rollout 卡 `sglang_mem_fraction=0.7` 已留出 UCCL 的余量。若 OOM，先把 `UCCL_P2P_COMPRESS_BUFFER_BYTES` 降到 512 MiB，或调小 `max_tokens_per_gpu`。

预算估计：单卡训练和单引擎 rollout 都更慢，每组 20 步：Qwen3-4B 短输出约 20–30 分钟，Qwen3-8B 短输出约 30–45 分钟，4096 输出约 2–3 小时；6～8 组加环境搭建，约 1–1.5 天。先跑短输出的 baseline/LZip 两组拿到主结果，再补长输出和 mooncake。

**两种 rollout 长度都要跑，并都报告**（避免被质疑挑设置）：

- `ROLLOUT_MAX_RESPONSE_LEN=100`（run.py 默认，权重同步占比高，对应高频同步/异步 RL 的场景）；
- `ROLLOUT_MAX_RESPONSE_LEN=4096`（典型 reasoning RL，同步占比被稀释）。

汇总：

```bash
python summarize_lzip_e2e.py --skip 2 \
  baseline=/root/results/e2e_qwen3_8b_uccl_none.log lzip=/root/results/e2e_qwen3_8b_uccl_split_only.log \
  mooncake=/root/results/e2e_qwen3_8b_mooncake.log
```

输出 `perf/update_weights_implementation_time`（权重更新）、`perf/rollout_time`、`perf/actor_train_time`、`perf/step_time` 的均值、加速比，以及权重更新在 step 中的占比。若某个 miles 版本不输出 `perf/step_time`，就用 `date` 记录整个 job 的墙钟时间除以 `NUM_ROLLOUT`。

确认压缩真的生效：rollout 日志里有 `Registered uccl transfer engine info`；用 `UCCL_P2P_LOG_LEVEL=INFO UCCL_DEBUG_SUBSYS=RDMA` 跑一步能看到 `Encode done ... ratio: 0.6x`。

## 6. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `publishes its weights for mooncake, but --p2p-transport is uccl` | rollout 没加 `--sglang-remote-instance-weight-loader-start-seed-via-uccl`；用 `run.py --p2p-transport uccl` 会自动加 |
| `UCCL could not connect` | RoCE 设 `UCCL_P2P_RDMA_GID_INDEX=3`；检查 `ibv_devinfo` 端口 ACTIVE；防火墙 |
| `decompression ... did not finish in time` | 接收端没设 `UCCL_P2P_COMPRESS_STRATEGY=split_only`（两端必须一致），或 `--p2p-transfer-timeout` 太小 |
| SGLang OOM | 每进程多占约 5×`UCCL_P2P_COMPRESS_BUFFER_BYTES` 显存；调低 `sglang_mem_fraction` 或 buffer |
| 压缩比例很低 / 没提速 | 权重分片 < 16 MiB 或 > compress buffer；看 `benchmark_weight_sync` 的 `compressed_path_fraction` |
| 同机测试 | `UcclTransport` 总走 RDMA（NIC loopback），不会走 IPC，压缩路径仍然生效 |

## 7. 本地已做/未做的验证

- 已做：UCCL 改动用 g++ `-fsyntax-only` 编译通过（含/不含 `USE_DIETGPU`）；miles `UcclTransport` 用 stub `uccl.p2p` 做了单元测试（压缩/非压缩路由、按槽缓存注册、栅栏、失败路径）；sglang/miles Python 编译通过；日志汇总脚本用合成日志测过。
- **未做**（开发机没有 torch/CUDA toolkit/docker 权限，且磁盘几乎满）：真实 RDMA 运行、DietGPU 编译、miles 全流程。第 3 节冒烟测试是第一道关。
