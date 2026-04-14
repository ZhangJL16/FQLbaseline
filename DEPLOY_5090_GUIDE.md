# FQLbaseline 5090 部署与运行手册

## 1. 当前仓库能不能直接部署到另一台 5090 机器

可以，但建议按下面前提执行：

- 推荐系统：`Ubuntu 22.04/24.04 x86_64`
- Python：`3.12`
- 显卡：`RTX 5090`
- JAX GPU 安装建议：`jax[cuda13]`
- NVIDIA Driver：建议 `>= 580`
- 不建议直接用原生 Windows 跑 JAX GPU；如果是 Windows，建议用 `WSL2 + Ubuntu`

当前仓库用于 `OGBench` 和 `Minari` 是可以直接部署的。我另外顺手修了 `D4RL` 分支里一个 `env` 初始化顺序问题，避免以后切回老 `D4RL` 环境时报错。

## 2. 远程仓库地址

```bash
git clone https://github.com/ZhangJL16/FQLbaseline.git
```

## 3. 新机器从零部署

### 3.1 克隆仓库

```bash
git clone https://github.com/ZhangJL16/FQLbaseline.git && cd FQLbaseline
```

### 3.2 安装 uv

官方安装命令：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
```

检查是否安装成功：

```bash
uv --version
```

### 3.3 创建虚拟环境

```bash
uv venv --python 3.12 .venv
```

```bash
source .venv/bin/activate
```

### 3.4 安装项目依赖

```bash
uv pip install --upgrade pip
```

```bash
uv pip install -r requirements.txt
```

### 3.5 安装 JAX GPU 版本

5090 建议优先用 CUDA 13 的 JAX wheel：

```bash
uv pip install --upgrade "jax[cuda13]"
```

如果目标机器驱动还不满足 CUDA 13，可以退回：

```bash
uv pip install --upgrade "jax[cuda12]"
```

### 3.6 验证 JAX 是否识别到 5090

```bash
uv run python -c "import jax; print(jax.devices())"
```

如果输出里能看到 `gpu` 设备，说明 JAX GPU 安装成功。

## 4. wandb 配置

本仓库默认会记录到 W&B，默认配置在 [config/main.yaml](/mnt/d/Projects/RL/FQLbaseline/config/main.yaml:1)：

- `wandb.project=gfp`
- `wandb.group=Default` 或 `Final`
- `log_on_wandb=true`

### 4.1 交互式登录

```bash
uv run wandb login --verify=True
```

### 4.2 用环境变量登录

```bash
export WANDB_API_KEY="你的_wandb_api_key"
```

### 4.3 在线/离线模式

在线同步：

```bash
export WANDB_MODE=online
```

离线记录，之后再同步：

```bash
export WANDB_MODE=offline
```

### 4.4 常用覆盖写法

把实验发到你自己的项目和分组：

```bash
WANDB_MODE=online uv run python main.py seed=0 agent=scsgfp env_name=humanoidmaze-medium-navigate-singletask-v0 wandb.project=fqlbaseline wandb.entity=你的账号或团队 wandb.group=scsgfp_vs_gfp
```

## 5. 数据集下载策略

核心原则：不要全量下载。哪个任务要跑，就只下载哪个数据集。

### 5.1 OGBench

`OGBench` 不走 [download_minari_datasets.py](/mnt/d/Projects/RL/FQLbaseline/download_minari_datasets.py:1)。

这套仓库里 **OGBench 没有单独的数据集下载脚本**，也不需要像 `Minari` 那样先执行 `download_minari_datasets.py`。  
也就是说，**OGBench 的“下载数据集”步骤就是：安装好依赖后，直接运行训练命令**。

建议按下面顺序做一次最小确认：

```bash
cd FQLbaseline
source .venv/bin/activate
```

先确认 JAX GPU 正常：

```bash
uv run python -c "import jax; print(jax.devices())"
```

再确认 agent 注册正常：

```bash
uv run python -c "from agents import agents; print(sorted(agents.keys()))"
```

最后做一个 OGBench 小步数 smoke test，这一步会触发环境和数据准备流程：

```bash
WANDB_MODE=offline uv run python main.py seed=0 agent=gfp env_name=humanoidmaze-medium-navigate-singletask-v0 offline_steps=1000 eval_interval=500 log_interval=500
```

如果上面这条可以正常开始训练，就说明当前机器上的 OGBench 路径已经打通，不需要再额外“下载 OGBench 数据集”。

常用 OGBench 单任务命令：

```bash
WANDB_MODE=online uv run python main.py seed=0 agent=fql env_name=humanoidmaze-medium-navigate-singletask-v0
WANDB_MODE=online uv run python main.py seed=0 agent=scsfql env_name=humanoidmaze-medium-navigate-singletask-v0
WANDB_MODE=online uv run python main.py seed=0 agent=gfp env_name=humanoidmaze-medium-navigate-singletask-v0
WANDB_MODE=online uv run python main.py seed=0 agent=scsgfp env_name=humanoidmaze-medium-navigate-singletask-v0
```

### 5.2 Minari 按需下载

查看当前脚本支持的 Minari 数据集：

```bash
uv run python download_minari_datasets.py --list
```

下载单个数据集的通用写法：

```bash
uv run python download_minari_datasets.py --datasets "数据集ID"
```

下面是这个仓库里已经支持的 Minari 数据集，一行一个命令。

#### Adroit

```bash
uv run python download_minari_datasets.py --datasets "D4RL/door/cloned-v2"
uv run python download_minari_datasets.py --datasets "D4RL/door/expert-v2"
uv run python download_minari_datasets.py --datasets "D4RL/door/human-v2"
uv run python download_minari_datasets.py --datasets "D4RL/hammer/cloned-v2"
uv run python download_minari_datasets.py --datasets "D4RL/hammer/expert-v2"
uv run python download_minari_datasets.py --datasets "D4RL/hammer/human-v2"
uv run python download_minari_datasets.py --datasets "D4RL/pen/cloned-v2"
uv run python download_minari_datasets.py --datasets "D4RL/pen/expert-v2"
uv run python download_minari_datasets.py --datasets "D4RL/pen/human-v2"
uv run python download_minari_datasets.py --datasets "D4RL/relocate/cloned-v2"
uv run python download_minari_datasets.py --datasets "D4RL/relocate/expert-v2"
uv run python download_minari_datasets.py --datasets "D4RL/relocate/human-v2"
```

#### MuJoCo

```bash
uv run python download_minari_datasets.py --datasets "mujoco/halfcheetah/expert-v0"
uv run python download_minari_datasets.py --datasets "mujoco/halfcheetah/medium-v0"
uv run python download_minari_datasets.py --datasets "mujoco/halfcheetah/simple-v0"
uv run python download_minari_datasets.py --datasets "mujoco/hopper/expert-v0"
uv run python download_minari_datasets.py --datasets "mujoco/hopper/medium-v0"
uv run python download_minari_datasets.py --datasets "mujoco/hopper/simple-v0"
uv run python download_minari_datasets.py --datasets "mujoco/walker2d/expert-v0"
uv run python download_minari_datasets.py --datasets "mujoco/walker2d/medium-v0"
uv run python download_minari_datasets.py --datasets "mujoco/walker2d/simple-v0"
```

### 5.3 只跑一个 Minari 任务时的完整示例

例如只跑 `D4RL/pen/expert-v2`：

```bash
uv run python download_minari_datasets.py --datasets "D4RL/pen/expert-v2"
```

```bash
WANDB_MODE=online uv run python main.py seed=0 agent=gfp env_name='D4RL/pen/expert-v2'
```

### 5.4 D4RL / Minari 常用下载命令速查

如果你只打算跑单个任务，直接下载对应数据集即可。

Adroit:

```bash
uv run python download_minari_datasets.py --datasets "D4RL/pen/expert-v2"
uv run python download_minari_datasets.py --datasets "D4RL/pen/cloned-v2"
uv run python download_minari_datasets.py --datasets "D4RL/pen/human-v2"
uv run python download_minari_datasets.py --datasets "D4RL/door/expert-v2"
uv run python download_minari_datasets.py --datasets "D4RL/hammer/expert-v2"
uv run python download_minari_datasets.py --datasets "D4RL/relocate/expert-v2"
```

MuJoCo:

```bash
uv run python download_minari_datasets.py --datasets "mujoco/halfcheetah/medium-v0"
uv run python download_minari_datasets.py --datasets "mujoco/hopper/medium-v0"
uv run python download_minari_datasets.py --datasets "mujoco/walker2d/medium-v0"
```

## 6. 常用训练命令

### 6.1 单任务单种算法

```bash
WANDB_MODE=online uv run python main.py seed=0 agent=gfp env_name=humanoidmaze-medium-navigate-singletask-v0
```

```bash
WANDB_MODE=online uv run python main.py seed=0 agent=scsgfp env_name=humanoidmaze-medium-navigate-singletask-v0
```

### 6.1.1 D4RL / Minari 单任务单种算法

下面这些命令都默认走仓库里的任务级超参恢复，不需要手工把 `alpha/q_agg/discount` 一项项补齐。

FQL:

```bash
WANDB_MODE=online uv run python main.py seed=0 agent=fql env_name='D4RL/pen/expert-v2'
WANDB_MODE=online uv run python main.py seed=0 agent=fql env_name='mujoco/halfcheetah/medium-v0'
```

SCSFQL:

```bash
WANDB_MODE=online uv run python main.py seed=0 agent=scsfql env_name='D4RL/pen/expert-v2'
WANDB_MODE=online uv run python main.py seed=0 agent=scsfql env_name='mujoco/halfcheetah/medium-v0'
```

GFP:

```bash
WANDB_MODE=online uv run python main.py seed=0 agent=gfp env_name='D4RL/pen/expert-v2'
WANDB_MODE=online uv run python main.py seed=0 agent=gfp env_name='mujoco/halfcheetah/medium-v0'
```

SCSGFP:

```bash
WANDB_MODE=online uv run python main.py seed=0 agent=scsgfp env_name='D4RL/pen/expert-v2'
WANDB_MODE=online uv run python main.py seed=0 agent=scsgfp env_name='mujoco/halfcheetah/medium-v0'
```

### 6.1.2 D4RL / Minari 多 seed 对比模板

对比 `gfp` 和 `scsgfp`：

```bash
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_d4rl env_name='D4RL/pen/expert-v2'
```

对比 `fql` 和 `scsfql`：

```bash
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=fql,scsfql wandb.group=scsfql_vs_fql_d4rl env_name='D4RL/pen/expert-v2'
```

MuJoCo 任务同理：

```bash
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_mujoco env_name='mujoco/halfcheetah/medium-v0'
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=fql,scsfql wandb.group=scsfql_vs_fql_mujoco env_name='mujoco/halfcheetah/medium-v0'
```

### 6.2 对比 scsgfp 和 gfp

最省事的方式是用 Hydra multirun，一条命令同时扫 `seed=0,1,2` 和 `agent=gfp,scsgfp`。

通用模板：

```bash
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp env_name=你的环境名
```

下面是 10 个 OGBench 环境的一行命令版本。

```bash
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_10env env_name=humanoidmaze-medium-navigate-singletask-v0
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_10env env_name=humanoidmaze-large-navigate-singletask-v0
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_10env env_name=antsoccer-arena-navigate-singletask-v0
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_10env env_name=antmaze-large-navigate-singletask-v0
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_10env env_name=antmaze-giant-navigate-singletask-v0
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_10env env_name=cube-single-play-singletask-v0
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_10env env_name=cube-double-play-singletask-v0
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_10env env_name=scene-play-singletask-v0
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_10env env_name=puzzle-3x3-play-singletask-v0
WANDB_MODE=online uv run python main.py -m seed=0,1,2 agent=gfp,scsgfp wandb.group=scsgfp_vs_gfp_10env env_name=puzzle-4x4-play-singletask-v0
```

如果你更喜欢直接跑脚本，仓库里也准备了可迁移脚本：

```bash
bash run_scsgfp_vs_gfp_10env_3seed.sh
```

## 7. 结果导出

对比任务结束后，可以把 `scsgfp vs gfp` 和 `scsfql vs fql` 导出到一个工作簿：

```bash
uv run python export_run_comparisons.py --output results/comparisons/scs_method_comparisons.xlsx
```

## 8. 部署后建议的最小检查流程

```bash
uv --version
```

```bash
uv run python -c "import jax; print(jax.devices())"
```

```bash
uv run python -c "from agents import agents; print(sorted(agents.keys()))"
```

```bash
WANDB_MODE=offline uv run python main.py seed=0 agent=gfp env_name=humanoidmaze-medium-navigate-singletask-v0 offline_steps=1000 eval_interval=500 log_interval=500
```

## 9. 参考

- uv 官方安装文档：https://docs.astral.sh/uv/getting-started/installation/
- JAX 官方安装文档：https://docs.jax.dev/en/latest/installation.html
- W&B 登录文档：https://docs.wandb.ai/ref/cli/wandb-login/
