# Git 管理手册

## 1. 远程仓库

当前远程仓库：

```bash
git remote -v
```

本仓库当前 `origin`：

```bash
origin  https://github.com/ZhangJL16/FQLbaseline.git
```

## 2. 第一次拉取仓库

```bash
git clone https://github.com/ZhangJL16/FQLbaseline.git && cd FQLbaseline
```

## 3. 每次开始工作前

先看状态：

```bash
git status --short --branch
```

拉远程最新提交但不改本地文件：

```bash
git fetch origin
```

把远程 `main` 同步到本地当前分支，推荐用 rebase：

```bash
git pull --rebase origin main
```

## 4. 日常开发推荐流程

### 4.1 开始前同步

```bash
git checkout main
```

```bash
git pull --rebase origin main
```

### 4.2 修改完先检查

```bash
git status
```

查看改了什么：

```bash
git diff
```

查看准备提交什么：

```bash
git diff --cached
```

### 4.3 提交并上传

添加指定文件：

```bash
git add 文件1 文件2 文件3
```

或者添加当前目录下所有改动：

```bash
git add .
```

提交：

```bash
git commit -m "写清楚这次修改做了什么"
```

上传到远程：

```bash
git push origin main
```

## 5. 在另一台机器上同步最新代码

进入仓库：

```bash
cd FQLbaseline
```

拉远程更新：

```bash
git pull --rebase origin main
```

如果只是想看远程更新了什么：

```bash
git fetch origin
```

```bash
git log --oneline HEAD..origin/main
```

## 6. 本地和远程都改了时怎么同步

推荐顺序：

```bash
git status
```

```bash
git add .
```

```bash
git commit -m "保存本地进行中的修改"
```

```bash
git pull --rebase origin main
```

```bash
git push origin main
```

## 7. 常用查看命令

看最近提交：

```bash
git log --oneline -n 10
```

看某个文件的修改：

```bash
git log --oneline -- 文件路径
```

看当前分支：

```bash
git branch --show-current
```

看远程分支：

```bash
git branch -r
```

## 8. 遇到冲突时

先看哪些文件冲突：

```bash
git status
```

解决冲突后重新加入暂存区：

```bash
git add 冲突文件
```

如果你是在 `pull --rebase` 过程中：

```bash
git rebase --continue
```

如果不想继续这次 rebase：

```bash
git rebase --abort
```

## 9. 适合这个项目的最简工作流

机器 A 开发并上传：

```bash
git checkout main
```

```bash
git pull --rebase origin main
```

```bash
git add .
```

```bash
git commit -m "update training code and deployment docs"
```

```bash
git push origin main
```

机器 B 拉取并运行：

```bash
git clone https://github.com/ZhangJL16/FQLbaseline.git && cd FQLbaseline
```

```bash
git pull --rebase origin main
```

```bash
source .venv/bin/activate
```

```bash
uv run python main.py seed=0 agent=scsgfp env_name=humanoidmaze-medium-navigate-singletask-v0
```
