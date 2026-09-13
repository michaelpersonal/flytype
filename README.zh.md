# FlyType / FlyBreak

用果蝇连接组（MaleCNS v1.0，**166,700 个神经元、2,560 万条连接**）的仿真来驱动两个任务：
打字，和打砖块。

[English README](README.md)

---

## 这是什么

一个本地渲染的 320×180 画面进入仿真的果蝇视觉系统，神经活动被一个**冻结的解码器**读出，
再变成动作。中间没有任何程序替它判断该做什么。

两个任务：

| 任务 | 命令 | 读出方式 |
| --- | --- | --- |
| 打字 | `flytype run` | DNp20 左右发放率之差 → 左 / 右 / 不动 |
| 打砖块 | `flytype web` | 视网膜光感受器群体向量 → `[-1, 1]` 连续控制 |

**结果**：球拍跟踪球的比例 **0.782**（随机为 0.5），6,947 次有效移动，
**超出随机 47 个标准差**，9 局里打中球 197 次。

---

## 怎么跑

需要 Python 3.11+、C++17 编译器、macOS 或 Linux，至少 10 GB 磁盘（建议 20 GB），
以及下载约 1.1 GB MaleCNS 数据集的网络。建议 16 GB 内存，不需要 GPU。

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m flytype prepare     # 下载并编译连接组，会先打印目标目录
python -m flytype verify
```

打砖块（实时网页，只监听 127.0.0.1）：

```sh
# 先拟合冻结的解码器（一次即可）
python -m flytype calibrate-play-decoder \
    --out calibration/retinal-50ms.json --neural-ms 50 --decoder retinal

# 启动
python -m flytype web --out runs/live --neural-ms 50 \
    --motor-decoder calibration/retinal-50ms.json
```

打字：

```sh
python -m flytype run --target 'hello world' --seed 42 --out runs/hello
python -m flytype status --out runs/hello
```

`--fixture` 用一个确定性的脚本替身代替 MaleCNS，方便离线快速测试；
它产生的每条记录都标着 `"source": "fixture"`，真实运行标着 `"source": "malecns"`。

---

## 果蝇的视觉通路

果蝇视觉系统是一条流水线，按**小眼（ommatidium）重复排列的柱状结构，约 1,770 根柱子**，
所以每种细胞几乎都正好 1,770 个：

**光感受器（photoreceptor，R1–R6 / R7 / R8）→ 视板（lamina，L1–L5、C2、C3、T1）
→ 视髓（medulla，Mi、Tm、TmY、Dm）→ T4/T5 运动检测 → 视小叶（lobula，LC、LPLC）
→ 中央脑 → 下行神经元（DN）**

本项目在**最前面那一层**读出：直接取视网膜光感受器的群体活动，
用每个光感受器**自己在屏幕上的解剖学位置**（来自连接组的视网膜拓扑）加权求重心
（**population vector**），并且只取发放最强的**前 1%** —— 球大约只盖住 40 个光感受器，
正是它们指向球。

定位精度 **r = 0.99，误差 17.5 像素**（球到球拍的距离范围是 ±284 像素）。

> **边界说明**：读的是光感受器这一层，属于**视觉层面**的神经读出 ——
> 眼睛看见了，动作跟上了，但不能说大脑在「思考」。
> 在这个仿真里，视板只剩很少信号，视髓、T4/T5、视小叶基本不放电；
> 原因与所需的模型改动记录在 [docs/validation.md](docs/validation.md)，
> `tools/probe_propagation.py` 可以复现整个诊断。

---

## 每个结论都有对照组

这是这个项目最重要的方法论约束：**任何数字都必须和「打乱标签的同一套流程」比较**。

同一套拟合流程喂随机标签也能得到 r ≈ 0.42 —— 没有对照组的话，噪声看起来会很像信号。
`calibrate-play-decoder` 每次都会同时打印：

- 留出集误差（held-out MAE）
- **打乱标签**的对照结果
- **永远预测平均值**（等于什么都没学到）的结果

跟踪率用实测的随机水平 0.5 作基准（`tests/test_breakout.py` 固定了这个定义），
并用精确二项检验，见 `tools/analyze_play.py`。

---

## 不做什么

- 不会在 X 或任何网络服务上发帖、关注、点赞或做任何操作，也不接受、不存储任何账号凭据。
- 不用大模型替神经网络做决定，不修正解码出来的控制信号，不用真实球位置覆盖它。
- 不声称这是真实的果蝇、不声称它理解语言、有意图、有痛觉或愉悦、有意识，
  也不声称已验证的学习行为。
- 强化信号（PAM11 奖励 / PPL101 厌恶）是**工程化的信号**，不是模拟的痛觉受体。

---

## 文档

- [docs/model.md](docs/model.md) — 模型与证据
- [docs/validation.md](docs/validation.md) — 所有实测数据、对照组，以及被撤回的结论
- [docs/continuous-game-web-spec.md](docs/continuous-game-web-spec.md) — 实时网页的规格

## 测试

```sh
python -m pytest -q
FLYTYPE_FULL_TEST=1 python -m pytest -q tests/test_neural_integration.py   # 需要完整连接组
```
