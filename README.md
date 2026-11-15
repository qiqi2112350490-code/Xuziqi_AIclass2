# 课业墨存 · 徐子祺

人工智能概论课程作业站点（GitHub Pages）。两卷作业，可在页面顶部随时切换。

| 卷 | 内容 | 页面 |
| --- | --- | --- |
| 卷壹 | 四种 AI Agent（TRAE / Claude / Codex / WorkBuddy）使用手记 | `homework1.html` |
| 卷贰 | 第二篇第 15 节 · MLP 学习参数的实现与可视化（异或 · 2—2—1） | `homework2.html` |

首页 `index.html` 为两卷索引。

## 目录

```
index.html            首页 · 两卷索引
homework1.html        卷壹 · Agent 使用手记
homework2.html        卷贰 · MLP 交互实验
assets/               共享样式与四张截图
code/xor_pytorch.py   异或 2—2—1 网络 · PyTorch 版
code/xor_tensorflow.py 异或 2—2—1 网络 · TensorFlow 版
data/run_*.json       两个脚本实跑后的逐轮损失（供网页对比）
```

## 运行两个 Python 脚本

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow

python code/xor_pytorch.py
python code/xor_tensorflow.py
```

两个脚本会分别在 `data/` 下写入 `run_pytorch.json` 与 `run_tensorflow.json`；
网页「真跑记录」一节读取这两个文件，把真实框架的损失曲线叠加到训练台上。

结构与超参数取自课程讲义第 15 节：2 输入 → 2 个 sigmoid 隐藏单元 → 1 输出，
四个 XOR 样本整批训练，二元交叉熵 + Adam（学习率 0.03），训练 300 轮，输出以 0.5 为阈值。

## 技术说明

- 纯静态：无框架、无构建步骤，CSS/JS 全部内联或本地，断网可用。
- 卷贰的训练台是**浏览器内的真实训练**（原生 JavaScript 实现前向、反向与 Adam），
  不是录像；它与两个 Python 脚本使用同一套数学，因此损失曲线几乎重合。
- 字体通过 Google Fonts 加载（马善政毛笔楷书 + 思源宋体），加载失败时回退到系统楷体 / 宋体。

## 部署

仓库开启 GitHub Pages（分支 `main`，根目录），推送后即更新。
