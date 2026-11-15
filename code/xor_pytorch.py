"""
异或（XOR）问题的 2—2—1 多层感知机 —— PyTorch 版
对应课程讲义第二篇 · 第 15 节「多层网络怎样学习参数」

结构：2 个输入 -> 2 个 sigmoid 隐藏单元 -> 1 个输出
损失：二元交叉熵（带 logits）；优化器：Adam，学习率 0.03；整批训练 300 轮

运行：
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python xor_pytorch.py

脚本会把逐轮损失与最终参数写成 JSON，供网站上的「真跑记录」对比用。
"""

import json
import os

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# ① 张量：x 有 4 行样本、每行 2 个输入 x1、x2
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
# target 每行 1 个正确答案：00→0，01→1，10→1，11→0
target = torch.tensor([[0.], [1.], [1.], [0.]])

# ② 模型参数：Parameter 表示这些数可由训练更新
W = torch.nn.Parameter(torch.tensor([[2., -2.], [-2., 2.]]))   # 2 输入 → 2 隐藏
b_hidden = torch.nn.Parameter(torch.tensor([-1., -1.]))
v = torch.nn.Parameter(torch.tensor([[3.], [3.]]))              # 2 隐藏 → 1 输出
b_output = torch.nn.Parameter(torch.tensor([-2.5]))
parameters = [W, b_hidden, v, b_output]

optimizer = torch.optim.Adam(parameters, lr=0.03)

history = []

for epoch in range(1, 301):
    optimizer.zero_grad(set_to_none=True)            # 清旧梯度
    hidden = torch.sigmoid(x @ W + b_hidden)         # 2 个隐藏值
    logits = hidden @ v + b_output                   # 1 个原始输出分数
    # ③ 损失：比较原始分数 logits 与正确答案 target
    loss = F.binary_cross_entropy_with_logits(logits, target)
    # ④ 自动求梯度：把各参数的梯度写入 .grad
    loss.backward()
    # ⑤ 更新参数：Adam 读取梯度并改变参数
    optimizer.step()

    history.append(round(float(loss.item()), 8))
    if epoch in (1, 2, 5, 10, 50, 100, 200, 300):
        with torch.no_grad():
            h = torch.sigmoid(x @ W + b_hidden)
            p = torch.sigmoid(h @ v + b_output).flatten().tolist()
        print(f"epoch {epoch:>3}  loss {loss.item():.6f}  "
              + "  ".join(f"p{i}={v_:.4f}" for i, v_ in enumerate(p)))

with torch.no_grad():
    hidden = torch.sigmoid(x @ W + b_hidden)
    prob = torch.sigmoid(hidden @ v + b_output)      # 0～1 概率

print("\n四个样本的输出概率：", [round(p, 4) for p in prob.flatten().tolist()])
print("阈值 0.5 后的预测：  ", (prob >= 0.5).int().flatten().tolist())
print("正确答案：            ", target.int().flatten().tolist())
print("最终隐藏层激活 h：    ", [[round(v_, 4) for v_ in row] for row in hidden.tolist()])
print("最终损失：            ", round(history[-1], 8))

result = {
    "framework": "PyTorch",
    "version": torch.__version__,
    "epochs": 300,
    "lr": 0.03,
    "loss": history,
    "final_probs": [round(p, 6) for p in prob.flatten().tolist()],
    "final_pred": (prob >= 0.5).int().flatten().tolist(),
    "final_params": {
        "W": [[round(v_, 6) for v_ in row] for row in W.tolist()],
        "b_hidden": [round(v_, 6) for v_ in b_hidden.tolist()],
        "v": [round(v_[0], 6) for v_ in v.tolist()],
        "b_output": round(float(b_output.item()), 6),
    },
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "run_pytorch.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
print("\n已写入", os.path.normpath(out))
