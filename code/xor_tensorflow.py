"""
异或（XOR）问题的 2—2—1 多层感知机 —— TensorFlow 版
对应课程讲义第二篇 · 第 15 节「多层网络怎样学习参数」

与 PyTorch 版同一套结构、同一组初值、同一个优化器与学习率，
差别只在框架写法：GradientTape 记录运算图 → tape.gradient 求梯度 → apply_gradients 更新。

运行：
    pip install tensorflow
    python xor_tensorflow.py

脚本会把逐轮损失与最终参数写成 JSON，供网站上的「真跑记录」对比用。
"""

import json
import os

import tensorflow as tf

tf.keras.backend.set_floatx("float64")
tf.random.set_seed(0)

# ① 张量：x 有 4 行样本、每行 2 个输入 x1、x2
x = tf.constant([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
# target 每行 1 个正确答案：00→0，01→1，10→1，11→0
target = tf.constant([[0.], [1.], [1.], [0.]])

# ② 模型参数：W、v 是连接权重；b 是偏置
W = tf.Variable([[2., -2.], [-2., 2.]])   # 2 输入 → 2 隐藏
b_hidden = tf.Variable([-1., -1.])
v = tf.Variable([[3.], [3.]])             # 2 隐藏 → 1 输出
b_output = tf.Variable([-2.5])
parameters = [W, b_hidden, v, b_output]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)

history = []

for epoch in range(1, 301):
    with tf.GradientTape() as tape:
        hidden = tf.sigmoid(tf.matmul(x, W) + b_hidden)      # 2 个隐藏值
        logits = tf.matmul(hidden, v) + b_output             # 1 个原始输出分数
        # ③ 损失：比较原始分数 logits 与正确答案 target
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits))
    # ④ 自动求梯度：每个参数稍变会怎样影响损失？
    gradients = tape.gradient(loss, parameters)
    # ⑤ 更新参数：Adam 按梯度改变 W、b、v
    optimizer.apply_gradients(zip(gradients, parameters))

    history.append(round(float(loss.numpy()), 8))
    if epoch in (1, 2, 5, 10, 50, 100, 200, 300):
        print(f"epoch {epoch:>3}  loss {float(loss.numpy()):.6f}  "
              + "  ".join(f"p{i}={float(p):.4f}" for i, p in
                          enumerate(tf.sigmoid(tf.matmul(tf.sigmoid(tf.matmul(x, W) + b_hidden), v) + b_output).numpy().ravel())))

hidden = tf.sigmoid(tf.matmul(x, W) + b_hidden)
prob = tf.sigmoid(tf.matmul(hidden, v) + b_output)           # 0～1 概率

print("\n四个样本的输出概率：", [round(float(p), 4) for p in prob.numpy().ravel()])
print("阈值 0.5 后的预测：  ", tf.cast(prob >= 0.5, tf.int32).numpy().ravel().tolist())
print("正确答案：            ", target.numpy().ravel().astype(int).tolist())
print("最终隐藏层激活 h：    ", [[round(float(v_), 4) for v_ in row] for row in hidden.numpy().tolist()])
print("最终损失：            ", round(history[-1], 8))

result = {
    "framework": "TensorFlow",
    "version": tf.__version__,
    "epochs": 300,
    "lr": 0.03,
    "loss": history,
    "final_probs": [round(float(p), 6) for p in prob.numpy().ravel()],
    "final_pred": tf.cast(prob >= 0.5, tf.int32).numpy().ravel().tolist(),
    "final_params": {
        "W": [[round(float(v_), 6) for v_ in row] for row in W.numpy().tolist()],
        "b_hidden": [round(float(v_), 6) for v_ in b_hidden.numpy().tolist()],
        "v": [round(float(v_), 6) for v_ in v.numpy().ravel()],
        "b_output": round(float(b_output.numpy()[0]), 6),
    },
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "run_tensorflow.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
print("\n已写入", os.path.normpath(out))
