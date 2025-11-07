# Simplex Solver (单纯形法求解器)

这是一个使用 Python 实现的单纯形法（Simplex Algorithm）求解器，用于解决标准形式的线性规划（Linear Programming, LP）问题。

本项目被构建为一个可安装的 Python 包，展示了现代 Python 项目的规范结构（`src` 布局）。

---

## 🚀 功能特性

* 基于 revised-simplex method 实现线性规划问题求解
* 使用大M来法进行初始化
* 使用bland's rule 避免 cycling


---

## 🔧 安装

你需要先将本项目克隆到你的本地：

```bash
git clone [https://github.com/YourUsername/simplex_solver.git](https://github.com/YourUsername/simplex_solver.git)
cd simplex_solver
```
*(请把上面的 URL 替换成你自己的 GitHub 仓库地址)*

然后，我们强烈推荐使用“可编辑模式” (`-e`) 来安装，这样你对 `src` 中源代码的任何修改都会立刻生效，无需重新安装。

```bash
# 安装本项目及其依赖
pip install -e .
```

(如果你只是想“使用”这个包，而不是“开发”它，你也可以使用 `pip install .` 来进行标准安装。)

---

## 💡 使用方法

一旦你通过 `pip install -e .` 安装了本包，你就可以在你电脑的**任何 Python 脚本**中导入并使用它，就像使用 `numpy` 一样。

我们默认的求解问题为

$$
\begin{aligned}
min \quad &z = c^T x\\
s.t.\quad
    &Ax \le b\\
    &Aeq x = beq\\
    &x \ge 0
\end{aligned}
$$

函数参数为 A, b, Aeq, beq, c. 可选参数为 A, b, Aeq, beq. 其中 A, b 为不等式约束系数矩阵和约束右端项向量，Aeq, beq 为等式约束系数矩阵和等式约束右端项向量，c 为目标函数系数向量。 函数返回值为一个字典，包含最优解、最优值、是否最优等信息。



下面是一个在 `examples/basic_usage.py` 中（或任何其他脚本中）的标准用法：

```python
import numpy as np

# 从你的包中导入你编写的求解器类
# (请确认你的类名和文件名是否是 SimplexSolver 和 solver.py)
try:
    from simplex_solver.solver import SimplexSolver
except ImportError:
    print("导入失败！")
    print("请确保你已在项目根目录运行 'pip install -e .'")
    exit()

# --- 定义一个线性规划问题 ---
# min z = -5x1 + -4x2
# s.t.
#      6x1 + 4x2 <= 24
#       x1 + 2x2 <= 6
#      -x1 +  x2 <= 1
#             x2 <= 2
#      x1, x2 >= 0
# -----------------------------

# 价值向量 c
c = np.array([-5, -4])

# 约束系数矩阵 A
A = np.array([
    [6, 4],
    [1, 2],
    [-1, 1],
    [0, 1]
])

# 约束右端项 b
b = np.array([24, 6, 1, 2])

# 1. 实例化求解器
# (这里的参数取决于你 __init__ 方法的设计)
print("正在初始化求解器...")
solver = SimplexSolver(c, A, b)

# 2. 运行求解
print("开始求解...")
result = solver.solve() # (假设你的求解方法叫 solve)

# 3. 打印结果
print(result)

```

---

## 📁 项目结构

```
.
├── src                     # 源代码包目录
│   └── simplex_solver
│       ├── __init__.py
│       └── solver.py       # 核心求解器实现
│
├── examples                # 用法示例
│   └── basic_usage.py
│
├── benchmarks              # 性能测试 
│   └── ...
│
├── .gitignore              # Git 忽略文件
├── pyproject.toml          # 项目构建和依赖配置文件
└── README.md               # 项目说明
```


