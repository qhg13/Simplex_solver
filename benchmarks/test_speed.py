import numpy as np
from numpy.random import randn
from simplex_solver.solver import SimplexSolver, SolverStatus

def main():
    n = 300
    m = n // 2
    meq = n // 4
    x_feasible = np.random.uniform(0, 20, n)
                    
    # 2. ## 升级 ## c, A, Aeq 使用 randn 生成正负数，范围更广
    c = np.random.randn(n) * 10
    A = np.random.randn(m, n) * 10 if m > 0 else np.array([[]])
    Aeq = np.random.randn(meq, n) * 10 if meq > 0 else np.array([[]])
                    
    # 3. 反向计算 b 和 beq
    b = A @ x_feasible + np.random.uniform(0, 5, m) if m > 0 else np.array([])
    beq = Aeq @ x_feasible if meq > 0 else np.array([])
    solver = SimplexSolver(c, A, b, Aeq, beq)
    result = solver.solve()
    print(result)

if __name__ == "__main__":
    main()
