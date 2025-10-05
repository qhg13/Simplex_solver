from simplex_solver.solver import SimplexSolver
import numpy as np

# 示例问题
# c = [1, 1, 1, 1, 1]  # 目标函数系数
# A = [[1, 1, 1, 1, 1], [1, 2, 3, 4, 1], [1, 3, 4, 6, 1]]   # 约束矩阵
# Aeq  = [[1, 1, 1, -1, 1], [2, 2, 2, -2, 2]]
# b = [-1, -7, 6]
# beq = [-1, 7]



c = [1, 1, -3]  # 目标函数系数
A = [[1, -2,1], [-2, -1, 4]]   # 约束矩阵
Aeq  = [[1, 0 ,-2]]
b = [11, -3]
beq = [1]

# c = [-10, -12,-12]
# A = [[1, 2, 2], [2, 1, 2], [2, 2, 1]]
# b = [20, 20, 20]



Solver = SimplexSolver(c, A, b, Aeq, beq)
# Solver.get_fullrank_matrix(Solver.A)
Solver.convert_to_standard_form()
Solver.Initiate_by_bigM_method()
result = Solver.solve()
print(result)

