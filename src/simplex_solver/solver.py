import numpy as np
import sympy as sp



## 基于revised simplex method
# 说明一下具体步骤，求解时, 第一步要计算 C_B B^{-1} = p
# 然后计算非基变量的检验数cj = c_j - p^T A_j
# 根据brand 规则选取下标最小的负数入基
# 然后 需要算出那个选择的入基变量所在列变成了 B^{-1}A_j  
# 然后 用基变量除ui 选出最小的 如果有多个最小，依旧选取下标最小的出基.\
# 更新基变量
# 算出新的B^{-1}


# 定义常数M 
M = 1000000
class SimplexSolver:
    def __init__(self, c, Aeq, beq, A, b):
        """
        构造函数，用于初始化线性规划问题。
        假设问题是标准型:
        max z = c'x
        s.t. Ax <= b
             x >= 0
        
        :param obj_coeffs: 目标函数的系数 c (一个列表或NumPy数组)
        :param constraint_coeffs: 约束矩阵 A (一个二维列表或NumPy数组)
        :param rhs_vals: 约束右侧的值 b (一个列表或NumPy数组)
        """
        self.c = np.array(c)
        self.Aeq = np.array(Aeq)
        self.beq = np.array(beq)
        self.A = np.array(A)
        self.b = np.array(b)
        print("单纯形法求解器已创建。")

    def convert_to_standard_form(self):
        A_merge = np.vstack((self.A, self.Aeq))
        b_merge = np.hstack((self.b, self.beq))
        newA = np.hstack((A_merge, np.zeros((A_merge.shape[0], self.A.shape[0]))))
        negative_b_indices = np.where(self.b < 0)[0]
        for i in range(self.A.shape[0]):
            if i in negative_b_indices:
                newA[i, :] *= -1
                b_merge[i] *= -1
                newA[i, self.A.shape[1] + i] = -1
                # self.Mfor_leq_where[i] = 1 #这一行需要M变量

            else:
                newA[i, self.A.shape[1] + i] = 1
        # 对等于的
        # 标准形式暂时不用管
        self.A_merge = newA
        self.c_merge = np.hstack((self.c, np.zeros(self.A.shape[0])))
        print("转换为标准形式完成。")
        self.A_merge, self.b_merge = self.get_fullrank_matrix(self.A_merge, b_merge)
        print(self.A_merge)
        print(self.b_merge)




    def get_fullrank_matrix(self, A, b):
        """
        """
        A_T = sp.Matrix(A).T
        rref_matrix_T, pivot_columns = A_T.rref()
        independent_row_indices = sorted(list(pivot_columns))
        non_redundant_A = A[independent_row_indices, :] 
        non_redundant_b = b[independent_row_indices] 
        return non_redundant_A, non_redundant_b
        
    def Initiate_by_bigM_method(self):
        for i in range(self.A_merge.shape[0]):
            if(self.A_merge[i, self.A.shape[1] : self.A_merge.shape[1] - 1].any() != 0):
                if(self.A_merge[i, self.A.shape[1] + i] == -1):
                    new_column = np.zeros(self.A_merge.shape[0])
                    new_column[i] = 1
                    self.A_merge = np.hstack((self.A_merge, new_column.reshape(-1, 1)))
                    self.c_merge = np.hstack((self.c_merge, np.array([M])))
            elif(self.A_merge[i, self.A.shape[1]] == 0):
                new_column = np.zeros(self.A_merge.shape[0])
                new_column[i] = 1
                self.A_merge = np.hstack((self.A_merge, new_column.reshape(-1, 1)))
                self.c_merge = np.hstack((self.c_merge, np.array([M])))
                
                self.A_M = self.A_merge
        print("初始化可行基解完成")
        print(self.A_M)
        print(self.b_merge)
        print(self.c_merge)
                