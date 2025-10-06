import numpy as np
import sympy as sp



## 基于revised simplex method 相比单纯形表一般更高效
# 说明一下具体步骤，求解时, 第一步要计算 C_B B^{-1} = p
# 然后计算非基变量的检验数cj = c_j - p^T A_j
# 根据brand 规则选取下标最小的负数入基
# 然后 需要算出那个选择的入基变量所在列变成了 B^{-1}A_j  
# 然后 用基变量除ui 选出最小的 如果有多个最小，依旧选取下标最小的出基.\
# 更新基变量
# 算出新的B^{-1} reinversion 每10次


# 定义常数M 
M = 1000000

from enum import Enum, auto

class SolverStatus(Enum):
    """
    定义线性规划求解器的所有可能最终状态。
    
    使用 auto() 可以让枚举成员自动分配一个唯一的值，
    我们只关心成员的名字，不关心它具体等于几。
    """
    OPTIMAL = auto()        # 找到最优解
    UNBOUNDED = auto()      # 问题是无界的
    INFEASIBLE = auto()     # 问题无可行解
    NOT_SOLVED = auto()     # 尚未求解
    MAX_ITERATIONS = auto() # 达到最大迭代次数




class SimplexSolver:
    def __init__(self, c, A, b ,Aeq = None, beq = None):
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
        if(Aeq is not None):
            self.Aeq = np.array(Aeq)
        else:
            self.Aeq = None
        if(beq is not None):
            self.beq = np.array(beq)
        else:
            self.beq = None
        self.A = np.array(A)
        self.b = np.array(b)
        print("单纯形法求解器已创建。")

    def convert_to_standard_form(self):
        # 合并等式和不等式约束
        if(self.Aeq is not None and self.beq is not None):
            A_merge = np.vstack((self.A, self.Aeq))
            b_merge = np.hstack((self.b, self.beq))
        else:
            A_merge = self.A
            b_merge = self.b
        # 引入松弛变量化为标准形式， 保证b为正
        A_merge = np.hstack((A_merge, np.zeros((A_merge.shape[0], self.A.shape[0]))))
        negative_b_indices = np.where(self.b < 0)[0]
        for i in range(self.A.shape[0]):
            if i in negative_b_indices:
                A_merge[i, :] *= -1
                b_merge[i] *= -1
                A_merge[i, self.A.shape[1] + i] = -1
            else:
                A_merge[i, self.A.shape[1] + i] = 1
        # 对等于的
        negative_b_indices_eq = np.where(self.beq < 0)[0]
        for i in range(self.Aeq.shape[0]):
            if i in negative_b_indices_eq:
                A_merge[i + self.A.shape[0], :] *= -1
                b_merge[i + self.A.shape[0]] *= -1

        self.A_merge = A_merge
        self.c_merge = np.hstack((self.c, np.zeros(self.A.shape[0])))
        print("转换为标准形式完成。")
        self.A_merge, self.b_merge = self.get_fullrank_matrix(self.A_merge, b_merge)
        print(self.A_merge)
        print(self.b_merge)




    def get_fullrank_matrix(self, A, b):
        """
        去除多余的行
        """
        A_T = sp.Matrix(A).T
        rref_matrix_T, pivot_columns = A_T.rref()
        independent_row_indices = sorted(list(pivot_columns))
        non_redundant_A = A[independent_row_indices, :] 
        non_redundant_b = b[independent_row_indices] 
        return non_redundant_A, non_redundant_b
        
    def Initiate_by_bigM_method(self):
        """
        初始化可行基解， 利用大M法
        """
        self.basis_record = []
        self.M_var = []
        for i in range(self.A_merge.shape[0]):
            # 不等式约束引入人工变量
            if(self.A_merge[i, self.A.shape[1] : self.A_merge.shape[1]].any() != 0):
                if(self.A_merge[i, self.A.shape[1] + i] == -1):
                    new_column = np.zeros(self.A_merge.shape[0])
                    new_column[i] = 1
                    self.A_merge = np.hstack((self.A_merge, new_column.reshape(-1, 1)))
                    self.c_merge = np.hstack((self.c_merge, np.array([M])))
                    self.M_var.append(self.A_merge.shape[1] - 1)
                    # 按照从小到大的顺序
                    self.basis_record.append(self.A_merge.shape[1] - 1)
                else:
                    # 标准形式已经引入过松弛变量， 不需要再引入人工变量
                    self.basis_record.append(self.A.shape[1] + i)
            # 等式约束引入人工变量
            elif(self.A_merge[i,self.A.shape[1] : self.A_merge.shape[1]].all() == 0):
                new_column = np.zeros(self.A_merge.shape[0])
                new_column[i] = 1
                self.A_merge = np.hstack((self.A_merge, new_column.reshape(-1, 1)))
                self.c_merge = np.hstack((self.c_merge, np.array([M])))
                self.M_var.append(self.A_merge.shape[1] - 1)
                self.basis_record.append(self.A_merge.shape[1] - 1)
                
        self.A_M = self.A_merge
        # 构造基矩阵
        self.B = self.A_M[:, self.basis_record]
        # 构造基变量对应的目标函数系数
        self.c_basis = self.c_merge[self.basis_record]
        self.Binv = np.linalg.inv(self.B)
        # print(self.Binv)
        self.init_x = self.Binv @ self.b_merge
        print("初始化可行基解完成")
        print(self.A_M)
        print(self.b_merge)
        print(self.c_merge)
        print(self.basis_record)
        print(self.M_var)


    def compute_cj(self):
        self.p = self.c_basis @ self.Binv 
        for i in range(self.A_M.shape[1]): # brand's rule 最小下标原则
            if(self.c_merge[i] - self.p @ self.A_M[:, i] < -1e-10): #入基
                return {"success": True, "status": SolverStatus.NOT_SOLVED, "message": "not solved yet", "index": i}
            
        # 检验数已经全部大于等于零，看看是否有M变量，如果有M变量，说明无可行解
        if(not (set(self.M_var).isdisjoint(set(self.basis_record)))):
            mask = np.isin(self.basis_record, self.M_var)
            intersection_index = np.where(mask)[0]
            if(self.init_x[intersection_index].any() != 0):
                return {"success": False, "status" : SolverStatus.INFEASIBLE, "message": "infeasible solution", "index":None} # -2意味着无可行解
            
        solution = np.zeros(self.A_M.shape[1])
        for i in range(self.A_M.shape[1]):
            if(i in self.basis_record):
                solution[i] = self.init_x[self.basis_record.index(i)]
        return {"success": True, "status" : SolverStatus.OPTIMAL, "message": "optimal solution found", "index":None, "solution": solution, "objective_value": self.init_x @ self.c_basis} # -1说明已经最优解
    
    def compute_u(self, i): #找出基
        self.u = self.Binv @ self.A_M[:, i]
        u_positive = self.u[self.u > 0]
        # 如果全为负，说明问题无界
        if(u_positive.size == 0):
            return {"success": True, "status" : SolverStatus.UNBOUNDED, "message": "unbounded", "index": None, "theta": None, "solution": None}
        else: 

            #找最小theta x / u
            ratios_where = np.where(self.u > 0, self.init_x / self.u, np.inf)
            min_ratio = np.min(ratios_where)
            all_theta_indices_where = np.where(ratios_where == min_ratio)[0]
            #  找最小的下标
            local_index = np.argmin(np.array(self.basis_record)[all_theta_indices_where])
            theta_index_where = all_theta_indices_where[local_index]
           
            # theta_index_where = np.argmin(ratios_where) #最小下标原则
            return {"success": True, "status" : SolverStatus.NOT_SOLVED, "message": "not solved yet", "index": theta_index_where, "theta": ratios_where[theta_index_where]}
            # 返回出基变量是第几个
            
    def pivot(self, theta_index_where, i, theta):
        # 入基
        self.B[:, theta_index_where] = self.A_M[:, i]
        # 系数
        self.c_basis[theta_index_where] = self.c_merge[i]
        #更新基变量的值
        self.init_x[theta_index_where] =  theta
        # init_x 遍历
        for j in range(self.init_x.shape[0]):
            if(j != theta_index_where):
                self.init_x[j] = self.init_x[j] - theta * self.u[j]

        
    def cumpute_newBinv(self, theta_index_where):
        # 计算新的基矩阵的逆
        self.Binv = np.hstack((self.Binv, self.u.reshape(-1, 1)))
        for j in range(self.Binv.shape[0]):
            if(j != theta_index_where):
                self.Binv[j, :] = self.Binv[j, :] - self.Binv[theta_index_where, :] * self.u[j] / self.u[theta_index_where]
        self.Binv[theta_index_where, :] = self.Binv[theta_index_where, :] / self.u[theta_index_where]
        #去除最后一列
        self.Binv = self.Binv[:, :-1]


    def solve(self):
        reinversion_num = 0
        while True:
            cj_result = self.compute_cj()
            if(cj_result["success"]):
                if(cj_result["status"] == SolverStatus.OPTIMAL):
                    return cj_result
                else:
                    i = cj_result["index"]
                    u_result = self.compute_u(i)
                    if(u_result["success"]):
                        if(u_result["status"] == SolverStatus.UNBOUNDED):
                            return u_result
                        else:
                            reinversion_num += 1
                            theta_index_where = u_result["index"]
                            theta = u_result["theta"]
                            self.pivot(theta_index_where, i, theta)
                            if(reinversion_num >= 15):
                                reinversion_num = 0
                                self.Binv = np.linalg.inv(self.B)
                            else:
                                self.cumpute_newBinv(theta_index_where)
                            self.basis_record[theta_index_where] = i
                        
                    else:
                        return u_result
            else:
                return cj_result

        

