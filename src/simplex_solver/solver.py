import numpy as np
import sympy as sp
import math 



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
    REDUNDANT_CONSTRAINTS_REMOVED = auto() # 冗余约束已移除



class SimplexSolver:
    def __init__(self, c, A = None, b = None, Aeq = None, beq = None):
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
        self.TOLERANCE = 1e-5
         # 1. 确定决策变量的数量
        self.c = np.array(c)
        num_vars = len(self.c)

        inputs_to_process = [
            ('A', A, (0, num_vars)),    # 矩阵 A
            ('b', b, (0,)),             # 向量 b
            ('Aeq', Aeq, (0, num_vars)), # 矩阵 Aeq
            ('beq', beq, (0,))            # 向量 beq
        ]

        for attr_name, value, empty_shape in inputs_to_process:
            if value is not None:
                # 如果传入了值，就将其转换为 NumPy 数组
                processed_value = np.array(value)
            else:
                # 如果值是 None，就创建一个指定形状的空数组
                processed_value = np.empty(empty_shape)
            # 使用 setattr 动态地设置 self 的属性
            setattr(self, attr_name, processed_value)
        # print("单纯形法求解器已创建。")

    def convert_to_standard_form(self):
        # 合并等式和不等式约束
        # if(self.Aeq is not None and self.beq is not None):
        #     A_merge = np.vstack((self.A, self.Aeq))
        #     b_merge = np.hstack((self.b, self.beq))
        # else:
        #     A_merge = self.A
        #     b_merge = self.b
        A_merge = np.vstack((self.A, self.Aeq))
        b_merge = np.hstack((self.b, self.beq))
        # 引入松弛变量化为标准形式， 保证b为正
        A_merge = np.hstack((A_merge, np.zeros((A_merge.shape[0], self.A.shape[0]))))
        negative_b_indices = np.where(self.b < 0)[0]
        # print("A大小",self.A.shape)
        for i in range(self.A.shape[0]):
            if i in negative_b_indices:
                A_merge[i, :] *= -1
                b_merge[i] *= -1
                A_merge[i, self.A.shape[1] + i] = -1
            else:
                A_merge[i, self.A.shape[1] + i] = 1
        # 对等于的
        # if(self.beq is not None):
        #     negative_b_indices_eq = np.where(self.beq < 0)[0]
        #     for i in range(self.Aeq.shape[0]):
        #         if i in negative_b_indices_eq:
        #             A_merge[i + self.A.shape[0], :] *= -1
        #             b_merge[i + self.A.shape[0]] *= -1
        negative_b_indices_eq = np.where(self.beq < 0)[0]
        # print(f"Aeq: {self.Aeq}")
        # print(f"A_merge: {A_merge}")
        for i in range(self.Aeq.shape[0]):
            if i in negative_b_indices_eq:
                A_merge[i + self.A.shape[0], :] *= -1
                b_merge[i + self.A.shape[0]] *= -1

        self.A_merge = A_merge
        self.c_merge = np.hstack((self.c, np.zeros(self.A.shape[0])))
        # print("转换为标准形式完成。")
        self.A_merge, self.b_merge = self.get_fullrank_matrix(self.A_merge, b_merge)
        # print(self.A_merge)
        # print(self.b_merge)


    def normalize_by_leading_one(self, coeffs):
        """
        将数组除以其第一个非零元素，使得该元素的位置变为1。
        如果数组所有元素都为0，则不进行任何操作。
        """
        # 步骤1: 找到第一个非零元素作为除数 (The "Finding" step)
        divisor = None
        for c in coeffs:
            # 使用 np.isclose 更安全地比较浮点数
            if not np.isclose(c, 0):
                divisor = c
                break  # 找到后立即跳出循环，不再继续搜索

        if divisor is not None:
            # 只有在找到了非零除数的情况下才进行除法
            return coeffs / abs(divisor), abs(divisor)
        else:
            # 如果循环结束都没找到 (说明全是0)，直接返回原数组
            return coeffs, 1

    def presolve(self):
        """
        使用哈希查找冗余约束，同时比较符号和RHS.
        返回:
        indices_to_keep: 一个集合，包含应该保留的约束的原始索引。
        """
    
        # constraint_map 的 key 是系数元组, value 是包含详细信息的字典
        constraint_map = {}
        indices_to_remove = set()
        # if(self.Aeq is not None and self.beq is not None):
        #     A_Aeq = np.vstack((self.A, self.Aeq))
        #     b_beq = np.hstack((self.b, self.beq))
        # else:
        #     A_Aeq = self.A
        #     b_beq = self.b
        A_Aeq = np.vstack((self.A, self.Aeq))
        b_beq = np.hstack((self.b, self.beq))
        for i in range(A_Aeq.shape[0]):
            current_row_coeffs = A_Aeq[i, :]
            current_row_coeffs, divisor = self.normalize_by_leading_one(current_row_coeffs)
            if(i < self.A.shape[0]):
                current_sign = '<='
            else:
                current_sign = '=='
            current_rhs = b_beq[i] / divisor
            
            # 为了哈希，将系数行转换为元组
            key = tuple(current_row_coeffs)
            # print(f"当前行系数: {current_row_coeffs}")
            
            if key in constraint_map:
                # 找到了一个具有相同系数 (LHS) 的约束，现在需要比较 sign 和 rhs
                existing_constraint_info = constraint_map[key]
                existing_idx = existing_constraint_info['idx']
                existing_sign = existing_constraint_info['sign']
                existing_rhs = existing_constraint_info['rhs']
                
                
                if current_sign == '<=' and existing_sign == '<=':
                    if current_rhs < existing_rhs:
                        # 当前行更严格，所以之前的行是冗余的
                        indices_to_remove.add(existing_idx)
                        # 用当前更严格的约束信息更新map
                        constraint_map[key] = {'idx': i, 'sign': current_sign, 'rhs': current_rhs}
                    else:
                        # 当前行更宽松，所以当前行是冗余的
                        # print(f"  - 决策: 第 {i} 行比第 {existing_idx} 行更宽松。标记第 {i} 行为待删除。")
                        indices_to_remove.add(i)
                
                elif current_sign == '==' and existing_sign == '==':
                    if current_rhs != existing_rhs:
                        # indices_to_remove.add(existing_idx)
                        # constraint_map[key] = {'idx': i, 'sign': current_sign, 'rhs': current_rhs}
                        # return ('infeasible', f"错误: 等式约束 {i} 与等式约束 {existing_idx} 冲突。")
                        return {"success": False, "status": SolverStatus.INFEASIBLE, "message": f"错误: 等式约束 {i} 与等式约束 {existing_idx} 冲突。"}
                        
                    else: 
                        indices_to_remove.add(i)

                elif current_sign == '==' and existing_sign == '<=':
                    if current_rhs <= existing_rhs:
                        # 当前行更严格，所以之前的行是冗余的
                        indices_to_remove.add(existing_idx)
                        # 用当前更严格的约束信息更新map
                        constraint_map[key] = {'idx': i, 'sign': current_sign, 'rhs': current_rhs}
                    else:
                        error_msg = f"错误: 等式约束 {i} 与小于等于约束 {existing_idx} 冲突。"
                        # return ('infeasible', error_msg)
                        return {"success": False, "status": SolverStatus.INFEASIBLE, "message": error_msg}
                        
                
            else:
                # 第一次遇到这个系数的行，将其信息存入 map
                constraint_map[key] = {'idx': i, 'sign': current_sign, 'rhs': current_rhs}

            # 从所有索引中减去要删除的索引
        all_indices = set(range(A_Aeq.shape[0]))
        # print(f"all_indices: {all_indices}")
        # print(f"indices_to_remove: {indices_to_remove}")
        indices_to_keep = all_indices - indices_to_remove
        for i in range(self.A.shape[0]):
            if(i not in indices_to_keep):
                self.A = np.delete(self.A, i, axis=0)
                self.b = np.delete(self.b, i, axis=0)
        # if(self.Aeq is not None and self.beq is not None):
        #     for i in range(self.Aeq.shape[0]):
        #         if(i not in indices_to_keep):
        #             self.Aeq = np.delete(self.Aeq, i - self.A.shape[0], axis=0)
        #             self.beq = np.delete(self.beq, i - self.A.shape[0], axis=0)
        for i in range(self.Aeq.shape[0]):
                if(i not in indices_to_keep):
                    self.Aeq = np.delete(self.Aeq, i - self.A.shape[0], axis=0)
                    self.beq = np.delete(self.beq, i - self.A.shape[0], axis=0)
        # return {'success', 'redundant constraints removed' }
        return {"success": True, "status": SolverStatus.REDUNDANT_CONSTRAINTS_REMOVED, "message": "冗余约束已移除"}
        
    

        
            
        
        


    def get_fullrank_matrix(self, A, b):
        """
        去除多余的行
        """
        # A和b拼接
        A_b = np.hstack((A, b.reshape(-1, 1)))
        A_T = sp.Matrix(A_b).T
        rref_matrix_T, pivot_columns = A_T.rref()
        independent_row_indices = sorted(list(pivot_columns))
        non_redundant_A = A[independent_row_indices, :] 
        non_redundant_b = b[independent_row_indices] 
        return non_redundant_A, non_redundant_b
        # A = np.hstack((A, b.reshape(-1, 1)))
        # A_T = sp.Matrix(A_b).T
        # rref_matrix_T, pivot_columns = A_T.rref()
        # independent_row_indices = sorted(list(pivot_columns))
        # non_redundant_A = A[independent_row_indices, :] 
        # non_redundant_b = b[independent_row_indices] 
        # return non_redundant_A, non_redundant_b
        
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
                    flag0 = 0
                    for j in range(self.A.shape[1]):
                        if((self.A_merge[:, j] == np.eye(self.A_merge.shape[0])[i]).all()):
                            flag = 1
                            self.basis_record.append(j)
                            break
                    if(flag0 == 0):
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
            # elif(self.A_merge[i,self.A.shape[1] : self.A_merge.shape[1]].all() == 0):
            else:
                # 循环之前的到A的列是否有e_{i + 1}, 如果已经有e_{i + 1}, 则不需要引入人工变量
                flag = 0
                for j in range(self.A.shape[1]):
                    if((self.A_merge[:, j] == np.eye(self.A_merge.shape[0])[i]).all()):
                        flag = 1
                        # self.M_var.append(j) 不是人工变量
                        self.basis_record.append(j)
                        break
                if(flag == 0):
                    new_column = np.zeros(self.A_merge.shape[0])
                    new_column[i] = 1
                    self.A_merge = np.hstack((self.A_merge, new_column.reshape(-1, 1)))
                    self.c_merge = np.hstack((self.c_merge, np.array([M])))
                    self.M_var.append(self.A_merge.shape[1] - 1)
                    self.basis_record.append(self.A_merge.shape[1] - 1)
            
        self.A_M = self.A_merge
        # print(f"self.A_M: {self.A_M}")
        # print(f"self.basis_record: {self.basis_record}")
        # 构造基矩阵
        self.B = self.A_M[:, self.basis_record]
        # 构造基变量对应的目标函数系数
        self.c_basis = self.c_merge[self.basis_record]
        self.Binv = np.linalg.inv(self.B)
        # print(self.Binv)
        self.init_x = self.Binv @ self.b_merge
        # print("初始化可行基解完成")
        # print(self.A_M)
        # print(self.b_merge)
        # print(self.c_merge)
        # print(self.basis_record)
        # print(self.M_var)


    def compute_cj(self):
        print("选择入基")
        self.p = self.c_basis @ self.Binv 
        for i in range(self.A_M.shape[1]): # brand's rule 最小下标原则
            
            if(self.c_merge[i] - self.p @ self.A_M[:, i] < -self.TOLERANCE): #入基
                print(self.c_merge[i] - self.p @ self.A_M[:, i])
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
            all_theta_indices_where = np.where(np.isclose(ratios_where, min_ratio, atol=self.TOLERANCE))[0]

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
        presolve_result = self.presolve()
        if(not presolve_result["success"]):
            return presolve_result
        self.convert_to_standard_form()
        self.Initiate_by_bigM_method()
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

        

