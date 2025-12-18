import numpy as np
import sympy as sp
import math 
import scipy.linalg
import sys

# 检查当前环境是否已经定义了 'profile' (即是否在 kernprof 下运行)
# 如果没有定义，就创建一个“什么都不做”的空装饰器来顶替，防止报错
# if 'profile' not in dir(__builtins__):
#     def profile(func):
#         return func
try:
    profile  
except NameError:
    # 没人应，说明 kernprof 没在跑，那我造个假的顶着
    def profile(func):
        return func



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
        self.TOLERANCE_INV = 1e-6
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
        negative_b_indices_eq = np.where(self.beq < 0)[0]
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
        将数组除以其第一个非零元素绝对值，使得该元素的位置变为1。
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
        # 这样做非常耗时 rref
        # A和b拼接
        # A_b = np.hstack((A, b.reshape(-1, 1)))
        # A_T = sp.Matrix(A_b).T
        # rref_matrix_T, pivot_columns = A_T.rref()
        # independent_row_indices = sorted(list(pivot_columns))
        # non_redundant_A = A[independent_row_indices, :] 
        # non_redundant_b = b[independent_row_indices] 
        # return non_redundant_A, non_redundant_b
    
        # 确保 b 是列向量
        A_b = np.hstack((A, b.reshape(-1, 1)))
        
        # QR 分解 (用LU分解也可以，但带行置换的LU分解数值不如带列置换的QR分解稳定)
        # 注意：QR 分解通常用来找“线性无关的列”。
        # 我们要找“线性无关的行”，所以必须先转置 (.T)
        # pivoting=True 会返回一个置换数组 P，告诉我们哪些列（即原矩阵的行）被选为主元了
        Q, R, P = scipy.linalg.qr(A_b.T, pivoting=True)
        
        # 4. 确定秩 (Rank)
        # R 是上三角矩阵，对角线元素的绝对值大于阈值的个数就是秩
        # 1e-5 是一个常用的数值容差
        rank = np.sum(np.abs(np.diag(R)) > 1e-5)
        
        # 5. 提取线性无关的行索引
        # P 是一个置换后的索引数组，前 rank 个就是线性无关的行的索引
        independent_row_indices = P[:rank]
        
        # 6. 排序索引 (保持原矩阵的相对顺序，虽然不是必须的，但方便调试)
        independent_row_indices = sorted(independent_row_indices)
        
        # 7. 根据索引切片
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
                    flag0 = 0
                    for j in range(self.A.shape[1]):
                        # if((self.A_merge[:, j] == np.eye(self.A_merge.shape[0])[i]).all()):
                        if(self.A_merge[i, j]  == 1 and np.count_nonzero(self.A_merge[:, j]) == 1):
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
                    # if((self.A_merge[:, j] == np.eye(self.A_merge.shape[0])[i]).all()):
                    if(self.A_merge[i, j]  == 1 and np.count_nonzero(self.A_merge[:, j]) == 1):
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
        # self.Binv = np.linalg.inv(self.B)
        self.Binv = np.eye(self.B.shape[0])
        # print(self.Binv)
        self.init_x = self.Binv @ self.b_merge
        # print("初始化可行基解完成")
        # print(self.A_M)
        # print(self.b_merge)
        # print(self.c_merge)
        # print(self.basis_record)
        # print(self.M_var)


    def compute_cj(self, use_brand_rule=True):
        # print("选择入基")
        self.p = self.c_basis @ self.Binv 
        # for i in range(self.A_M.shape[1]): # brand's rule 最小下标原则
            
        #     if(self.c_merge[i] - self.p @ self.A_M[:, i] < -self.TOLERANCE): #入基
        #         # print(self.c_merge[i] - self.p @ self.A_M[:, i])
        #         return {"success": True, "status": SolverStatus.NOT_SOLVED, "message": "not solved yet", "index": i}
        all_cost = self.c_merge - self.p @ self.A_M
        candidate = np.where(all_cost < -self.TOLERANCE)[0]
        if candidate.size > 0:
            if use_brand_rule:
                i = candidate[0] # brand's rule
                # print(self.c_merge[i] - self.p @ self.A_M[:, i])

                return {"success": True, "status": SolverStatus.NOT_SOLVED, "message": "not solved yet", "index": i}
            else :
                idx = np.argmin(all_cost[candidate])
                i = candidate[idx]
                # print(self.c_merge[i] - self.p @ self.A_M[:, i])
                return {"success": True, "status": SolverStatus.NOT_SOLVED, "message": "not solved yet", "index": i}
        # 检验数已经全部大于等于零，看看是否有M变量，如果有M变量，说明无可行解
        if(not (set(self.M_var).isdisjoint(set(self.basis_record)))):
            mask = np.isin(self.basis_record, self.M_var)
            intersection_index = np.where(mask)[0]
            if(self.init_x[intersection_index].any() != 0):
                return {"success": False, "status" : SolverStatus.INFEASIBLE, "message": "infeasible solution", "index":None} # -2意味着无可行解
            
        # solution = np.zeros(self.A_M.shape[1])
        # for i in range(self.A_M.shape[1]):
        #     if(i in self.basis_record):
        #         solution[i] = self.init_x[self.basis_record.index(i)]
        solution = np.zeros(self.A_M.shape[1])
        solution[self.basis_record] = self.init_x
        return {"success": True, "status" : SolverStatus.OPTIMAL, "message": "optimal solution found", "index":None, "solution": solution, "objective_value": self.init_x @ self.c_basis} # -1说明已经最优解
    
    def compute_u(self, i): #找出基
        # self.u = self.Binv @ self.A_M[:, i]
        # u_positive = self.u[self.u > 0]
        # # 如果全为负，说明问题无界
        # if(u_positive.size == 0):
        #     return {"success": True, "status" : SolverStatus.UNBOUNDED, "message": "unbounded", "index": None, "theta": None, "solution": None}
        # else: 

        #     #找最小theta x / u
        #     ratios_where = np.where(self.u > 0, self.init_x / self.u, np.inf)
        #     min_ratio = np.min(ratios_where)
        #     all_theta_indices_where = np.where(ratios_where == min_ratio)[0]
        #     all_theta_indices_where = np.where(np.isclose(ratios_where, min_ratio, atol=self.TOLERANCE))[0]

        #     #  找最小的下标
        #     local_index = np.argmin(np.array(self.basis_record)[all_theta_indices_where])
        #     theta_index_where = all_theta_indices_where[local_index]
           
        #     # theta_index_where = np.argmin(ratios_where) #最小下标原则
        #     return {"success": True, "status" : SolverStatus.NOT_SOLVED, "message": "not solved yet", "index": theta_index_where, "theta": ratios_where[theta_index_where]}
            # 返回出基变量是第几个
        self.u = self.Binv @ self.A_M[:, i]
        
        # 2. 生成掩码：只关心 u > 0 的部分 (大于容差)
        # 这一步比 where 快，因为它只做比较
        # positive_mask = self.u > self.TOLERANCE
        positive_mask = self.u > 0
        
        # 如果没有正的 u，说明无界
        if not np.any(positive_mask):
            return {"success": True, "status": SolverStatus.UNBOUNDED}
            
        # 3. 【关键加速】只计算有效部分的 theta
        # u[positive_mask] 只有几十个或几百个元素，比全量除法快得多
        valid_u = self.u[positive_mask]
        valid_x = self.init_x[positive_mask]
        thetas = valid_x / valid_u
        
        min_theta = np.min(thetas)
        candidates_mask = thetas <= min_theta + self.TOLERANCE
        candidates_local_idx = np.where(candidates_mask)[0]
        if len(candidates_local_idx) == 1:
            best_idx_local = candidates_local_idx[0]
        # 寻找最小下标
        else:
            original_row_indices = np.where(positive_mask)[0][candidates_local_idx]
            candidates_basis_vars = [self.basis_record[idx] for idx in original_row_indices]
            winner_idx = np.argmin(candidates_basis_vars)
            best_idx_local = candidates_local_idx[winner_idx]
        theta_index_where = np.where(positive_mask)[0][best_idx_local]
 
        return {
            "success": True, 
            "status": SolverStatus.NOT_SOLVED, 
            "index": theta_index_where, 
            "theta": thetas[best_idx_local]
        }
            
    def pivot(self, theta_index_where, i, theta):
        # 入基
        self.B[:, theta_index_where] = self.A_M[:, i]
        # 系数
        self.c_basis[theta_index_where] = self.c_merge[i]
        #更新基变量的值
        self.init_x[theta_index_where] =  theta
        
        self.init_x -= theta * self.u
        self.init_x[theta_index_where] = theta
        

        
    def cumpute_newBinv(self, theta_index_where):
        # 计算新的基矩阵的逆
        self.Binv = np.hstack((self.Binv, self.u.reshape(-1, 1)))
        multiplyer = self.u / self.u[theta_index_where]
        multiplyer[theta_index_where] = 0
        self.Binv -= multiplyer.reshape(-1, 1) * self.Binv[theta_index_where, :]
        self.Binv[theta_index_where, : ] /= self.u[theta_index_where]
        self.Binv = self.Binv[:, :-1]


    @profile
    def solve(self):
        '''
        return {"success": , "status" : , "message": , "index":, "solution": , "objective_value":}
        '''
        
        presolve_result = self.presolve()
        if(not presolve_result["success"]):
            return presolve_result
        self.convert_to_standard_form()
        self.Initiate_by_bigM_method()
        reinversion_num = 0
        iter_num = 0    
        MAX_DANTZI_ITERATION = 3000
        use_brand_rule = False
        need_reinversion = False
        

        while True:
            iter_num += 1
            if iter_num > MAX_DANTZI_ITERATION:
                use_brand_rule = True
            cj_result = self.compute_cj(use_brand_rule)
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

                            if reinversion_num % 100 == 0:
                                if(np.max(np.abs(self.B @ self.init_x - self.b_merge)) > self.TOLERANCE_INV):
                                    need_reinversion = True

                            if(reinversion_num >= 2000 or need_reinversion):
                                reinversion_num = 0
                                self.Binv = np.linalg.inv(self.B)
                            else:
                                self.cumpute_newBinv(theta_index_where)
                            self.basis_record[theta_index_where] = i
                        
                    else:
                        return u_result
            else:
                return cj_result

        

