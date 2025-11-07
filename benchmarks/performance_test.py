import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import traceback
matplotlib.rc("font",family='KaiTi')



from simplex_solver.solver import SimplexSolver, SolverStatus
N_START = 10      # n 的起始值
N_END = 50   # n 的结束值
N_STEP = 10       # n 的步长
NUM_CASES = 20    # 每个规模n的测试案例数量
FEASIBLE_PROBABILITY = 0.8  # 80% 概率生成可行解
if __name__ == "__main__":
    # 设置中文字体，以防绘图时出现乱码
    # 你可能需要根据你的操作系统选择一个存在的字体
    # try:
    #     matplotlib.rcParams['font.sans-serif'] = ['Heiti TC'] # Mac
    #     matplotlib.rcParams['axes.unicode_minus'] = False
    # except:
    #     try:
    #         matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # Windows
    #         matplotlib.rcParams['axes.unicode_minus'] = False
    #     except:
    #         print("未找到指定中文字体，绘图可能出现乱码。")
    # try:
    #     # 在 Windows 上，优先尝试微软雅黑、等线或黑体
    #     # Matplotlib会依次尝试列表中的字体，直到找到一个可用的
    #     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Dengxian', 'SimHei'] 
    #     # 下面这行代码确保负号也能正常显示
    #     plt.rcParams['axes.unicode_minus'] = False 
    # except Exception as e:
    #     print(f"设置中文字体失败: {e}")
    #     print("图表中的中文可能显示为方框。")


    problem_sizes = []
    avg_times = []
    std_times = []

# ... (脚本的其他部分不变) ...

# 外层循环：遍历不同的问题规模 n
    for n in range(N_START, N_END + 1, N_STEP):
        m = n // 2
        meq = n // 4
        times_for_current_size = []
        print(f"正在测试问题规模 n = {n}, m = {m} ...")

        # 内层循环：对每个规模 n 运行 NUM_CASES 次
        for i in range(NUM_CASES):
            print(f"  正在测试 n={n}, case={i+1} ...")
            try:
                # 生成随机问题数据
                if np.random.rand() < FEASIBLE_PROBABILITY:
                    x_feasible = np.random.uniform(0, 20, n)
                    
                    # 2. ## 升级 ## c, A, Aeq 使用 randn 生成正负数，范围更广
                    c = np.random.randn(n) * 10
                    A = np.random.randn(m, n) * 10 if m > 0 else np.array([[]])
                    Aeq = np.random.randn(meq, n) * 10 if meq > 0 else np.array([[]])
                    
                    # 3. 反向计算 b 和 beq
                    b = A @ x_feasible + np.random.uniform(0, 5, m) if m > 0 else np.array([])
                    beq = Aeq @ x_feasible if meq > 0 else np.array([])
                else:
                    c = np.random.rand(n)
                    A = np.random.rand(m, n)
                    b = np.random.rand(m) * n 
                    Aeq = np.random.rand(meq, n)
                    beq = np.random.rand(meq) * n 
                # print(f"    生成随机数据: c={c}, A={A}, b={b}, Aeq={Aeq}, beq={beq}")
                # print(f"--- Running case n={n}, i={i+1} ---")
                # print("c =", repr(c))
                # print("A =", repr(A))
                # print("b =", repr(b))
                # print("Aeq =", repr(Aeq))
                # print("beq =", repr(beq))
                

                # --- 计时开始 ---
                start_time = time.perf_counter()
                solver = SimplexSolver(c, A = A, b = b, Aeq = Aeq, beq = beq)

                    
                result = solver.solve()
                    
                end_time = time.perf_counter()
                # --- 计时结束 ---
                # 判断是否可行，如果不可行舍弃这次计时
                status = result['status']
                if status != SolverStatus.INFEASIBLE:
                    elapsed_time = end_time - start_time
                    times_for_current_size.append(elapsed_time)
            except Exception as e:
                # print(f"  在 n={n}, case={i+1} 时发生错误: {e}")
                # 这个增强版的 except 会捕获致命错误并打印所有你需要的信息
                print(f"\n!!!!!!!!!!\n在 n={n}, case={i+1} 时捕获到一个严重错误!\n!!!!!!!!!!")
                print("错误类型:", type(e).__name__)
                print("错误信息:", e)
                print("--- 详细追溯信息 (Traceback) ---")
                traceback.print_exc() # 这一行会打印出完整的错误路径和代码行号
                print("----------------------------------\n")
                # 如果你想让程序在第一个错误发生时就彻底停止，可以取消下面这行代码的注释
                # raise 
        print(f"  测试 n={n} 完成, 成功运行 {len(times_for_current_size)} 次")
        if times_for_current_size:
                mean_time = np.mean(times_for_current_size)
                std_dev_time = np.std(times_for_current_size)
                
                problem_sizes.append(n)
                avg_times.append(mean_time)
                std_times.append(std_dev_time)
                
                print(f"  -> 平均时间: {mean_time:.6f} 秒, 标准差: {std_dev_time:.6f} 秒\n")

    print(f"problem_sizes = {problem_sizes}")
    print(f"avg_times = {avg_times}")
    print(f"std_times = {std_times}")
    # 3. 结果可视化
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

        # 使用 errorbar 函数绘制带误差棒的图
    #英文
    ax.errorbar(problem_sizes, avg_times, yerr=std_times, fmt='-o', 
                    capsize=5, label='Average Solving Time (with Std Dev)')

        
    #换成英文
    ax.set_title('Simplex Solving Time vs. Problem Size', fontsize=16)

    # ax.set_xlabel('问题规模 (变量数量 n)', fontsize=12)
    # ax.set_ylabel('平均求解时间 (秒)', fontsize=12)
    #换成英文
    ax.set_xlabel('Problem Size (n)', fontsize=12)
    ax.set_ylabel('Average Solving Time (seconds)', fontsize=12)

    ax.legend()
        
        # 设置坐标轴从0开始，使图像更清晰
    ax.set_xlim(0, N_END + 20)
    ax.set_ylim(0, max(avg_times) * 1.2 if avg_times else 1)

    plt.tight_layout()
    plt.show()
