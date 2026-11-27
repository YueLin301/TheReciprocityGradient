import os
import sys

# 将项目根目录添加到 sys.path，以便能够导入 agent 和 env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from demo_agent import DemoAgent
from env import ContinuousRandomMatchingDonationGame
from torchviz import make_dot

def main():
    print("==========================================================================")
    print(" 验证 Reputation Gradient (反向传播) 的直观 Demo (3人局)")
    print(" 我们设置一个极简的 3步 交互历史，展示计算图的因果链条，并用有限差分验证梯度")
    print(" 并在最后导出 Computation Graph 的可视化图")
    print("==========================================================================\n")

    # 1. 强制设随机种子，保证每次运行结果一致
    torch.manual_seed(42)

    # 封装一个创建 Agent 的函数，并在评估时关闭 Dropout 保证确定性
    def create_agents():
        # 我们需要3个 Agent： i=0, j=1, k=2
        agents = [
            DemoAgent(name=f"Agent_{idx}", signal_dim=1, second_order=False)
            for idx in range(3)
        ]
        for a in agents:
            a.eval() # 关闭 Transformer 中的 Dropout，确保前向传播的确定性
        return agents

    agents_base = create_agents()

    # 保存初始的网络权重状态，用于后续的有限差分测试
    base_state_dicts = [a.state_dict() for a in agents_base]

    # 3. 初始化环境
    b_value = 2.0
    env = ContinuousRandomMatchingDonationGame(
        agents=agents_base,
        horizon=3,
        b=b_value,
        initial_reputation_dist="constant",
        initial_reputation_params={"value": 0.5}
    )

    print("【第一部分：因果链条与交互过程】")
    # 固定匹配序列：
    # t=0: i (0) -> k (2)
    # t=1: j (1) -> i (0)
    # t=2: i (0) -> j (1)
    fixed_matching = [(0, 2), (1, 0), (0, 1)]

    # 运行一次游戏
    cumulative_rewards, matching_sequence, history = env.play_episode(matching_sequence=fixed_matching)

    # 提取关键变量，这些是从 history 里拿出来的记录值
    # 命名规则: a_{donor}_{recipient}
    a_0_to_2 = history[0]["donor_action"]
    sig_2_about_0 = history[0]["recipient_signal"]

    a_1_to_0 = history[1]["donor_action"]
    sig_0_about_1 = history[1]["recipient_signal"]

    a_0_to_1 = history[2]["donor_action"]
    sig_1_about_0 = history[2]["recipient_signal"]

    # Agent 0 的总奖励: R0 = -a_0_to_2 + b * a_1_to_0 - a_0_to_1
    # cumulative_rewards 保存了完整的计算图 (Computation Graph)
    R0 = cumulative_rewards[0]

    print("预设的固定匹配序列: ")
    print("  t=0: Agent 0 (i) -> Agent 2 (k)")
    print("  t=1: Agent 1 (j) -> Agent 0 (i)")
    print("  t=2: Agent 0 (i) -> Agent 1 (j)\n")

    print("因果传递链条 (Computation Graph) 如下:")
    print("  (1) t=0: Agent 0 观察 Agent 2 的初始声誉，采取行动 a_0_to_2 = ActionNet0(Rep2)")
    print(f"      => a_0_to_2 实际输出值为: {a_0_to_2.item():.4f}")
    print("      Agent 2 观察到 a_0_to_2，给出对 Agent 0 的评价信号 sig_2_about_0 = SignalNet2(a_0_to_2)")
    print(f"      => sig_2_about_0 实际输出值为: {sig_2_about_0.item():.4f}")
    print("           (此信号被追加到 Agent 0 的声誉记录中)")

    print("\n  (2) t=1: Agent 1 观察 Agent 0 的声誉 (包含 sig_2_about_0)，采取行动 a_1_to_0 = ActionNet1(Rep0)")
    print(f"      => a_1_to_0 实际输出值为: {a_1_to_0.item():.4f}")
    print("      Agent 0 观察到 a_1_to_0，给出对 Agent 1 的评价信号 sig_0_about_1 = SignalNet0(a_1_to_0)")
    print(f"      => sig_0_about_1 实际输出值为: {sig_0_about_1.item():.4f}")
    print("           (此信号被追加到 Agent 1 的声誉记录中)")

    print("\n  (3) t=2: Agent 0 观察 Agent 1 的声誉 (包含 sig_0_about_1)，采取行动 a_0_to_1 = ActionNet0(Rep1)")
    print(f"      => a_0_to_1 实际输出值为: {a_0_to_1.item():.4f}")
    print("      Agent 1 观察到 a_0_to_1，给出对 Agent 0 的评价信号 sig_1_about_0 = SignalNet1(a_0_to_1)")
    print(f"      => sig_1_about_0 实际输出值为: {sig_1_about_0.item():.4f}")
    print("           (此信号被追加到 Agent 0 的声誉记录中)")

    print(f"\n  (4) Agent 0 的总奖励 R0 = -a_0_to_2 + {b_value} * a_1_to_0 - a_0_to_1")
    print(f"      => R0 = {-a_0_to_2.item():.4f} + {b_value * a_1_to_0.item():.4f} - {a_0_to_1.item():.4f} = {R0.item():.4f}\n")

    print("【第二部分：计算图反向传播 (Autograd)】")
    print("我们对 Agent 0 的总奖励 R0，求关于 Agent 0 所有网络参数的梯度。")
    print("Agent 0 (i) 的参数在哪些地方参与了计算：")
    print("  1. 决定 t=0 时的 a_0_to_2 (直接影响自身奖励 -a_0_to_2，并改变 sig_2_about_0)")
    print("  2. a_0_to_2 导致 sig_2_about_0 变化，进而影响了别人在 t=1 给我的回报 a_1_to_0 (影响自身奖励 b*a_1_to_0)")
    print("  3. 在 t=1 时，作为评价者给出信号 sig_0_about_1，这个信号记录在 Agent 1 身上")
    print("  4. 在 t=2 时，决定 a_0_to_1 (观察 Agent 1 的声誉，该声誉里有我上一步给出的 sig_0_about_1)")
    print("  5. a_0_to_1 直接影响自身奖励 -a_0_to_1")
    print("这就构成了一个高度交织的递归博弈计算图！\n")

    # 取 Agent 0 action_net 的第一层权重矩阵中的一个参数作为代表进行有限差分验证
    # 注意：我们的求导是求整个 Agent 0 的参数（包括 action_net 和 signal_net），
    # 但由于终端打印篇幅原因，我们只选取 action_net[0].weight 的其中一个元素做精细数值对比。
    target_param = agents_base[0].action_net[0].weight

    # 也可以拿到所有涉及 Agent 0 的参数
    # params_agent_0 = list(agents_base[0].parameters())

    # 获取 Autograd 梯度
    grads = torch.autograd.grad(
        outputs=R0,
        inputs=target_param,
        retain_graph=True
    )
    # 我们只关注第 0 行第 0 列的权重的梯度
    autograd_grad = grads[0][0, 0].item()

    print(f"通过 Autograd 求得 action_net[0].weight[0,0] 的梯度: {autograd_grad:.6f}\n")

    print("【第三部分：有限差分法验证 (Finite Difference)】")
    print("为了验证 Autograd 不是瞎算的，我们手动给该权重加上一个微小的扰动 epsilon，")
    print("重新运行一遍游戏，通过 (R0_new - R0_old) / epsilon 计算近似梯度，看看是否吻合。\n")

    epsilon = 1e-4

    # 封装一个函数：改变特定参数后重新运行一整局游戏，返回新的 R0
    def evaluate_R0_with_perturbed_weight(eps):
        agents_copy = create_agents()
        # 恢复所有 agent 的初始权重
        for i in range(3):
            sd = base_state_dicts[i].copy()
            # 忽略 reputation_history buffer 的形状不匹配问题 (每个 episode 会自行重置)
            if "reputation_history" in sd:
                del sd["reputation_history"]
            agents_copy[i].load_state_dict(sd, strict=False)

        # 给特定的参数加上扰动 eps
        agents_copy[0].action_net[0].weight.data[0, 0] += eps

        env_copy = ContinuousRandomMatchingDonationGame(
            agents=agents_copy,
            horizon=3,
            b=b_value,
            initial_reputation_dist="constant",
            initial_reputation_params={"value": 0.5}
        )

        rewards, _, _ = env_copy.play_episode(matching_sequence=fixed_matching)
        return rewards[0].item()

    R0_base = evaluate_R0_with_perturbed_weight(0.0)
    R0_plus = evaluate_R0_with_perturbed_weight(epsilon)

    fd_grad = (R0_plus - R0_base) / epsilon

    print(f"基础 R0: {R0_base:.6f}")
    print(f"扰动后 R0 (+eps): {R0_plus:.6f}")
    print(f"有限差分近似求得的梯度 [(R0_new - R0_base) / epsilon]: {fd_grad:.6f}\n")

    error = abs(autograd_grad - fd_grad)
    print(f"Autograd 与有限差分的误差: {error:.6e}")
    if error < 1e-3:
        print("结论: 误差极小！成功证明 PyTorch 计算图完好无损地跨越了 Agent 的交互，反向传播是绝对正确的。\n")
    else:
        print("结论: 误差较大，可能存在计算图断裂的问题！\n")

    print("【第四部分：导出 Computation Graph 可视化】")
    print("正在使用 torchviz 生成多个变量的计算图...")

    # 为了方便追踪哪些节点属于哪个 Agent，我们把所有 Agent 的参数传入 params 字典
    # 并且我们可以给参数重命名，方便在图里认出来。
    all_params = {}
    for i, agent in enumerate(agents_base):
        for name, p in agent.named_parameters():
            all_params[f"Agent_{i}_{name}"] = p

    def save_computation_graph(target_tensor, output_name):
        dot = make_dot(target_tensor, params=all_params)

        # 重新组织计算图中的节点：按 Agent 和 Network 分组，并涂上不同颜色
        lines = dot.body
        new_body = []

        # 结构: agent_nodes[agent_id][network_type] = list_of_nodes
        # network_type: "action" 或 "signal"
        agent_nodes = {
            0: {"action": [], "signal": []},
            1: {"action": [], "signal": []},
            2: {"action": [], "signal": []}
        }
        edges = []
        other_nodes = []

        for line in lines:
            if "->" in line:
                edges.append(line)
            elif "Agent_" in line:
                # 解析到底是哪个 Agent
                agent_id = None
                if "Agent_0" in line: agent_id = 0
                elif "Agent_1" in line: agent_id = 1
                elif "Agent_2" in line: agent_id = 2

                if agent_id is not None:
                    # 涂色 (保持同一个 agent 使用同一种颜色)
                    if agent_id == 0:
                        colored_line = line.replace("lightblue", '"#ffb3ba"')
                    elif agent_id == 1:
                        colored_line = line.replace("lightblue", '"#baffc9"')
                    elif agent_id == 2:
                        colored_line = line.replace("lightblue", '"#bae1ff"')
                    else:
                        colored_line = line

                    # 区分 action_net 和 signal_net
                    if "action_net" in line:
                        agent_nodes[agent_id]["action"].append(colored_line)
                    elif "signal_net" in line:
                        agent_nodes[agent_id]["signal"].append(colored_line)
                    else:
                        # 对于其他的参数也放在 action 中 (如果有的话)
                        agent_nodes[agent_id]["action"].append(colored_line)
                else:
                    other_nodes.append(line)
            else:
                other_nodes.append(line)

        new_body.extend(other_nodes)

        cluster_idx = 0
        for i in range(3):
            # Action Net 的框
            if agent_nodes[i]["action"]:
                new_body.append(f'\tsubgraph cluster_{cluster_idx} {{\n')
                new_body.append(f'\t\tlabel="Agent {i} Action Net";\n')
                new_body.append(f'\t\tstyle="filled,rounded";\n')
                new_body.append(f'\t\tfillcolor="gray95";\n')
                for node in agent_nodes[i]["action"]:
                    new_body.append('\t' + node)
                new_body.append('\t}\n')
                cluster_idx += 1

            # Signal Net 的框
            if agent_nodes[i]["signal"]:
                new_body.append(f'\tsubgraph cluster_{cluster_idx} {{\n')
                new_body.append(f'\t\tlabel="Agent {i} Signal Net";\n')
                new_body.append(f'\t\tstyle="filled,rounded";\n')
                new_body.append(f'\t\tfillcolor="gray95";\n')
                for node in agent_nodes[i]["signal"]:
                    new_body.append('\t' + node)
                new_body.append('\t}\n')
                cluster_idx += 1

        new_body.extend(edges)

        dot.body = new_body

        # 获取当前脚本所在目录，确保输出文件保存在该目录下
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, output_name)

        # 如果系统里安装了 graphviz，可以直接渲染成 PDF 或者 PNG
        try:
            dot.render(output_base, format="pdf")
            print(f"-> 成功生成 {output_base}.pdf")
        except Exception as e:
            print(f"-> 无法生成 PDF (可能缺少 Graphviz 系统依赖): {e}")
            print("-> 尝试仅保存 .dot 源码文件...")
            try:
                dot.save(f"{output_base}.dot")
                print(f"-> 成功保存 {output_base}.dot")
            except Exception as e2:
                print(f"-> 保存 .dot 文件也失败了: {e2}")

    print("\n[图1] 生成 R0 (Agent 0 的总奖励) 对 Agent 0 参数的计算图...")
    save_computation_graph(R0, "computation_graph_R0_to_pi0")

    print("\n[图2] 生成 a^i (Agent 0 在 t=2 的动作 a_0_to_1) 对 Agent 0 参数的计算图...")
    save_computation_graph(a_0_to_1, "computation_graph_a0_to_pi0")

    print("\n[图3] 生成 a^j (Agent 1 在 t=1 的动作 a_1_to_0) 对 Agent 0 参数的计算图...")
    save_computation_graph(a_1_to_0, "computation_graph_a1_to_pi0")

if __name__ == "__main__":
    main()
