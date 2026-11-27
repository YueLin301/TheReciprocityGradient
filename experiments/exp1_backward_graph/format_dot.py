import re

def format_dot(dot_file):
    with open(dot_file, 'r') as f:
        content = f.read()

    # Find all node definitions that represent parameters
    # The format is something like: 140327339893968 [label="Agent_0_action_net.0.weight\n(10, 1)", fillcolor=lightblue, style=filled];
    param_pattern = re.compile(r'(\d+)\s+\[label="(Agent_\d+_([a-z_]+)\.[^"]+)"[^\]]*\];')

    clusters = {}
    for match in param_pattern.finditer(content):
        node_id = match.group(1)
        full_name = match.group(2)
        agent_and_net = match.group(3) # e.g. Agent_0_action_net -> actually the regex matched it differently

        # better parsing:
        name_parts = full_name.split('_')
        agent_id = name_parts[1]
        net_type = "action" if "action_net" in full_name else "signal"
        cluster_name = f"Agent_{agent_id}_{net_type}_net"

        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name].append(match.group(0))

    if not clusters:
        return content

    # Add subgraph definitions to the end of the graph, right before the closing '}'
    subgraphs = ""
    colors = {
        "Agent_0_action_net": "lightpink",
        "Agent_0_signal_net": "lightcoral",
        "Agent_1_action_net": "lightgreen",
        "Agent_1_signal_net": "lightseagreen",
        "Agent_2_action_net": "lightskyblue",
        "Agent_2_signal_net": "lightcyan",
    }

    for cluster_name, nodes in clusters.items():
        color = colors.get(cluster_name, "lightgrey")
        subgraphs += f'\n  subgraph cluster_{cluster_name} {{\n'
        subgraphs += f'    label="{cluster_name}";\n'
        subgraphs += f'    style=filled;\n'
        subgraphs += f'    color="{color}";\n'
        for node in nodes:
            # We need to extract just the node ID to group them in the subgraph,
            # or keep the node definition inside the subgraph and remove it from outside.
            # Easiest is to just list the node IDs inside the subgraph
            node_id = node.split()[0]
            subgraphs += f'    {node_id};\n'
        subgraphs += '  }\n'

    # Insert subgraphs right before the last closing brace
    last_brace_idx = content.rfind('}')
    if last_brace_idx != -1:
        content = content[:last_brace_idx] + subgraphs + content[last_brace_idx:]

    return content

if __name__ == "__main__":
    formatted = format_dot('computation_graph.dot')
    with open('computation_graph_grouped.dot', 'w') as f:
        f.write(formatted)
