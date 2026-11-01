import sys, random, tempfile, os
sys.path.insert(0, "/opt/aos")
from kernel.page_table import PageTable
from kernel.state_dag import StateDAG

tmp = tempfile.mkdtemp()
pt = PageTable(os.path.join(tmp, "pages.db"))
dag = StateDAG(os.path.join(tmp, "dag.db"))

def vec(seed):
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(16)]

p1 = pt.add("用户要求实现登录接口，用 Go 标准库", vec=vec(1), importance=0.8, tag="goal")
p2 = pt.add("已写好 handler，未测试", vec=vec(2), importance=0.6)
p3 = pt.add("测试报错：nil pointer on session", vec=vec(3), importance=0.9)
p4 = pt.add("早期讨论：数据库选型 MySQL", vec=vec(9), importance=0.2)
pt.pin(p1.page_id)

prompt, used = pt.compose(query_vec=vec(2), budget_tokens=25)
print("=== 上下文 ===")
print(prompt)
print("入选页:", used)
print("页表状态:", pt.stats())

n1 = dag.add("目标：加登录接口", kind="goal")
n2 = dag.add("写 handler", parent_id=n1.node_id)
n3 = dag.add("先改数据库 schema", parent_id=n1.node_id)
n4 = dag.add("handler 完成，测试通过", parent_id=n2.node_id)
exp = dag.prune(n4.node_id)
print("\n=== DAG 活跃路径 ===")
print(dag.render())
print("经验节点:", exp.content)
print("节点状态:", dag.stats())

h = dag.cow(n4.node_id, "人类改口：改用 JWT 而不是 session", kind="human_interrupt")
dag.add("按 JWT 重写中间件", parent_id=h.node_id)
print("\n=== CoW 之后 ===")
print(dag.render())
print("节点状态:", dag.stats())
