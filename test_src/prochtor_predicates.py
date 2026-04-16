from procthor_adapter import ProcTHORAdapter

# 方法1: 使用 prior 库加载
import prior
dataset = prior.load_dataset("procthor-10k")
adapter = ProcTHORAdapter(dataset["train"])
scene = adapter.load_scene("scene_0")
predicates = adapter.to_predicates(scene)
print(predicates)