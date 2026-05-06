import torch
import numpy as np
from your_model_file import EffortDetector  # 替换为你的模型定义文件

# -------------------------- 1. 配置参数 --------------------------
weights_path = "/path/to/your/weights.pth"  # 替换为你的权重文件路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型（按你的EffortDetector初始化参数）
model = EffortDetector(
    backbone_name="clip-vit-large-patch14",  # 示例参数，替换为实际参数
    num_classes=2,  # 示例参数
    ...
).to(device)

# -------------------------- 2. 加载权重文件 --------------------------
# 加载权重并打印基础信息
weights = torch.load(weights_path, map_location=device)
print("="*50 + " 权重文件基础信息 " + "="*50)
# 若权重是字典（多数情况），打印键的数量和前10个键
if isinstance(weights, dict):
    print(f"权重文件包含 {len(weights.keys())} 个参数键")
    print("前10个参数键：")
    for i, k in enumerate(list(weights.keys())[:10]):
        print(f"  {i+1}. {k} | 形状: {weights[k].shape if isinstance(weights[k], torch.Tensor) else type(weights[k])}")
else:
    print(f"权重文件类型：{type(weights)}（非字典，可能是模型对象）")
    weights = weights.state_dict()  # 若直接保存的模型，提取state_dict

# -------------------------- 3. 对比模型与权重的键 --------------------------
print("\n" + "="*50 + " 模型与权重键对比 " + "="*50)
model_state_dict = model.state_dict()

# 提取双方的键
weights_keys = set(weights.keys())
model_keys = set(model_state_dict.keys())

# 1. 模型有但权重没有的键（缺失的键，对应你的报错）
missing_keys = model_keys - weights_keys
print(f"\n❌ 模型需要但权重缺失的键（共 {len(missing_keys)} 个）：")
# 筛选出你报错的S_r/U_r/V_r相关键
missing_target_keys = [k for k in missing_keys if any(s in k for s in ["S_r", "U_r", "V_r"])]
if missing_target_keys:
    print(f"  核心缺失键（S_r/U_r/V_r）前20个：")
    for i, k in enumerate(missing_target_keys[:20]):
        print(f"    {i+1}. {k}")
    if len(missing_target_keys) > 20:
        print(f"    ... 还有 {len(missing_target_keys)-20} 个同类缺失键")
else:
    print("  无S_r/U_r/V_r相关缺失键")

# 2. 权重有但模型没有的键（多余的键）
extra_keys = weights_keys - model_keys
print(f"\n⚠️  权重有但模型不需要的键（共 {len(extra_keys)} 个）：")
print(f"  前10个多余键：{list(extra_keys)[:10]}")

# 3. 双方都有的键（匹配的键）
matched_keys = weights_keys & model_keys
print(f"\n✅ 模型与权重匹配的键（共 {len(matched_keys)} 个）：")
print(f"  前10个匹配键：{list(matched_keys)[:10]}")

# -------------------------- 4. 验证匹配键的形状 --------------------------
print("\n" + "="*50 + " 匹配键的形状验证 " + "="*50)
shape_mismatch = []
for k in list(matched_keys)[:20]:  # 仅检查前20个匹配键
    w_shape = weights[k].shape
    m_shape = model_state_dict[k].shape
    if w_shape != m_shape:
        shape_mismatch.append((k, w_shape, m_shape))

if shape_mismatch:
    print(f"❌ 形状不匹配的键（前10个）：")
    for k, w_shape, m_shape in shape_mismatch[:10]:
        print(f"  {k} | 权重形状: {w_shape} | 模型形状: {m_shape}")
else:
    print("✅ 前20个匹配键形状完全一致")

# -------------------------- 5. 尝试加载权重（带日志） --------------------------
print("\n" + "="*50 + " 尝试加载权重 " + "="*50)
try:
    # 先尝试严格加载（会报错，验证）
    model.load_state_dict(weights, strict=True)
    print("✅ 严格加载成功！权重与模型完全匹配")
except RuntimeError as e:
    print(f"❌ 严格加载失败（预期结果）：{str(e)[:200]}...")
    # 尝试非严格加载
    print("\n🔄 尝试非严格加载（忽略缺失/多余键）：")
    model.load_state_dict(weights, strict=False)
    print("✅ 非严格加载成功！缺失的键会随机初始化，多余的键会被忽略")