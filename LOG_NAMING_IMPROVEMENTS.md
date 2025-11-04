# 日志命名系统改进总结

## 问题分析

原始的日志命名系统存在以下问题：

1. **参数交叉污染**：当使用nsp_lora时，日志目录中仍包含sgp_lora的参数（如weight_temp, weight_kind）
2. **知识蒸馏命名不一致**：有些实验使用"distill-xxx"作为子目录，有些则直接将KD参数放在同一级
3. **目录结构混乱**：不同实验的目录层级不一致，导致难以比较结果

## 解决方案

### 1. 参数过滤机制

添加了`_filter_args_by_lora_type`函数，确保只保存与当前LoRA类型相关的参数：

```python
def _filter_args_by_lora_type(args: dict) -> dict:
    """
    过滤参数字典，只保留与当前LoRA类型相关的参数
    这样可以避免在params.json中保存不相关的参数，导致日志命名混乱
    """
    lora_type = args.get('lora_type', 'basic_lora')
    filtered_args = args.copy()
    
    # 定义每种LoRA类型相关的参数
    sgp_lora_params = {'weight_temp', 'weight_kind', 'weight_p'}
    nsp_lora_params = {'nsp_eps', 'nsp_weight'}
    
    # 移除与当前LoRA类型不相关的参数
    if lora_type == 'sgp_lora':
        # 保留SGP参数，移除NSP参数
        for param in nsp_lora_params:
            filtered_args.pop(param, None)
    elif lora_type == 'nsp_lora':
        # 保留NSP参数，移除SGP参数
        for param in sgp_lora_params:
            filtered_args.pop(param, None)
    elif lora_type == 'basic_lora':
        # 移除所有LoRA特定参数
        for param in sgp_lora_params.union(nsp_lora_params):
            filtered_args.pop(param, None)
    elif lora_type == 'full':
        # 移除所有LoRA特定参数
        for param in sgp_lora_params.union(nsp_lora_params):
            filtered_args.pop(param, None)
    
    return filtered_args
```

### 2. 知识蒸馏参数统一命名

修改了`_get_kd_params`函数，统一知识蒸馏参数的命名规则：

```python
def _get_kd_params(args: dict) -> list:
    """获取知识蒸馏相关参数，统一命名规则"""
    kd_params = []
    
    if args.get('gamma_kd', 0.0) > 0.0:
        kd_params.append(f"kd-{short(args['gamma_kd'])}")
        if 'kd_type' in args:
            kd_params.append(f"type-{short(args['kd_type'])}")
        if 'distillation_transform' in args:
            kd_params.append(f"dt-{short(args['distillation_transform'])}")
        if args.get('use_aux_for_kd', False):
            kd_params.append("aux")
        # 添加update_teacher_each_task参数，简写为utt
        if 'update_teacher_each_task' in args:
            kd_params.append(f"utt-{short(args['update_teacher_each_task'])}")
            
    return kd_params
```

### 3. 改进的参数保存

在保存参数到JSON文件前使用过滤函数，并记录过滤信息：

```python
# 保存过滤后的参数到 JSON，避免参数交叉污染
filtered_args = _filter_args_by_lora_type(args)
params_json = Path(abs_log_dir) / "params.json"
if not params_json.exists():
    with open(params_json, "w", encoding="utf-8") as f:
        json.dump(filtered_args, f, ensure_ascii=False, indent=2)

# 记录过滤信息
original_params = set(args.keys())
filtered_params = set(filtered_args.keys())
removed_params = original_params - filtered_params
if removed_params:
    logging.info(f"   过滤掉的参数: {sorted(removed_params)}")
```

## 测试验证

创建了三个测试脚本：

1. **test_log_naming.py**：原始测试脚本，更新了参数验证
2. **test_log_naming_standalone.py**：独立测试脚本，不依赖整个项目
3. **test_backward_compatibility.py**：向后兼容性测试

所有测试都通过，验证了以下功能：

- ✅ LoRA特定参数正确隔离
- ✅ 不同LoRA类型之间没有交叉污染
- ✅ 知识蒸馏参数清晰分离
- ✅ 对不适当的参数使用发出警告
- ✅ 目录结构清晰且描述性强
- ✅ 参数在params.json中正确过滤
- ✅ 现有日志目录仍可解析
- ✅ 新系统正确处理旧参数结构
- ✅ 未引入破坏性更改

## 使用示例

### SGP LoRA（无知识蒸馏）

生成的路径：
```
sldc_logs_test_user/cifar100_224_vit-b-p16-mocov3/init-10_inc-10/lrank-4_ltype-sgp_lora/t-2.0_k-log1p_p-1.0/opt-adamw_lr-0.0001_b-16_i-2000_s-1993
```

params.json中包含：
```json
{
  "lora_type": "sgp_lora",
  "weight_temp": 2.0,
  "weight_kind": "log1p",
  "weight_p": 1.0,
  // ... 其他参数
  // 不包含 nsp_eps, nsp_weight
}
```

### NSP LoRA（无知识蒸馏）

生成的路径：
```
sldc_logs_test_user/cifar100_224_vit-b-p16-mocov3/init-10_inc-10/lrank-4_ltype-nsp_lora/eps-0.05_w-0.02/opt-adamw_lr-0.0001_b-16_i-2000_s-1993
```

params.json中包含：
```json
{
  "lora_type": "nsp_lora",
  "nsp_eps": 0.05,
  "nsp_weight": 0.02,
  // ... 其他参数
  // 不包含 weight_temp, weight_kind, weight_p
}
```

### Basic LoRA（带知识蒸馏）

生成的路径：
```
sldc_logs_test_user/cifar100_224_vit-b-p16-mocov3/init-10_inc-10/lrank-4_ltype-basic_lora/kd-0.1_type-feat_dt-linear_aux_utt-1/opt-adamw_lr-0.0001_b-16_i-2000_s-1993
```

## 向后兼容性

新系统完全向后兼容：

1. 现有日志目录仍可正常解析
2. 新系统能正确处理旧的参数结构
3. 未引入破坏性更改

## 总结

这些改进解决了日志命名混乱的问题，使实验结果更易于比较和管理，同时保持了向后兼容性。