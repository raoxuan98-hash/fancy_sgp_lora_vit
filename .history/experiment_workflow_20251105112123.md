# 实验工作流程图

## 整体实验流程

```mermaid
graph TD
    A[开始实验] --> B[主实验阶段]
    A --> C[参数优化阶段]
    A --> D[消融研究阶段]
    A --> E[补充实验阶段]
    
    B --> B1[Basic LoRA实验]
    B --> B2[LoRA + KD实验]
    B --> B3[LoRA-NSP实验]
    B --> B4[SGP-LoRA实验]
    
    B1 --> F[结果收集与分析]
    B2 --> F
    B3 --> F
    B4 --> F
    
    C --> C1[SGP参数网格搜索]
    C1 --> C2[参数敏感性分析]
    C2 --> F
    
    D --> D1[组件消融实验]
    D --> D2[AMDC消融实验]
    D --> D3[分类器消融实验]
    
    D1 --> F
    D2 --> F
    D3 --> F
    
    E --> E1[长序列任务实验]
    E --> E2[跨架构泛化实验]
    
    E1 --> F
    E2 --> F
    
    F --> G[结果汇总与可视化]
    G --> H[论文撰写]
```

## 主实验详细流程

```mermaid
graph TD
    A[主实验开始] --> B[数据集选择]
    
    B --> C[CIFAR-100]
    B --> D[ImageNet-R]
    B --> E[CUB-200]
    B --> F[Cars-196]
    
    C --> G[方法选择]
    D --> G
    E --> G
    F --> G
    
    G --> H[Basic LoRA]
    G --> I[LoRA + KD]
    G --> J[LoRA-NSP]
    G --> K[SGP-LoRA]
    
    H --> L[种子选择]
    I --> L
    J --> L
    K --> L
    
    L --> M[Seed 1993]
    L --> N[Seed 1996]
    L --> O[Seed 1997]
    
    M --> P[并行执行]
    N --> P
    O --> P
    
    P --> Q[结果收集]
    Q --> R[统计分析]
    R --> S[结果可视化]
```

## 参数优化流程

```mermaid
graph TD
    A[参数优化开始] --> B[选择数据集]
    
    B --> C[参数网格定义]
    C --> D[weight_temp: 1.0, 2.0, 4.0]
    C --> E[weight_p: 1.0, 2.0]
    C --> F[weight_kind: log1p, exp, rational1]
    
    D --> G[生成参数组合]
    E --> G
    F --> G
    
    G --> H[并行执行实验]
    H --> I[结果收集]
    I --> J[性能分析]
    J --> K[最优参数选择]
    K --> L[验证实验]
```

## 消融研究流程

```mermaid
graph TD
    A[消融研究开始] --> B[组件消融]
    A --> C[AMDC消融]
    A --> D[SGP消融]
    
    B --> B1[完整方法]
    B --> B2[w/o SGP]
    B --> B3[w/o AMDC]
    B --> B4[w/o both]
    
    B1 --> E[结果对比]
    B2 --> E
    B3 --> E
    B4 --> E
    
    C --> C1[完整AMDC]
    C --> C2[仅均值补偿]
    C --> C3[仅协方差补偿]
    C --> C4[线性变换]
    C --> C5[弱非线性变换]
    
    C1 --> F[效率对比]
    C2 --> F
    C3 --> F
    C4 --> F
    C5 --> F
    
    D --> D1[不同weight_temp]
    D --> D2[不同weight_p]
    D --> D3[不同weight_kind]
    
    D1 --> G[敏感性分析]
    D2 --> G
    D3 --> G
    
    E --> H[组件贡献分析]
    F --> I[补偿策略分析]
    G --> J[参数影响分析]
    
    H --> K[消融结论]
    I --> K
    J --> K
```

## 结果分析流程

```mermaid
graph TD
    A[结果分析开始] --> B[数据收集]
    
    B --> C[解析aggregate_results.json]
    C --> D[提取关键指标]
    D --> E[Last-Accuracy]
    D --> F[Avg-Accuracy]
    D --> G[执行时间]
    
    E --> H[统计分析]
    F --> H
    G --> H
    
    H --> I[计算均值和标准差]
    I --> J[显著性检验]
    J --> K[效果量计算]
    
    K --> L[结果可视化]
    L --> M[性能对比图]
    L --> N[参数敏感性图]
    L --> O[消融效果图]
    
    M --> P[表格生成]
    N --> P
    O --> P
    
    P --> Q[LaTeX表格]
    P --> R[Markdown表格]
    P --> S[CSV数据]
    
    Q --> T[论文图表]
    R --> T
    S --> T
```

## 实验管理流程

```mermaid
graph TD
    A[实验管理开始] --> B[资源检查]
    
    B --> C[GPU可用性检查]
    B --> D[存储空间检查]
    B --> E[内存检查]
    
    C --> F[GPU分配策略]
    D --> G[存储管理策略]
    E --> H[内存优化策略]
    
    F --> I[实验队列管理]
    G --> I
    H --> I
    
    I --> J[并行执行控制]
    J --> K[实验监控]
    K --> L[进度跟踪]
    
    L --> M[异常检测]
    M --> N[自动重试]
    N --> O[故障恢复]
    
    O --> P[结果验证]
    P --> Q[数据完整性检查]
    Q --> R[结果一致性验证]
    
    R --> S[实验完成]
    S --> T[资源清理]
    T --> U[结果归档]
```

## 实施时间线

```mermaid
gantt
    title 实验实施时间线
    dateFormat  YYYY-MM-DD
    section 阶段1
    核心实验实施    :a1, 2024-01-01, 14d
    结果收集系统    :a2, after a1, 7d
    
    section 阶段2
    SGP参数搜索    :b1, after a2, 14d
    实验管理优化    :b2, after b1, 7d
    
    section 阶段3
    组件消融实验    :c1, after b2, 14d
    AMDC消融实验    :c2, after c1, 7d
    SGP消融实验     :c3, after c2, 7d
    
    section 阶段4
    长序列任务      :d1, after c3, 7d
    跨架构泛化      :d2, after d1, 7d
    结果完善        :d3, after d2, 7d
```

## 关键决策点

```mermaid
graph TD
    A[实验决策点] --> B{主实验是否成功?}
    
    B -->|是| C[继续参数优化]
    B -->|否| D[问题诊断与修复]
    
    C --> E{参数是否有明显改进?}
    E -->|是| F[继续消融研究]
    E -->|否| G[调整参数范围]
    
    D --> H[检查代码和环境]
    H --> I[重新运行实验]
    I --> B
    
    F --> J{消融结果是否符合预期?}
    J -->|是| K[继续补充实验]
    J -->|否| L[分析方法设计]
    
    G --> M[重新设计参数网格]
    M --> C
    
    K --> N[完成所有实验]
    L --> O[调整消融设计]
    O --> F
    
    N --> P[结果分析与论文撰写]