# 第二章：核心架构总览

> 本章将从整体视角分析 Nano-vLLM 的架构设计，理解各模块之间的关系和数据流向。

## 2.1 整体架构图

```mermaid
graph TB
    subgraph "用户接口"
        User[用户程序]
        LLM[LLM 类]
    end
    
    subgraph "推理引擎 LLMEngine"
        Engine[LLMEngine]
        Scheduler[Scheduler 调度器]
        BlockMgr[BlockManager 块管理器]
    end
    
    subgraph "模型执行 ModelRunner"
        Runner[ModelRunner]
        Model[Qwen3ForCausalLM]
        Sampler[Sampler 采样器]
        KVCache[(KV Cache)]
    end
    
    subgraph "神经网络层"
        Attention[Attention]
        Linear[Linear 层]
        RoPE[RoPE 位置编码]
        Norm[RMSNorm]
        Embed[Embedding]
    end
    
    User -->|"generate(prompts)"| LLM
    LLM --> Engine
    Engine -->|"调度请求"| Scheduler
    Scheduler -->|"管理内存块"| BlockMgr
    Engine -->|"执行推理"| Runner
    Runner --> Model
    Runner --> Sampler
    Runner -->|"读写"| KVCache
    Model --> Attention
    Model --> Linear
    Model --> RoPE
    Model --> Norm
    Model --> Embed
    Attention -->|"存储/读取"| KVCache
```

---

## 2.2 核心概念解释

### 2.2.1 Prefill 与 Decode 两阶段

LLM 推理分为两个阶段：

```mermaid
sequenceDiagram
    participant User as 用户
    participant Engine as 推理引擎
    participant Model as 模型
    
    User->>Engine: 输入 prompt
    
    Note over Engine,Model: Prefill 阶段
    Engine->>Model: 并行处理所有 prompt tokens
    Model-->>Engine: 生成第一个 token + KV Cache
    
    Note over Engine,Model: Decode 阶段 (循环)
    loop 直到生成结束
        Engine->>Model: 输入上一个 token
        Model-->>Engine: 生成下一个 token
    end
    
    Engine-->>User: 返回完整输出
```

| 阶段 | 输入 | 计算特点 | 主要瓶颈 |
|:---|:---|:---|:---|
| **Prefill** | 完整 prompt | 并行计算，计算密集 | 计算量 |
| **Decode** | 单个 token | 顺序执行，访存密集 | 内存带宽 |

### 2.2.2 KV Cache

KV Cache 是加速 LLM 推理的关键技术：

```mermaid
graph LR
    subgraph "无 KV Cache"
        A1[Token 1] --> C1[计算 K,V]
        A2[Token 2] --> C2[重新计算所有 K,V]
        A3[Token 3] --> C3[再次重新计算]
    end
    
    subgraph "有 KV Cache"
        B1[Token 1] --> D1[计算 K,V] --> E1[(缓存)]
        B2[Token 2] --> D2[仅计算新 K,V] --> E1
        B3[Token 3] --> D3[仅计算新 K,V] --> E1
    end
```

**核心思想**：缓存已计算的 Key 和 Value，避免重复计算。

> 💡 **设计思想**：KV Cache 是典型的「空间换时间」策略。Decode 阶段的计算复杂度从 O(n²) 降低到 O(n)，代价是需要额外的显存存储历史 KV。

### 2.2.3 Continuous Batching

传统批处理 vs Continuous Batching：

```mermaid
gantt
    title 传统批处理
    dateFormat X
    axisFormat %s
    
    section 请求1
    ████████████████    :0, 16
    section 请求2 (短)
    ████    等待    :0, 16
    section 请求3
    ████████████        等待 :0, 16
```

```mermaid
gantt
    title Continuous Batching
    dateFormat X
    axisFormat %s
    
    section 请求1
    ████████████████    :0, 16
    section 请求2 (短)
    ████    :0, 4
    section 请求3
    ████████████ :0, 12
    section 请求4 (新加入)
        ████████████ :4, 16
```

**优势**：请求完成后立即释放资源，新请求可以立即加入。

> 💡 **设计思想**：Continuous Batching 将 GPU 资源的分配单位从「批次」细化到「token」，大幅提升了 GPU 利用率。这是 LLM 推理引擎的核心创新之一。

### 2.2.4 Prefix Caching

相同前缀的请求可以共享 KV Cache：

```mermaid
graph TB
    subgraph "请求1"
        A1["系统提示: 你是一个助手"] --> B1["用户: 什么是AI?"]
    end
    
    subgraph "请求2"
        A2["系统提示: 你是一个助手"] --> B2["用户: 天气如何?"]
    end
    
    subgraph "KV Cache"
        C[共享缓存: 系统提示]
        D1[请求1专有缓存]
        D2[请求2专有缓存]
    end
    
    A1 -.->|"共享"| C
    A2 -.->|"共享"| C
    B1 --> D1
    B2 --> D2
```

---

## 2.3 模块职责

### 2.3.1 各模块职责表

| 模块 | 文件 | 核心职责 |
|:---|:---|:---|
| **LLM** | `llm.py` | 用户接口，继承 LLMEngine |
| **LLMEngine** | `llm_engine.py` | 推理引擎入口，协调调度和执行 |
| **Scheduler** | `scheduler.py` | 请求调度，决定哪些序列参与推理 |
| **BlockManager** | `block_manager.py` | KV Cache 内存管理，Prefix Caching |
| **Sequence** | `sequence.py` | 序列数据结构，状态管理 |
| **ModelRunner** | `model_runner.py` | 模型执行，CUDA Graph，张量并行 |
| **Qwen3ForCausalLM** | `qwen3.py` | Transformer 模型实现 |
| **Attention** | `attention.py` | 注意力计算，Flash Attention |
| **Linear** | `linear.py` | 并行线性层 |
| **RMSNorm** | `layernorm.py` | 归一化层 |
| **Sampler** | `sampler.py` | Token 采样 |

> 💡 **设计思想**：Nano-vLLM 的模块划分遵循「单一职责原则」——Scheduler 只负责调度，BlockManager 只负责内存，ModelRunner 只负责执行。这种分离让代码更易理解和维护。

### 2.3.2 模块依赖关系

```mermaid
graph TD
    A[LLM] --> B[LLMEngine]
    B --> C[Scheduler]
    B --> D[ModelRunner]
    C --> E[BlockManager]
    C --> F[Sequence]
    D --> G[Qwen3ForCausalLM]
    D --> H[Sampler]
    D --> I[Context]
    D --> J[Loader]
    G --> K[Attention]
    G --> L[Linear]
    G --> M[RMSNorm]
    G --> N[RoPE]
    G --> O[Embedding]
    K --> I
    O --> I
```

---

## 2.4 数据流分析

### 2.4.1 完整推理流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant LLM as LLM.generate()
    participant Sched as Scheduler
    participant BM as BlockManager
    participant Runner as ModelRunner
    participant Model as Model
    participant Sampler as Sampler
    
    User->>LLM: generate(prompts, params)
    
    loop 每个 prompt
        LLM->>LLM: add_request(prompt, params)
        LLM->>Sched: scheduler.add(seq)
    end
    
    loop 直到所有序列完成
        LLM->>Sched: schedule()
        Sched->>BM: allocate/can_append
        Sched-->>LLM: (seqs, is_prefill)
        
        LLM->>Runner: run(seqs, is_prefill)
        Runner->>Runner: prepare_prefill/decode
        Runner->>Model: forward(input_ids, positions)
        Model-->>Runner: hidden_states
        Runner->>Sampler: sample(logits, temperatures)
        Sampler-->>Runner: token_ids
        Runner-->>LLM: token_ids
        
        LLM->>Sched: postprocess(seqs, token_ids)
        Sched->>Sched: 更新序列状态
    end
    
    LLM-->>User: outputs
```

### 2.4.2 单步推理详解

每次 `step()` 调用的内部流程：

```python
def step(self):
    # 1. 调度：决定本轮处理哪些序列
    seqs, is_prefill = self.scheduler.schedule()
    
    # 2. 执行：运行模型生成 token
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    
    # 3. 后处理：更新序列状态
    self.scheduler.postprocess(seqs, token_ids)
    
    # 4. 收集完成的序列
    outputs = [(seq.seq_id, seq.completion_token_ids) 
               for seq in seqs if seq.is_finished]
    
    return outputs, num_tokens
```

> 💡 **设计思想**：`step()` 方法将一次推理循环分为三步：调度→执行→后处理。这种流水线式设计让各阶段职责清晰，也便于单独优化每个阶段。

---

## 2.5 内存布局

### 2.5.1 KV Cache 内存结构

```mermaid
graph TB
    subgraph "KV Cache 张量"
        direction LR
        A["kv_cache[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]"]
    end
    
    subgraph "维度说明"
        B["2: Key 和 Value"]
        C["num_layers: 模型层数"]
        D["num_blocks: 总块数"]
        E["block_size: 每块 token 数 (256)"]
        F["num_kv_heads: KV 头数"]
        G["head_dim: 每头维度"]
    end
    
    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    A --> G
```

### 2.5.2 Block 分配示例

假设 `block_size=256`，一个 512 token 的序列：

```
序列 tokens: [t0, t1, t2, ..., t511]
            |-------- Block 0 --------|-------- Block 1 --------|
            [t0 ... t255]              [t256 ... t511]
            
Block Table: [block_id_0, block_id_1]
```

---

## 2.6 张量并行

### 2.6.1 并行策略

Nano-vLLM 使用 **张量并行（Tensor Parallelism）**：

```mermaid
graph LR
    subgraph "GPU 0"
        A0[Attention Head 0-3]
        M0[MLP 切片 0]
    end
    
    subgraph "GPU 1"
        A1[Attention Head 4-7]
        M1[MLP 切片 1]
    end
    
    Input[输入] --> A0
    Input --> A1
    A0 --> |AllReduce| Output[输出]
    A1 --> |AllReduce| Output
```

### 2.6.2 进程通信

多 GPU 通过共享内存进行通信：

```mermaid
sequenceDiagram
    participant R0 as Rank 0 (主进程)
    participant SHM as 共享内存
    participant R1 as Rank 1
    participant R2 as Rank 2
    
    R0->>SHM: 写入 (方法名, 参数)
    R0->>R1: event.set()
    R0->>R2: event.set()
    
    par 并行执行
        R0->>R0: 执行方法
        R1->>SHM: 读取数据
        R1->>R1: 执行方法
        R2->>SHM: 读取数据
        R2->>R2: 执行方法
    end
    
    Note over R0,R2: NCCL AllReduce 同步
```

---

## 2.7 本章小结

本章我们学习了：

1. **整体架构**：从用户接口到底层模型的完整调用链
2. **核心概念**：
   - Prefill/Decode 两阶段推理
   - KV Cache 缓存机制
   - Continuous Batching 动态批处理
   - Prefix Caching 前缀共享
3. **模块职责**：各模块的核心功能和依赖关系
4. **数据流**：一次推理请求的完整处理流程
5. **内存布局**：KV Cache 的存储结构
6. **张量并行**：多 GPU 协同工作方式

---

**下一章** → [03 配置与采样参数](03_config_and_params.md)
