// 知识图谱层级数据结构
// 按照学习路径重组，层级深度统一为3-4层
export const hierarchy = {
  name: 'AI大模型学习',
  value: '从理论、数据到算法、应用、安全、运维与算力的全栈知识地图',
  children: [
    {
      name: 'AI基础理论',
      tag: 'Theory',
      value: '神经网络与深度学习的基础概念与架构。',
      children: [
        {
          name: '核心概念',
          tag: 'Concepts',
          value: '梯度、损失函数、反向传播、优化器、激活函数、正则化、残差链接、位置编码、归一化等核心概念。',
          children: [
            { name: '梯度', tag: 'Gradient', value: '损失函数对参数的偏导数,指示参数更新的方向。' },
            { name: '损失函数', tag: 'Loss', value: '衡量模型预测与真实值差异的函数,如交叉熵、MSE等。' },
            { name: '反向传播', tag: 'Backpropagation', value: '通过链式法则计算梯度,从输出层向输入层传播。' },
            { name: '优化器', tag: 'Optimizer', value: '更新模型参数的算法,如SGD、Adam、AdamW等。' },
            { name: '激活函数', tag: 'Activation', value: '引入非线性,如ReLU、Sigmoid、Tanh、GELU等。' },
            { name: '正则化', tag: 'Regularization', value: '防止过拟合的技术,包括L1/L2、Dropout、权重衰减等。' },
            { name: '残差链接', tag: 'Residual', value: '跳跃连接,解决梯度消失,实现恒等映射。' },
            { name: '位置编码', tag: 'Position', value: '为序列数据添加位置信息,包括绝对和相对位置编码。' },
            { name: 'RoPE', tag: 'RoPE', value: '旋转位置编码,通过复数旋转实现相对位置编码,支持外推。' },
            { name: 'ALiBi', tag: 'ALiBi', value: '注意力线性偏置,通过线性偏置实现位置编码,无需额外参数。' },
            { name: 'GQA', tag: 'GQA', value: '分组查询注意力,减少KV缓存,提升推理效率。' },
            { name: 'FlashAttention-3', tag: 'FlashAttn3', value: '第三代FlashAttention,进一步优化IO效率与计算性能。' },
            { name: '归一化', tag: 'Normalization', value: 'BatchNorm、LayerNorm、GroupNorm等归一化技术。' }
          ]
        },
        {
          name: '数学函数',
          tag: 'MathFunctions',
          value: '深度学习中常用的数学函数及其数学表达,包括激活函数、损失函数、归一化函数、距离函数等。',
          children: [
            { name: 'ReLU', tag: 'ReLU', value: 'ReLU激活函数: f(x) = max(0, x),最常用的激活函数。' },
            { name: 'Sigmoid', tag: 'Sigmoid', value: 'Sigmoid函数: σ(x) = 1/(1+e^(-x)),输出范围(0,1)。' },
            { name: 'Tanh', tag: 'Tanh', value: 'Tanh函数: tanh(x) = (e^x-e^(-x))/(e^x+e^(-x)),输出范围(-1,1)。' },
            { name: 'GELU', tag: 'GELU', value: 'GELU激活函数: x·Φ(x),Transformer中常用。' },
            { name: 'Swish', tag: 'Swish', value: 'Swish/SiLU激活函数: x·σ(x),自门控激活函数。' },
            { name: 'SwiGLU', tag: 'SwiGLU', value: 'SwiGLU激活函数: Swish门控线性单元,LLaMA核心激活函数。' },
            { name: 'LeakyReLU', tag: 'LeakyReLU', value: 'Leaky ReLU: 解决ReLU死神经元问题。' },
            { name: 'ELU', tag: 'ELU', value: 'ELU激活函数: 指数线性单元,负值部分平滑。' },
            { name: 'Mish', tag: 'Mish', value: 'Mish激活函数: x·tanh(softplus(x)),自正则化。' },
            { name: 'Softmax', tag: 'Softmax', value: 'Softmax函数: 将logits转换为概率分布。' },
            { name: 'Logit Scaling', tag: 'LogitScale', value: 'Logit缩放技术,用于多模态模型中的温度参数调节。' },
            { name: '交叉熵损失', tag: 'CrossEntropy', value: '交叉熵损失函数: 用于分类任务。' },
            { name: 'MSE损失', tag: 'MSE', value: '均方误差损失: 用于回归任务。' },
            { name: '余弦相似度', tag: 'CosineSimilarity', value: '余弦相似度: 衡量向量间夹角。' }
          ]
        },
        {
          name: '优化理论',
          tag: 'Optimization',
          value: '梯度下降、优化器原理、学习率调度等优化理论基础。',
          children: [
            { name: '梯度下降', tag: 'GradientDescent', value: '梯度下降算法原理、批量梯度下降、随机梯度下降、小批量梯度下降。' },
            { name: '优化器原理', tag: 'OptimizerTheory', value: 'SGD、Momentum、Adam、AdamW等优化器的数学原理与适用场景。' },
            { name: 'SAM', tag: 'SAM', value: '锐度感知最小化,通过同时优化损失值与损失景观的平坦度提升泛化能力。' },
            { name: '二阶优化算法', tag: 'SecondOrder', value: '牛顿法、拟牛顿法、自然梯度等二阶优化方法。' },
            { name: '学习率调度', tag: 'LRScheduler', value: '学习率衰减策略、Warmup、Cosine Annealing等调度方法。' }
          ]
        }
      ]
    },
    {
      name: '网络架构',
      tag: 'Architecture',
      value: '基础、进阶、生成式架构及典型大模型实例。',
      children: [
        {
          name: '经典架构',
          tag: 'Basic',
          value: 'MLP、CNN、ResNet、YOLO、RNN 等经典结构。',
          children: [
            { name: 'MLP', tag: 'MLP', value: '多层感知机,前馈神经网络基础。' },
            { name: 'CNN', tag: 'CNN', value: '卷积神经网络,擅长图像处理。' },
            { name: 'ResNet', tag: 'ResNet', value: '残差网络,缓解深层退化。' },
            { name: 'YOLO', tag: 'YOLO', value: '实时目标检测网络。' },
            { name: 'RNN', tag: 'RNN', value: '循环神经网络,处理序列数据。' }
          ]
        },
        {
          name: 'LLM架构',
          tag: 'LLM-Core',
          value: 'Transformer、LSTM/GRU、BERT、MoE 等架构。',
          children: [
            { name: 'Transformer', tag: 'Transformer', value: '自注意力架构,LLM 核心基础。' },
            { name: 'LSTM', tag: 'LSTM', value: '长短期记忆网络,缓解长序列梯度问题。' },
            { name: 'GRU', tag: 'GRU', value: '门控循环单元,LSTM 轻量变体。' },
            { name: 'BERT', tag: 'BERT', value: '双向 Transformer 编码器,预训练基础。' },
            { name: 'MoE', tag: 'MoE', value: '专家混合模型,通过稀疏路由提升容量。' },
            { name: 'Mixture of Depths', tag: 'MoD', value: '深度混合模型,动态分配计算资源到不同token。' },
            { name: 'DeepSeek-V3', tag: 'DeepSeekV3', value: 'DeepSeek V3架构,高效长上下文处理。' },
            { name: 'Llama-3', tag: 'Llama3', value: 'Llama-3架构改进,包括GQA、RoPE等优化。' },
            { name: 'Mamba', tag: 'Mamba', value: '线性时序模块,兼具效率与长依赖。' },
            { name: 'Miras', tag: 'Miras', value: '通用架构设计框架,基于关联记忆与注意力偏差。' },
            { name: 'Titans', tag: 'Titans', value: '仿生记忆架构,融合短期/长期记忆,突破200万Token上下文。' },
            { name: 'RWKV', tag: 'RWKV', value: 'RNN-Transformer 混合架构。' },
            { name: 'ViT', tag: 'ViT', value: '视觉 Transformer,处理图像补丁序列。' },
            { name: 'CLIP', tag: 'CLIP', value: '图文对齐模型,联合嵌入空间。' },
            { name: 'DSA', tag: 'DSA', value: 'DeepSeek稀疏注意力,显著降低长文本场景下的计算复杂度,在API成本上实现50%以上的降幅。' }
          ]
        },
        {
          name: '生成式架构',
          tag: 'Generative',
          value: 'Diffusion、VAE、GAN、U-Net、GNN 等。',
          children: [
            { name: 'Diffusion', tag: 'Diffusion', value: '扩散模型,逐步去噪生成图像。' },
            { name: 'VAE', tag: 'VAE', value: '变分自编码器,概率生成模型。' },
            { name: 'GAN', tag: 'GAN', value: '生成对抗网络,生成逼真样本。' },
            { name: 'U-Net', tag: 'U-Net', value: '编码器-解码器结构,用于分割/生成。' },
            { name: 'GNN', tag: 'GNN', value: '图神经网络,处理图结构数据。' },
            { name: 'DQN', tag: 'DQN', value: '深度 Q 网络,强化学习价值网络。' },
            { name: 'DBN', tag: 'DBN', value: '深度置信网络,由多层 RBM 组成。' }
          ]
        },
        {
          name: '模型实例',
          tag: 'Models',
          value: '具代表性的开源/商用大模型。',
          children: [
            { name: 'LLaMA', tag: 'LLaMA', value: 'Meta 开源大模型系列。' },
            { name: 'ChatGLM', tag: 'ChatGLM', value: '智谱 AI 中文对话大模型。' },
            { name: 'QWen', tag: 'QWen', value: '阿里通义千问系列大模型。' },
            { name: 'Moneta', tag: 'Moneta', value: '基于Miras框架的高效关联记忆架构。' },
            { name: 'Yaad', tag: 'Yaad', value: '基于Miras框架的优化注意力偏差模型。' },
            { name: 'Memora', tag: 'Memora', value: '基于Miras框架的长期记忆管理模型。' }
          ]
        },
        {
          name: '多模态架构',
          tag: 'Multimodal',
          value: '视觉、视频、语音等多模态大模型架构。',
          children: [
            {
              name: '视觉模型',
              tag: 'Vision',
              value: '图像理解与生成的视觉大模型。',
              children: [
                { name: 'SigLIP', tag: 'SigLIP', value: 'Sigmoid损失优化的图文对齐模型。' },
                { name: 'LLaVA', tag: 'LLaVA', value: '大型语言和视觉助手,结合视觉编码器与LLM。' },
                { name: 'Qwen-VL', tag: 'QwenVL', value: '通义千问视觉语言模型,支持多图输入。' }
              ]
            },
            {
              name: '视频模型',
              tag: 'Video',
              value: '视频理解与生成的视频大模型。',
              children: [
                { name: 'DiT', tag: 'DiT', value: 'Diffusion Transformer,用于视频生成的扩散Transformer架构。' },
                { name: 'Space-Time Latent Patch', tag: 'STLP', value: '时空潜在补丁,高效视频表示学习。' }
              ]
            },
            {
              name: '语音模型',
              tag: 'Audio',
              value: '语音识别、合成与理解的语音大模型。',
              children: [
                { name: 'Whisper', tag: 'Whisper', value: 'OpenAI多语言语音识别模型。' },
                { name: 'AudioLM', tag: 'AudioLM', value: '语言模型方法生成高质量音频。' },
                { name: 'GPT-4o Omni', tag: 'GPT4o', value: 'GPT-4 Omni多模态机制,统一处理文本、图像、音频。' }
              ]
            }
          ]
        },
        {
          name: '极致长文本',
          tag: 'LongContext',
          value: '支持超长上下文的高效架构与技术。',
          children: [
            { name: 'StreamingLLM', tag: 'StreamingLLM', value: '流式LLM,通过注意力池化实现无限长度处理。' },
            { name: 'Activation Beacon', tag: 'ActivationBeacon', value: '激活信标,压缩历史激活实现长上下文。' },
            { name: 'Ring Attention', tag: 'RingAttn', value: '环形注意力,通过环形通信实现超长序列并行处理。' }
          ]
        },
        {
          name: '拓扑架构革新',
          tag: 'Topology',
          value: '超深网络架构的拓扑连接优化技术。',
          children: [
            { name: 'mHC', tag: 'mHC', value: '流形约束超连接,通过流形约束恢复恒等映射特性,解决超深网络信号发散问题,仅增加约6.7%训练开销即可显著提升性能。' }
          ]
        }
      ]
    },
    {
      name: '数据工程',
      tag: 'Data',
      value: '数据采集、清洗、增强、质量评估与管理能力。',
      children: [
        {
          name: '数据收集',
          tag: 'Collect',
          value: '公开数据、抓取、人工标注与合成策略。',
          children: [
            { name: '公开数据集', tag: 'Public', value: 'CommonCrawl、The Pile、C4等公开数据集的使用。' },
            { name: '数据抓取', tag: 'Scraping', value: '网络爬虫、API调用、数据清洗与去重。' },
            { name: '人工标注', tag: 'Annotation', value: '标注流程、质量控制、众包平台与工具链。' },
            {
              name: '合成数据',
              tag: 'Synthetic',
              value: '数据生成、增强、模拟与合成策略。',
              children: [
                { name: 'Self-Instruct', tag: 'SelfInstruct', value: '自我指令生成,模型自主生成训练数据。' },
                { name: 'Evol-Instruct', tag: 'EvolInstruct', value: '指令演化,通过迭代演化生成复杂指令。' },
                { name: '算术合成数据', tag: 'MathSynthetic', value: '数学问题与解答的合成数据生成。' },
                { name: '代码合成数据', tag: 'CodeSynthetic', value: '代码示例与解释的合成数据生成。' },
                { name: '高质量合成数据流', tag: 'HighQualitySynthetic', value: '利用模型生成的教科书级数据解决语料瓶颈,通过自动评估过滤机制确保合成数据质量,降低对人类标注的依赖。' }
              ]
            }
          ]
        },
        {
          name: '数据处理',
          tag: 'Process',
          value: '数据清洗、格式转换、增强三段式流水线。',
          children: [
            { name: '数据清洗', tag: 'Clean', value: '去重、过滤、标准化与数据验证流程。' },
            { name: '格式转换', tag: 'Format', value: 'JSONL/Parquet/ShareGPT 等格式互转。' },
            { name: '数据增强', tag: 'Augment', value: '回译、同义替换、模板填充、排序等技巧。' }
          ]
        },
        {
          name: '质量保证',
          tag: 'QA',
          value: '数据质量评估与数据集资产治理。',
          children: [
            { name: '质量评估', tag: 'Quality', value: '准确性、相关性、多样性与一致性度量。' },
            { name: '数据管理', tag: 'Dataset', value: '版本化、元数据、权限与生命周期管理。' }
          ]
        },
        {
          name: '数据治理',
          tag: 'Governance',
          value: '数据隐私、偏见消除与多语言平衡等治理技术。',
          children: [
            { name: 'PII脱敏', tag: 'PII', value: '个人身份信息脱敏,保护用户隐私。' },
            { name: '去偏见', tag: 'Debias', value: '数据去偏见技术,消除性别、种族等偏见。' },
            { name: '多语言平衡', tag: 'Multilang', value: '多语言数据比例平衡,确保模型公平性。' }
          ]
        }
      ]
    },
    {
      name: '模型训练',
      tag: 'Training',
      value: '分布式并行、混合精度、ZeRO、梯度累积等训练技术。',
      children: [
        {
          name: '并行训练',
          tag: 'Parallel',
          value: '数据并行、模型并行、流水线并行等分布式训练策略。',
          children: [
            { name: '数据并行', tag: 'DP', value: 'AllReduce、梯度同步、数据分片原理与实践。' },
            { name: '模型并行', tag: 'MP', value: '张量并行、列并行、行并行、专家并行拆分方法。' },
            { name: '流水线并行', tag: 'PP', value: '流水线分段、气泡抑制、调度策略。' },
            { name: 'Context Parallelism', tag: 'CP', value: '上下文并行,将长序列分片到不同设备并行处理。' },
            { name: 'Expert Parallelism', tag: 'EP', value: '专家并行,MoE模型中专家路由到不同设备。' },
            { name: '通信优化', tag: 'Comm', value: '梯度压缩、异步更新、通信与计算重叠。' }
          ]
        },
        {
          name: '训练优化',
          tag: 'Optimize',
          value: 'ZeRO、梯度累积、混合精度等优化技术。',
          children: [
            { name: 'ZeRO优化器', tag: 'ZeRO', value: '参数/梯度/优化器三路分片。' },
            { name: '梯度累积与检查点', tag: 'Grad', value: '小显存累积与激活重计算。' },
            { name: '混合精度训练', tag: 'AMP', value: 'FP16/BF16 + Grad Scaling。' }
          ]
        },
        {
          name: '训练稳定性',
          tag: 'Stability',
          value: 'Loss Spike处理、权重衰减诊断、Epsilon预测等训练稳定性技术。',
          children: [
            { name: 'Loss Spike处理', tag: 'LossSpike', value: '训练过程中的损失突增检测与恢复策略。' },
            { name: '权重衰减诊断', tag: 'WeightDecay', value: '权重衰减参数调优与诊断方法。' },
            { name: 'Epsilon预测', tag: 'EpsilonPred', value: '优化器epsilon参数的自适应预测与调整。' }
          ]
        },
        {
          name: '分布式训练协同优化',
          tag: 'DistributedOpt',
          value: '软硬一体协同优化技术,包括KV缓存优化、多Token预测、FP8混合精度等。',
          children: [
            { name: 'MLA', tag: 'MLA', value: '多头潜在注意力,通过低秩压缩技术大幅削减KV Cache显存占用。' },
            { name: 'MTP', tag: 'MTP', value: '多Token预测,在训练时预测未来多个Token,提升推理时的并行度,加速1.8倍。' },
            { name: 'FP8混合精度训练', tag: 'FP8', value: 'FP8混合精度训练,在有限资源下实现超大规模模型训练,显著降低显存占用。' }
          ]
        },
        {
          name: 'Minimind实践',
          tag: 'Project',
          value: '从零复现 GPT 的工程项目。',
          children: [
            { name: '项目架构', tag: 'Arch', value: '代码结构、模块设计、依赖管理。' },
            { name: '训练流程', tag: 'Pipeline', value: '数据准备、模型训练、评估与部署流程。' },
            { name: '工程实践', tag: 'Practice', value: '调试技巧、性能分析、最佳实践。' },
            { name: '性能优化', tag: 'Perf', value: '训练加速、内存优化、收敛策略。' }
          ]
        }
      ]
    },
    {
      name: '模型优化',
      tag: 'Optimization',
      value: '微调、量化、对齐、合并等模型优化技术。',
      children: [
        {
          name: '模型微调',
          tag: 'Finetune',
          value: 'SFT、LoRA、QLoRA、PEFT等微调方法与Axolotl、Unsloth等微调工具。',
          children: [
            { name: 'SFT', tag: 'SFT', value: '监督微调流程与数据要点。' },
            { name: 'LoRA', tag: 'LoRA', value: '低秩适配微调,冻结主干。' },
            { name: 'LoRA+', tag: 'LoRAPlus', value: 'LoRA改进版,不同层使用不同学习率。' },
            { name: 'DoRA', tag: 'DoRA', value: '权重分解低秩适配,将权重分解为幅度与方向分别优化。' },
            { name: 'LongLoRA', tag: 'LongLoRA', value: '长上下文LoRA,通过移位注意力实现长上下文微调。' },
            { name: 'QLoRA', tag: 'QLoRA', value: '4bit 量化 + LoRA,节省显存。' },
            { name: 'PEFT', tag: 'PEFT', value: '参数高效微调方法总览。' },
            { name: 'Axolotl', tag: 'Tool', value: '多模型多框架微调工具。' },
            { name: 'Unsloth', tag: 'Tool', value: '轻量高效微调套件。' }
          ]
        },
        {
          name: '模型量化',
          tag: 'Quant',
          value: '量化基础、方法、格式与引擎。包括PTQ/QAT、GPTQ、AWQ、GGUF、ExLlamaV2等量化方案。',
          children: [
            { name: '量化基础', tag: 'Basics', value: 'PTQ/QAT、位宽选择、误差指标。' },
            { name: 'GPTQ', tag: 'GPTQ', value: '梯度/Hessian 感知的 4bit 权重量化。' },
            { name: 'AWQ', tag: 'AWQ', value: 'Activation-aware 权重量化方案。' },
            { name: 'SmoothQuant', tag: 'Smooth', value: '激活/权重同步缩放,便于推理部署。' },
            { name: 'PTQ', tag: 'PTQ', value: '后训练量化流程与校准策略。' },
            { name: 'HQQ', tag: 'HQQ', value: '半二次优化,零校准极速量化。' },
            { name: 'GGUF', tag: 'GGUF', value: 'llama.cpp 通用格式,整合多种位宽。' },
            { name: 'ExLlamaV2', tag: 'Runtime', value: 'INT4 推理引擎,结合分页 KV Cache。' }
          ]
        },
        {
          name: '模型对齐',
          tag: 'Alignment',
          value: '强化学习基础、PPO、TRPO、RLHF、DPO、ORPO、RLAIF等对齐方法与CoT、逻辑推理优化等推理增强技术。',
          children: [
            { name: '强化学习基础', tag: 'RL-Base', value: 'MDP、策略、价值函数与 Bellman 方程。' },
            { name: 'PPO', tag: 'PPO', value: 'Clip-Objective 策略梯度,稳定高效。' },
            { name: 'TRPO', tag: 'TRPO', value: '信赖域约束,确保策略更新安全。' },
            { name: 'RLHF', tag: 'Align', value: '人类偏好反馈 + 奖励模型 + PPO。' },
            { name: 'DPO', tag: 'Align', value: '无需奖励模型的直接偏好优化。' },
            { name: 'SimPO', tag: 'SimPO', value: '简单偏好优化,简化DPO无需参考模型。' },
            { name: 'Iterative DPO', tag: 'IterDPO', value: '迭代DPO,通过多轮迭代提升对齐效果。' },
            { name: 'ORPO', tag: 'Align', value: 'Odds Ratio 约束的偏好优化。' },
            { name: 'RLAIF', tag: 'Align', value: 'AI 反馈替代人工偏好。' },
            { name: 'GRPO', tag: 'GRPO', value: '组相对策略优化,无评论家架构的强化学习,通过组内相对评分实现偏好对齐,极大节省显存,支持自我演进和中间阶段模型。' },
            { name: 'CoT', tag: 'Reason', value: '链式思维 + 搜索/PRM 的推理强化。' },
            { name: '逻辑推理优化', tag: 'Reason', value: '逻辑奖励模型、自博弈、过程评估。' }
          ]
        },
        {
          name: '推理增强',
          tag: 'Reasoning',
          value: '提升模型在数学、代码与逻辑领域的深度思考能力。',
          children: [
            { name: 'PRM', tag: 'PRM', value: '过程奖励模型,对思维链中的每一步进行打分,而非仅对最终结果评分。' },
            { name: 'MCTS', tag: 'MCTS', value: '蒙特卡洛树搜索,在推理空间内进行启发式搜索,结合Value Head寻找最优路径。' },
            { name: 'Self-Correction', tag: 'SelfCorr', value: '自我纠错,模型通过生成多个回答并由自身判别,实现自我进化。' }
          ]
        },
        {
          name: '模型合并',
          tag: 'Merge',
          value: '模型权重合并与融合技术，无需重新训练即可组合多个模型优势。',
          children: [
            { name: '线性合并', tag: 'LinearMerge', value: '简单的加权平均，适合相似模型的合并。' },
            { name: '任务向量合并', tag: 'TaskVector', value: '基于任务向量的合并，适合任务特化模型。' },
            { name: '分层合并', tag: 'LayerWise', value: '对不同层使用不同策略，精细控制合并过程。' },
            { name: '参数空间合并', tag: 'ParamSpace', value: '直接对模型参数进行加权平均或其他算术操作。' },
            { name: '功能锚点合并', tag: 'FuncAnchor', value: '利用功能锚点捕捉任务特定功能变化。' },
            { name: 'MergeKit', tag: 'Tool', value: '模型合并工具，支持多种合并策略。' }
          ]
        },
        {
          name: '模型安全',
          tag: 'Security',
          value: '模型安全、去审查化、内容过滤等安全技术。',
          children: [
            { name: '去审查化', tag: 'Uncensored', value: '模型去审查化技术。' },
            {
              name: '防御技术',
              tag: 'Defense',
              value: '提示词注入防御、对抗性攻击测试等安全防御技术。',
              children: [
                { name: '提示词注入防御', tag: 'PromptInjection', value: '防御提示词注入攻击的技术与方法。' },
                { name: '对抗性攻击测试', tag: 'Adversarial', value: '对抗性样本生成与模型鲁棒性测试。' },
                { name: '红色对抗', tag: 'RedTeaming', value: '红色对抗测试,主动发现模型安全漏洞。' }
              ]
            },
            {
              name: '合规与伦理',
              tag: 'Compliance',
              value: '机器版权保护、水印技术等合规与伦理技术。',
              children: [
                { name: '机器版权保护', tag: 'Copyright', value: 'AI生成内容的版权保护与归属识别。' },
                { name: '水印技术', tag: 'Watermarking', value: '模型输出水印,用于内容溯源与防伪。' }
              ]
            }
          ]
        }
      ]
    },
    {
      name: '模型推理',
      tag: 'Inference',
      value: '推理流程、性能指标与系统要点。推理优化、引擎与文本生成技术。',
      children: [
        {
          name: '推理优化',
          tag: 'Optimize',
          value: '图/算子优化、量化、剪枝、缓存、注意力、加速等推理优化技术。',
          children: [
            { name: '图优化', tag: 'GraphOpt', value: '计算图优化、算子融合。' },
            { name: '量化推理', tag: 'QuantInf', value: '推理时量化、INT8/INT4推理。' },
            { name: '模型剪枝', tag: 'Pruning', value: '结构化/非结构化剪枝、稀疏化。' },
            { name: 'KV Cache', tag: 'Cache', value: '注意力键值缓存复用。' },
            { name: 'PagedAttention', tag: 'Cache', value: '分页注意力,长序 KV 管理。' },
            { name: 'FlashAttention', tag: 'Kernel', value: 'IO 感知注意力算子。' },
            { name: 'MLA', tag: 'MLA', value: '多头潜在注意力,通过低秩压缩技术大幅削减KV Cache显存占用,提升推理效率。' },
            { name: 'MTP', tag: 'MTP', value: '多Token预测,在训练时预测未来多个Token,提升推理时的并行度,加速1.8倍。' },
            { name: 'Speculative Decoding', tag: 'Speed', value: '草稿模型 + 验证模型加速。' },
            { name: 'Medusa', tag: 'Medusa', value: '多头草稿解码,并行生成多个候选token提升吞吐。' },
            { name: 'Lookahead Decoding', tag: 'Lookahead', value: '前瞻解码,通过n-gram匹配实现2-3倍加速。' }
          ]
        },
        {
          name: '端侧优化',
          tag: 'Edge',
          value: '移动端与边缘设备的模型优化与部署技术。',
          children: [
            { name: 'BitNet', tag: 'BitNet', value: '1.58-bit量化,极低比特量化技术。' },
            { name: 'W4A8量化', tag: 'W4A8', value: '权重4bit、激活8bit的混合量化方案。' },
            { name: 'Executive', tag: 'Executive', value: '移动端高效推理框架,专为边缘设备优化。' }
          ]
        },
        {
          name: '推理引擎',
          tag: 'Engines',
          value: 'TensorRT-LLM、vLLM 等推理引擎。',
          children: [
            { name: 'TensorRT-LLM', tag: 'Engine', value: 'NVIDIA TensorRT-LLM 部署。' },
            { name: 'vLLM', tag: 'Engine', value: '连续批处理 + PagedAttention 框架。' }
          ]
        },
        {
          name: '文本生成',
          tag: 'TextGen',
          value: '解码策略、提示工程、流式生成。',
          children: [
            { name: '解码策略', tag: 'Decoding', value: 'Greedy、Beam、Sampling、Top-k/p、温度控制。' },
            { name: '提示工程', tag: 'Prompt', value: '指令模板、少样本/思维链提示设计。' },
            { name: '流式生成', tag: 'Streaming', value: '逐 token 推流、交互体验优化。' }
          ]
        }
      ]
    },
    {
      name: '应用开发',
      tag: 'Application',
      value: '知识增强、智能体开发与开发框架。',
      children: [
        {
          name: '知识增强',
          tag: 'KG',
          value: 'RAG、知识图谱、向量数据库等知识增强技术。',
          children: [
            { name: 'RAG系统', tag: 'RAG', value: '检索、索引、重排、DSPy、RAG-Fusion。' },
            { name: 'GraphRAG', tag: 'GraphRAG', value: '知识图谱结合RAG,利用图结构增强检索与推理。' },
            { name: 'Long-Context RAG', tag: 'LongRAG', value: '长上下文RAG,处理超长文档与多轮对话。' },
            { name: '多向量检索', tag: 'MultiVector', value: '多向量检索,通过多个向量表示提升检索精度。' },
            { name: '知识图谱增强', tag: 'KG', value: '知识图谱增强、检索优化。' },
            { name: '向量数据库', tag: 'VectorDB', value: '向量检索基础原理与优化。Milvus、Pinecone、Weaviate、Qdrant 等产品与优化。' }
          ]
        },
        {
          name: '智能体开发',
          tag: 'Agent',
          value: 'AI智能体开发框架与技术。',
          children: [
            { name: 'AI智能体', tag: 'Agent', value: 'ReAct、LangGraph、smolagents、CrewAI 等框架。' },
            { name: '智能体框架', tag: 'AgentFramework', value: 'LangGraph、CrewAI、AutoGen 等多智能体协作框架。' },
            { name: '工具调用', tag: 'ToolCalling', value: 'Function Calling、Tool Use、API 调用等工具使用技术。' },
            { name: '多智能体系统', tag: 'MultiAgent', value: '多智能体协作、通信、协调与竞争机制。' },
            {
              name: 'Agent记忆体系',
              tag: 'AgentMemory',
              value: '智能体的记忆管理机制,包括层次化记忆、向量缓存等。',
              children: [
                { name: '层次化记忆', tag: 'HierMemory', value: '短期、中期、长期记忆的分层管理。' },
                { name: '向量数据库缓存', tag: 'VectorCache', value: '使用向量数据库缓存智能体历史交互。' },
                { name: '记忆刷新机制', tag: 'MemoryRefresh', value: '长短期记忆的刷新与遗忘策略。' }
              ]
            }
          ]
        },
        {
          name: '开发框架',
          tag: 'Frameworks',
          value: 'LangChain、HuggingFace Transformers、Datasets、Tokenizers、Accelerate、Hub等开发框架与工具。',
          children: [
            { name: 'LangChain框架', tag: 'LangChain', value: '模型、Prompt、Chain、Agent、Callback。' },
            { name: 'Transformers', tag: 'HF-Transformers', value: 'Pipeline、模型加载、Encoder-Decoder 架构。' },
            { name: 'Datasets', tag: 'HF-Datasets', value: '数据加载、处理、创建与切分。' },
            { name: 'Tokenizers', tag: 'HF-Tokenizers', value: '分词器训练、编码/解码、批处理。' },
            { name: 'Accelerate', tag: 'HF-Accelerate', value: '分布式训练、混合精度、设备管理。' },
            { name: 'HuggingFace Hub', tag: 'HF-Hub', value: '模型/数据集管理、Spaces 部署。' }
          ]
        }
      ]
    },
    {
      name: '模型评估',
      tag: 'Evaluation',
      value: '模型能力评估与性能分析。',
      children: [
        {
          name: '指标体系',
          tag: 'Metrics',
          value: '分类指标、生成指标、任务特定指标等评估指标体系。',
          children: [
            { name: '分类指标', tag: 'ClassMetrics', value: 'Accuracy、Precision、Recall、F1、AUC-ROC等分类任务指标。' },
            { name: '生成指标', tag: 'GenMetrics', value: 'BLEU、ROUGE、METEOR、BERTScore等文本生成指标。' },
            { name: '任务特定指标', tag: 'TaskMetrics', value: '问答、代码生成、RAG、数学推理等任务特定指标。' }
          ]
        },
        {
          name: '评估方法',
          tag: 'Methods',
          value: '自动评估、人工评估等模型评估方法。',
          children: [
            { name: '自动评估', tag: 'AutoEval', value: '参考答案类、模型裁判、规则检查等自动评估方法。' },
            { name: '人工评估', tag: 'HumanEval', value: '单点评估、对比评估、多维面板等人工评估方法。' }
          ]
        },
        {
          name: '基准测试',
          tag: 'Benchmarks',
          value: 'GLUE、MMLU、HellaSwag等标准评估基准。',
          children: [
            { name: '语言理解基准', tag: 'NLU', value: 'GLUE、SuperGLUE等自然语言理解基准。' },
            { name: '知识推理基准', tag: 'Knowledge', value: 'MMLU、HellaSwag等多学科知识与推理基准。' },
            { name: '代码生成基准', tag: 'Code', value: 'HumanEval、MBPP等代码生成与执行基准。' }
          ]
        },
        {
          name: '评估工具',
          tag: 'Tools',
          value: 'LM Evaluation Harness等评估工具与自动化框架。',
          children: [
            { name: 'LM Evaluation Harness', tag: 'LMEval', value: '覆盖主流基准的统一评估框架。' },
            { name: '评估工具链', tag: 'EvalTools', value: '评估流程自动化、结果分析与报告生成工具。' }
          ]
        }
      ]
    },
    {
      name: '基础设施与运维',
      tag: 'InfraOps',
      value: '从硬件基础设施到模型部署运维的全生命周期管理能力。',
      children: [
        { name: '硬件与集群', tag: 'Hardware', value: '加速卡、集群拓扑、网络通信与算力规划。' },
        { name: 'AI编译器', tag: 'Compiler', value: '编译原理、前后端技术、主流框架与案例。' },
        { name: 'LLMOps', tag: 'LLMOps', value: '部署策略、K8s、版本治理、监控与实践案例。' },
        { name: '性能分析', tag: 'Perf', value: 'PyTorch Profiler、Nsight、优化路线与案例。' },
        {
          name: '国产适配',
          tag: 'Domestic',
          value: '国产AI芯片与框架的适配与优化技术。',
          children: [
            { name: '昇腾CANN', tag: 'Ascend', value: '华为昇腾CANN架构,NPU加速与模型适配。' },
            { name: '海光DCU', tag: 'Hygon', value: '海光DCU适配,GPU替代方案优化。' },
            { name: '摩尔线程MUSA', tag: 'MUSA', value: '摩尔线程MUSA框架,国产GPU生态适配。' }
          ]
        },
        {
          name: '算力优化',
          tag: 'ComputeOpt',
          value: '算力网络调度、异构计算并行等算力优化技术。',
          children: [
            { name: '算力网络调度', tag: 'ComputeSched', value: '分布式算力资源的智能调度与分配。' },
            { name: '异构计算并行', tag: 'HeteroParallel', value: 'CPU、GPU、NPU等异构设备的协同并行计算。' }
          ]
        }
      ]
    }
  ]
};
