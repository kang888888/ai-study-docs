# AI大模型学习知识图谱系统

这是一个基于 React + Vite + ECharts 构建的交互式知识图谱可视化系统，用于展示AI大模型学习的完整知识体系。

## ✨ 核心特性

- 🎨 **交互式知识图谱**: 基于 ECharts 的径向树状图展示
- 📄 **完整技术文档**: 
  - ✅ React组件方式（已重构BERT、ChatGLM等核心节点）
  - ✅ iframe方式（兼容100+其他技术节点）
- 🔍 **多层级导航**: 支持从总览到具体技术的多层级浏览
- 📱 **响应式设计**: 适配桌面和移动设备
- 🎯 **侧边栏详情**: 点击节点查看详细技术文档
  - ✅ 数学公式渲染（KaTeX）
  - ✅ 代码高亮（SyntaxHighlighter）
  - ✅ 图片放大查看
  - ✅ 完整内容展示
- 🌈 **美观UI**: 现代化的深色主题设计
- ⚡ **极速开发**: Vite热更新，开发体验极佳
- 🔄 **渐进式重构**: 支持逐步将HTML内容重构为React组件

## 🛠️ 技术栈

- React 18 - 现代化UI框架
- Vite 5 - 极速构建工具
- ECharts 5 - 数据可视化
- 原生CSS - 轻量级样式

## 快速开始

### 安装依赖

```bash
npm install
```

### 开发模式

```bash
npm run dev
```

### 构建生产版本

```bash
npm run build
```

### 预览生产构建

```bash
npm run preview
```

## 项目结构

```
ai-study/
├── src/
│   ├── components/        # React组件
│   │   ├── KnowledgeGraph.jsx    # 知识图谱主组件
│   │   ├── Sidebar.jsx           # 侧边栏组件
│   │   └── ControlBar.jsx        # 控制栏组件
│   ├── data/             # 数据文件
│   │   ├── hierarchy.js          # 知识层级数据
│   │   ├── techPages.js          # 技术页面映射
│   │   └── descriptions.js       # 节点描述
│   ├── styles/           # 样式文件
│   │   └── index.css
│   ├── App.jsx           # 主应用组件
│   └── main.jsx          # 入口文件
├── index.html
├── package.json
└── vite.config.js
```

## 📊 知识体系

系统涵盖7大核心模块，100+技术节点，每个技术节点都有详细的HTML文档：

1. **AI基础理论** (30+ HTML文档)
   - 神经网络架构: MLP, CNN, ResNet, Transformer, BERT, LLaMA, ChatGLM等
   - 生成式架构: Diffusion, VAE, GAN等

2. **数据工程** (7 HTML文档)
   - 数据收集、清洗、转换、增强
   - 质量评估、数据集管理

3. **大模型核心技术** (50+ HTML文档)
   - 微调: LoRA, QLoRA, DPO, RLHF, SFT等
   - 量化: GPTQ, AWQ, GGUF, HQQ等
   - 训练: 分布式训练、并行技术、ZeRO优化
   - 推理: vLLM, TensorRT-LLM, FlashAttention, KV Cache等
   - 强化学习: PPO, TRPO, 对齐技术

4. **应用开发与框架** (10 HTML文档)
   - RAG系统、AI智能体、向量数据库
   - HuggingFace工具链（Transformers, Datasets, Tokenizers等）

5. **评估与安全** (3 HTML文档)
   - 模型评估、安全防御

6. **部署与运维** (3 HTML文档)
   - LLMOps、性能分析

7. **AI基础设施** (4 HTML文档)
   - 硬件集群、AI编译器、国产化适配

## 🎯 核心功能

### 1. 交互式知识图谱
- 径向树状图展示完整知识体系
- 支持缩放、拖拽、旋转
- 颜色编码区分不同模块
- 悬停高亮显示节点路径

### 2. 详细技术文档（⭐核心特性）

#### React组件方式（优先）
- **已重构节点**: BERT、ChatGLM等核心节点
- **完整功能**: 
  - ✅ 数学公式渲染（react-katex）
  - ✅ 代码高亮（react-syntax-highlighter）
  - ✅ 图片放大查看
  - ✅ 响应式布局
  - ✅ 更好的性能

#### iframe方式（兼容）
- **100+技术节点**: 其他节点的HTML文档
- **完整内容保留**: 
  - ✅ 数学公式（MathJax渲染）
  - ✅ 代码示例（Prism.js高亮）
  - ✅ 架构图解（可放大查看）
  - ✅ 核心概念、特点、应用场景

### 3. 智能侧边栏
- 自动识别节点类型
- **优先使用React组件**（已重构的节点）
- **兼容iframe方式**（未重构的节点）
- 分类节点：显示节点信息和子节点列表

### 4. 控制面板
- 重新布局：重置图谱
- 展开全部：查看完整知识体系
- 折叠全部：聚焦主要模块

## 📖 文档

- [启动说明.md](./启动说明.md) - 详细启动指南
- [使用指南.md](./使用指南.md) - 完整使用教程
- [项目说明.md](./项目说明.md) - 技术架构文档
- [重构完成说明.md](./重构完成说明.md) - 重构总结报告

## 🔧 开发说明

### 项目结构
- `src/data/` - 数据层（知识层级、描述、HTML页面映射）
- `src/components/` - React组件
- `src/utils/` - 工具函数（颜色计算等）
- `src/styles/` - 全局样式

### 添加新技术节点
1. 创建HTML文档
2. 在 `src/data/techPages.js` 添加映射
3. 在 `src/data/hierarchy.js` 添加节点数据

### 技术特点
- 组件化架构，易于维护
- 数据与视图分离
- iframe加载HTML，保留所有原有功能
- 响应式设计，适配多种设备

## 🌟 与原HTML的区别

| 特性 | 原HTML | 重构后 |
|------|--------|--------|
| 文件结构 | 单文件1200+行 | 模块化组件 |
| HTML内容 | 内嵌 | iframe加载 |
| 维护性 | ❌ 困难 | ✅ 简单 |
| 开发体验 | ❌ 无热更新 | ✅ Vite热更新 |
| 扩展性 | ❌ 低 | ✅ 高 |

## 📝 License

MIT
