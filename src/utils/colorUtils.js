// 颜色工具函数

// 根据tag和level获取节点颜色
export function getNodeColor(tag, level) {
  const colorMap = {
    // 基础理论相关 - 浅色系
    'Theory': '#7dd3fc',
    'Architecture': '#93c5fd',
    'Basic': '#93c5fd',
    'LLM-Core': '#60a5fa',
    'Generative': '#60a5fa',
    'Models': '#60a5fa',
    
    // 数据工程相关 - 浅色系
    'Data': '#6ee7b7',
    'Pipeline': '#6ee7b7',
    'Collect': '#a7f3d0',
    'Process': '#86efac',
    'Clean': '#86efac',
    'Format': '#6ee7b7',
    'Augment': '#6ee7b7',
    'QA': '#6ee7b7',
    'Quality': '#6ee7b7',
    'Dataset': '#6ee7b7',
    
    // 核心技术相关 - 浅色系
    'CoreTech': '#f9a8d4',
    'Finetune': '#f9a8d4',
    'SFT': '#fbcfe8',
    'RLHF': '#f9a8d4',
    'LoRA': '#f9a8d4',
    'QLoRA': '#f9a8d4',
    'DPO': '#f9a8d4',
    'ORPO': '#f9a8d4',
    'Quant': '#c4b5fd',
    'GPTQ': '#c4b5fd',
    'AWQ': '#c4b5fd',
    'Training': '#fcd34d',
    'Dist': '#fde68a',
    'DP': '#fde68a',
    'MP': '#fde68a',
    'PP': '#fde68a',
    'ZeRO': '#fde68a',
    'Inference': '#67e8f9',
    'RL': '#c4b5fd',
    'Advanced': '#5eead4',
    
    // 应用开发相关 - 浅色系
    'Application': '#fef08a',
    'App': '#fef08a',
    'VectorDB': '#fef08a',
    'LangChain': '#fef08a',
    'Agent': '#fef08a',
    'RAG': '#fef08a',
    'HF-Transformers': '#fef08a',
    'HF-Datasets': '#fef08a',
    'HF-Tokenizers': '#fef08a',
    'HF-Accelerate': '#fef08a',
    'HF-Hub': '#fef08a',
    
    // 评估与安全相关 - 浅色系
    'Evaluation': '#fca5a5',
    'Security': '#fca5a5',
    'Eval': '#fca5a5',
    'Defense': '#fca5a5',
    
    // 部署与运维相关 - 浅色系
    'Ops': '#a5b4fc',
    'OpsMap': '#a5b4fc',
    'LLMOps': '#a5b4fc',
    'Perf': '#a5b4fc',
    
    // 基础设施相关 - 浅色系
    'Infrastructure': '#cbd5e1',
    'InfraMap': '#cbd5e1',
    'Hardware': '#cbd5e1',
    'Compiler': '#cbd5e1',
    'Localization': '#cbd5e1'
  };
  
  // 如果找到tag对应的颜色，使用它；否则根据level返回默认颜色
  if (tag && colorMap[tag]) {
    return colorMap[tag];
  }
  
  // 默认颜色方案（按层级）- 浅色系
  const defaultColors = ['#7dd3fc', '#f9a8d4', '#fef08a', '#fde68a', '#6ee7b7', '#a5b4fc'];
  return defaultColors[level % defaultColors.length];
}

// 获取线条颜色（使用节点颜色的较浅版本，但保持较高可见性）
export function getLineColor(tag, level) {
  const nodeColor = getNodeColor(tag, level);
  // 将颜色转换为rgba，保持较高的不透明度以确保可见性
  if (nodeColor && nodeColor.startsWith('#')) {
    const r = parseInt(nodeColor.slice(1, 3), 16);
    const g = parseInt(nodeColor.slice(3, 5), 16);
    const b = parseInt(nodeColor.slice(5, 7), 16);
    // 使用较高的不透明度，确保线条清晰可见
    if (!isNaN(r) && !isNaN(g) && !isNaN(b)) {
      return `rgba(${r}, ${g}, ${b}, 0.75)`;
    }
  }
  // 默认返回灰色线条
  return 'rgba(148,163,184,0.75)';
}

// 为树形数据添加层级信息和样式
export function addLevelToTree(node, level = 0, parentTag = '') {
  node.level = level;
  
  // 为节点添加 label 配置，根据层级设置字体大小
  const fontSizeMap = { 0: 28, 1: 24, 2: 20, 3: 16, 4: 16 };
  node.label = {
    fontSize: fontSizeMap[level] || 12,
    fontWeight: level === 0 ? 'bold' : 'normal'
  };
  
  // 为节点添加线条颜色信息
  if (level === 0) {
    node.lineStyle = {
      color: 'rgba(56, 189, 248, 0.9)',
      width: 5,
      opacity: 0.9
    };
  } else if (level === 1) {
    node.lineStyle = {
      color: 'rgba(168, 85, 247, 0.85)',
      width: 4,
      opacity: 0.85
    };
  } else if (parentTag) {
    const parentLevel = level > 0 ? level - 1 : 0;
    const lineColor = getLineColor(parentTag, parentLevel);
    node.lineStyle = {
      color: lineColor,
      width: level === 2 ? 3.5 : 3,
      opacity: 0.8
    };
  }
  
  if (node.children && node.children.length > 0) {
    const currentTag = node.tag || parentTag;
    node.children.forEach(child => addLevelToTree(child, level + 1, currentTag));
  }
  return node;
}
