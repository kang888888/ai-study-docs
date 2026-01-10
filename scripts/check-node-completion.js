// 检查节点补全情况
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 要检查的节点列表
const nodesToCheck = {
  'AI基础理论': [
    'RoPE',
    'ALiBi',
    'GQA',
    'FlashAttention-3',
    'SwiGLU',
    'Logit Scaling',
    'SAM',
    '二阶优化算法'
  ],
  '网络架构': [
    'DeepSeek-V3',
    'Llama-3',
    'Mixture of Depths'
  ],
  '多模态架构': [
    'SigLIP',
    'LLaVA',
    'Qwen-VL'
  ],
  '推理增强': [
    'PRM',
    'MCTS',
    'Self-Correction'
  ],
  '模型微调': [
    'DoRA',
    'LoRA+',
    'LongLoRA'
  ],
  '模型对齐': [
    'SimPO',
    'Iterative DPO'
  ],
  '推理优化': [
    'Medusa',
    'Lookahead Decoding'
  ],
  'RAG系统': [
    'GraphRAG',
    'Long-Context RAG',
    '多向量检索'
  ],
  '并行训练': [
    'Context Parallelism',
    'Expert Parallelism'
  ],
  '端侧优化': [
    'BitNet',
    'W4A8量化'
  ],
  '数据工程': [
    'Self-Instruct',
    'Evol-Instruct',
    '算术合成数据',
    '代码合成数据'
  ],
  'Agent记忆': [
    '层次化记忆',
    '向量数据库缓存'
  ]
};

// 读取 hierarchy.js
const hierarchyPath = path.join(__dirname, '../src/data/hierarchy.js');
const hierarchyContent = fs.readFileSync(hierarchyPath, 'utf-8');

// 读取 knowledgeMap.js
const knowledgeMapPath = path.join(__dirname, '../src/data/knowledge/knowledgeMap.js');
const knowledgeMapContent = fs.readFileSync(knowledgeMapPath, 'utf-8');

// 知识文档目录
const knowledgeDir = path.join(__dirname, '../src/data/knowledge');

// 检查结果
const results = {
  total: 0,
  completed: 0,
  missing: [],
  details: {}
};

// 扁平化所有节点
const allNodes = [];
Object.values(nodesToCheck).forEach(category => {
  allNodes.push(...category);
});
results.total = allNodes.length;

// 检查每个节点
allNodes.forEach(nodeName => {
  const result = {
    nodeName,
    inHierarchy: false,
    hasJsonFile: false,
    inKnowledgeMap: false,
    isLeafNode: false
  };

  // 检查是否在 hierarchy.js 中
  const escapedName = nodeName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const hierarchyPattern = new RegExp(`name:\\s*['"]${escapedName}['"]`, 'g');
  if (hierarchyPattern.test(hierarchyContent)) {
    result.inHierarchy = true;
    
    // 使用更简单的方法：查找节点定义，检查是否在同一行或紧邻的几行内包含children
    // 节点定义格式通常是: { name: 'xxx', tag: 'xxx', value: 'xxx' } 或 { name: 'xxx', ..., children: [...] }
    const lines = hierarchyContent.split('\n');
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes(`name:`) && lines[i].includes(escapedName)) {
        // 检查这一行和接下来的几行是否包含children
        let nodeDef = lines[i];
        let j = i + 1;
        // 如果这一行以}结尾，说明节点定义在这一行就结束了
        if (lines[i].trim().endsWith('}')) {
          // 节点定义在这一行就结束了
          if (!nodeDef.includes('children:')) {
            result.isLeafNode = true;
          }
        } else {
          // 节点定义可能跨多行，检查接下来的几行
          while (j < lines.length && j < i + 5) {
            nodeDef += '\n' + lines[j];
            if (lines[j].trim().endsWith('}')) {
              break;
            }
            j++;
          }
          // 检查完整的节点定义中是否包含children
          if (!nodeDef.includes('children:')) {
            result.isLeafNode = true;
          } else {
            // 检查children是否是空数组
            const childrenMatch = nodeDef.match(/children:\s*\[\s*\]/);
            if (childrenMatch) {
              result.isLeafNode = true;
            }
          }
        }
        break;
      }
    }
  }

  // 检查是否有 JSON 文件
  const jsonFileName = `${nodeName}.json`;
  const jsonFilePath = path.join(knowledgeDir, jsonFileName);
  if (fs.existsSync(jsonFilePath)) {
    result.hasJsonFile = true;
  }

  // 检查是否在 knowledgeMap.js 中
  const mapPattern = new RegExp(`['"]${nodeName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}['"]\\s*:`, 'g');
  if (mapPattern.test(knowledgeMapContent)) {
    result.inKnowledgeMap = true;
  }

  // 判断是否完整
  const isComplete = result.inHierarchy && result.isLeafNode && result.hasJsonFile && result.inKnowledgeMap;
  if (isComplete) {
    results.completed++;
  } else {
    results.missing.push({
      nodeName,
      issues: []
    });
    if (!result.inHierarchy) results.missing[results.missing.length - 1].issues.push('不在hierarchy.js中');
    if (!result.isLeafNode) results.missing[results.missing.length - 1].issues.push('不是末端节点');
    if (!result.hasJsonFile) results.missing[results.missing.length - 1].issues.push('缺少JSON文件');
    if (!result.inKnowledgeMap) results.missing[results.missing.length - 1].issues.push('未在knowledgeMap.js中映射');
  }

  results.details[nodeName] = result;
});

// 输出结果
console.log('='.repeat(80));
console.log('节点补全情况检查报告');
console.log('='.repeat(80));
console.log(`\n总计: ${results.total} 个节点`);
console.log(`已完成: ${results.completed} 个节点`);
console.log(`未完成: ${results.missing.length} 个节点\n`);

if (results.missing.length > 0) {
  console.log('未完成的节点:');
  console.log('-'.repeat(80));
  results.missing.forEach(({ nodeName, issues }) => {
    console.log(`\n❌ ${nodeName}`);
    console.log(`   问题: ${issues.join(', ')}`);
  });
} else {
  console.log('✅ 所有节点都已补全！');
}

console.log('\n' + '='.repeat(80));
console.log('详细检查结果:');
console.log('='.repeat(80));

// 按分类输出
Object.entries(nodesToCheck).forEach(([category, nodes]) => {
  console.log(`\n## ${category} (${nodes.length} 个)`);
  nodes.forEach(nodeName => {
    const detail = results.details[nodeName];
    const status = detail.inHierarchy && detail.isLeafNode && detail.hasJsonFile && detail.inKnowledgeMap 
      ? '✅' 
      : '❌';
    console.log(`  ${status} ${nodeName}`);
    if (status === '❌') {
      const issues = [];
      if (!detail.inHierarchy) issues.push('不在hierarchy');
      if (!detail.isLeafNode) issues.push('非末端节点');
      if (!detail.hasJsonFile) issues.push('缺少JSON');
      if (!detail.inKnowledgeMap) issues.push('未映射');
      console.log(`     问题: ${issues.join(', ')}`);
    }
  });
});
