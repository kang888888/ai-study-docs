// 添加缺失的知识文档映射
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 需要添加映射的节点列表（已确认JSON文件存在）
const missingMappings = [
  { nodeName: 'StreamingLLM', importName: 'StreamingLLM', fileName: 'StreamingLLM.json' },
  { nodeName: 'Activation Beacon', importName: 'ActivationBeacon', fileName: 'Activation Beacon.json' },
  { nodeName: 'Ring Attention', importName: 'RingAttention', fileName: 'Ring Attention.json' },
  { nodeName: 'PII脱敏', importName: 'PIIDesensitization', fileName: 'PII脱敏.json' },
  { nodeName: '去偏见', importName: 'Debias', fileName: '去偏见.json' },
  { nodeName: '多语言平衡', importName: 'MultilingualBalance', fileName: '多语言平衡.json' },
  { nodeName: '梯度累积与检查点', importName: 'GradientAccumulation', fileName: '梯度累积与检查点.json' },
  { nodeName: '混合精度训练', importName: 'MixedPrecisionTraining', fileName: '混合精度训练.json' },
  { nodeName: 'Loss Spike处理', importName: 'LossSpikeHandling', fileName: 'Loss Spike处理.json' },
  { nodeName: '权重衰减诊断', importName: 'WeightDecayDiagnosis', fileName: '权重衰减诊断.json' },
  { nodeName: 'Epsilon预测', importName: 'EpsilonPrediction', fileName: 'Epsilon预测.json' },
  { nodeName: '强化学习基础', importName: 'RLBasics', fileName: '强化学习基础.json' },
  { nodeName: '逻辑推理优化', importName: 'LogicalReasoning', fileName: '逻辑推理优化.json' },
  { nodeName: '提示词注入防御', importName: 'PromptInjectionDefense', fileName: '提示词注入防御.json' },
  { nodeName: '对抗性攻击测试', importName: 'AdversarialAttackTesting', fileName: '对抗性攻击测试.json' },
  { nodeName: '红色对抗', importName: 'RedTeaming', fileName: '红色对抗.json' },
  { nodeName: '机器版权保护', importName: 'MachineCopyrightProtection', fileName: '机器版权保护.json' },
  { nodeName: '水印技术', importName: 'Watermarking', fileName: '水印技术.json' },
  { nodeName: 'Executive', importName: 'Executive', fileName: 'Executive.json' },
  { nodeName: '知识图谱增强', importName: 'KnowledgeGraphEnhancement', fileName: '知识图谱增强.json' },
  { nodeName: '向量数据库', importName: 'VectorDatabase', fileName: '向量数据库.json' },
  { nodeName: '记忆刷新机制', importName: 'MemoryRefresh', fileName: '记忆刷新机制.json' },
  { nodeName: 'LangChain框架', importName: 'LangChainFramework', fileName: 'LangChain框架.json' },
  { nodeName: '硬件与集群', importName: 'HardwareCluster', fileName: '硬件与集群.json' },
  { nodeName: 'AI编译器', importName: 'AICompiler', fileName: 'AI编译器.json' },
  { nodeName: '昇腾CANN', importName: 'AscendCANN', fileName: '昇腾CANN.json' },
  { nodeName: '海光DCU', importName: 'HygonDCU', fileName: '海光DCU.json' },
  { nodeName: '摩尔线程MUSA', importName: 'MooreThreadsMUSA', fileName: '摩尔线程MUSA.json' },
  { nodeName: '算力网络调度', importName: 'ComputeNetworkScheduling', fileName: '算力网络调度.json' },
  { nodeName: '异构计算并行', importName: 'HeterogeneousComputingParallelism', fileName: '异构计算并行.json' },
];

const knowledgeMapPath = path.join(__dirname, '../src/data/knowledge/knowledgeMap.js');
let content = fs.readFileSync(knowledgeMapPath, 'utf-8');

// 检查哪些导入已存在
const existingImports = new Set();
missingMappings.forEach(({ importName, fileName }) => {
  const importPattern = new RegExp(`import\\s+${importName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s+from`, 'i');
  if (importPattern.test(content)) {
    existingImports.add(importName);
    console.log(`✅ 导入已存在: ${importName}`);
  }
});

// 检查哪些映射已存在
const existingMappings = new Set();
missingMappings.forEach(({ nodeName, importName }) => {
  // 检查映射（考虑中文节点名）
  const mappingPattern = new RegExp(`['"]${nodeName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}['"]\\s*:`, 'i');
  if (mappingPattern.test(content)) {
    existingMappings.add(nodeName);
    console.log(`✅ 映射已存在: ${nodeName}`);
  }
});

// 添加缺失的导入
const importsToAdd = [];
missingMappings.forEach(({ importName, fileName }) => {
  if (!existingImports.has(importName)) {
    importsToAdd.push({ importName, fileName });
  }
});

// 找到导入部分的结束位置（在export const knowledgeMap之前）
const exportPattern = /export\s+const\s+knowledgeMap\s*=/;
const exportMatch = content.match(exportPattern);
if (!exportMatch) {
  console.error('无法找到knowledgeMap导出位置');
  process.exit(1);
}

const insertPosition = exportMatch.index;

// 添加导入语句
if (importsToAdd.length > 0) {
  const importStatements = importsToAdd.map(({ importName, fileName }) => {
    return `import ${importName} from './${fileName}';`;
  }).join('\n');
  
  content = content.slice(0, insertPosition) + 
            '\n// 添加缺失的导入\n' +
            importStatements + '\n\n' +
            content.slice(insertPosition);
  
  console.log(`\n✅ 已添加 ${importsToAdd.length} 个导入语句`);
}

// 添加缺失的映射
const mappingsToAdd = [];
missingMappings.forEach(({ nodeName, importName }) => {
  if (!existingMappings.has(nodeName)) {
    mappingsToAdd.push({ nodeName, importName });
  }
});

// 找到knowledgeMap对象的结束位置（在};之前）
const mapEndPattern = /^\s*\};\s*$/m;
const mapEndMatch = content.slice(insertPosition).match(mapEndPattern);
if (!mapEndMatch) {
  console.error('无法找到knowledgeMap对象结束位置');
  process.exit(1);
}

const mapEndPosition = insertPosition + mapEndMatch.index;

// 在knowledgeMap对象结束前添加映射
if (mappingsToAdd.length > 0) {
  const mappingStatements = mappingsToAdd.map(({ nodeName, importName }) => {
    return `  '${nodeName}': ${importName},`;
  }).join('\n');
  
  content = content.slice(0, mapEndPosition) + 
            mappingStatements + '\n' +
            content.slice(mapEndPosition);
  
  console.log(`\n✅ 已添加 ${mappingsToAdd.length} 个映射`);
}

// 保存文件
fs.writeFileSync(knowledgeMapPath, content, 'utf-8');
console.log('\n✅ 已更新 knowledgeMap.js');

// 输出总结
console.log('\n' + '='.repeat(80));
console.log('总结');
console.log('='.repeat(80));
console.log(`总共有 ${missingMappings.length} 个节点需要映射`);
console.log(`已存在导入: ${existingImports.size} 个`);
console.log(`已存在映射: ${existingMappings.size} 个`);
console.log(`新增导入: ${importsToAdd.length} 个`);
console.log(`新增映射: ${mappingsToAdd.length} 个`);
console.log('='.repeat(80));
