// 将新创建的文档添加到knowledgeMap.js
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const knowledgeMapPath = path.join(__dirname, '../src/data/knowledge/knowledgeMap.js');

// 需要添加的文档
const newDocs = [
  { importName: 'DiT', fileName: 'DiT.json', mapKey: 'DiT' },
  { importName: 'SpaceTimeLatentPatch', fileName: 'Space-Time Latent Patch.json', mapKey: 'Space-Time Latent Patch' },
  { importName: 'Whisper', fileName: 'Whisper.json', mapKey: 'Whisper' },
  { importName: 'AudioLM', fileName: 'AudioLM.json', mapKey: 'AudioLM' },
  { importName: 'GPT4oOmni', fileName: 'GPT-4o Omni.json', mapKey: 'GPT-4o Omni' },
  { importName: 'FormatConversion', fileName: '格式转换.json', mapKey: '格式转换' },
  { importName: 'QualityAssessment', fileName: '质量评估.json', mapKey: '质量评估' },
  { importName: 'DataManagement', fileName: '数据管理.json', mapKey: '数据管理' },
  { importName: 'DataParallel', fileName: '数据并行.json', mapKey: '数据并行' },
  { importName: 'ModelParallel', fileName: '模型并行.json', mapKey: '模型并行' },
  { importName: 'PipelineParallel', fileName: '流水线并行.json', mapKey: '流水线并行' },
  { importName: 'GraphOptimization', fileName: '图优化.json', mapKey: '图优化' },
  { importName: 'QuantizationInference', fileName: '量化推理.json', mapKey: '量化推理' },
  { importName: 'ModelPruning', fileName: '模型剪枝.json', mapKey: '模型剪枝' },
  { importName: 'AgentFramework', fileName: '智能体框架.json', mapKey: '智能体框架' },
  { importName: 'ToolCalling', fileName: '工具调用.json', mapKey: '工具调用' },
  { importName: 'MultiAgentSystem', fileName: '多智能体系统.json', mapKey: '多智能体系统' },
  { importName: 'TransformersHF', fileName: 'Transformers.json', mapKey: 'Transformers' },
  { importName: 'AccelerateHF', fileName: 'Accelerate.json', mapKey: 'Accelerate' },
  { importName: 'PerformanceAnalysis', fileName: '性能分析.json', mapKey: '性能分析' },
];

// 读取文件
let content = fs.readFileSync(knowledgeMapPath, 'utf-8');

// 找到导入语句的位置（在AICompiler导入之后）
const importInsertPoint = content.indexOf("import AICompiler from './AI编译器.json';");
if (importInsertPoint === -1) {
  console.error('❌ 找不到导入插入点');
  process.exit(1);
}

// 找到导入语句的结束位置（下一行）
let importEnd = content.indexOf('\n', importInsertPoint);
while (content[importEnd + 1] === ' ' || content[importEnd + 1] === '\t') {
  importEnd = content.indexOf('\n', importEnd + 1);
}

// 生成导入语句
const importStatements = newDocs.map(({ importName, fileName }) => {
  return `import ${importName} from './${fileName}';`;
}).join('\n');

// 插入导入语句
content = content.slice(0, importEnd + 1) + importStatements + '\n' + content.slice(importEnd + 1);

// 找到knowledgeMap对象的位置
const mapStart = content.indexOf('export const knowledgeMap = {');
if (mapStart === -1) {
  console.error('❌ 找不到knowledgeMap对象');
  process.exit(1);
}

// 找到knowledgeMap对象的结束位置（最后一个}之前）
let mapEnd = content.lastIndexOf('};');
if (mapEnd === -1) {
  console.error('❌ 找不到knowledgeMap对象结束');
  process.exit(1);
}

// 检查是否已有映射
const existingMappings = {};
newDocs.forEach(({ mapKey }) => {
  const pattern = new RegExp(`['"]${mapKey.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}['"]\\s*:\\s*\\w+`, 'g');
  if (pattern.test(content)) {
    existingMappings[mapKey] = true;
  }
});

// 生成映射语句（只添加不存在的）
const mappingStatements = newDocs
  .filter(({ mapKey }) => !existingMappings[mapKey])
  .map(({ importName, mapKey }) => {
    // 处理特殊字符
    const escapedKey = mapKey.includes("'") ? `"${mapKey}"` : `'${mapKey}'`;
    return `  ${escapedKey}: ${importName},`;
  })
  .join('\n');

if (mappingStatements) {
  // 在最后一个映射之前插入
  const lastMapping = content.lastIndexOf(',', mapEnd);
  content = content.slice(0, lastMapping + 1) + '\n' + mappingStatements + content.slice(lastMapping + 1);
}

// 处理特殊映射（替换已有的错误映射）
// 替换 'Accelerate': README
content = content.replace(/'Accelerate'\s*:\s*README,/, `'Accelerate': AccelerateHF,`);
// 替换 '性能分析': LLM
content = content.replace(/'性能分析'\s*:\s*LLM,/, `'性能分析': PerformanceAnalysis,`);
// 替换 '数据并行': DataParallelBasics 或 Knowledge13
content = content.replace(/'数据并行'\s*:\s*(DataParallelBasics|Knowledge13),/, `'数据并行': DataParallel,`);
// 替换 '模型并行': ModelParallelBasics
content = content.replace(/'模型并行'\s*:\s*ModelParallelBasics,/, `'模型并行': ModelParallel,`);
// 替换 '流水线并行': PipelineParallelBasics
content = content.replace(/'流水线并行'\s*:\s*PipelineParallelBasics,/, `'流水线并行': PipelineParallel,`);
// 替换 '格式转换': Knowledge15
content = content.replace(/'格式转换'\s*:\s*Knowledge15,/, `'格式转换': FormatConversion,`);
// 替换 '质量评估': Knowledge17
content = content.replace(/'质量评估'\s*:\s*Knowledge17,/, `'质量评估': QualityAssessment,`);
// 替换 '数据管理': Knowledge18
content = content.replace(/'数据管理'\s*:\s*Knowledge18,/, `'数据管理': DataManagement,`);

// 保存文件
fs.writeFileSync(knowledgeMapPath, content, 'utf-8');

console.log('✅ 已添加导入语句:');
newDocs.forEach(({ importName, fileName }) => {
  console.log(`   import ${importName} from './${fileName}';`);
});

console.log('\n✅ 已添加映射:');
newDocs.forEach(({ mapKey, importName }) => {
  if (!existingMappings[mapKey]) {
    console.log(`   '${mapKey}': ${importName}`);
  } else {
    console.log(`   ⏭️  '${mapKey}': 已存在映射`);
  }
});

console.log('\n✅ 已更新knowledgeMap.js！');
