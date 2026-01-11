// 为模型合并相关节点添加关键技术模块
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const knowledgeDir = path.join(__dirname, '../src/data/knowledge');

// 需要修复的文件
const fixes = [
  { file: '线性合并.json', keyTech: '加权平均、参数合并、权重分配、逐层合并、模型平均' },
  { file: '任务向量合并.json', keyTech: '任务向量提取、向量合并、参数差异、多任务融合、alpha缩放' },
  { file: '分层合并.json', keyTech: '分层策略、逐层合并、层选择、权重分配、架构适配' },
  { file: '参数空间合并.json', keyTech: '参数空间、空间映射、参数对齐、空间融合、几何方法' },
  { file: '功能锚点合并.json', keyTech: '功能锚点、锚点选择、功能对齐、合并策略、性能优化' },
  { file: 'MergeKit.json', keyTech: '模型合并、多种策略、格式支持、配置管理、批量处理' },
];

// 处理每个文件
fixes.forEach(({ file, keyTech }) => {
  const filePath = path.join(knowledgeDir, file);
  
  if (!fs.existsSync(filePath)) {
    console.log(`❌ 文件不存在: ${file}`);
    return;
  }
  
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const doc = JSON.parse(content);
    
    let sections = doc.content || [];
    
    // 检查是否已有关键技术
    const hasKeyTech = sections.some(s => 
      s.title && (s.title.includes('关键技术') || s.title.includes('⚙️ 关键技术'))
    );
    
    if (hasKeyTech) {
      console.log(`⏭️  ${file}: 已有关键技术模块`);
      return;
    }
    
    // 找到应用场景的位置（在它之前插入关键技术）
    let appSceneIndex = sections.findIndex(s => 
      s.title && (s.title.includes('应用场景') || s.title.includes('🚀 应用场景'))
    );
    
    // 创建关键技术section
    const keyTechSection = {
      type: "section",
      title: "⚙️ 关键技术",
      content: [
        {
          type: "tech-box",
          content: keyTech
        }
      ]
    };
    
    // 在应用场景之前插入，如果没有应用场景则在最后插入
    if (appSceneIndex >= 0) {
      sections.splice(appSceneIndex, 0, keyTechSection);
    } else {
      sections.push(keyTechSection);
    }
    
    // 保存文件
    doc.content = sections;
    const newContent = JSON.stringify(doc, null, 2);
    fs.writeFileSync(filePath, newContent, 'utf-8');
    console.log(`✅ ${file}: 已添加关键技术模块`);
    
  } catch (error) {
    console.error(`❌ 处理 ${file} 时出错:`, error.message);
  }
});

console.log('\n✅ 批量修复完成！');
