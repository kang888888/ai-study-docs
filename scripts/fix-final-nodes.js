// 为最后几个节点添加缺失模块
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const knowledgeDir = path.join(__dirname, '../src/data/knowledge');

// 需要修复的文件
const fixes = [
  { 
    file: 'Miras.json', 
    addCoreFeatures: true, 
    addKeyTech: true 
  },
  { 
    file: 'Titans.json', 
    addCoreFeatures: true, 
    addKeyTech: true 
  },
  { 
    file: 'Moneta.json', 
    addKeyTech: true 
  },
  { 
    file: 'Yaad.json', 
    addKeyTech: true 
  },
  { 
    file: 'Memora.json', 
    addKeyTech: true 
  },
  { 
    file: '向量数据库.json', 
    addCoreFeatures: true 
  },
];

// 获取内容模板
const getContent = (fileName, type) => {
  const name = fileName.replace('.json', '');
  
  // 核心特点
  if (type === 'coreFeatures') {
    const features = {
      'Miras': [
        '统一框架：统一理解神经架构为关联记忆模块',
        '注意力偏差：通过注意力偏差优化信息检索',
        '架构指导：指导新架构的设计和优化',
        '理论统一：统一理解Transformers、Titans等架构',
        '性能提升：在多个任务上超越现有架构'
      ],
      'Titans': [
        '超长上下文：支持200万+ Token的上下文长度',
        '仿生设计：融合短期记忆、长期记忆和注意力',
        '记忆系统：内置长期记忆，无需外部记忆模块',
        '多架构变体：MAC、MEC、MEC+三种变体',
        '性能优异：在长文本任务上表现卓越'
      ],
      '向量数据库': [
        '高维向量：专门存储和检索高维向量',
        '相似度搜索：基于向量相似度的快速检索',
        '语义理解：支持语义搜索和语义匹配',
        '可扩展性：支持大规模向量存储和检索',
        '多种产品：Milvus、Pinecone、Weaviate等'
      ],
    };
    return features[name] || [];
  }
  
  // 关键技术
  if (type === 'keyTech') {
    const techs = {
      'Miras': '关联记忆架构、注意力偏差、记忆组织、信息检索、架构设计框架',
      'Titans': '短期记忆、长期记忆、注意力机制、记忆编码器、记忆检索器、记忆更新器',
      'Moneta': '高效检索、快速更新、关联记忆、实时推理、KV Cache优化',
      'Yaad': '注意力优化、内存管理、KV Cache、推理加速',
      'Memora': '记忆管理、KV Cache优化、长上下文支持',
    };
    return techs[name] || '';
  }
  
  return '';
};

// 处理每个文件
fixes.forEach(({ file, addCoreFeatures, addKeyTech }) => {
  const filePath = path.join(knowledgeDir, file);
  
  if (!fs.existsSync(filePath)) {
    console.log(`❌ 文件不存在: ${file}`);
    return;
  }
  
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const doc = JSON.parse(content);
    
    let sections = doc.content || [];
    let hasChanges = false;
    
    // 检查现有模块
    const hasCoreFeatures = sections.some(s => 
      s.title && (s.title.includes('核心特点') || s.title.includes('🌟 核心特点'))
    );
    const hasKeyTech = sections.some(s => 
      s.title && (s.title.includes('关键技术') || s.title.includes('⚙️ 关键技术'))
    );
    
    // 找到核心概念的位置
    let coreConceptIndex = sections.findIndex(s => 
      s.title && s.title.includes('核心概念')
    );
    
    // 找到应用场景的位置
    let appSceneIndex = sections.findIndex(s => 
      s.title && (s.title.includes('应用场景') || s.title.includes('🚀 应用场景'))
    );
    
    // 构建新的sections数组
    const newSections = [];
    
    // 保留核心概念
    if (coreConceptIndex >= 0) {
      newSections.push(sections[coreConceptIndex]);
    }
    
    // 添加核心特点
    if (addCoreFeatures && !hasCoreFeatures) {
      const features = getContent(file, 'coreFeatures');
      if (features.length > 0) {
        newSections.push({
          type: "section",
          title: "🌟 核心特点",
          content: [
            {
              type: "features",
              items: features
            }
          ]
        });
        hasChanges = true;
      }
    } else if (hasCoreFeatures) {
      const existing = sections.find(s => s.title && (s.title.includes('核心特点') || s.title.includes('🌟 核心特点')));
      if (existing) newSections.push(existing);
    }
    
    // 添加关键技术
    if (addKeyTech && !hasKeyTech) {
      const keyTech = getContent(file, 'keyTech');
      if (keyTech) {
        newSections.push({
          type: "section",
          title: "⚙️ 关键技术",
          content: [
            {
              type: "tech-box",
              content: keyTech
            }
          ]
        });
        hasChanges = true;
      }
    } else if (hasKeyTech) {
      const existing = sections.find(s => s.title && (s.title.includes('关键技术') || s.title.includes('⚙️ 关键技术')));
      if (existing) newSections.push(existing);
    }
    
    // 添加其他现有section（应用场景、架构图解等）
    sections.forEach((section, index) => {
      if (index !== coreConceptIndex && 
          section.title && 
          !section.title.includes('核心特点') && 
          !section.title.includes('关键技术')) {
        newSections.push(section);
      }
    });
    
    // 保存文件
    if (hasChanges) {
      doc.content = newSections;
      const newContent = JSON.stringify(doc, null, 2);
      fs.writeFileSync(filePath, newContent, 'utf-8');
      console.log(`✅ ${file}: 已添加缺失模块`);
    } else {
      console.log(`⏭️  ${file}: 无需修改`);
    }
    
  } catch (error) {
    console.error(`❌ 处理 ${file} 时出错:`, error.message);
  }
});

console.log('\n✅ 批量修复完成！');
