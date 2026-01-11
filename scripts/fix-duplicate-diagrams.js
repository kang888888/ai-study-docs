// 修复重复的架构图解和data: null问题
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 读取hierarchy.js
const hierarchyPath = path.join(__dirname, '../src/data/hierarchy.js');

// 提取所有末端节点
function extractLeafNodes(node, parentPath = []) {
  const nodes = [];
  const currentPath = [...parentPath, node.name];
  
  if (!node.children || node.children.length === 0) {
    nodes.push({
      name: node.name,
      path: currentPath.join(' > ')
    });
  } else {
    node.children.forEach(child => {
      nodes.push(...extractLeafNodes(child, currentPath));
    });
  }
  
  return nodes;
}

// 解析hierarchy（使用动态导入）
const hierarchyModule = await import('../src/data/hierarchy.js');
const hierarchy = hierarchyModule.hierarchy;

const leafNodes = extractLeafNodes(hierarchy);

console.log(`\n找到 ${leafNodes.length} 个末端节点\n`);

// 检查并修复每个末端节点
const knowledgeDir = path.join(__dirname, '../src/data/knowledge');
const knowledgeMapPath = path.join(__dirname, '../src/data/knowledge/knowledgeMap.js');
const knowledgeMapContent = fs.readFileSync(knowledgeMapPath, 'utf-8');

const fixedFiles = [];
const skippedFiles = [];

leafNodes.forEach(({ name }) => {
  // 检查是否在knowledgeMap中
  const mapPattern = new RegExp(`['"]${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}['"]`, 'g');
  if (!mapPattern.test(knowledgeMapContent)) {
    return; // 不在映射中，跳过
  }
  
  const fileName = `${name}.json`;
  const filePath = path.join(knowledgeDir, fileName);
  
  if (!fs.existsSync(filePath)) {
    return; // 文件不存在，跳过
  }
  
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const doc = JSON.parse(content);
    
    let hasChanges = false;
    
    // 遍历所有section
    (doc.content || []).forEach(section => {
      // 找到所有diagram-gallery
      const galleries = (section.content || []).filter(c => c.type === 'diagram-gallery');
      
      galleries.forEach(gallery => {
        const images = gallery.images || [];
        if (images.length === 0) return;
        
        // 检查重复：相同component和props.type的组合
        const seen = new Set();
        const uniqueImages = [];
        
        images.forEach(img => {
          // 删除data: null属性
          if (img.props && img.props.data === null) {
            delete img.props.data;
            hasChanges = true;
          }
          
          // 生成唯一键
          const key = `${img.component || ''}_${(img.props && img.props.type) || ''}_${img.caption || ''}`;
          
          // 如果还没见过这个组合，保留它
          if (!seen.has(key)) {
            seen.add(key);
            uniqueImages.push(img);
          } else {
            hasChanges = true;
            console.log(`  - 删除重复图表: ${name} - ${img.caption || '未命名'}`);
          }
        });
        
        // 更新images数组
        if (uniqueImages.length !== images.length) {
          gallery.images = uniqueImages;
          hasChanges = true;
        }
      });
    });
    
    // 如果有更改，保存文件
    if (hasChanges) {
      const newContent = JSON.stringify(doc, null, 2);
      fs.writeFileSync(filePath, newContent, 'utf-8');
      fixedFiles.push(name);
      console.log(`✅ 已修复: ${name}`);
    }
    
  } catch (error) {
    console.error(`❌ 处理 ${fileName} 时出错:`, error.message);
    skippedFiles.push({ name, error: error.message });
  }
});

console.log('\n' + '='.repeat(80));
console.log('修复完成');
console.log('='.repeat(80));
console.log(`\n✅ 已修复 ${fixedFiles.length} 个文件`);
if (skippedFiles.length > 0) {
  console.log(`\n⚠️  跳过 ${skippedFiles.length} 个文件:`);
  skippedFiles.forEach(({ name, error }) => {
    console.log(`  - ${name}: ${error}`);
  });
}
console.log('\n' + '='.repeat(80));
