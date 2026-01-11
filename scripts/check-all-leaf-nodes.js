// 检查所有末端节点的文档规范
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 读取hierarchy.js
const hierarchyPath = path.join(__dirname, '../src/data/hierarchy.js');
const hierarchyContent = fs.readFileSync(hierarchyPath, 'utf-8');

// 提取所有末端节点
function extractLeafNodes(node, parentPath = []) {
  const nodes = [];
  const currentPath = [...parentPath, node.name];
  
  if (!node.children || node.children.length === 0) {
    // 这是末端节点
    nodes.push({
      name: node.name,
      path: currentPath.join(' > ')
    });
  } else {
    // 递归处理子节点
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

// 检查每个末端节点
const knowledgeDir = path.join(__dirname, '../src/data/knowledge');
const knowledgeMapPath = path.join(__dirname, '../src/data/knowledge/knowledgeMap.js');
const knowledgeMapContent = fs.readFileSync(knowledgeMapPath, 'utf-8');

const issues = {
  missingFiles: [],
  missingInMap: [],
  missingRequiredSections: [],
  diagramIssues: [],
  duplicateDiagrams: []
};

// 必需模块
const requiredSections = ['核心概念', '核心特点', '关键技术', '应用场景'];
const requiredSectionTypes = {
  '核心概念': 'desc-box',
  '核心特点': 'features',
  '关键技术': 'tech-box',
  '应用场景': 'app-box'
};

leafNodes.forEach(node => {
  const fileName = `${node.name}.json`;
  const filePath = path.join(knowledgeDir, fileName);
  
  // 检查文件是否存在
  if (!fs.existsSync(filePath)) {
    issues.missingFiles.push(node);
    return;
  }
  
  // 检查是否在knowledgeMap中映射
  if (!knowledgeMapContent.includes(`'${node.name}'`)) {
    issues.missingInMap.push(node);
  }
  
  // 检查文档结构
  try {
    const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    const sections = content.content || [];
    
    // 检查必需模块
    const sectionTitles = sections.map(s => s.title || '').filter(t => t);
    const missingSections = requiredSections.filter(req => 
      !sectionTitles.some(title => title.includes(req.replace('📖 ', '').replace('🌟 ', '').replace('⚙️ ', '').replace('🚀 ', '')))
    );
    
    if (missingSections.length > 0) {
      issues.missingRequiredSections.push({
        node: node.name,
        missing: missingSections
      });
    }
    
    // 检查架构图解
    const diagramSections = sections.filter(s => s.title && s.title.includes('架构图解'));
    diagramSections.forEach(section => {
      const galleries = (section.content || []).filter(c => c.type === 'diagram-gallery');
      galleries.forEach(gallery => {
        const images = gallery.images || [];
        
        // 检查是否有重复的组件、type和title组合（caption或title不同应视为不同图表）
        const imageKeys = images.map(img => {
          const component = img.component || '';
          const type = (img.props && img.props.type) || '';
          const title = (img.props && img.props.title) || img.caption || '';
          return `${component}_${type}_${title}`;
        });
        const uniqueKeys = [...new Set(imageKeys)];
        if (imageKeys.length !== uniqueKeys.length) {
          issues.duplicateDiagrams.push({
            node: node.name,
            count: imageKeys.length - uniqueKeys.length
          });
        }
        
        // 检查是否有data: null（不必要的属性）
        images.forEach(img => {
          if (img.props && img.props.data === null) {
            issues.diagramIssues.push({
              node: node.name,
              issue: '包含不必要的 data: null 属性'
            });
          }
        });
      });
    });
    
  } catch (error) {
    console.error(`解析 ${fileName} 时出错:`, error.message);
  }
});

// 输出结果
console.log('='.repeat(80));
console.log('检查结果汇总');
console.log('='.repeat(80));

if (issues.missingFiles.length > 0) {
  console.log(`\n❌ 缺少知识文档文件 (${issues.missingFiles.length}个):`);
  issues.missingFiles.forEach(node => {
    console.log(`  - ${node.name} (路径: ${node.path})`);
  });
}

if (issues.missingInMap.length > 0) {
  console.log(`\n❌ 未在knowledgeMap.js中映射 (${issues.missingInMap.length}个):`);
  issues.missingInMap.forEach(node => {
    console.log(`  - ${node.name}`);
  });
}

if (issues.missingRequiredSections.length > 0) {
  console.log(`\n❌ 缺少必需模块 (${issues.missingRequiredSections.length}个):`);
  issues.missingRequiredSections.forEach(({node, missing}) => {
    console.log(`  - ${node}: 缺少 ${missing.join(', ')}`);
  });
}

if (issues.duplicateDiagrams.length > 0) {
  console.log(`\n❌ 架构图解重复展示 (${issues.duplicateDiagrams.length}个):`);
  issues.duplicateDiagrams.forEach(({node, count}) => {
    console.log(`  - ${node}: 有 ${count} 个重复的图表`);
  });
}

if (issues.diagramIssues.length > 0) {
  console.log(`\n⚠️  架构图解配置问题 (${issues.diagramIssues.length}个):`);
  issues.diagramIssues.forEach(({node, issue}) => {
    console.log(`  - ${node}: ${issue}`);
  });
}

const totalIssues = issues.missingFiles.length + 
                   issues.missingInMap.length + 
                   issues.missingRequiredSections.length + 
                   issues.duplicateDiagrams.length + 
                   issues.diagramIssues.length;

if (totalIssues === 0) {
  console.log('\n✅ 所有末端节点检查通过！');
} else {
  console.log(`\n总计发现 ${totalIssues} 个问题需要修复`);
}

console.log('\n' + '='.repeat(80));
