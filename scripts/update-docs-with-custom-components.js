// 更新新文档使用自定义组件
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const knowledgeDir = path.join(__dirname, '../src/data/knowledge');

// 需要更新的文档配置
const updates = [
  {
    file: 'DiT.json',
    component: 'DiffusionDiagram',
    props: { type: 'architecture', title: 'DiT架构' }
  },
  {
    file: '数据并行.json',
    component: 'ParallelTrainingDiagram',
    props: { type: 'data', title: '数据并行原理' }
  },
  {
    file: '模型并行.json',
    component: 'ParallelTrainingDiagram',
    props: { type: 'model', title: '模型并行原理' }
  },
  {
    file: '流水线并行.json',
    component: 'ParallelTrainingDiagram',
    props: { type: 'pipeline', title: '流水线并行原理' }
  },
  {
    file: '量化推理.json',
    component: 'QuantizationDiagram',
    props: { title: '量化推理' }
  }
];

// 处理每个文档
updates.forEach(({ file, component, props }) => {
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
    
    // 找到架构图解section
    const diagramSectionIndex = sections.findIndex(s => 
      s.title && s.title.includes('架构图解')
    );
    
    if (diagramSectionIndex >= 0) {
      const diagramSection = sections[diagramSectionIndex];
      if (diagramSection.content && diagramSection.content[0] && 
          diagramSection.content[0].images && diagramSection.content[0].images[0]) {
        const image = diagramSection.content[0].images[0];
        
        // 更新组件和props
        if (image.component !== component || JSON.stringify(image.props) !== JSON.stringify(props)) {
          image.component = component;
          image.props = props;
          hasChanges = true;
        }
      }
    }
    
    // 保存文件
    if (hasChanges) {
      const newContent = JSON.stringify(doc, null, 2);
      fs.writeFileSync(filePath, newContent, 'utf-8');
      console.log(`✅ ${file}: 已更新为使用 ${component}`);
    } else {
      console.log(`⏭️  ${file}: 无需修改`);
    }
    
  } catch (error) {
    console.error(`❌ 处理 ${file} 时出错:`, error.message);
  }
});

console.log('\n✅ 批量更新完成！');
