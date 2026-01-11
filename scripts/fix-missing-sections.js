// 批量修复缺失的模块
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const knowledgeDir = path.join(__dirname, '../src/data/knowledge');

// 需要修复的文件和缺失的模块
const fixes = [
  // 激活函数 - 需要添加核心特点和关键技术
  { file: 'Tanh.json', addCoreFeatures: true, addKeyTech: true },
  { file: 'GELU.json', addCoreFeatures: true, addKeyTech: true },
  { file: 'Swish.json', addCoreFeatures: true, addKeyTech: true },
  { file: 'LeakyReLU.json', addCoreFeatures: true, addKeyTech: true },
  { file: 'ELU.json', addCoreFeatures: true, addKeyTech: true },
  { file: 'Mish.json', addCoreFeatures: true, addKeyTech: true },
  { file: 'Softmax.json', addCoreFeatures: true, addKeyTech: true },
  
  // 损失函数 - 需要添加核心特点和关键技术
  { file: '交叉熵损失.json', addCoreFeatures: true, addKeyTech: true },
  { file: 'MSE损失.json', addCoreFeatures: true, addKeyTech: true },
  { file: '余弦相似度.json', addCoreFeatures: true, addKeyTech: true },
];

// 核心特点模板（根据文件类型）
const getCoreFeatures = (fileName) => {
  if (fileName.includes('Tanh')) {
    return [
      "零中心化：输出范围(-1,1)，均值为0，有利于梯度传播",
      "平滑可导：处处可导，梯度连续",
      "对称性：关于原点对称，f(-x) = -f(x)",
      "饱和问题：当|x|较大时，梯度接近0",
      "计算开销：涉及双曲正切函数，计算相对较慢"
    ];
  }
  if (fileName.includes('GELU')) {
    return [
      "平滑非线性：结合ReLU和Sigmoid的优点",
      "自适应门控：根据输入值自适应调整激活",
      "零中心化：输出均值接近0",
      "Transformer标准：BERT、GPT等模型广泛使用",
      "计算开销：涉及误差函数，计算相对较慢"
    ];
  }
  if (fileName.includes('Swish')) {
    return [
      "自门控机制：输入作为门控信号",
      "平滑激活：处处可导，无死神经元问题",
      "性能优异：在多个任务上优于ReLU",
      "计算开销：涉及Sigmoid运算，计算相对较慢",
      "可调参数：Swish-β变体可调整门控强度"
    ];
  }
  if (fileName.includes('LeakyReLU')) {
    return [
      "解决死神经元：负值区域有小的正梯度",
      "计算简单：只需比较和线性变换",
      "参数可调：泄漏系数α通常为0.01",
      "非零中心：输出均值不为0",
      "性能提升：在某些任务上优于ReLU"
    ];
  }
  if (fileName.includes('ELU')) {
    return [
      "平滑负值：负值区域平滑可导",
      "零中心化：输出均值接近0",
      "无死神经元：负值区域有非零梯度",
      "计算开销：涉及指数运算，计算相对较慢",
      "性能优异：在某些任务上优于ReLU"
    ];
  }
  if (fileName.includes('Mish')) {
    return [
      "平滑激活：处处可导，无死神经元",
      "自门控机制：类似Swish但更平滑",
      "性能优异：在多个任务上表现优秀",
      "计算开销：涉及多个函数，计算相对较慢",
      "无界输出：正值区域无上界"
    ];
  }
  if (fileName.includes('Softmax')) {
    return [
      "概率归一化：将输出转换为概率分布",
      "多分类标准：多分类任务的输出层标准选择",
      "可导性：处处可导，适合反向传播",
      "数值稳定性：需要特殊处理避免溢出",
      "注意力机制：Transformer注意力计算的核心"
    ];
  }
  if (fileName.includes('交叉熵损失')) {
    return [
      "概率分布差异：衡量预测分布与真实分布的差异",
      "分类任务标准：多分类任务的标准损失函数",
      "梯度友好：梯度计算简单，训练稳定",
      "信息论基础：基于信息熵和KL散度",
      "数值稳定性：需要特殊处理避免数值问题"
    ];
  }
  if (fileName.includes('MSE损失')) {
    return [
      "回归任务标准：回归任务的标准损失函数",
      "平方误差：对大误差惩罚更重",
      "可导性：处处可导，梯度计算简单",
      "对异常值敏感：平方项放大异常值影响",
      "计算简单：实现简单，计算效率高"
    ];
  }
  if (fileName.includes('余弦相似度')) {
    return [
      "方向相似性：衡量向量方向而非大小",
      "归一化输出：输出范围[-1,1]",
      "度量学习：用于相似度计算和检索",
      "注意力机制：Transformer中计算注意力分数",
      "计算高效：只需点积和范数计算"
    ];
  }
  return [];
};

// 关键技术模板
const getKeyTech = (fileName) => {
  if (fileName.includes('Tanh')) {
    return "双曲正切函数、零中心化、对称激活、梯度传播、饱和区域处理";
  }
  if (fileName.includes('GELU')) {
    return "高斯误差线性单元、自适应门控、误差函数、平滑激活、Transformer激活";
  }
  if (fileName.includes('Swish')) {
    return "自门控激活、Sigmoid门控、Swish-β变体、平滑非线性";
  }
  if (fileName.includes('LeakyReLU')) {
    return "泄漏修正线性单元、负值梯度、可调泄漏系数、死神经元解决";
  }
  if (fileName.includes('ELU')) {
    return "指数线性单元、平滑负值、零中心化、无死神经元";
  }
  if (fileName.includes('Mish')) {
    return "Mish激活函数、自门控机制、平滑非线性、无界输出";
  }
  if (fileName.includes('Softmax')) {
    return "概率归一化、多分类输出、数值稳定性、注意力计算、温度缩放";
  }
  if (fileName.includes('交叉熵损失')) {
    return "交叉熵、KL散度、信息熵、概率分布、标签平滑、类别权重";
  }
  if (fileName.includes('MSE损失')) {
    return "均方误差、平方损失、L2损失、回归任务、异常值处理";
  }
  if (fileName.includes('余弦相似度')) {
    return "余弦相似度、向量归一化、点积计算、相似度度量、注意力分数";
  }
  return "";
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
    
    let hasChanges = false;
    const sections = doc.content || [];
    
    // 检查是否已有核心特点
    const hasCoreFeatures = sections.some(s => 
      s.title && (s.title.includes('核心特点') || s.title.includes('🌟 核心特点'))
    );
    
    // 检查是否已有关键技术
    const hasKeyTech = sections.some(s => 
      s.title && (s.title.includes('关键技术') || s.title.includes('⚙️ 关键技术'))
    );
    
    // 找到核心概念部分的位置
    let coreConceptIndex = sections.findIndex(s => 
      s.title && s.title.includes('核心概念')
    );
    
    // 添加核心特点
    if (addCoreFeatures && !hasCoreFeatures && coreConceptIndex >= 0) {
      const features = getCoreFeatures(file);
      if (features.length > 0) {
        const featuresSection = {
          type: "section",
          title: "🌟 核心特点",
          content: [
            {
              type: "features",
              items: features
            }
          ]
        };
        sections.splice(coreConceptIndex + 1, 0, featuresSection);
        hasChanges = true;
        console.log(`✅ ${file}: 已添加核心特点`);
      }
    }
    
    // 找到应用场景部分的位置（在它之前插入关键技术）
    let appSceneIndex = sections.findIndex(s => 
      s.title && s.title.includes('应用场景')
    );
    
    // 添加关键技术
    if (addKeyTech && !hasKeyTech) {
      const keyTech = getKeyTech(file);
      if (keyTech) {
        const techSection = {
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
          sections.splice(appSceneIndex, 0, techSection);
        } else {
          sections.push(techSection);
        }
        hasChanges = true;
        console.log(`✅ ${file}: 已添加关键技术`);
      }
    }
    
    // 保存文件
    if (hasChanges) {
      doc.content = sections;
      const newContent = JSON.stringify(doc, null, 2);
      fs.writeFileSync(filePath, newContent, 'utf-8');
    }
    
  } catch (error) {
    console.error(`❌ 处理 ${file} 时出错:`, error.message);
  }
});

console.log('\n✅ 批量修复完成！');
