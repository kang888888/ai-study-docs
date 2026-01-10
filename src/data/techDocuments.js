// 技术文档数据结构
// 这个文件包含从HTML提取的技术文档内容

// BERT文档示例
export const bertDocument = {
  title: 'BERT (Bidirectional Encoder Representations from Transformers)',
  subtitle: 'Google的预训练语言模型',
  content: [
    {
      type: 'section',
      title: '📖 核心概念',
      content: [
        {
          type: 'desc-box',
          content: [
            'Google在2018年提出的预训练模型，只使用Transformer的Encoder部分。通过掩码语言模型（MLM）和下一句预测（NSP）任务进行预训练，学习双向上下文表示。'
          ]
        }
      ]
    },
    {
      type: 'section',
      title: '🌟 核心特点',
      content: [
        {
          type: 'features',
          items: [
            '双向理解：同时利用左侧和右侧的上下文信息',
            '掩码语言模型（MLM）：随机遮盖15%的词，预测被遮盖的词',
            '预训练+微调：在大规模语料上预训练，然后在下游任务微调',
            '只有Encoder：不包含Decoder，不适合生成任务',
            'SOTA性能：在多个NLP理解任务上刷新记录'
          ]
        }
      ]
    },
    {
      type: 'section',
      title: '⚙️ 关键技术',
      content: [
        {
          type: 'tech-box',
          content: 'Masked Language Model、Next Sentence Prediction、WordPiece分词、[CLS]和[SEP]特殊Token'
        }
      ]
    },
    {
      type: 'section',
      title: '🚀 应用场景',
      content: [
        {
          type: 'app-box',
          content: '文本分类、命名实体识别（NER）、问答系统（QA）、语义相似度、情感分析'
        }
      ]
    },
    {
      type: 'section',
      title: '📊 架构图解',
      content: [
        {
          type: 'diagram-gallery',
          images: [
            {
              type: 'svg-d3',
              component: 'BERTDiagram',
              caption: 'BERT架构图',
              width: 1000,
              height: 800,
              interactive: true,
              props: {
                type: 'architecture',
                title: 'BERT架构图'
              }
            },
            {
              type: 'svg-d3',
              component: 'BERTDiagram',
              caption: 'BERT MLM可视化',
              width: 1000,
              height: 800,
              interactive: true,
              props: {
                type: 'mlm',
                title: 'BERT MLM可视化'
              }
            },
            {
              type: 'svg-d3',
              component: 'BERTDiagram',
              caption: 'BERT双向注意力',
              width: 1000,
              height: 800,
              interactive: true,
              props: {
                type: 'attention',
                title: 'BERT双向注意力'
              }
            }
          ]
        }
      ]
    },
    {
      type: 'section',
      title: '📐 数学原理',
      content: [
        {
          type: 'math-box',
          title: '掩码语言模型（MLM）损失',
          formulas: [
            {
              text: '对于被掩码的位置 $m$，预测被掩码的词：'
            },
            {
              display: 'L_{MLM} = -\\sum_{m \\in M} \\log P(x_m | x_{\\backslash m})'
            },
            {
              text: '其中 $M$ 是被掩码的位置集合，$x_{\\backslash m}$ 是除位置 $m$ 外的所有词'
            }
          ]
        },
        {
          type: 'math-box',
          title: '下一句预测（NSP）损失',
          formulas: [
            {
              text: '预测句子B是否是句子A的下一句：'
            },
            {
              display: 'L_{NSP} = -\\log P(\\text{IsNext} | \\text{CLS})'
            },
            {
              text: '总损失：$L = L_{MLM} + L_{NSP}$'
            }
          ]
        },
        {
          type: 'math-box',
          title: '双向注意力',
          formulas: [
            {
              text: 'BERT使用双向自注意力，每个词可以同时看到左右两侧的上下文：'
            },
            {
              display: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V'
            },
            {
              text: '与GPT的单向注意力不同，BERT可以同时利用前后文信息'
            }
          ]
        }
      ]
    },
    {
      type: 'section',
      title: '💻 Python 代码示例',
      content: [
        {
          type: 'code-box',
          title: '使用 Transformers 库加载 BERT',
          language: 'python',
          code: `from transformers import BertModel, BertTokenizer, BertForMaskedLM
import torch

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 输入文本
text = "The cat sat on the [MASK]."

# 分词和编码
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# 前向传播
with torch.no_grad():
    outputs = model(**inputs)

# 获取词嵌入
embeddings = outputs.last_hidden_state
print(f"词嵌入形状: {embeddings.shape}")  # [batch_size, seq_len, hidden_size]

# 使用MLM模型进行掩码预测
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
with torch.no_grad():
    mlm_outputs = mlm_model(**inputs)
    predictions = mlm_outputs.logits

# 预测被掩码的词
masked_index = inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)
predicted_token_id = predictions[0, masked_index].argmax().item()
predicted_token = tokenizer.decode([predicted_token_id])
print(f"预测的词: {predicted_token}")`
        }
      ]
    }
  ]
};

// ChatGLM文档示例
export const chatglmDocument = {
  title: 'ChatGLM (智谱AI)',
  subtitle: '智谱AI开源的中英双语对话模型',
  content: [
    {
      type: 'section',
      title: '📖 核心概念',
      content: [
        {
          type: 'desc-box',
          content: [
            '智谱AI开源的中英双语对话模型，基于GLM（General Language Model）架构。采用混合注意力机制，在中文理解和生成上表现优异。'
          ]
        }
      ]
    },
    {
      type: 'section',
      title: '🌟 核心特点',
      content: [
        {
          type: 'features',
          items: [
            '中文优化：在大规模中文语料上训练，中文能力突出',
            'GLM架构：混合自回归和自编码的预训练目标',
            '工具调用：ChatGLM3支持Function Calling',
            '多模态：GLM-4支持图像理解',
            '开源可商用：6B参数，可在消费级GPU上运行'
          ]
        }
      ]
    },
    {
      type: 'section',
      title: '⚙️ 关键技术',
      content: [
        {
          type: 'tech-box',
          content: 'GLM架构、双向注意力、旋转位置编码、Flash Attention'
        }
      ]
    },
    {
      type: 'section',
      title: '🚀 应用场景',
      content: [
        {
          type: 'app-box',
          content: '中文对话、知识问答、代码生成、工具调用、多模态理解'
        }
      ]
    },
    {
      type: 'section',
      title: '📐 数学原理',
      content: [
        {
          type: 'math-box',
          title: 'GLM 预训练目标',
          formulas: [
            {
              text: 'GLM 结合自回归和自编码：'
            },
            {
              display: 'L = -\\sum_{i \\in M} \\log P(x_i | x_{\\backslash M}, M)'
            },
            {
              text: '其中 $M$ 是被掩码的连续span，模型需要自回归地预测这些span'
            }
          ]
        },
        {
          type: 'math-box',
          title: '双向注意力',
          formulas: [
            {
              text: 'ChatGLM使用双向注意力，可以同时利用前后文信息：'
            },
            {
              display: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V'
            },
            {
              text: '与GPT的单向注意力不同，ChatGLM可以双向理解上下文'
            }
          ]
        }
      ]
    },
    {
      type: 'section',
      title: '💻 Python 代码示例',
      content: [
        {
          type: 'code-box',
          title: '使用 Transformers 库加载 ChatGLM',
          language: 'python',
          code: `from transformers import AutoTokenizer, AutoModel
import torch

# 加载模型和分词器
model_path = "THUDM/chatglm-6b"  # 需要HuggingFace访问权限
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

# 对话
query = "你好"
response, history = model.chat(tokenizer, query, history=[])
print(response)

# 继续对话
query = "介绍一下深度学习"
response, history = model.chat(tokenizer, query, history=history)
print(response)`
        }
      ]
    }
  ]
};

// 技术文档映射表
export const techDocuments = {
  'BERT': bertDocument,
  'ChatGLM': chatglmDocument,
  // 其他文档可以在这里添加
};

// 获取技术文档
export function getTechDocument(nodeName) {
  return techDocuments[nodeName] || null;
}

// 检查是否有技术文档
export function hasTechDocument(nodeName) {
  return techDocuments.hasOwnProperty(nodeName);
}
