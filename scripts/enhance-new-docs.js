// 为新创建的文档添加架构图解和代码示例
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const knowledgeDir = path.join(__dirname, '../src/data/knowledge');

// 需要增强的文档配置
const enhancements = [
  {
    file: 'DiT.json',
    diagram: {
      type: 'architecture',
      title: 'DiT架构',
      caption: 'DiT架构图'
    },
    code: {
      title: 'DiT模型使用',
      language: 'python',
      code: `from diffusers import DiffusionPipeline
import torch

# 加载DiT模型
pipe = DiffusionPipeline.from_pretrained("facebook/dit-base")

# 生成视频
prompt = "A beautiful sunset over the ocean"
video = pipe(prompt, num_inference_steps=50).images[0]`
    }
  },
  {
    file: '数据并行.json',
    diagram: {
      type: 'architecture',
      title: '数据并行原理',
      caption: '数据并行原理'
    },
    code: {
      title: 'PyTorch数据并行',
      language: 'python',
      code: `import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 创建模型
model = nn.Linear(10, 1)
model = model.cuda()
model = DDP(model)

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()`
    }
  },
  {
    file: '模型并行.json',
    diagram: {
      type: 'architecture',
      title: '模型并行原理',
      caption: '模型并行原理'
    },
    code: {
      title: '模型并行示例',
      language: 'python',
      code: `import torch
import torch.nn as nn
from torch.nn.parallel import parallel_apply

# 将模型拆分到多个GPU
device_ids = [0, 1]
model_part1 = nn.Sequential(...).to(device_ids[0])
model_part2 = nn.Sequential(...).to(device_ids[1])

# 前向传播
def forward(input):
    intermediate = model_part1(input)
    output = model_part2(intermediate)
    return output`
    }
  },
  {
    file: '流水线并行.json',
    diagram: {
      type: 'flow',
      title: '流水线并行流程',
      caption: '流水线并行流程'
    },
    code: {
      title: '流水线并行示例',
      language: 'python',
      code: `import torch
from torch.distributed.pipeline.sync import Pipe

# 创建模型分段
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.Linear(20, 30),
    nn.Linear(30, 1)
)

# 创建流水线
model = Pipe(model, chunks=4)

# 训练
output = model(input)`
    }
  },
  {
    file: '图优化.json',
    diagram: {
      type: 'architecture',
      title: '计算图优化',
      caption: '计算图优化'
    },
    code: {
      title: 'TensorRT图优化',
      language: 'python',
      code: `import tensorrt as trt

# 创建TensorRT引擎
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

# 解析ONNX模型
parser.parse_from_file("model.onnx")

# 构建优化引擎
builder.max_batch_size = 1
builder.max_workspace_size = 1 << 30
engine = builder.build_cuda_engine(network)`
    }
  },
  {
    file: '量化推理.json',
    diagram: {
      type: 'comparison',
      title: '量化对比',
      caption: '量化对比'
    },
    code: {
      title: 'INT8量化推理',
      language: 'python',
      code: `import torch
from torch.quantization import quantize_dynamic

# 加载模型
model = torch.load("model.pth")

# 动态量化
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 推理
with torch.no_grad():
    output = quantized_model(input)`
    }
  },
  {
    file: '模型剪枝.json',
    diagram: {
      type: 'architecture',
      title: '模型剪枝流程',
      caption: '模型剪枝流程'
    },
    code: {
      title: '模型剪枝示例',
      language: 'python',
      code: `import torch
import torch.nn.utils.prune as prune

# 创建模型
model = nn.Linear(10, 1)

# 剪枝
prune.l1_unstructured(model, name="weight", amount=0.2)

# 永久移除剪枝
prune.remove(model, "weight")`
    }
  },
  {
    file: '智能体框架.json',
    diagram: {
      type: 'architecture',
      title: '智能体框架架构',
      caption: '智能体框架架构'
    },
    code: {
      title: 'LangGraph示例',
      language: 'python',
      code: `from langgraph.graph import StateGraph, END

# 定义状态图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# 添加边
workflow.add_edge("agent", "tools")
workflow.add_edge("tools", END)

# 编译并运行
app = workflow.compile()
result = app.invoke({"messages": [("user", "Hello")]})`
    }
  },
  {
    file: '工具调用.json',
    diagram: {
      type: 'flow',
      title: '工具调用流程',
      caption: '工具调用流程'
    },
    code: {
      title: 'Function Calling示例',
      language: 'python',
      code: `from openai import OpenAI

client = OpenAI()

# 定义工具
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

# 调用模型
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools
)`
    }
  },
  {
    file: '多智能体系统.json',
    diagram: {
      type: 'architecture',
      title: '多智能体系统架构',
      caption: '多智能体系统架构'
    },
    code: {
      title: '多智能体协作示例',
      language: 'python',
      code: `from crewai import Agent, Task, Crew

# 创建智能体
researcher = Agent(
    role='Researcher',
    goal='Research information',
    backstory='Expert researcher'
)

writer = Agent(
    role='Writer',
    goal='Write content',
    backstory='Expert writer'
)

# 创建任务
task1 = Task(description='Research topic', agent=researcher)
task2 = Task(description='Write article', agent=writer)

# 创建团队
crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff()`
    }
  },
  {
    file: 'Transformers.json',
    diagram: {
      type: 'architecture',
      title: 'Transformers框架',
      caption: 'Transformers框架'
    },
    code: {
      title: '使用Pipeline',
      language: 'python',
      code: `from transformers import pipeline

# 创建Pipeline
classifier = pipeline("sentiment-analysis")

# 使用Pipeline
result = classifier("I love this product!")
print(result)`
    }
  },
  {
    file: 'Accelerate.json',
    diagram: {
      type: 'architecture',
      title: 'Accelerate架构',
      caption: 'Accelerate架构'
    },
    code: {
      title: 'Accelerate使用',
      language: 'python',
      code: `from accelerate import Accelerator

# 初始化Accelerator
accelerator = Accelerator()

# 准备模型和数据
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()`
    }
  },
  {
    file: '性能分析.json',
    diagram: {
      type: 'architecture',
      title: '性能分析流程',
      caption: '性能分析流程'
    },
    code: {
      title: 'PyTorch Profiler',
      language: 'python',
      code: `import torch
from torch.profiler import profile, record_function, ProfilerActivity

# 使用Profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    with record_function("model_inference"):
        output = model(input)

# 查看结果
print(prof.key_averages().table(sort_by="cuda_time_total"))`
    }
  },
  {
    file: 'Space-Time Latent Patch.json',
    diagram: {
      type: 'architecture',
      title: 'Space-Time Latent Patch架构',
      caption: 'Space-Time Latent Patch架构'
    },
    code: {
      title: '视频编码示例',
      language: 'python',
      code: `import torch
import torch.nn as nn

# Space-Time Latent Patch编码
class STLPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv3d(3, 768, kernel_size=(2, 16, 16))
    
    def forward(self, video):
        # 提取时空补丁
        patches = self.patch_embed(video)
        return patches`
    }
  },
  {
    file: 'Whisper.json',
    diagram: {
      type: 'architecture',
      title: 'Whisper架构',
      caption: 'Whisper架构'
    },
    code: {
      title: 'Whisper使用',
      language: 'python',
      code: `import whisper

# 加载模型
model = whisper.load_model("base")

# 转录音频
result = model.transcribe("audio.mp3")
print(result["text"])`
    }
  },
  {
    file: 'AudioLM.json',
    diagram: {
      type: 'architecture',
      title: 'AudioLM架构',
      caption: 'AudioLM架构'
    },
    code: {
      title: 'AudioLM生成',
      language: 'python',
      code: `import torch
from audiolm import AudioLM

# 加载模型
model = AudioLM.from_pretrained("google/audiolm")

# 生成音频
audio = model.generate(prompt="A piano melody", duration=5.0)`
    }
  },
  {
    file: 'GPT-4o Omni.json',
    diagram: {
      type: 'architecture',
      title: 'GPT-4o Omni架构',
      caption: 'GPT-4o Omni架构'
    },
    code: {
      title: '多模态调用',
      language: 'python',
      code: `from openai import OpenAI

client = OpenAI()

# 多模态调用
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "image.jpg"}}
        ]}
    ]
)`
    }
  },
  {
    file: '格式转换.json',
    diagram: {
      type: 'flow',
      title: '格式转换流程',
      caption: '格式转换流程'
    },
    code: {
      title: '格式转换示例',
      language: 'python',
      code: `import json
import pandas as pd

# JSONL转Parquet
def jsonl_to_parquet(jsonl_file, parquet_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df.to_parquet(parquet_file)`
    }
  },
  {
    file: '质量评估.json',
    diagram: {
      type: 'architecture',
      title: '质量评估流程',
      caption: '质量评估流程'
    },
    code: {
      title: '数据质量评估',
      language: 'python',
      code: `from datasets import load_dataset
from evaluate import load

# 加载数据集
dataset = load_dataset("dataset_name")

# 评估质量
accuracy = evaluate_accuracy(dataset)
diversity = evaluate_diversity(dataset)
consistency = evaluate_consistency(dataset)`
    }
  },
  {
    file: '数据管理.json',
    diagram: {
      type: 'architecture',
      title: '数据管理架构',
      caption: '数据管理架构'
    },
    code: {
      title: '数据版本管理',
      language: 'python',
      code: `import dvc.api

# 使用DVC管理数据版本
data = dvc.api.read(
    'data/dataset.csv',
    repo='https://github.com/user/repo',
    rev='v1.0'
)`
    }
  }
];

// 处理每个文档
enhancements.forEach(({ file, diagram, code }) => {
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
    
    // 检查是否已有架构图解
    const hasDiagram = sections.some(s => 
      s.title && s.title.includes('架构图解')
    );
    
    // 检查是否已有代码示例
    const hasCode = sections.some(s => 
      s.title && (s.title.includes('代码示例') || s.title.includes('💻'))
    );
    
    // 找到应用场景的位置
    let appSceneIndex = sections.findIndex(s => 
      s.title && (s.title.includes('应用场景') || s.title.includes('🚀 应用场景'))
    );
    
    // 添加架构图解（在应用场景之前）
    if (!hasDiagram && diagram) {
      const diagramSection = {
        type: "section",
        title: "📊 架构图解",
        content: [
          {
            type: "diagram-gallery",
            images: [
              {
                type: "svg-d3",
                component: "GenericDiagram",
                caption: diagram.caption,
                width: 1000,
                height: 800,
                interactive: true,
                props: {
                  type: diagram.type,
                  title: diagram.title
                }
              }
            ]
          }
        ]
      };
      
      if (appSceneIndex >= 0) {
        sections.splice(appSceneIndex, 0, diagramSection);
      } else {
        sections.push(diagramSection);
      }
      hasChanges = true;
    }
    
    // 添加代码示例（在最后）
    if (!hasCode && code) {
      const codeSection = {
        type: "section",
        title: "💻 Python 代码示例",
        content: [
          {
            type: "code-box",
            title: code.title,
            language: code.language,
            code: code.code
          }
        ]
      };
      sections.push(codeSection);
      hasChanges = true;
    }
    
    // 保存文件
    if (hasChanges) {
      doc.content = sections;
      const newContent = JSON.stringify(doc, null, 2);
      fs.writeFileSync(filePath, newContent, 'utf-8');
      console.log(`✅ ${file}: 已添加架构图解和代码示例`);
    } else {
      console.log(`⏭️  ${file}: 无需修改`);
    }
    
  } catch (error) {
    console.error(`❌ 处理 ${file} 时出错:`, error.message);
  }
});

console.log('\n✅ 批量增强完成！');
