// 交互式SVG组件映射
import TransformerDiagram from './TransformerDiagram';
import GenericDiagram from './GenericDiagram';
import RNNDiagram from './RNNDiagram';
import CNNDiagram from './CNNDiagram';
import LSTMDiagram from './LSTMDiagram';
import BERTDiagram from './BERTDiagram';
import MLPDiagram from './MLPDiagram';
import ResNetDiagram from './ResNetDiagram';
import YOLODiagram from './YOLODiagram';
import GRUDiagram from './GRUDiagram';
import MoEDiagram from './MoEDiagram';
import MambaDiagram from './MambaDiagram';
import ViTDiagram from './ViTDiagram';
import CLIPDiagram from './CLIPDiagram';
import DiffusionDiagram from './DiffusionDiagram';
import VAEDiagram from './VAEDiagram';
import GANDiagram from './GANDiagram';
import UNetDiagram from './UNetDiagram';
import GNNDiagram from './GNNDiagram';
import DQNDiagram from './DQNDiagram';
import DBNDiagram from './DBNDiagram';
import RWKVDiagram from './RWKVDiagram';
import MirasDiagram from './MirasDiagram';
import TitansDiagram from './TitansDiagram';
import SpeculativeDecodingDiagram from './SpeculativeDecodingDiagram';
import MathFunctionDiagram from './MathFunctionDiagram';
import ParallelTrainingDiagram from './ParallelTrainingDiagram';
import QuantizationDiagram from './QuantizationDiagram';

// 组件映射表
export const diagramComponents = {
  TransformerDiagram,
  GenericDiagram,
  RNNDiagram,
  CNNDiagram,
  LSTMDiagram,
  BERTDiagram,
  MLPDiagram,
  ResNetDiagram,
  YOLODiagram,
  GRUDiagram,
  MoEDiagram,
  MambaDiagram,
  ViTDiagram,
  CLIPDiagram,
  DiffusionDiagram,
  VAEDiagram,
  GANDiagram,
  UNetDiagram,
  GNNDiagram,
  DQNDiagram,
  DBNDiagram,
  RWKVDiagram,
  MirasDiagram,
  TitansDiagram,
  SpeculativeDecodingDiagram,
  MathFunctionDiagram,
  ParallelTrainingDiagram,
  QuantizationDiagram,
};

// 根据组件名称获取组件
export const getDiagramComponent = (componentName) => {
  return diagramComponents[componentName] || null;
};

// 检查组件是否存在
export const hasDiagramComponent = (componentName) => {
  return componentName in diagramComponents;
};

export default diagramComponents;
