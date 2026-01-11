import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * 并行训练架构图解组件
 * 展示数据并行、模型并行、流水线并行等并行训练策略
 */
const ParallelTrainingDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'data', // data, model, pipeline
  title = '并行训练架构',
  ...props 
}) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 60, right: 80, bottom: 80, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const colors = {
      primary: '#667eea',
      secondary: '#764ba2',
      accent: '#4facfe',
      success: '#43e97b',
      text: '#2d3748',
      background: '#f7fafc'
    };

    // 添加标题
    if (title) {
      g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', -20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '24px')
        .attr('font-weight', 'bold')
        .attr('fill', colors.text)
        .text(title);
    }

    switch (type) {
      case 'data':
        renderDataParallel(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'model':
        renderModelParallel(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'pipeline':
        renderPipelineParallel(g, innerWidth, innerHeight, colors, interactive);
        break;
      default:
        renderDataParallel(g, innerWidth, innerHeight, colors, interactive);
    }

  }, [width, height, type, interactive, title]);

  // 渲染数据并行
  const renderDataParallel = (g, width, height, colors, interactive) => {
    const numGPUs = 4;
    const gpuWidth = (width - 100) / numGPUs;
    const gpuHeight = height * 0.6;
    const startY = height * 0.2;

    // 绘制GPU节点
    for (let i = 0; i < numGPUs; i++) {
      const x = 50 + i * (gpuWidth + 20);
      const y = startY;

      // GPU框
      const gpuBox = g.append('g')
        .attr('class', 'gpu-node')
        .attr('transform', `translate(${x}, ${y})`);

      gpuBox.append('rect')
        .attr('width', gpuWidth)
        .attr('height', gpuHeight)
        .attr('fill', colors.background)
        .attr('stroke', colors.primary)
        .attr('stroke-width', 2)
        .attr('rx', 8);

      gpuBox.append('text')
        .attr('x', gpuWidth / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('font-weight', 'bold')
        .attr('fill', colors.text)
        .text(`GPU ${i}`);

      // 数据分片
      gpuBox.append('rect')
        .attr('x', 10)
        .attr('y', 50)
        .attr('width', gpuWidth - 20)
        .attr('height', 40)
        .attr('fill', colors.accent)
        .attr('opacity', 0.7);

      gpuBox.append('text')
        .attr('x', gpuWidth / 2)
        .attr('y', 75)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#fff')
        .text(`Data Shard ${i}`);

      // 模型
      gpuBox.append('rect')
        .attr('x', 10)
        .attr('y', 100)
        .attr('width', gpuWidth - 20)
        .attr('height', 60)
        .attr('fill', colors.secondary)
        .attr('opacity', 0.7);

      gpuBox.append('text')
        .attr('x', gpuWidth / 2)
        .attr('y', 135)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#fff')
        .text('Model');

      // 梯度
      gpuBox.append('rect')
        .attr('x', 10)
        .attr('y', 170)
        .attr('width', gpuWidth - 20)
        .attr('height', 30)
        .attr('fill', colors.success)
        .attr('opacity', 0.7);

      gpuBox.append('text')
        .attr('x', gpuWidth / 2)
        .attr('y', 190)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#fff')
        .text('Gradients');
    }

    // AllReduce操作
    const allReduceY = startY + gpuHeight + 30;
    g.append('text')
      .attr('x', width / 2)
      .attr('y', allReduceY)
      .attr('text-anchor', 'middle')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .attr('fill', colors.primary)
      .text('AllReduce (Gradient Synchronization)');

    // 连接线
    for (let i = 0; i < numGPUs; i++) {
      const x = 50 + i * (gpuWidth + 20) + gpuWidth / 2;
      g.append('line')
        .attr('x1', x)
        .attr('y1', startY + gpuHeight)
        .attr('x2', width / 2)
        .attr('y2', allReduceY - 10)
        .attr('stroke', colors.primary)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5');
    }
  };

  // 渲染模型并行
  const renderModelParallel = (g, width, height, colors, interactive) => {
    const numGPUs = 4;
    const layerHeight = height / (numGPUs + 1);
    const layerWidth = width * 0.8;
    const startX = width * 0.1;

    // 绘制模型层
    for (let i = 0; i < numGPUs; i++) {
      const y = i * layerHeight + 50;

      // 层框
      const layerBox = g.append('g')
        .attr('class', 'layer-node')
        .attr('transform', `translate(${startX}, ${y})`);

      layerBox.append('rect')
        .attr('width', layerWidth)
        .attr('height', layerHeight - 20)
        .attr('fill', colors.background)
        .attr('stroke', colors.secondary)
        .attr('stroke-width', 2)
        .attr('rx', 8);

      layerBox.append('text')
        .attr('x', layerWidth / 2)
        .attr('y', (layerHeight - 20) / 2)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('font-weight', 'bold')
        .attr('fill', colors.text)
        .text(`Layer ${i + 1} (GPU ${i})`);

      // 连接线
      if (i < numGPUs - 1) {
        g.append('line')
          .attr('x1', startX + layerWidth / 2)
          .attr('y1', y + layerHeight - 20)
          .attr('x2', startX + layerWidth / 2)
          .attr('y2', y + layerHeight)
          .attr('stroke', colors.primary)
          .attr('stroke-width', 2)
          .attr('marker-end', 'url(#arrowhead)');
      }
    }

    // 添加箭头标记
    const defs = g.append('defs');
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 5)
      .attr('refY', 5)
      .attr('orient', 'auto')
      .append('polygon')
      .attr('points', '0 0, 10 5, 0 10')
      .attr('fill', colors.primary);
  };

  // 渲染流水线并行
  const renderPipelineParallel = (g, width, height, colors, interactive) => {
    const numStages = 4;
    const stageWidth = (width - 100) / numStages;
    const stageHeight = height * 0.6;
    const startY = height * 0.2;

    // 绘制流水线阶段
    for (let i = 0; i < numStages; i++) {
      const x = 50 + i * (stageWidth + 20);
      const y = startY;

      // 阶段框
      const stageBox = g.append('g')
        .attr('class', 'stage-node')
        .attr('transform', `translate(${x}, ${y})`);

      stageBox.append('rect')
        .attr('width', stageWidth)
        .attr('height', stageHeight)
        .attr('fill', colors.background)
        .attr('stroke', colors.accent)
        .attr('stroke-width', 2)
        .attr('rx', 8);

      stageBox.append('text')
        .attr('x', stageWidth / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('font-weight', 'bold')
        .attr('fill', colors.text)
        .text(`Stage ${i + 1}`);

      // 微批次
      for (let j = 0; j < 3; j++) {
        const batchY = 50 + j * (stageHeight - 60) / 3;
        stageBox.append('rect')
          .attr('x', 10)
          .attr('y', batchY)
          .attr('width', stageWidth - 20)
          .attr('height', (stageHeight - 60) / 3 - 10)
          .attr('fill', colors.primary)
          .attr('opacity', 0.3 + j * 0.2);

        stageBox.append('text')
          .attr('x', stageWidth / 2)
          .attr('y', batchY + (stageHeight - 60) / 6)
          .attr('text-anchor', 'middle')
          .attr('font-size', '12px')
          .attr('fill', colors.text)
          .text(`Batch ${j + 1}`);
      }

      // 连接线
      if (i < numStages - 1) {
        g.append('line')
          .attr('x1', x + stageWidth)
          .attr('y1', y + stageHeight / 2)
          .attr('x2', x + stageWidth + 20)
          .attr('y2', y + stageHeight / 2)
          .attr('stroke', colors.accent)
          .attr('stroke-width', 2)
          .attr('marker-end', 'url(#arrowhead)');
      }
    }

    // 添加箭头标记
    const defs = g.append('defs');
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 5)
      .attr('refY', 5)
      .attr('orient', 'auto')
      .append('polygon')
      .attr('points', '0 0, 10 5, 0 10')
      .attr('fill', colors.accent);
  };

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{ display: 'block', margin: '0 auto' }}
    />
  );
};

export default ParallelTrainingDiagram;
