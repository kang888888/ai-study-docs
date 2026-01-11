import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * 量化推理架构图解组件
 * 展示模型量化过程和量化对比
 */
const QuantizationDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  title = '量化推理',
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
      fp32: '#667eea',
      fp16: '#764ba2',
      int8: '#4facfe',
      int4: '#43e97b',
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

    // 绘制量化对比
    const precisionTypes = [
      { name: 'FP32', color: colors.fp32, size: 100, speed: 1, memory: 100 },
      { name: 'FP16', color: colors.fp16, size: 50, speed: 2, memory: 50 },
      { name: 'INT8', color: colors.int8, size: 25, speed: 4, memory: 25 },
      { name: 'INT4', color: colors.int4, size: 12.5, speed: 8, memory: 12.5 }
    ];

    const boxWidth = innerWidth / precisionTypes.length - 40;
    const boxHeight = innerHeight * 0.6;
    const startY = innerHeight * 0.2;

    precisionTypes.forEach((type, i) => {
      const x = 20 + i * (boxWidth + 40);
      const y = startY;

      // 精度框
      const typeBox = g.append('g')
        .attr('class', 'precision-box')
        .attr('transform', `translate(${x}, ${y})`);

      typeBox.append('rect')
        .attr('width', boxWidth)
        .attr('height', boxHeight)
        .attr('fill', type.color)
        .attr('opacity', 0.7)
        .attr('stroke', type.color)
        .attr('stroke-width', 2)
        .attr('rx', 8);

      typeBox.append('text')
        .attr('x', boxWidth / 2)
        .attr('y', 40)
        .attr('text-anchor', 'middle')
        .attr('font-size', '20px')
        .attr('font-weight', 'bold')
        .attr('fill', '#fff')
        .text(type.name);

      // 模型大小
      typeBox.append('text')
        .attr('x', boxWidth / 2)
        .attr('y', 80)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', '#fff')
        .text(`Size: ${type.size}%`);

      // 速度
      typeBox.append('text')
        .attr('x', boxWidth / 2)
        .attr('y', 110)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', '#fff')
        .text(`Speed: ${type.speed}x`);

      // 内存
      typeBox.append('text')
        .attr('x', boxWidth / 2)
        .attr('y', 140)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', '#fff')
        .text(`Memory: ${type.memory}%`);

      // 精度条
      const accuracy = 100 - i * 5;
      typeBox.append('rect')
        .attr('x', 10)
        .attr('y', boxHeight - 60)
        .attr('width', (boxWidth - 20) * (accuracy / 100))
        .attr('height', 20)
        .attr('fill', '#fff')
        .attr('opacity', 0.8);

      typeBox.append('text')
        .attr('x', boxWidth / 2)
        .attr('y', boxHeight - 35)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#fff')
        .text(`Accuracy: ${accuracy}%`);
    });

    // 量化流程
    const processY = startY + boxHeight + 40;
    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', processY)
      .attr('text-anchor', 'middle')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .attr('fill', colors.text)
      .text('量化流程: FP32 → Calibration → Quantization → INT8/INT4');

    // 流程箭头
    const processSteps = ['FP32', 'Calibration', 'Quantization', 'INT8'];
    const stepWidth = innerWidth / processSteps.length;
    for (let i = 0; i < processSteps.length - 1; i++) {
      const x1 = (i + 1) * stepWidth;
      const x2 = (i + 1) * stepWidth;
      g.append('line')
        .attr('x1', x1)
        .attr('y1', processY + 20)
        .attr('x2', x2)
        .attr('y2', processY + 20)
        .attr('stroke', colors.int8)
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrowhead)');
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
      .attr('fill', colors.int8);

  }, [width, height, interactive, title]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{ display: 'block', margin: '0 auto' }}
    />
  );
};

export default QuantizationDiagram;
