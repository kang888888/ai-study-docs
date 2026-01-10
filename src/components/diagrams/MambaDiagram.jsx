import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * Mamba状态空间模型架构图解组件
 * 展示Mamba的SSM架构、选择性扫描机制和线性复杂度
 */
const MambaDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture', // architecture, ssm, selective
  title = 'Mamba架构图',
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
      input: '#4facfe',
      ssm: '#667eea',
      selective: '#f093fb',
      output: '#43e97b',
      scan: '#764ba2',
      text: '#2d3748'
    };

    switch (type) {
      case 'ssm':
        renderSSM(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'selective':
        renderSelective(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'architecture':
      default:
        renderArchitecture(g, innerWidth, innerHeight, colors, interactive);
    }

    if (title) {
      g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', -30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '24px')
        .attr('font-weight', 'bold')
        .attr('fill', colors.text)
        .text(title);
    }
  }, [width, height, type, interactive, title]);

  // 渲染Mamba整体架构
  function renderArchitecture(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const inputY = 80;
    const expandY = 180;
    const ssmY = 320;
    const projectY = 460;
    const outputY = 600;

    // 输入
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', inputY - 30)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.input)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', inputY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Input x');

    // 扩展投影
    g.append('rect')
      .attr('x', centerX - 120)
      .attr('y', expandY - 30)
      .attr('width', 240)
      .attr('height', 60)
      .attr('fill', colors.input)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', expandY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Expand Projection');

    // SSM块（核心）
    g.append('rect')
      .attr('x', centerX - 150)
      .attr('y', ssmY - 50)
      .attr('width', 300)
      .attr('height', 100)
      .attr('fill', colors.ssm)
      .attr('stroke', colors.text)
      .attr('stroke-width', 3)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', ssmY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Selective SSM');

    g.append('text')
      .attr('x', centerX)
      .attr('y', ssmY + 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('h_t = A h_{t-1} + B x_t');

    g.append('text')
      .attr('x', centerX)
      .attr('y', ssmY + 35)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('y_t = C h_t');

    // 压缩投影
    g.append('rect')
      .attr('x', centerX - 120)
      .attr('y', projectY - 30)
      .attr('width', 240)
      .attr('height', 60)
      .attr('fill', colors.input)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', projectY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Compress Projection');

    // 输出
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', outputY - 30)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.output)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', outputY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Output y');

    // 箭头定义
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-mamba')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    // 连接箭头
    const arrows = [
      [centerX, inputY + 30, centerX, expandY - 30],
      [centerX, expandY + 30, centerX, ssmY - 50],
      [centerX, ssmY + 50, centerX, projectY - 30],
      [centerX, projectY + 30, centerX, outputY - 30]
    ];

    arrows.forEach(([x1, y1, x2, y2]) => {
      g.append('path')
        .attr('d', `M ${x1} ${y1} L ${x2} ${y2}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-mamba)');
    });

    // 复杂度标注
    g.append('text')
      .attr('x', centerX + 200)
      .attr('y', ssmY)
      .attr('font-size', '14px')
      .attr('fill', colors.selective)
      .attr('font-weight', 'bold')
      .text('O(L) 线性复杂度');

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('Mamba架构：通过选择性SSM实现线性复杂度，无KV Cache');
  }

  // 渲染SSM机制
  function renderSSM(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 状态空间方程
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 100)
      .attr('text-anchor', 'middle')
      .attr('font-size', '18px')
      .attr('fill', colors.text)
      .attr('font-weight', 'bold')
      .text('状态空间模型（SSM）');

    // 状态方程
    g.append('rect')
      .attr('x', centerX - 200)
      .attr('y', centerY - 50)
      .attr('width', 400)
      .attr('height', 60)
      .attr('fill', colors.ssm)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .text('h_t = A h_{t-1} + B x_t  (状态更新)');

    // 输出方程
    g.append('rect')
      .attr('x', centerX - 200)
      .attr('y', centerY + 30)
      .attr('width', 400)
      .attr('height', 60)
      .attr('fill', colors.ssm)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 55)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .text('y_t = C h_t  (输出)');

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 150)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('A, B, C 是可学习参数，h_t 是隐藏状态');
  }

  // 渲染选择性机制
  function renderSelective(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 输入序列
    const tokens = ['Token 1', 'Token 2', 'Token 3', 'Token 4', 'Token 5'];
    const tokenSpacing = 120;
    const startX = centerX - (tokens.length - 1) * tokenSpacing / 2;

    tokens.forEach((token, i) => {
      const x = startX + i * tokenSpacing;
      const y = centerY - 150;

      g.append('rect')
        .attr('x', x - 40)
        .attr('y', y - 20)
        .attr('width', 80)
        .attr('height', 40)
        .attr('fill', colors.input)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5);

      g.append('text')
        .attr('x', x)
        .attr('y', y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(token);
    });

    // 选择性参数（根据输入动态变化）
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 50)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', colors.text)
      .attr('font-weight', 'bold')
      .text('选择性参数：B, C 根据输入 x_t 动态计算');

    // 动态参数示例
    tokens.forEach((token, i) => {
      const x = startX + i * tokenSpacing;
      const y = centerY;

      g.append('rect')
        .attr('x', x - 50)
        .attr('y', y - 30)
        .attr('width', 100)
        .attr('height', 60)
        .attr('fill', colors.selective)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5);

      g.append('text')
        .attr('x', x)
        .attr('y', y - 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .attr('fill', 'white')
        .text(`B_${i + 1}, C_${i + 1}`);

      g.append('text')
        .attr('x', x)
        .attr('y', y + 15)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', 'white')
        .text('= f(x_t)');
    });

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 150)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('选择性机制：参数根据输入内容动态调整，类似Attention的内容选择能力');
  }

  return (
    <div className="generic-diagram-container">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          backgroundColor: '#ffffff'
        }}
      />
    </div>
  );
};

export default MambaDiagram;
