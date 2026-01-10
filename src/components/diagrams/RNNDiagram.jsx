import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * RNN架构图解组件
 * 展示循环神经网络的架构、展开形式和工作原理
 */
const RNNDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture', // architecture, unfolded, cell
  title = 'RNN架构图',
  ...props 
}) => {
  const svgRef = useRef(null);
  const [selectedElement, setSelectedElement] = useState(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 60, right: 80, bottom: 80, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const colors = {
      primary: '#667eea',
      secondary: '#764ba2',
      input: '#4facfe',
      hidden: '#f093fb',
      output: '#43e97b',
      text: '#2d3748',
      background: '#f7fafc',
      border: '#e2e8f0'
    };

    // 根据类型渲染不同的图表
    switch (type) {
      case 'unfolded':
        renderUnfoldedRNN(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'cell':
        renderRNNCell(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'architecture':
      default:
        renderRNNArchitecture(g, innerWidth, innerHeight, colors, interactive);
    }

    // 添加标题
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

  // 渲染RNN架构（循环形式）
  function renderRNNArchitecture(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.25;

    // 输入层
    const inputX = centerX - radius * 1.5;
    const inputY = centerY;
    const inputBox = g.append('rect')
      .attr('x', inputX - 60)
      .attr('y', inputY - 30)
      .attr('width', 120)
      .attr('height', 60)
      .attr('fill', colors.input)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('rx', 8)
      .attr('opacity', 0.9);

    g.append('text')
      .attr('x', inputX)
      .attr('y', inputY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('x_t');

    // RNN单元（循环）
    const cellX = centerX;
    const cellY = centerY;
    const cellBox = g.append('rect')
      .attr('x', cellX - 80)
      .attr('y', cellY - 50)
      .attr('width', 160)
      .attr('height', 100)
      .attr('fill', colors.hidden)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 3)
      .attr('rx', 10)
      .attr('opacity', 0.9);

    g.append('text')
      .attr('x', cellX)
      .attr('y', cellY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '18px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('RNN');

    g.append('text')
      .attr('x', cellX)
      .attr('y', cellY + 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('h_t');

    // 输出层
    const outputX = centerX + radius * 1.5;
    const outputY = centerY;
    const outputBox = g.append('rect')
      .attr('x', outputX - 60)
      .attr('y', outputY - 30)
      .attr('width', 120)
      .attr('height', 60)
      .attr('fill', colors.output)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('rx', 8)
      .attr('opacity', 0.9);

    g.append('text')
      .attr('x', outputX)
      .attr('y', outputY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('y_t');

    // 输入到RNN的箭头
    const arrow1 = g.append('path')
      .attr('d', `M ${inputX + 60} ${inputY} L ${cellX - 80} ${cellY}`)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 3)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead)');

    // RNN到输出的箭头
    const arrow2 = g.append('path')
      .attr('d', `M ${cellX + 80} ${cellY} L ${outputX - 60} ${outputY}`)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 3)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead)');

    // 循环连接（隐藏状态反馈）
    const loopPath = `M ${cellX} ${cellY + 50} 
                      Q ${cellX - radius * 0.8} ${cellY + radius * 0.6} ${cellX - radius * 0.3} ${cellY + radius * 0.3}
                      Q ${cellX} ${cellY + radius * 0.1} ${cellX + radius * 0.3} ${cellY + radius * 0.3}
                      Q ${cellX + radius * 0.8} ${cellY + radius * 0.6} ${cellX} ${cellY + 50}`;
    
    const loopArrow = g.append('path')
      .attr('d', loopPath)
      .attr('stroke', colors.secondary)
      .attr('stroke-width', 3)
      .attr('fill', 'none')
      .attr('stroke-dasharray', '5,5')
      .attr('marker-end', 'url(#arrowhead)');

    g.append('text')
      .attr('x', cellX)
      .attr('y', cellY + radius * 0.7 + 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.secondary)
      .attr('font-weight', 'bold')
      .text('h_{t-1} → h_t');

    // 定义箭头标记
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.primary);

    // 添加公式说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('h_t = tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)');

    // 交互效果
    if (interactive) {
      [inputBox, cellBox, outputBox].forEach((box, i) => {
        box
          .on('mouseenter', function() {
            d3.select(this).attr('opacity', 1).attr('stroke-width', 4);
          })
          .on('mouseleave', function() {
            d3.select(this).attr('opacity', 0.9).attr('stroke-width', i === 1 ? 3 : 2);
          });
      });
    }
  }

  // 渲染展开的RNN
  function renderUnfoldedRNN(g, width, height, colors, interactive) {
    const timeSteps = 4;
    const stepWidth = width / (timeSteps + 1);
    const centerY = height / 2;

    // 绘制每个时间步
    for (let t = 0; t < timeSteps; t++) {
      const x = stepWidth * (t + 1);
      
      // 输入
      const inputBox = g.append('rect')
        .attr('x', x - 50)
        .attr('y', centerY - 120)
        .attr('width', 100)
        .attr('height', 40)
        .attr('fill', colors.input)
        .attr('stroke', colors.primary)
        .attr('stroke-width', 2)
        .attr('rx', 5);

      g.append('text')
        .attr('x', x)
        .attr('y', centerY - 95)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', 'white')
        .text(`x_${t}`);

      // RNN单元
      const cellBox = g.append('rect')
        .attr('x', x - 60)
        .attr('y', centerY - 40)
        .attr('width', 120)
        .attr('height', 80)
        .attr('fill', colors.hidden)
        .attr('stroke', colors.primary)
        .attr('stroke-width', 2)
        .attr('rx', 8);

      g.append('text')
        .attr('x', x)
        .attr('y', centerY)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text('RNN');

      g.append('text')
        .attr('x', x)
        .attr('y', centerY + 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(`h_${t}`);

      // 输出
      const outputBox = g.append('rect')
        .attr('x', x - 50)
        .attr('y', centerY + 60)
        .attr('width', 100)
        .attr('height', 40)
        .attr('fill', colors.output)
        .attr('stroke', colors.primary)
        .attr('stroke-width', 2)
        .attr('rx', 5);

      g.append('text')
        .attr('x', x)
        .attr('y', centerY + 85)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', 'white')
        .text(`y_${t}`);

      // 时间步标签
      g.append('text')
        .attr('x', x)
        .attr('y', centerY - 150)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('fill', colors.text)
        .attr('font-weight', 'bold')
        .text(`t=${t}`);

      // 连接箭头（水平）
      if (t < timeSteps - 1) {
        g.append('path')
          .attr('d', `M ${x + 60} ${centerY} L ${x + stepWidth - 60} ${centerY}`)
          .attr('stroke', colors.secondary)
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead-unfolded)');
      }
    }

    // 定义箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-unfolded')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.secondary);

    // 说明文字
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height - 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('RNN展开形式：每个时间步共享相同的参数');
  }

  // 渲染RNN单元内部结构
  function renderRNNCell(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 输入
    const inputX = centerX - 150;
    const inputY = centerY;
    g.append('rect')
      .attr('x', inputX - 40)
      .attr('y', inputY - 20)
      .attr('width', 80)
      .attr('height', 40)
      .attr('fill', colors.input)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', inputX)
      .attr('y', inputY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('x_t');

    // 前一个隐藏状态
    const prevHX = centerX;
    const prevHY = centerY - 100;
    g.append('rect')
      .attr('x', prevHX - 40)
      .attr('y', prevHY - 20)
      .attr('width', 80)
      .attr('height', 40)
      .attr('fill', colors.secondary)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', prevHX)
      .attr('y', prevHY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('h_{t-1}');

    // 权重矩阵
    const wX = centerX;
    const wY = centerY;
    g.append('rect')
      .attr('x', wX - 60)
      .attr('y', wY - 30)
      .attr('width', 120)
      .attr('height', 60)
      .attr('fill', colors.hidden)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 3)
      .attr('rx', 8);

    g.append('text')
      .attr('x', wX)
      .attr('y', wY - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('W_{xh}·x_t +');

    g.append('text')
      .attr('x', wX)
      .attr('y', wY + 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('W_{hh}·h_{t-1}');

    // 激活函数
    const actX = centerX + 150;
    const actY = centerY;
    g.append('rect')
      .attr('x', actX - 40)
      .attr('y', actY - 20)
      .attr('width', 80)
      .attr('height', 40)
      .attr('fill', colors.output)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', actX)
      .attr('y', actY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('tanh');

    // 输出隐藏状态
    const outHX = centerX;
    const outHY = centerY + 100;
    g.append('rect')
      .attr('x', outHX - 40)
      .attr('y', outHY - 20)
      .attr('width', 80)
      .attr('height', 40)
      .attr('fill', colors.secondary)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', outHX)
      .attr('y', outHY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('h_t');

    // 连接箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-cell')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.primary);

    // 输入到权重
    g.append('path')
      .attr('d', `M ${inputX + 40} ${inputY} L ${wX - 60} ${wY}`)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-cell)');

    // 前一个隐藏状态到权重
    g.append('path')
      .attr('d', `M ${prevHX} ${prevHY + 20} L ${wX} ${wY - 30}`)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-cell)');

    // 权重到激活函数
    g.append('path')
      .attr('d', `M ${wX + 60} ${wY} L ${actX - 40} ${actY}`)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-cell)');

    // 激活函数到输出
    g.append('path')
      .attr('d', `M ${actX} ${actY + 20} L ${outHX} ${outHY - 20}`)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-cell)');
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

export default RNNDiagram;
