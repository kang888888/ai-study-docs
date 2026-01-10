import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * CNN架构图解组件
 * 展示卷积神经网络的架构、卷积操作和特征提取过程
 */
const CNNDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture', // architecture, convolution, pooling
  title = 'CNN架构图',
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

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const colors = {
      input: '#4facfe',
      conv: '#667eea',
      pool: '#764ba2',
      fc: '#43e97b',
      output: '#f093fb',
      text: '#2d3748',
      background: '#f7fafc'
    };

    // 根据类型渲染不同的图表
    switch (type) {
      case 'convolution':
        renderConvolution(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'pooling':
        renderPooling(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'architecture':
      default:
        renderCNNArchitecture(g, innerWidth, innerHeight, colors, interactive);
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

  // 渲染CNN整体架构
  function renderCNNArchitecture(g, width, height, colors, interactive) {
    const layers = [
      { name: 'Input', type: 'input', size: '32×32×3', y: 50 },
      { name: 'Conv1', type: 'conv', size: '32×32×32', y: 150 },
      { name: 'Pool1', type: 'pool', size: '16×16×32', y: 250 },
      { name: 'Conv2', type: 'conv', size: '16×16×64', y: 350 },
      { name: 'Pool2', type: 'pool', size: '8×8×64', y: 450 },
      { name: 'Conv3', type: 'conv', size: '8×8×128', y: 550 },
      { name: 'Pool3', type: 'pool', size: '4×4×128', y: 650 },
      { name: 'FC', type: 'fc', size: '512', y: 750 },
      { name: 'Output', type: 'output', size: '10', y: 850 }
    ];

    const centerX = width / 2;
    const layerWidth = 200;

    layers.forEach((layer, i) => {
      const y = layer.y;
      let fillColor, strokeColor;
      
      switch (layer.type) {
        case 'input':
          fillColor = colors.input;
          break;
        case 'conv':
          fillColor = colors.conv;
          break;
        case 'pool':
          fillColor = colors.pool;
          break;
        case 'fc':
          fillColor = colors.fc;
          break;
        case 'output':
          fillColor = colors.output;
          break;
        default:
          fillColor = colors.background;
      }

      const box = g.append('rect')
        .attr('x', centerX - layerWidth / 2)
        .attr('y', y - 30)
        .attr('width', layerWidth)
        .attr('height', 60)
        .attr('fill', fillColor)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 8)
        .attr('opacity', 0.9);

      g.append('text')
        .attr('x', centerX)
        .attr('y', y)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text(layer.name);

      g.append('text')
        .attr('x', centerX)
        .attr('y', y + 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(layer.size);

      // 连接箭头
      if (i < layers.length - 1) {
        const arrow = g.append('path')
          .attr('d', `M ${centerX} ${y + 30} L ${centerX} ${layers[i + 1].y - 30}`)
          .attr('stroke', colors.text)
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead-cnn)');
      }

      if (interactive) {
        box
          .on('mouseenter', function() {
            d3.select(this).attr('opacity', 1).attr('stroke-width', 3);
          })
          .on('mouseleave', function() {
            d3.select(this).attr('opacity', 0.9).attr('stroke-width', 2);
          });
      }
    });

    // 定义箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-cnn')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);
  }

  // 渲染卷积操作
  function renderConvolution(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 输入特征图
    const inputSize = 80;
    const inputX = centerX - 200;
    const inputY = centerY;
    
    // 绘制输入网格
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        g.append('rect')
          .attr('x', inputX + i * 16)
          .attr('y', inputY + j * 16)
          .attr('width', 14)
          .attr('height', 14)
          .attr('fill', colors.input)
          .attr('stroke', colors.text)
          .attr('stroke-width', 1)
          .attr('opacity', 0.7);
      }
    }

    g.append('text')
      .attr('x', inputX + 40)
      .attr('y', inputY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('输入特征图 5×5');

    // 卷积核
    const kernelX = centerX;
    const kernelY = centerY;
    const kernelSize = 48;
    
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        g.append('rect')
          .attr('x', kernelX - 24 + i * 16)
          .attr('y', kernelY - 24 + j * 16)
          .attr('width', 14)
          .attr('height', 14)
          .attr('fill', colors.conv)
          .attr('stroke', colors.text)
          .attr('stroke-width', 2)
          .attr('opacity', 0.9);
      }
    }

    g.append('text')
      .attr('x', kernelX)
      .attr('y', kernelY - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('卷积核 3×3');

    // 输出特征图
    const outputX = centerX + 200;
    const outputY = centerY;
    
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        g.append('rect')
          .attr('x', outputX + i * 16)
          .attr('y', outputY + j * 16)
          .attr('width', 14)
          .attr('height', 14)
          .attr('fill', colors.output)
          .attr('stroke', colors.text)
          .attr('stroke-width', 1)
          .attr('opacity', 0.7);
      }
    }

    g.append('text')
      .attr('x', outputX + 24)
      .attr('y', outputY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('输出特征图 3×3');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-conv')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    g.append('path')
      .attr('d', `M ${inputX + 80} ${centerY} L ${kernelX - 24} ${kernelY}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-conv)');

    g.append('path')
      .attr('d', `M ${kernelX + 24} ${kernelY} L ${outputX} ${outputY}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-conv)');

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('卷积操作：滑动窗口在输入上计算点积');
  }

  // 渲染池化操作
  function renderPooling(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 输入特征图
    const inputX = centerX - 200;
    const inputY = centerY;
    
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const value = Math.floor(Math.random() * 9) + 1;
        g.append('rect')
          .attr('x', inputX + i * 20)
          .attr('y', inputY + j * 20)
          .attr('width', 18)
          .attr('height', 18)
          .attr('fill', colors.input)
          .attr('stroke', colors.text)
          .attr('stroke-width', 1)
          .attr('opacity', 0.7);

        g.append('text')
          .attr('x', inputX + i * 20 + 9)
          .attr('y', inputY + j * 20 + 13)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', colors.text)
          .text(value);
      }
    }

    g.append('text')
      .attr('x', inputX + 40)
      .attr('y', inputY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('输入 4×4');

    // 池化窗口
    const poolX = centerX;
    const poolY = centerY;
    
    // 绘制2×2池化窗口
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        g.append('rect')
          .attr('x', poolX - 20 + i * 40)
          .attr('y', poolY - 20 + j * 40)
          .attr('width', 38)
          .attr('height', 38)
          .attr('fill', 'none')
          .attr('stroke', colors.pool)
          .attr('stroke-width', 3)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.7);
      }
    }

    g.append('text')
      .attr('x', poolX)
      .attr('y', poolY - 50)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('Max Pooling 2×2');

    // 输出特征图
    const outputX = centerX + 200;
    const outputY = centerY;
    
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        g.append('rect')
          .attr('x', outputX + i * 20)
          .attr('y', outputY + j * 20)
          .attr('width', 18)
          .attr('height', 18)
          .attr('fill', colors.output)
          .attr('stroke', colors.text)
          .attr('stroke-width', 1)
          .attr('opacity', 0.7);

        g.append('text')
          .attr('x', outputX + i * 20 + 9)
          .attr('y', outputY + j * 20 + 13)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', colors.text)
          .text('9');
      }
    }

    g.append('text')
      .attr('x', outputX + 20)
      .attr('y', outputY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('输出 2×2');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-pool')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    g.append('path')
      .attr('d', `M ${inputX + 80} ${centerY} L ${poolX - 20} ${poolY}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-pool)');

    g.append('path')
      .attr('d', `M ${poolX + 20} ${poolY} L ${outputX} ${outputY}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-pool)');
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

export default CNNDiagram;
