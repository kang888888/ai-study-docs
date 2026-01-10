import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * MLP架构图解组件
 * 展示多层感知机的结构、前向传播和反向传播
 */
const MLPDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture', // architecture, forward, backward
  title = 'MLP架构图',
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
      hidden: '#667eea',
      output: '#43e97b',
      activation: '#f093fb',
      text: '#2d3748'
    };

    switch (type) {
      case 'forward':
        renderForward(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'backward':
        renderBackward(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'architecture':
      default:
        renderMLPArchitecture(g, innerWidth, innerHeight, colors, interactive);
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

  // 渲染MLP架构
  function renderMLPArchitecture(g, width, height, colors, interactive) {
    const layers = [
      { name: 'Input', size: 4, y: 100, color: colors.input },
      { name: 'Hidden 1', size: 8, y: 250, color: colors.hidden },
      { name: 'Hidden 2', size: 6, y: 400, color: colors.hidden },
      { name: 'Output', size: 2, y: 550, color: colors.output }
    ];

    const centerX = width / 2;
    const neuronRadius = 15;
    const layerSpacing = 200;

    layers.forEach((layer, layerIdx) => {
      const layerX = centerX - (layerIdx - 1.5) * layerSpacing;
      
      // 绘制神经元
      for (let i = 0; i < layer.size; i++) {
        const neuronY = layer.y + (i - layer.size / 2) * 40;
        const circle = g.append('circle')
          .attr('cx', layerX)
          .attr('cy', neuronY)
          .attr('r', neuronRadius)
          .attr('fill', layer.color)
          .attr('stroke', colors.text)
          .attr('stroke-width', 2)
          .attr('opacity', 0.9);

        if (interactive) {
          circle
            .on('mouseenter', function() {
              d3.select(this).attr('opacity', 1).attr('r', neuronRadius + 3);
            })
            .on('mouseleave', function() {
              d3.select(this).attr('opacity', 0.9).attr('r', neuronRadius);
            });
        }

        // 连接线（到下一层）
        if (layerIdx < layers.length - 1) {
          const nextLayer = layers[layerIdx + 1];
          const nextLayerX = centerX - (layerIdx - 0.5) * layerSpacing;
          
          for (let j = 0; j < nextLayer.size; j++) {
            const nextNeuronY = nextLayer.y + (j - nextLayer.size / 2) * 40;
            g.append('line')
              .attr('x1', layerX + neuronRadius)
              .attr('y1', neuronY)
              .attr('x2', nextLayerX - neuronRadius)
              .attr('y2', nextNeuronY)
              .attr('stroke', colors.text)
              .attr('stroke-width', 1)
              .attr('opacity', 0.3);
          }
        }
      }

      // 层标签
      g.append('text')
        .attr('x', layerX)
        .attr('y', layer.y - 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('fill', colors.text)
        .attr('font-weight', 'bold')
        .text(layer.name);
    });
  }

  // 渲染前向传播
  function renderForward(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 输入
    g.append('rect')
      .attr('x', centerX - 200)
      .attr('y', centerY - 30)
      .attr('width', 100)
      .attr('height', 60)
      .attr('fill', colors.input)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX - 150)
      .attr('y', centerY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('x');

    // 权重矩阵
    g.append('rect')
      .attr('x', centerX - 50)
      .attr('y', centerY - 40)
      .attr('width', 100)
      .attr('height', 80)
      .attr('fill', colors.hidden)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('W');

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('z = Wx + b');

    // 激活函数
    g.append('rect')
      .attr('x', centerX + 100)
      .attr('y', centerY - 30)
      .attr('width', 100)
      .attr('height', 60)
      .attr('fill', colors.activation)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX + 150)
      .attr('y', centerY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('σ(z)');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-mlp')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    g.append('path')
      .attr('d', `M ${centerX - 100} ${centerY} L ${centerX - 50} ${centerY}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-mlp)');

    g.append('path')
      .attr('d', `M ${centerX + 50} ${centerY} L ${centerX + 100} ${centerY}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-mlp)');
  }

  // 渲染反向传播
  function renderBackward(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 输出层误差
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', centerY - 150)
      .attr('width', 200)
      .attr('height', 40)
      .attr('fill', colors.output)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 125)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('δ^(L) = ∇_a J ⊙ σ\'(z^(L))');

    // 隐藏层误差
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', centerY - 50)
      .attr('width', 200)
      .attr('height', 40)
      .attr('fill', colors.hidden)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('δ^(l) = (W^(l+1))^T δ^(l+1) ⊙ σ\'(z^(l))');

    // 梯度
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', centerY + 50)
      .attr('width', 200)
      .attr('height', 40)
      .attr('fill', colors.activation)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 75)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('∇W = δ^(l) (a^(l-1))^T');

    // 反向箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-backward')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    g.append('path')
      .attr('d', `M ${centerX} ${centerY - 110} L ${centerX} ${centerY - 50}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-backward)');

    g.append('path')
      .attr('d', `M ${centerX} ${centerY - 10} L ${centerX} ${centerY + 50}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-backward)');
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

export default MLPDiagram;
