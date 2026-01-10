import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * MoE混合专家模型架构图解组件
 * 展示MoE的路由机制、专家网络和稀疏激活
 */
const MoEDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture', // architecture, routing, sparse
  title = 'MoE架构图',
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
      router: '#f093fb',
      expert: '#667eea',
      active: '#43e97b',
      inactive: '#cbd5e0',
      output: '#764ba2',
      text: '#2d3748'
    };

    switch (type) {
      case 'routing':
        renderRouting(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'sparse':
        renderSparse(g, innerWidth, innerHeight, colors, interactive);
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

  // 渲染MoE整体架构
  function renderArchitecture(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const inputY = 80;
    const routerY = 200;
    const expertY = 400;
    const outputY = 650;

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

    // 门控网络（Router）
    g.append('rect')
      .attr('x', centerX - 120)
      .attr('y', routerY - 40)
      .attr('width', 240)
      .attr('height', 80)
      .attr('fill', colors.router)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', routerY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Gating Network (Router)');

    g.append('text')
      .attr('x', centerX)
      .attr('y', routerY + 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('G(x) = softmax(W_g x + b_g)');

    // 专家网络（8个专家，Top-2激活）
    const numExperts = 8;
    const expertWidth = 80;
    const expertHeight = 100;
    const expertSpacing = 100;
    const startX = centerX - (numExperts - 1) * expertSpacing / 2;

    for (let i = 0; i < numExperts; i++) {
      const expertX = startX + i * expertSpacing;
      const isActive = i < 2; // Top-2激活

      // 专家框
      g.append('rect')
        .attr('x', expertX - expertWidth / 2)
        .attr('y', expertY - expertHeight / 2)
        .attr('width', expertWidth)
        .attr('height', expertHeight)
        .attr('fill', isActive ? colors.expert : colors.inactive)
        .attr('stroke', colors.text)
        .attr('stroke-width', isActive ? 3 : 1)
        .attr('rx', 5)
        .attr('opacity', isActive ? 1 : 0.5);

      g.append('text')
        .attr('x', expertX)
        .attr('y', expertY - 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', isActive ? 'white' : colors.text)
        .attr('font-weight', 'bold')
        .text(`Expert ${i + 1}`);

      if (isActive) {
        g.append('text')
          .attr('x', expertX)
          .attr('y', expertY + 10)
          .attr('text-anchor', 'middle')
          .attr('font-size', '12px')
          .attr('fill', 'white')
          .text('Active');

        // 从Router到专家的连接
        g.append('path')
          .attr('d', `M ${centerX} ${routerY + 40} L ${expertX} ${expertY - expertHeight / 2}`)
          .attr('stroke', colors.active)
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead-moe)');
      } else {
        // 虚线表示未激活
        g.append('path')
          .attr('d', `M ${centerX} ${routerY + 40} L ${expertX} ${expertY - expertHeight / 2}`)
          .attr('stroke', colors.inactive)
          .attr('stroke-width', 1)
          .attr('stroke-dasharray', '5,5')
          .attr('fill', 'none')
          .attr('opacity', 0.3);
      }

      // 从专家到输出的连接
      if (isActive) {
        g.append('path')
          .attr('d', `M ${expertX} ${expertY + expertHeight / 2} L ${centerX} ${outputY - 30}`)
          .attr('stroke', colors.active)
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead-moe)');
      }
    }

    // 输出（加权求和）
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
      .text('Output');

    g.append('text')
      .attr('x', centerX)
      .attr('y', outputY + 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('Σ G_i(x) · E_i(x)');

    // 箭头定义
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-moe')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.active);

    // 主连接箭头
    g.append('path')
      .attr('d', `M ${centerX} ${inputY + 30} L ${centerX} ${routerY - 40}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-moe)');

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('MoE架构：Router选择Top-K专家（图中Top-2），只有激活的专家参与计算');
  }

  // 渲染路由机制
  function renderRouting(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 输入
    g.append('rect')
      .attr('x', centerX - 200)
      .attr('y', centerY - 40)
      .attr('width', 120)
      .attr('height', 80)
      .attr('fill', colors.input)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX - 140)
      .attr('y', centerY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Input x');

    // Router计算
    g.append('rect')
      .attr('x', centerX - 60)
      .attr('y', centerY - 40)
      .attr('width', 120)
      .attr('height', 80)
      .attr('fill', colors.router)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('G(x) =');

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('softmax(...)');

    // 权重分布（8个专家的权重）
    const numExperts = 8;
    const barWidth = 20;
    const barSpacing = 10;
    const maxBarHeight = 100;
    const startX = centerX + 100;
    const barY = centerY - maxBarHeight / 2;

    const weights = [0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.08, 0.07]; // 示例权重

    weights.forEach((weight, i) => {
      const barHeight = weight * maxBarHeight;
      const x = startX + i * (barWidth + barSpacing);
      const isTop2 = i < 2;

      g.append('rect')
        .attr('x', x)
        .attr('y', barY + maxBarHeight - barHeight)
        .attr('width', barWidth)
        .attr('height', barHeight)
        .attr('fill', isTop2 ? colors.active : colors.inactive)
        .attr('stroke', colors.text)
        .attr('stroke-width', 1)
        .attr('opacity', isTop2 ? 1 : 0.6);

      g.append('text')
        .attr('x', x + barWidth / 2)
        .attr('y', barY + maxBarHeight + 15)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', colors.text)
        .text(`E${i + 1}`);

      g.append('text')
        .attr('x', x + barWidth / 2)
        .attr('y', barY + maxBarHeight - barHeight - 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', colors.text)
        .text(weight.toFixed(2));
    });

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-routing')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    g.append('path')
      .attr('d', `M ${centerX - 80} ${centerY} L ${centerX - 60} ${centerY}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-routing)');

    g.append('path')
      .attr('d', `M ${centerX + 60} ${centerY} L ${startX} ${centerY}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-routing)');

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('Router计算每个专家的权重，选择Top-K（图中Top-2）激活');
  }

  // 渲染稀疏激活
  function renderSparse(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 总参数量 vs 激活参数量
    const totalParams = 100; // 100%
    const activeParams = 13; // 13%（Top-2 of 8）

    // 总参数框
    g.append('rect')
      .attr('x', centerX - 200)
      .attr('y', centerY - 150)
      .attr('width', 400)
      .attr('height', 60)
      .attr('fill', colors.inactive)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 120)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', colors.text)
      .attr('font-weight', 'bold')
      .text(`总参数量: ${totalParams}B (100%)`);

    // 激活参数框（部分）
    g.append('rect')
      .attr('x', centerX - 200)
      .attr('y', centerY - 50)
      .attr('width', (activeParams / totalParams) * 400)
      .attr('height', 60)
      .attr('fill', colors.active)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX - 200 + (activeParams / totalParams) * 200)
      .attr('y', centerY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text(`激活参数: ${activeParams}B (${activeParams}%)`);

    // 未激活部分
    g.append('rect')
      .attr('x', centerX - 200 + (activeParams / totalParams) * 400)
      .attr('y', centerY - 50)
      .attr('width', ((totalParams - activeParams) / totalParams) * 400)
      .attr('height', 60)
      .attr('fill', colors.inactive)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5)
      .attr('opacity', 0.5);

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 100)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('稀疏激活：8个专家中只激活Top-2，计算量仅为总参数的25%');
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

export default MoEDiagram;
