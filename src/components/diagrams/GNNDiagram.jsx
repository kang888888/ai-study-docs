import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * GNN图神经网络架构图解组件
 */
const GNNDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture',
  title = 'GNN架构图',
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
      node: '#4facfe',
      edge: '#667eea',
      message: '#f093fb',
      aggregate: '#43e97b',
      update: '#764ba2',
      text: '#2d3748'
    };

    renderArchitecture(g, innerWidth, innerHeight, colors, interactive);

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

  function renderArchitecture(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 绘制图结构（5个节点）
    const nodes = [
      { id: 0, x: centerX, y: centerY - 150, label: 'A' },
      { id: 1, x: centerX - 120, y: centerY - 50, label: 'B' },
      { id: 2, x: centerX + 120, y: centerY - 50, label: 'C' },
      { id: 3, x: centerX - 120, y: centerY + 100, label: 'D' },
      { id: 4, x: centerX + 120, y: centerY + 100, label: 'E' }
    ];

    const edges = [
      [0, 1], [0, 2], [1, 3], [2, 4], [1, 2]
    ];

    // 绘制边
    edges.forEach(([src, dst]) => {
      g.append('line')
        .attr('x1', nodes[src].x)
        .attr('y1', nodes[src].y)
        .attr('x2', nodes[dst].x)
        .attr('y2', nodes[dst].y)
        .attr('stroke', colors.edge)
        .attr('stroke-width', 2)
        .attr('opacity', 0.6);
    });

    // 绘制节点
    nodes.forEach(node => {
      g.append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', 30)
        .attr('fill', colors.node)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2);

      g.append('text')
        .attr('x', node.x)
        .attr('y', node.y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text(node.label);
    });

    // 消息传递示例（从节点A）
    const centerNode = nodes[0];
    nodes.slice(1, 3).forEach(neighbor => {
      g.append('path')
        .attr('d', `M ${neighbor.x} ${neighbor.y} Q ${centerX} ${centerY - 100} ${centerNode.x} ${centerNode.y}`)
        .attr('stroke', colors.message)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('stroke-dasharray', '5,5')
        .attr('opacity', 0.7);
    });

    // 消息聚合
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', centerY + 200)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.aggregate)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 225)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Message Aggregation');

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 245)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('h_A = AGG({h_B, h_C, ...})');

    // 节点更新
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', centerY + 280)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.update)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 305)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Node Update');

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 325)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('h_A^{new} = UPDATE(h_A, m_A)');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-gnn')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    g.append('path')
      .attr('d', `M ${centerNode.x} ${centerNode.y + 30} L ${centerX} ${centerY + 200}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-gnn)');

    g.append('path')
      .attr('d', `M ${centerX} ${centerY + 260} L ${centerX} ${centerY + 280}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-gnn)');

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('GNN：通过消息传递机制，节点聚合邻居信息更新自身表示');
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

export default GNNDiagram;
