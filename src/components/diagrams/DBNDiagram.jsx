import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * DBN深度信念网络架构图解组件
 */
const DBNDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture',
  title = 'DBN架构图',
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
      visible: '#4facfe',
      hidden: '#667eea',
      rbm: '#f093fb',
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
    const layers = [
      { name: 'Visible Layer', y: 150, color: colors.visible },
      { name: 'Hidden Layer 1', y: 300, color: colors.hidden },
      { name: 'Hidden Layer 2', y: 450, color: colors.hidden },
      { name: 'Hidden Layer 3', y: 600, color: colors.hidden }
    ];

    // 绘制RBM层
    layers.forEach((layer, i) => {
      if (i < layers.length - 1) {
        // RBM框
        g.append('rect')
          .attr('x', centerX - 150)
          .attr('y', layer.y)
          .attr('width', 300)
          .attr('height', layers[i + 1].y - layer.y)
          .attr('fill', 'none')
          .attr('stroke', colors.rbm)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.5);

        g.append('text')
          .attr('x', centerX + 160)
          .attr('y', layer.y + (layers[i + 1].y - layer.y) / 2)
          .attr('font-size', '12px')
          .attr('fill', colors.text)
          .text(`RBM ${i + 1}`);
      }

      // 层节点
      g.append('rect')
        .attr('x', centerX - 120)
        .attr('y', layer.y - 30)
        .attr('width', 240)
        .attr('height', 60)
        .attr('fill', layer.color)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 8);

      g.append('text')
        .attr('x', centerX)
        .attr('y', layer.y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', 'white')
        .text(layer.name);
    });

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-dbn')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    for (let i = 0; i < layers.length - 1; i++) {
      g.append('path')
        .attr('d', `M ${centerX} ${layers[i].y + 30} L ${centerX} ${layers[i + 1].y - 30}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-dbn)');
    }

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('DBN：多个RBM堆叠，逐层无监督预训练，最后有监督微调');
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

export default DBNDiagram;
