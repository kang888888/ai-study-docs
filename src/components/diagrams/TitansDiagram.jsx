import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * Titans架构图解组件
 */
const TitansDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture',
  title = 'Titans架构图',
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
      layer: '#667eea',
      attention: '#f093fb',
      output: '#43e97b',
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
    const inputY = 100;
    const layer1Y = 220;
    const layer2Y = 340;
    const layer3Y = 460;
    const outputY = 580;

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
      .text('Input');

    // 多层架构
    const layers = [
      { y: layer1Y, name: 'Titan Layer 1' },
      { y: layer2Y, name: 'Titan Layer 2' },
      { y: layer3Y, name: 'Titan Layer 3' }
    ];

    layers.forEach((layer, i) => {
      g.append('rect')
        .attr('x', centerX - 150)
        .attr('y', layer.y - 40)
        .attr('width', 300)
        .attr('height', 80)
        .attr('fill', colors.layer)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 8);

      g.append('text')
        .attr('x', centerX)
        .attr('y', layer.y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text(layer.name);

      // 内部组件
      g.append('text')
        .attr('x', centerX)
        .attr('y', layer.y + 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text('Attention + FFN');
    });

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
      .text('Output');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-titans')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    const arrows = [
      [centerX, inputY + 30, centerX, layer1Y - 40],
      [centerX, layer1Y + 40, centerX, layer2Y - 40],
      [centerX, layer2Y + 40, centerX, layer3Y - 40],
      [centerX, layer3Y + 40, centerX, outputY - 30]
    ];

    arrows.forEach(([x1, y1, x2, y2]) => {
      g.append('path')
        .attr('d', `M ${x1} ${y1} L ${x2} ${y2}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-titans)');
    });

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('Titans：多层Transformer-like架构');
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

export default TitansDiagram;
