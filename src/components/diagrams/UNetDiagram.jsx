import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * U-Net架构图解组件
 */
const UNetDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture',
  title = 'U-Net架构图',
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
      encoder: '#667eea',
      bottleneck: '#f093fb',
      decoder: '#43e97b',
      output: '#764ba2',
      skip: '#fa709a',
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
    const leftX = width / 4;
    const rightX = (width * 3) / 4;
    const centerX = width / 2;
    
    // U形结构：左侧编码器，右侧解码器
    const layers = [
      { name: 'Input', y: 100, width: 200 },
      { name: 'Conv+Pool', y: 200, width: 180 },
      { name: 'Conv+Pool', y: 300, width: 160 },
      { name: 'Bottleneck', y: 400, width: 140 },
      { name: 'Up+Conv', y: 500, width: 160 },
      { name: 'Up+Conv', y: 600, width: 180 },
      { name: 'Output', y: 700, width: 200 }
    ];

    // 编码器（左侧）
    layers.slice(0, 3).forEach((layer, i) => {
      const x = leftX - layer.width / 2;
      g.append('rect')
        .attr('x', x)
        .attr('y', layer.y - 30)
        .attr('width', layer.width)
        .attr('height', 60)
        .attr('fill', colors.encoder)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5);

      g.append('text')
        .attr('x', leftX)
        .attr('y', layer.y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(layer.name);
    });

    // Bottleneck（中间底部）
    const bottleneck = layers[3];
    g.append('rect')
      .attr('x', centerX - bottleneck.width / 2)
      .attr('y', bottleneck.y - 30)
      .attr('width', bottleneck.width)
      .attr('height', 60)
      .attr('fill', colors.bottleneck)
      .attr('stroke', colors.text)
      .attr('stroke-width', 3)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX)
      .attr('y', bottleneck.y + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text(bottleneck.name);

    // 解码器（右侧）
    layers.slice(4).forEach((layer, i) => {
      const x = rightX - layer.width / 2;
      g.append('rect')
        .attr('x', x)
        .attr('y', layer.y - 30)
        .attr('width', layer.width)
        .attr('height', 60)
        .attr('fill', colors.decoder)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5);

      g.append('text')
        .attr('x', rightX)
        .attr('y', layer.y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(layer.name);
    });

    // 跳跃连接（Skip Connections）
    const skipConnections = [
      [leftX, layers[0].y, rightX, layers[6].y],
      [leftX, layers[1].y, rightX, layers[5].y],
      [leftX, layers[2].y, rightX, layers[4].y]
    ];

    skipConnections.forEach(([x1, y1, x2, y2]) => {
      g.append('path')
        .attr('d', `M ${x1 + layers[0].width / 2} ${y1} Q ${centerX} ${(y1 + y2) / 2} ${x2 - layers[6].width / 2} ${y2}`)
        .attr('stroke', colors.skip)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('stroke-dasharray', '5,5');
    });

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-unet')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    // 编码器箭头
    for (let i = 0; i < 3; i++) {
      g.append('path')
        .attr('d', `M ${leftX} ${layers[i].y + 30} L ${leftX} ${layers[i + 1].y - 30}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-unet)');
    }

    // 到Bottleneck
    g.append('path')
      .attr('d', `M ${leftX} ${layers[2].y + 30} L ${centerX} ${bottleneck.y - 30}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-unet)');

    // 解码器箭头
    g.append('path')
      .attr('d', `M ${centerX} ${bottleneck.y + 30} L ${rightX} ${layers[4].y - 30}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-unet)');

    for (let i = 4; i < layers.length - 1; i++) {
      g.append('path')
        .attr('d', `M ${rightX} ${layers[i].y + 30} L ${rightX} ${layers[i + 1].y - 30}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-unet)');
    }

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('U-Net：编码器-解码器架构，跳跃连接保留细节信息，用于图像分割');
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

export default UNetDiagram;
