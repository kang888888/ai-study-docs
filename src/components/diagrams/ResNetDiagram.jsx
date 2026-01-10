import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

const ResNetDiagram = ({ width = 1000, height = 800, interactive = true, type = 'block', title = 'ResNet残差块', ...props }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 60, right: 80, bottom: 80, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const colors = { conv: '#667eea', relu: '#43e97b', skip: '#f093fb', text: '#2d3748' };

    if (type === 'architecture') {
      renderArchitecture(g, innerWidth, innerHeight, colors);
    } else if (type === 'gradient') {
      renderGradient(g, innerWidth, innerHeight, colors);
    } else {
      renderResidualBlock(g, innerWidth, innerHeight, colors, interactive);
    }

    if (title) {
      g.append('text').attr('x', innerWidth / 2).attr('y', -30).attr('text-anchor', 'middle')
        .attr('font-size', '24px').attr('font-weight', 'bold').attr('fill', colors.text).text(title);
    }
  }, [width, height, type, interactive, title]);

  function renderResidualBlock(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    const blockWidth = 300;
    const blockHeight = 200;

    // 输入
    const inputX = centerX - 200;
    g.append('rect').attr('x', inputX - 40).attr('y', centerY - 20).attr('width', 80).attr('height', 40)
      .attr('fill', colors.conv).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', inputX).attr('y', centerY + 5).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', 'white').text('x');

    // 主路径：Conv -> BN -> ReLU -> Conv -> BN
    const mainPathY = centerY;
    const conv1X = centerX - 100;
    g.append('rect').attr('x', conv1X - 30).attr('y', mainPathY - 20).attr('width', 60).attr('height', 40)
      .attr('fill', colors.conv).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', conv1X).attr('y', mainPathY + 5).attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('fill', 'white').text('Conv');

    const reluX = centerX;
    g.append('rect').attr('x', reluX - 30).attr('y', mainPathY - 20).attr('width', 60).attr('height', 40)
      .attr('fill', colors.relu).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', reluX).attr('y', mainPathY + 5).attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('fill', 'white').text('ReLU');

    const conv2X = centerX + 100;
    g.append('rect').attr('x', conv2X - 30).attr('y', mainPathY - 20).attr('width', 60).attr('height', 40)
      .attr('fill', colors.conv).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', conv2X).attr('y', mainPathY + 5).attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('fill', 'white').text('Conv');

    // 残差连接（跳跃连接）
    const skipPath = `M ${inputX + 40} ${centerY} Q ${centerX - 150} ${centerY - 80} ${centerX - 150} ${centerY - 100} 
                      L ${centerX + 150} ${centerY - 100} Q ${centerX + 150} ${centerY - 80} ${conv2X + 30} ${centerY}`;
    g.append('path').attr('d', skipPath).attr('stroke', colors.skip).attr('stroke-width', 3)
      .attr('fill', 'none').attr('stroke-dasharray', '5,5');

    // 相加
    const addX = centerX + 200;
    g.append('circle').attr('cx', addX).attr('cy', centerY).attr('r', 20)
      .attr('fill', colors.skip).attr('stroke', colors.text).attr('stroke-width', 2);
    g.append('text').attr('x', addX).attr('y', centerY + 5).attr('text-anchor', 'middle')
      .attr('font-size', '16px').attr('fill', 'white').text('+');

    // 输出
    const outputX = centerX + 300;
    g.append('rect').attr('x', outputX - 40).attr('y', centerY - 20).attr('width', 80).attr('height', 40)
      .attr('fill', colors.conv).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', outputX).attr('y', centerY + 5).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', 'white').text('F(x)+x');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker').attr('id', 'arrowhead-resnet').attr('markerWidth', 10)
      .attr('markerHeight', 10).attr('refX', 9).attr('refY', 3).attr('orient', 'auto');
    marker.append('polygon').attr('points', '0 0, 10 3, 0 6').attr('fill', colors.text);

    const arrows = [
      [inputX + 40, centerY, conv1X - 30, mainPathY],
      [conv1X + 30, mainPathY, reluX - 30, mainPathY],
      [reluX + 30, mainPathY, conv2X - 30, mainPathY],
      [conv2X + 30, mainPathY, addX - 20, centerY],
      [addX + 20, centerY, outputX - 40, centerY]
    ];
    arrows.forEach(([x1, y1, x2, y2]) => {
      g.append('path').attr('d', `M ${x1} ${y1} L ${x2} ${y2}`).attr('stroke', colors.text)
        .attr('stroke-width', 2).attr('fill', 'none').attr('marker-end', 'url(#arrowhead-resnet)');
    });

    g.append('text').attr('x', centerX).attr('y', height - 40).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', colors.text).text('H(x) = F(x) + x');
  }

  function renderArchitecture(g, width, height, colors) {
    const layers = ['Conv1', 'Pool', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'FC'];
    const centerX = width / 2;
    layers.forEach((layer, i) => {
      const y = 100 + i * 80;
      g.append('rect').attr('x', centerX - 100).attr('y', y - 20).attr('width', 200).attr('height', 40)
        .attr('fill', colors.conv).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
      g.append('text').attr('x', centerX).attr('y', y + 5).attr('text-anchor', 'middle')
        .attr('font-size', '14px').attr('fill', 'white').text(layer);
      if (i < layers.length - 1) {
        g.append('path').attr('d', `M ${centerX} ${y + 20} L ${centerX} ${y + 60}`)
          .attr('stroke', colors.text).attr('stroke-width', 2).attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead-resnet)');
      }
    });
    const defs = g.append('defs');
    const marker = defs.append('marker').attr('id', 'arrowhead-resnet').attr('markerWidth', 10)
      .attr('markerHeight', 10).attr('refX', 9).attr('refY', 3).attr('orient', 'auto');
    marker.append('polygon').attr('points', '0 0, 10 3, 0 6').attr('fill', colors.text);
  }

  function renderGradient(g, width, height, colors) {
    const centerX = width / 2;
    const centerY = height / 2;
    g.append('text').attr('x', centerX).attr('y', centerY - 100).attr('text-anchor', 'middle')
      .attr('font-size', '16px').attr('fill', colors.text).text('梯度流动：');
    g.append('text').attr('x', centerX).attr('y', centerY - 60).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', colors.text)
      .text('∂L/∂x = ∂L/∂H(x) · (1 + ∂F(x)/∂x)');
    g.append('text').attr('x', centerX).attr('y', centerY - 20).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', colors.text)
      .text('即使 ∂F(x)/∂x ≈ 0，梯度仍可通过恒等项 1 传播');
  }

  return (
    <div className="generic-diagram-container">
      <svg ref={svgRef} width={width} height={height}
        style={{ border: '1px solid #e2e8f0', borderRadius: '8px', backgroundColor: '#ffffff' }} />
    </div>
  );
};

export default ResNetDiagram;
