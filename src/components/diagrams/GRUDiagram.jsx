import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

const GRUDiagram = ({ width = 1000, height = 800, interactive = true, type = 'cell', title = 'GRU单元结构', ...props }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 60, right: 80, bottom: 80, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const colors = { input: '#4facfe', reset: '#fa709a', update: '#f093fb', hidden: '#667eea', text: '#2d3748' };

    if (type === 'unfolded') {
      renderUnfolded(g, innerWidth, innerHeight, colors);
    } else if (type === 'comparison') {
      renderComparison(g, innerWidth, innerHeight, colors);
    } else {
      renderGRUCell(g, innerWidth, innerHeight, colors, interactive);
    }

    if (title) {
      g.append('text').attr('x', innerWidth / 2).attr('y', -30).attr('text-anchor', 'middle')
        .attr('font-size', '24px').attr('font-weight', 'bold').attr('fill', colors.text).text(title);
    }
  }, [width, height, type, interactive, title]);

  function renderGRUCell(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 输入
    const inputX = centerX - 200;
    g.append('rect').attr('x', inputX - 40).attr('y', centerY - 20).attr('width', 80).attr('height', 40)
      .attr('fill', colors.input).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', inputX).attr('y', centerY + 5).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', 'white').text('x_t');

    // 前一个隐藏状态
    const prevHX = centerX;
    const prevHY = centerY - 120;
    g.append('rect').attr('x', prevHX - 40).attr('y', prevHY - 20).attr('width', 80).attr('height', 40)
      .attr('fill', colors.hidden).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', prevHX).attr('y', prevHY + 5).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', 'white').text('h_{t-1}');

    // 重置门
    const resetX = centerX - 100;
    const resetY = centerY - 40;
    g.append('rect').attr('x', resetX - 35).attr('y', resetY - 15).attr('width', 70).attr('height', 30)
      .attr('fill', colors.reset).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', resetX).attr('y', resetY + 3).attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('fill', 'white').text('重置门 r_t');

    // 更新门
    const updateX = centerX - 100;
    const updateY = centerY + 40;
    g.append('rect').attr('x', updateX - 35).attr('y', updateY - 15).attr('width', 70).attr('height', 30)
      .attr('fill', colors.update).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', updateX).attr('y', updateY + 3).attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('fill', 'white').text('更新门 z_t');

    // 候选隐藏状态
    const candidateX = centerX;
    const candidateY = centerY;
    g.append('rect').attr('x', candidateX - 50).attr('y', candidateY - 25).attr('width', 100).attr('height', 50)
      .attr('fill', colors.hidden).attr('stroke', colors.text).attr('stroke-width', 3).attr('rx', 8);
    g.append('text').attr('x', candidateX).attr('y', candidateY - 5).attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('fill', 'white').text('候选值');
    g.append('text').attr('x', candidateX).attr('y', candidateY + 12).attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('fill', 'white').text('h̃_t');

    // 输出隐藏状态
    const outputHX = centerX + 200;
    g.append('rect').attr('x', outputHX - 40).attr('y', centerY - 20).attr('width', 80).attr('height', 40)
      .attr('fill', colors.hidden).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', outputHX).attr('y', centerY + 5).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', 'white').text('h_t');

    // 定义箭头
    const defs = g.append('defs');
    const marker = defs.append('marker').attr('id', 'arrowhead-gru').attr('markerWidth', 10)
      .attr('markerHeight', 10).attr('refX', 9).attr('refY', 3).attr('orient', 'auto');
    marker.append('polygon').attr('points', '0 0, 10 3, 0 6').attr('fill', colors.text);

    // 连接线
    const connections = [
      [inputX + 40, centerY, resetX - 35, resetY],
      [inputX + 40, centerY, updateX - 35, updateY],
      [inputX + 40, centerY, candidateX - 50, candidateY],
      [prevHX, prevHY + 20, resetX - 35, resetY],
      [prevHX, prevHY + 20, updateX - 35, updateY],
      [prevHX, prevHY + 20, candidateX - 50, candidateY],
      [resetX + 35, resetY, candidateX - 50, candidateY],
      [candidateX + 50, candidateY, outputHX - 40, centerY],
      [updateX + 35, updateY, outputHX - 40, centerY]
    ];

    connections.forEach(([x1, y1, x2, y2]) => {
      g.append('path').attr('d', `M ${x1} ${y1} L ${x2} ${y2}`).attr('stroke', colors.text)
        .attr('stroke-width', 2).attr('fill', 'none').attr('marker-end', 'url(#arrowhead-gru)');
    });

    g.append('text').attr('x', centerX).attr('y', height - 40).attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('fill', colors.text)
      .text('h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t');
  }

  function renderUnfolded(g, width, height, colors) {
    const timeSteps = 4;
    const stepWidth = width / (timeSteps + 1);
    const centerY = height / 2;

    for (let t = 0; t < timeSteps; t++) {
      const x = stepWidth * (t + 1);
      const cellBox = g.append('rect').attr('x', x - 60).attr('y', centerY - 50).attr('width', 120).attr('height', 100)
        .attr('fill', colors.hidden).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 8);
      g.append('text').attr('x', x).attr('y', centerY).attr('text-anchor', 'middle')
        .attr('font-size', '16px').attr('fill', 'white').attr('font-weight', 'bold').text('GRU');
      g.append('text').attr('x', x).attr('y', centerY + 20).attr('text-anchor', 'middle')
        .attr('font-size', '12px').attr('fill', 'white').text(`h_${t}`);
      if (t < timeSteps - 1) {
        g.append('path').attr('d', `M ${x + 60} ${centerY} L ${x + stepWidth - 60} ${centerY}`)
          .attr('stroke', colors.hidden).attr('stroke-width', 2).attr('fill', 'none')
          .attr('stroke-dasharray', '5,5').attr('marker-end', 'url(#arrowhead-gru-unfolded)');
      }
    }
    const defs = g.append('defs');
    const marker = defs.append('marker').attr('id', 'arrowhead-gru-unfolded').attr('markerWidth', 10)
      .attr('markerHeight', 10).attr('refX', 9).attr('refY', 3).attr('orient', 'auto');
    marker.append('polygon').attr('points', '0 0, 10 3, 0 6').attr('fill', colors.hidden);
  }

  function renderComparison(g, width, height, colors) {
    const centerX = width / 2;
    const centerY = height / 2;
    g.append('text').attr('x', centerX).attr('y', centerY - 100).attr('text-anchor', 'middle')
      .attr('font-size', '18px').attr('fill', colors.text).attr('font-weight', 'bold')
      .text('GRU vs LSTM');
    g.append('text').attr('x', centerX).attr('y', centerY - 50).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', colors.text)
      .text('GRU: 2个门（重置门、更新门），参数量减少约25%');
    g.append('text').attr('x', centerX).attr('y', centerY).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', colors.text)
      .text('LSTM: 3个门（遗忘门、输入门、输出门）+ 细胞状态');
    g.append('text').attr('x', centerX).attr('y', centerY + 50).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', colors.text)
      .text('性能相近，但GRU训练更快');
  }

  return (
    <div className="generic-diagram-container">
      <svg ref={svgRef} width={width} height={height}
        style={{ border: '1px solid #e2e8f0', borderRadius: '8px', backgroundColor: '#ffffff' }} />
    </div>
  );
};

export default GRUDiagram;
