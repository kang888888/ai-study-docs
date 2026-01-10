import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * LSTM架构图解组件
 * 展示长短期记忆网络的门控机制、细胞状态和展开形式
 */
const LSTMDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'cell', // cell, unfolded, gates
  title = 'LSTM单元结构',
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
      forget: '#fa709a',
      inputGate: '#f093fb',
      cell: '#764ba2',
      output: '#43e97b',
      outputGate: '#fee140',
      hidden: '#667eea',
      text: '#2d3748'
    };

    // 根据类型渲染不同的图表
    switch (type) {
      case 'unfolded':
        renderUnfoldedLSTM(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'gates':
        renderGates(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'cell':
      default:
        renderLSTMCell(g, innerWidth, innerHeight, colors, interactive);
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

  // 渲染LSTM单元结构
  function renderLSTMCell(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 输入
    const inputX = centerX - 250;
    const inputY = centerY;
    g.append('rect')
      .attr('x', inputX - 40)
      .attr('y', inputY - 20)
      .attr('width', 80)
      .attr('height', 40)
      .attr('fill', colors.input)
      .attr('stroke', colors.text)
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
    const prevHY = centerY - 120;
    g.append('rect')
      .attr('x', prevHX - 40)
      .attr('y', prevHY - 20)
      .attr('width', 80)
      .attr('height', 40)
      .attr('fill', colors.hidden)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', prevHX)
      .attr('y', prevHY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('h_{t-1}');

    // 前一个细胞状态
    const prevCX = centerX;
    const prevCY = centerY + 120;
    g.append('rect')
      .attr('x', prevCX - 40)
      .attr('y', prevCY - 20)
      .attr('width', 80)
      .attr('height', 40)
      .attr('fill', colors.cell)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', prevCX)
      .attr('y', prevCY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('C_{t-1}');

    // 遗忘门
    const forgetX = centerX - 100;
    const forgetY = centerY - 40;
    g.append('rect')
      .attr('x', forgetX - 35)
      .attr('y', forgetY - 15)
      .attr('width', 70)
      .attr('height', 30)
      .attr('fill', colors.forget)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', forgetX)
      .attr('y', forgetY + 3)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('遗忘门 f_t');

    // 输入门
    const inputGateX = centerX - 100;
    const inputGateY = centerY + 40;
    g.append('rect')
      .attr('x', inputGateX - 35)
      .attr('y', inputGateY - 15)
      .attr('width', 70)
      .attr('height', 30)
      .attr('fill', colors.inputGate)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', inputGateX)
      .attr('y', inputGateY + 3)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('输入门 i_t');

    // 候选值
    const candidateX = centerX;
    const candidateY = centerY;
    g.append('rect')
      .attr('x', candidateX - 50)
      .attr('y', candidateY - 25)
      .attr('width', 100)
      .attr('height', 50)
      .attr('fill', colors.cell)
      .attr('stroke', colors.text)
      .attr('stroke-width', 3)
      .attr('rx', 8);

    g.append('text')
      .attr('x', candidateX)
      .attr('y', candidateY - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('候选值');

    g.append('text')
      .attr('x', candidateX)
      .attr('y', candidateY + 12)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('C̃_t');

    // 细胞状态更新
    const cellX = centerX + 100;
    const cellY = centerY;
    g.append('rect')
      .attr('x', cellX - 40)
      .attr('y', cellY - 20)
      .attr('width', 80)
      .attr('height', 40)
      .attr('fill', colors.cell)
      .attr('stroke', colors.text)
      .attr('stroke-width', 3)
      .attr('rx', 5);

    g.append('text')
      .attr('x', cellX)
      .attr('y', cellY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('C_t');

    // 输出门
    const outputGateX = centerX + 100;
    const outputGateY = centerY - 60;
    g.append('rect')
      .attr('x', outputGateX - 35)
      .attr('y', outputGateY - 15)
      .attr('width', 70)
      .attr('height', 30)
      .attr('fill', colors.outputGate)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', outputGateX)
      .attr('y', outputGateY + 3)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('输出门 o_t');

    // 输出隐藏状态
    const outputHX = centerX + 250;
    const outputHY = centerY;
    g.append('rect')
      .attr('x', outputHX - 40)
      .attr('y', outputHY - 20)
      .attr('width', 80)
      .attr('height', 40)
      .attr('fill', colors.hidden)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', outputHX)
      .attr('y', outputHY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('h_t');

    // 定义箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-lstm')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    // 连接线
    const connections = [
      { from: [inputX + 40, inputY], to: [forgetX - 35, forgetY] },
      { from: [inputX + 40, inputY], to: [inputGateX - 35, inputGateY] },
      { from: [inputX + 40, inputY], to: [candidateX - 50, candidateY] },
      { from: [prevHX, prevHY + 20], to: [forgetX - 35, forgetY] },
      { from: [prevHX, prevHY + 20], to: [inputGateX - 35, inputGateY] },
      { from: [prevHX, prevHY + 20], to: [candidateX - 50, candidateY] },
      { from: [prevCX, prevCY - 20], to: [forgetX + 35, forgetY] },
      { from: [forgetX + 35, forgetY], to: [cellX - 40, cellY] },
      { from: [inputGateX + 35, inputGateY], to: [cellX - 40, cellY] },
      { from: [candidateX + 50, candidateY], to: [cellX - 40, cellY] },
      { from: [cellX + 40, cellY], to: [outputGateX - 35, outputGateY] },
      { from: [outputGateX + 35, outputGateY], to: [outputHX - 40, outputHY] },
      { from: [cellX, cellY + 20], to: [outputHX - 40, outputHY] }
    ];

    connections.forEach(conn => {
      g.append('path')
        .attr('d', `M ${conn.from[0]} ${conn.from[1]} L ${conn.to[0]} ${conn.to[1]}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-lstm)');
    });

    // 公式说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .text('C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t    h_t = o_t ⊙ tanh(C_t)');
  }

  // 渲染展开的LSTM
  function renderUnfoldedLSTM(g, width, height, colors, interactive) {
    const timeSteps = 4;
    const stepWidth = width / (timeSteps + 1);
    const centerY = height / 2;

    for (let t = 0; t < timeSteps; t++) {
      const x = stepWidth * (t + 1);
      
      // LSTM单元
      const cellBox = g.append('rect')
        .attr('x', x - 60)
        .attr('y', centerY - 50)
        .attr('width', 120)
        .attr('height', 100)
        .attr('fill', colors.cell)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 8);

      g.append('text')
        .attr('x', x)
        .attr('y', centerY)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text('LSTM');

      g.append('text')
        .attr('x', x)
        .attr('y', centerY + 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(`h_${t}, C_${t}`);

      // 输入
      g.append('rect')
        .attr('x', x - 40)
        .attr('y', centerY - 100)
        .attr('width', 80)
        .attr('height', 30)
        .attr('fill', colors.input)
        .attr('stroke', colors.text)
        .attr('stroke-width', 1)
        .attr('rx', 5);

      g.append('text')
        .attr('x', x)
        .attr('y', centerY - 80)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(`x_${t}`);

      // 输出
      g.append('rect')
        .attr('x', x - 40)
        .attr('y', centerY + 70)
        .attr('width', 80)
        .attr('height', 30)
        .attr('fill', colors.output)
        .attr('stroke', colors.text)
        .attr('stroke-width', 1)
        .attr('rx', 5);

      g.append('text')
        .attr('x', x)
        .attr('y', centerY + 90)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(`y_${t}`);

      // 时间步标签
      g.append('text')
        .attr('x', x)
        .attr('y', centerY - 130)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('fill', colors.text)
        .attr('font-weight', 'bold')
        .text(`t=${t}`);

      // 连接箭头
      if (t < timeSteps - 1) {
        g.append('path')
          .attr('d', `M ${x + 60} ${centerY} L ${x + stepWidth - 60} ${centerY}`)
          .attr('stroke', colors.hidden)
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .attr('stroke-dasharray', '5,5')
          .attr('marker-end', 'url(#arrowhead-lstm-unfolded)');
      }
    }

    // 定义箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-lstm-unfolded')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.hidden);
  }

  // 渲染门控机制
  function renderGates(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    const gates = [
      { name: '遗忘门', formula: 'f_t = σ(W_f·[h_{t-1}, x_t] + b_f)', y: centerY - 100, color: colors.forget },
      { name: '输入门', formula: 'i_t = σ(W_i·[h_{t-1}, x_t] + b_i)', y: centerY, color: colors.inputGate },
      { name: '输出门', formula: 'o_t = σ(W_o·[h_{t-1}, x_t] + b_o)', y: centerY + 100, color: colors.outputGate }
    ];

    gates.forEach((gate, i) => {
      const box = g.append('rect')
        .attr('x', centerX - 200)
        .attr('y', gate.y - 30)
        .attr('width', 400)
        .attr('height', 60)
        .attr('fill', gate.color)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 8)
        .attr('opacity', 0.9);

      g.append('text')
        .attr('x', centerX)
        .attr('y', gate.y - 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text(gate.name);

      g.append('text')
        .attr('x', centerX)
        .attr('y', gate.y + 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(gate.formula);
    });
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

export default LSTMDiagram;
