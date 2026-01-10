import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * DQN深度Q网络架构图解组件
 */
const DQNDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture',
  title = 'DQN架构图',
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
      state: '#4facfe',
      network: '#667eea',
      qvalues: '#f093fb',
      action: '#43e97b',
      replay: '#764ba2',
      target: '#fa709a',
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
    const stateY = 100;
    const networkY = 250;
    const qvaluesY = 400;
    const actionY = 550;
    const replayY = 650;
    const targetY = 750;

    // 状态输入
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', stateY - 30)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.state)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', stateY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .text('State s');

    // Q网络
    g.append('rect')
      .attr('x', centerX - 150)
      .attr('y', networkY - 50)
      .attr('width', 300)
      .attr('height', 100)
      .attr('fill', colors.network)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', networkY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Deep Q-Network');

    g.append('text')
      .attr('x', centerX)
      .attr('y', networkY + 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('Q(s, a; θ)');

    // Q值输出
    g.append('rect')
      .attr('x', centerX - 200)
      .attr('y', qvaluesY - 30)
      .attr('width', 400)
      .attr('height', 60)
      .attr('fill', colors.qvalues)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', qvaluesY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Q-values: [Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)]');

    // 动作选择
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', actionY - 30)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.action)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', actionY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .text('Action a* = argmax Q(s,a)');

    // 经验回放
    g.append('rect')
      .attr('x', centerX - 120)
      .attr('y', replayY - 30)
      .attr('width', 240)
      .attr('height', 60)
      .attr('fill', colors.replay)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', replayY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Experience Replay Buffer');

    // 目标网络
    g.append('rect')
      .attr('x', centerX - 120)
      .attr('y', targetY - 30)
      .attr('width', 240)
      .attr('height', 60)
      .attr('fill', colors.target)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', targetY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Target Network Q(s,a; θ⁻)');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-dqn')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    const arrows = [
      [centerX, stateY + 30, centerX, networkY - 50],
      [centerX, networkY + 50, centerX, qvaluesY - 30],
      [centerX, qvaluesY + 30, centerX, actionY - 30]
    ];

    arrows.forEach(([x1, y1, x2, y2]) => {
      g.append('path')
        .attr('d', `M ${x1} ${y1} L ${x2} ${y2}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-dqn)');
    });

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('DQN：使用深度网络近似Q函数，经验回放和目标网络稳定训练');
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

export default DQNDiagram;
