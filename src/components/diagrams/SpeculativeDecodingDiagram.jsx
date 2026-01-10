import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import './SpeculativeDecodingDiagram.css';

const SpeculativeDecodingDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'flow',
  title = 'Speculative Decoding 流程',
  ...props 
}) => {
  const svgRef = useRef(null);
  const [selectedStep, setSelectedStep] = useState(null);

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
      draft: '#667eea',
      target: '#764ba2',
      accept: '#43e97b',
      reject: '#fa709a',
      text: '#2d3748',
      background: '#f7fafc',
      border: '#e2e8f0'
    };

    // 渲染整体流程
    const renderFlow = (g, width, height, colors, interactive) => {
    const centerY = height / 2;
    const stepWidth = width / 5;

    // 步骤1: Draft 模型生成
    const draftX = stepWidth;
    const draftBox = g.append('rect')
      .attr('x', draftX - 80)
      .attr('y', centerY - 60)
      .attr('width', 160)
      .attr('height', 120)
      .attr('fill', colors.draft)
      .attr('opacity', 0.8)
      .attr('rx', 8)
      .attr('stroke', colors.draft)
      .attr('stroke-width', 2);

    g.append('text')
      .attr('x', draftX)
      .attr('y', centerY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text('Draft 模型');

    g.append('text')
      .attr('x', draftX)
      .attr('y', centerY + 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('生成 K 个 token');

    // 箭头1
    g.append('line')
      .attr('x1', draftX + 80)
      .attr('y1', centerY)
      .attr('x2', stepWidth * 2 - 80)
      .attr('y2', centerY)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#arrowhead-spec-reject)');

    // 步骤2: Target 模型验证
    const targetX = stepWidth * 2;
    const targetBox = g.append('rect')
      .attr('x', targetX - 80)
      .attr('y', centerY - 60)
      .attr('width', 160)
      .attr('height', 120)
      .attr('fill', colors.target)
      .attr('opacity', 0.8)
      .attr('rx', 8)
      .attr('stroke', colors.target)
      .attr('stroke-width', 2);

    g.append('text')
      .attr('x', targetX)
      .attr('y', centerY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text('Target 模型');

    g.append('text')
      .attr('x', targetX)
      .attr('y', centerY + 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('并行验证');

    // 箭头2
    g.append('line')
      .attr('x1', targetX + 80)
      .attr('y1', centerY)
      .attr('x2', stepWidth * 3 - 80)
      .attr('y2', centerY)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#arrowhead-spec-reject)');

    // 步骤3: 接受/拒绝决策
    const decisionX = stepWidth * 3;
    const decisionBox = g.append('rect')
      .attr('x', decisionX - 80)
      .attr('y', centerY - 60)
      .attr('width', 160)
      .attr('height', 120)
      .attr('fill', colors.accept)
      .attr('opacity', 0.8)
      .attr('rx', 8)
      .attr('stroke', colors.accept)
      .attr('stroke-width', 2);

    g.append('text')
      .attr('x', decisionX)
      .attr('y', centerY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text('验证结果');

    g.append('text')
      .attr('x', decisionX)
      .attr('y', centerY + 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('接受/拒绝');

    // 箭头3
    g.append('line')
      .attr('x1', decisionX + 80)
      .attr('y1', centerY)
      .attr('x2', stepWidth * 4 - 80)
      .attr('y2', centerY)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#arrowhead-spec-reject)');

    // 步骤4: 输出
    const outputX = stepWidth * 4;
    const outputBox = g.append('rect')
      .attr('x', outputX - 80)
      .attr('y', centerY - 60)
      .attr('width', 160)
      .attr('height', 120)
      .attr('fill', colors.accept)
      .attr('opacity', 0.8)
      .attr('rx', 8)
      .attr('stroke', colors.accept)
      .attr('stroke-width', 2);

    g.append('text')
      .attr('x', outputX)
      .attr('y', centerY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text('输出 Token');

    g.append('text')
      .attr('x', outputX)
      .attr('y', centerY + 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('采纳或重新采样');

    // 添加箭头标记（在 svg 层面）
    const svgDefs = svg.append('defs');
    const arrowMarker = svgDefs.append('marker')
      .attr('id', 'arrowhead-spec-flow')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    
    arrowMarker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);
    
    // 更新引用
    g.selectAll('line[marker-end]').attr('marker-end', 'url(#arrowhead-spec-flow)');

    // 添加数据流动动画
    if (interactive) {
      const flowParticle = g.append('circle')
        .attr('r', 5)
        .attr('fill', colors.draft)
        .attr('opacity', 0.9)
        .attr('class', 'flow-particle');

      const animateFlow = () => {
        flowParticle
          .attr('cx', draftX + 80)
          .attr('cy', centerY)
          .transition()
          .duration(2000)
          .ease(d3.easeLinear)
          .attr('cx', outputX - 80)
          .on('end', animateFlow);
      };

      setTimeout(() => {
        animateFlow();
      }, 500);
    }
    };

    // 渲染拒绝与回退
    const renderReject = (g, width, height, colors, interactive, svg) => {
      const centerY = height / 2;
      const centerX = width / 2;

    // Draft 输出
    const draftBox = g.append('rect')
      .attr('x', centerX - 200)
      .attr('y', centerY - 100)
      .attr('width', 150)
      .attr('height', 80)
      .attr('fill', colors.draft)
      .attr('opacity', 0.8)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX - 125)
      .attr('y', centerY - 50)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Draft 输出');

    // 验证点
    const verifyX = centerX;
    const verifyCircle = g.append('circle')
      .attr('cx', verifyX)
      .attr('cy', centerY - 60)
      .attr('r', 30)
      .attr('fill', colors.target)
      .attr('opacity', 0.8);

    g.append('text')
      .attr('x', verifyX)
      .attr('y', centerY - 55)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('验证');

    // 接受路径
    g.append('line')
      .attr('x1', verifyX + 30)
      .attr('y1', centerY - 60)
      .attr('x2', centerX + 200)
      .attr('y2', centerY - 60)
      .attr('stroke', colors.accept)
      .attr('stroke-width', 3)
      .attr('marker-end', 'url(#arrowhead-spec-reject)');

    g.append('text')
      .attr('x', centerX + 100)
      .attr('y', centerY - 70)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.accept)
      .text('接受');

    // 拒绝路径
    g.append('line')
      .attr('x1', verifyX)
      .attr('y1', centerY - 30)
      .attr('x2', verifyX)
      .attr('y2', centerY + 60)
      .attr('stroke', colors.reject)
      .attr('stroke-width', 3)
      .attr('marker-end', 'url(#arrowhead-spec-reject)');

    g.append('text')
      .attr('x', verifyX + 40)
      .attr('y', centerY + 20)
      .attr('font-size', '14px')
      .attr('fill', colors.reject)
      .text('拒绝');

    // 回退到重新采样
    const resampleBox = g.append('rect')
      .attr('x', centerX - 200)
      .attr('y', centerY + 100)
      .attr('width', 150)
      .attr('height', 80)
      .attr('fill', colors.draft)
      .attr('opacity', 0.6)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX - 125)
      .attr('y', centerY + 150)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('重新采样');

    // 添加箭头标记
    const svgDefs = svg.append('defs');
    const arrowMarker = svgDefs.append('marker')
      .attr('id', 'arrowhead-spec-reject')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    
    arrowMarker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);
    
    // 更新引用
    g.selectAll('line[marker-end]').attr('marker-end', 'url(#arrowhead-spec-reject)');
    };

    // 渲染并行调度
    const renderParallel = (g, width, height, colors, interactive, svg) => {
      const centerY = height / 2;
    const stepHeight = height / 4;

    // Draft 模型并行生成
    const draftGroup = g.append('g').attr('class', 'draft-group');
    for (let i = 0; i < 4; i++) {
      const y = stepHeight * (i + 0.5);
      const box = draftGroup.append('rect')
        .attr('x', width / 4 - 60)
        .attr('y', y - 20)
        .attr('width', 120)
        .attr('height', 40)
        .attr('fill', colors.draft)
        .attr('opacity', 0.7)
        .attr('rx', 5);

      g.append('text')
        .attr('x', width / 4)
        .attr('y', y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(`Token ${i + 1}`);
    }

    g.append('text')
      .attr('x', width / 4)
      .attr('y', stepHeight * 0.5 - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .attr('fill', colors.text)
      .text('Draft 模型');

    // 箭头
    g.append('line')
      .attr('x1', width / 4 + 60)
      .attr('y1', centerY)
      .attr('x2', width / 2 - 60)
      .attr('y2', centerY)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#arrowhead-spec-reject)');

    // Target 模型并行验证
    const targetGroup = g.append('g').attr('class', 'target-group');
    for (let i = 0; i < 4; i++) {
      const y = stepHeight * (i + 0.5);
      const box = targetGroup.append('rect')
        .attr('x', width / 2 - 60)
        .attr('y', y - 20)
        .attr('width', 120)
        .attr('height', 40)
        .attr('fill', colors.target)
        .attr('opacity', 0.7)
        .attr('rx', 5);

      g.append('text')
        .attr('x', width / 2)
        .attr('y', y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(`验证 ${i + 1}`);
    }

    g.append('text')
      .attr('x', width / 2)
      .attr('y', stepHeight * 0.5 - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .attr('fill', colors.text)
      .text('Target 模型');

    // 箭头
    g.append('line')
      .attr('x1', width / 2 + 60)
      .attr('y1', centerY)
      .attr('x2', width * 3 / 4 - 60)
      .attr('y2', centerY)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#arrowhead-spec-reject)');

    // 输出
    const outputBox = g.append('rect')
      .attr('x', width * 3 / 4 - 60)
      .attr('y', centerY - 40)
      .attr('width', 120)
      .attr('height', 80)
      .attr('fill', colors.accept)
      .attr('opacity', 0.8)
      .attr('rx', 8);

    g.append('text')
      .attr('x', width * 3 / 4)
      .attr('y', centerY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('并行输出');

    // 添加箭头标记（在 svg 层面）
    const svgDefs = svg.append('defs');
    const arrowMarker = svgDefs.append('marker')
      .attr('id', 'arrowhead-spec-parallel')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    
    arrowMarker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);
    
    // 更新引用
    g.selectAll('line[marker-end]').attr('marker-end', 'url(#arrowhead-spec-parallel)');
    };

    // 根据类型渲染不同的图表
    switch (type) {
      case 'flow':
        renderFlow(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'reject':
        renderReject(g, innerWidth, innerHeight, colors, interactive, svg);
        break;
      case 'parallel':
        renderParallel(g, innerWidth, innerHeight, colors, interactive, svg);
        break;
      default:
        renderFlow(g, innerWidth, innerHeight, colors, interactive);
    }

    // 添加标题
    if (title) {
      g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', -20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '24px')
        .attr('font-weight', 'bold')
        .attr('fill', colors.text)
        .text(title);
    }

    // 清理函数
    return () => {
      if (svgRef.current) {
        d3.select(svgRef.current).selectAll('*').remove();
      }
    };
  }, [width, height, type, interactive, title]);

  return (
    <div className="speculative-decoding-diagram-container">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="speculative-decoding-diagram-svg"
      />
    </div>
  );
};

export default SpeculativeDecodingDiagram;
