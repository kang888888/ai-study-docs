import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * GAN生成对抗网络架构图解组件
 */
const GANDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture',
  title = 'GAN架构图',
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
      noise: '#4facfe',
      generator: '#667eea',
      fake: '#f093fb',
      real: '#43e97b',
      discriminator: '#764ba2',
      output: '#fa709a',
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
    const leftX = width / 3;
    const rightX = (width * 2) / 3;
    const noiseY = 100;
    const generatorY = 250;
    const fakeY = 400;
    const realY = 400;
    const discriminatorY = 550;
    const outputY = 700;

    // 左侧：生成器路径
    // 噪声输入
    g.append('rect')
      .attr('x', leftX - 80)
      .attr('y', noiseY - 30)
      .attr('width', 160)
      .attr('height', 60)
      .attr('fill', colors.noise)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', leftX)
      .attr('y', noiseY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('z ~ N(0, I)');

    // 生成器
    g.append('rect')
      .attr('x', leftX - 100)
      .attr('y', generatorY - 50)
      .attr('width', 200)
      .attr('height', 100)
      .attr('fill', colors.generator)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', leftX)
      .attr('y', generatorY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Generator G');

    g.append('text')
      .attr('x', leftX)
      .attr('y', generatorY + 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('G(z) → x_fake');

    // 生成数据
    g.append('rect')
      .attr('x', leftX - 80)
      .attr('y', fakeY - 30)
      .attr('width', 160)
      .attr('height', 60)
      .attr('fill', colors.fake)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', leftX)
      .attr('y', fakeY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Fake Data');

    // 右侧：真实数据
    g.append('rect')
      .attr('x', rightX - 80)
      .attr('y', realY - 30)
      .attr('width', 160)
      .attr('height', 60)
      .attr('fill', colors.real)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', rightX)
      .attr('y', realY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Real Data');

    // 判别器（中间）
    g.append('rect')
      .attr('x', width / 2 - 120)
      .attr('y', discriminatorY - 50)
      .attr('width', 240)
      .attr('height', 100)
      .attr('fill', colors.discriminator)
      .attr('stroke', colors.text)
      .attr('stroke-width', 3)
      .attr('rx', 8);

    g.append('text')
      .attr('x', width / 2)
      .attr('y', discriminatorY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Discriminator D');

    g.append('text')
      .attr('x', width / 2)
      .attr('y', discriminatorY + 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('D(x) → [0, 1] (Real/Fake)');

    // 输出
    g.append('rect')
      .attr('x', width / 2 - 100)
      .attr('y', outputY - 30)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.output)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', width / 2)
      .attr('y', outputY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .text('Adversarial Loss');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-gan')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    const arrows = [
      [leftX, noiseY + 30, leftX, generatorY - 50],
      [leftX, generatorY + 50, leftX, fakeY - 30],
      [leftX, fakeY + 30, width / 2 - 120, discriminatorY],
      [rightX, realY + 30, width / 2 + 120, discriminatorY],
      [width / 2, discriminatorY + 50, width / 2, outputY - 30]
    ];

    arrows.forEach(([x1, y1, x2, y2]) => {
      g.append('path')
        .attr('d', `M ${x1} ${y1} L ${x2} ${y2}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-gan)');
    });

    // 对抗训练标注
    g.append('text')
      .attr('x', width / 2)
      .attr('y', discriminatorY - 80)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .attr('font-weight', 'bold')
      .text('Adversarial Training: min_G max_D V(D, G)');

    g.append('text')
      .attr('x', width / 2)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('GAN：生成器与判别器对抗训练，达到纳什均衡');
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

export default GANDiagram;
