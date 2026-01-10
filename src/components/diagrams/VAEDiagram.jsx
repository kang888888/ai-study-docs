import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * VAE变分自编码器架构图解组件
 */
const VAEDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture',
  title = 'VAE架构图',
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
      latent: '#f093fb',
      decoder: '#43e97b',
      output: '#764ba2',
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
    const encoderY = 250;
    const latentY = 400;
    const decoderY = 550;
    const outputY = 700;

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
      .text('Input x');

    // 编码器
    g.append('rect')
      .attr('x', centerX - 120)
      .attr('y', encoderY - 50)
      .attr('width', 240)
      .attr('height', 100)
      .attr('fill', colors.encoder)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', encoderY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .text('Encoder q_φ(z|x)');

    g.append('text')
      .attr('x', centerX)
      .attr('y', encoderY + 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('μ, σ = Encoder(x)');

    // 潜在空间（采样）
    g.append('rect')
      .attr('x', centerX - 150)
      .attr('y', latentY - 50)
      .attr('width', 300)
      .attr('height', 100)
      .attr('fill', colors.latent)
      .attr('stroke', colors.text)
      .attr('stroke-width', 3)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', latentY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .text('Latent Space z');

    g.append('text')
      .attr('x', centerX)
      .attr('y', latentY + 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('z ~ N(μ, σ²) = μ + σ ⊙ ε, ε ~ N(0, I)');

    // 解码器
    g.append('rect')
      .attr('x', centerX - 120)
      .attr('y', decoderY - 50)
      .attr('width', 240)
      .attr('height', 100)
      .attr('fill', colors.decoder)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', decoderY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .text('Decoder p_θ(x|z)');

    g.append('text')
      .attr('x', centerX)
      .attr('y', decoderY + 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('x̂ = Decoder(z)');

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
      .text('Reconstructed x̂');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-vae')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    const arrows = [
      [centerX, inputY + 30, centerX, encoderY - 50],
      [centerX, encoderY + 50, centerX, latentY - 50],
      [centerX, latentY + 50, centerX, decoderY - 50],
      [centerX, decoderY + 50, centerX, outputY - 30]
    ];

    arrows.forEach(([x1, y1, x2, y2]) => {
      g.append('path')
        .attr('d', `M ${x1} ${y1} L ${x2} ${y2}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-vae)');
    });

    // 损失函数标注
    g.append('text')
      .attr('x', centerX + 200)
      .attr('y', latentY)
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .text('Loss = Reconstruction + KL Divergence');

    g.append('text')
      .attr('x', width / 2)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('VAE：编码器学习潜在分布，解码器从潜在空间重构数据');
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

export default VAEDiagram;
