import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * Diffusion扩散模型架构图解组件
 * 展示前向扩散、反向去噪和U-Net网络
 */
const DiffusionDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture', // architecture, forward, reverse, unet
  title = 'Diffusion架构图',
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
      data: '#4facfe',
      noise: '#667eea',
      unet: '#f093fb',
      denoise: '#43e97b',
      output: '#764ba2',
      text: '#2d3748'
    };

    switch (type) {
      case 'forward':
        renderForward(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'reverse':
        renderReverse(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'unet':
        renderUNet(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'architecture':
      default:
        renderArchitecture(g, innerWidth, innerHeight, colors, interactive);
    }

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

  // 渲染Diffusion整体架构
  function renderArchitecture(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const dataY = 100;
    const forwardY = 220;
    const noiseY = 340;
    const unetY = 460;
    const denoiseY = 580;
    const outputY = 700;

    // 原始数据
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', dataY - 30)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.data)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', dataY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('x_0 (Data)');

    // 前向扩散
    g.append('rect')
      .attr('x', centerX - 120)
      .attr('y', forwardY - 30)
      .attr('width', 240)
      .attr('height', 60)
      .attr('fill', colors.noise)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', forwardY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Forward Diffusion');

    g.append('text')
      .attr('x', centerX)
      .attr('y', forwardY + 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)');

    // 纯噪声
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', noiseY - 30)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.noise)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', noiseY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('x_T (Pure Noise)');

    // U-Net去噪网络
    g.append('rect')
      .attr('x', centerX - 150)
      .attr('y', unetY - 50)
      .attr('width', 300)
      .attr('height', 100)
      .attr('fill', colors.unet)
      .attr('stroke', colors.text)
      .attr('stroke-width', 3)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', unetY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('U-Net Denoising Network');

    g.append('text')
      .attr('x', centerX)
      .attr('y', unetY + 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('Predicts ε_θ(x_t, t)');

    // 反向去噪
    g.append('rect')
      .attr('x', centerX - 120)
      .attr('y', denoiseY - 30)
      .attr('width', 240)
      .attr('height', 60)
      .attr('fill', colors.denoise)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', denoiseY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Reverse Denoising');

    g.append('text')
      .attr('x', centerX)
      .attr('y', denoiseY + 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('p_θ(x_{t-1} | x_t)');

    // 生成数据
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
      .attr('font-weight', 'bold')
      .text('x_0 (Generated)');

    // 箭头定义
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-diffusion')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    // 前向箭头（训练时）
    g.append('path')
      .attr('d', `M ${centerX} ${dataY + 30} L ${centerX} ${forwardY - 30}`)
      .attr('stroke', colors.noise)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-diffusion)');

    g.append('path')
      .attr('d', `M ${centerX} ${forwardY + 30} L ${centerX} ${noiseY - 30}`)
      .attr('stroke', colors.noise)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-diffusion)');

    // 反向箭头（生成时）
    g.append('path')
      .attr('d', `M ${centerX} ${noiseY + 30} L ${centerX} ${unetY - 50}`)
      .attr('stroke', colors.denoise)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-diffusion)');

    g.append('path')
      .attr('d', `M ${centerX} ${unetY + 50} L ${centerX} ${denoiseY - 30}`)
      .attr('stroke', colors.denoise)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-diffusion)');

    g.append('path')
      .attr('d', `M ${centerX} ${denoiseY + 30} L ${centerX} ${outputY - 30}`)
      .attr('stroke', colors.denoise)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-diffusion)');

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('Diffusion：前向加噪声训练，反向去噪声生成');
  }

  // 渲染前向扩散过程
  function renderForward(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    const numSteps = 5;
    const stepSpacing = 150;
    const startX = centerX - (numSteps - 1) * stepSpacing / 2;

    // 显示从数据到噪声的渐进过程
    for (let i = 0; i < numSteps; i++) {
      const x = startX + i * stepSpacing;
      const noiseLevel = i / (numSteps - 1);
      const opacity = 1 - noiseLevel * 0.7;

      g.append('rect')
        .attr('x', x - 40)
        .attr('y', centerY - 40)
        .attr('width', 80)
        .attr('height', 80)
        .attr('fill', colors.data)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5)
        .attr('opacity', opacity);

      g.append('text')
        .attr('x', x)
        .attr('y', centerY + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(`t=${i}`);

      if (i < numSteps - 1) {
        g.append('path')
          .attr('d', `M ${x + 40} ${centerY} L ${x + stepSpacing - 40} ${centerY}`)
          .attr('stroke', colors.noise)
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead-forward)');
      }
    }

    // 箭头定义
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-forward')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.noise);

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 100)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('前向扩散：逐步添加高斯噪声，T步后变成纯噪声');
  }

  // 渲染反向去噪过程
  function renderReverse(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    const numSteps = 5;
    const stepSpacing = 150;
    const startX = centerX - (numSteps - 1) * stepSpacing / 2;

    // 显示从噪声到数据的渐进过程
    for (let i = 0; i < numSteps; i++) {
      const x = startX + i * stepSpacing;
      const noiseLevel = (numSteps - 1 - i) / (numSteps - 1);
      const opacity = 0.3 + noiseLevel * 0.7;

      g.append('rect')
        .attr('x', x - 40)
        .attr('y', centerY - 40)
        .attr('width', 80)
        .attr('height', 80)
        .attr('fill', colors.denoise)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5)
        .attr('opacity', opacity);

      g.append('text')
        .attr('x', x)
        .attr('y', centerY + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(`t=${numSteps - 1 - i}`);

      if (i < numSteps - 1) {
        g.append('path')
          .attr('d', `M ${x + 40} ${centerY} L ${x + stepSpacing - 40} ${centerY}`)
          .attr('stroke', colors.denoise)
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead-reverse)');
      }
    }

    // 箭头定义
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-reverse')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.denoise);

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 100)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('反向去噪：U-Net逐步预测并去除噪声，T步后恢复数据');
  }

  // 渲染U-Net结构
  function renderUNet(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // U-Net的U形结构
    // 编码器（下采样）
    const encoderLayers = ['Input', 'Conv', 'Down', 'Conv', 'Down', 'Bottleneck'];
    const encoderY = centerY - 150;
    const encoderSpacing = 100;

    encoderLayers.forEach((layer, i) => {
      const x = centerX - 200;
      const y = encoderY + i * encoderSpacing;
      g.append('rect')
        .attr('x', x - 60)
        .attr('y', y - 20)
        .attr('width', 120)
        .attr('height', 40)
        .attr('fill', colors.unet)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5);

      g.append('text')
        .attr('x', x)
        .attr('y', y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(layer);
    });

    // 解码器（上采样）
    const decoderLayers = ['Bottleneck', 'Up', 'Conv', 'Up', 'Conv', 'Output'];
    decoderLayers.forEach((layer, i) => {
      const x = centerX + 200;
      const y = encoderY + i * encoderSpacing;
      g.append('rect')
        .attr('x', x - 60)
        .attr('y', y - 20)
        .attr('width', 120)
        .attr('height', 40)
        .attr('fill', colors.denoise)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5);

      g.append('text')
        .attr('x', x)
        .attr('y', y + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(layer);
    });

    // 跳跃连接
    for (let i = 0; i < encoderLayers.length - 1; i++) {
      const y1 = encoderY + i * encoderSpacing;
      const y2 = encoderY + (decoderLayers.length - 1 - i) * encoderSpacing;
      g.append('path')
        .attr('d', `M ${centerX - 140} ${y1} Q ${centerX} ${(y1 + y2) / 2} ${centerX + 140} ${y2}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 1)
        .attr('fill', 'none')
        .attr('stroke-dasharray', '5,5')
        .attr('opacity', 0.5);
    }

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', encoderY + encoderLayers.length * encoderSpacing + 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('U-Net：编码器-解码器架构，跳跃连接保留细节信息');
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

export default DiffusionDiagram;
