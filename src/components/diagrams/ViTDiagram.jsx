import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * ViT视觉Transformer架构图解组件
 * 展示ViT的Patch Embedding、Transformer Encoder和分类头
 */
const ViTDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture', // architecture, patch, encoder
  title = 'ViT架构图',
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
      image: '#4facfe',
      patch: '#667eea',
      embedding: '#f093fb',
      encoder: '#43e97b',
      cls: '#764ba2',
      output: '#fa709a',
      text: '#2d3748'
    };

    switch (type) {
      case 'patch':
        renderPatch(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'encoder':
        renderEncoder(g, innerWidth, innerHeight, colors, interactive);
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

  // 渲染ViT整体架构
  function renderArchitecture(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const imageY = 80;
    const patchY = 180;
    const embeddingY = 280;
    const encoderY = 420;
    const clsY = 580;
    const outputY = 680;

    // 输入图像
    g.append('rect')
      .attr('x', centerX - 120)
      .attr('y', imageY - 40)
      .attr('width', 240)
      .attr('height', 80)
      .attr('fill', colors.image)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', imageY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Input Image');

    g.append('text')
      .attr('x', centerX)
      .attr('y', imageY + 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('224×224×3');

    // Patch分割
    g.append('rect')
      .attr('x', centerX - 150)
      .attr('y', patchY - 30)
      .attr('width', 300)
      .attr('height', 60)
      .attr('fill', colors.patch)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', patchY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Patch Embedding (16×16 patches → 196 patches)');

    // 位置编码 + Embedding
    g.append('rect')
      .attr('x', centerX - 150)
      .attr('y', embeddingY - 30)
      .attr('width', 300)
      .attr('height', 60)
      .attr('fill', colors.embedding)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', embeddingY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('+ Position Embedding');

    // Transformer Encoder (多层)
    const numLayers = 6;
    const layerHeight = 20;
    const encoderStartY = encoderY - (numLayers * layerHeight) / 2;

    g.append('rect')
      .attr('x', centerX - 150)
      .attr('y', encoderStartY)
      .attr('width', 300)
      .attr('height', numLayers * layerHeight)
      .attr('fill', colors.encoder)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    for (let i = 0; i < numLayers; i++) {
      const layerY = encoderStartY + i * layerHeight + layerHeight / 2;
      g.append('text')
        .attr('x', centerX)
        .attr('y', layerY + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(`Transformer Encoder Layer ${i + 1}`);
    }

    // [CLS] Token
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', clsY - 30)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.cls)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', clsY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('[CLS] Token');

    // 分类头
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
      .text('Classification Head');

    // 箭头定义
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-vit')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    // 连接箭头
    const arrows = [
      [centerX, imageY + 40, centerX, patchY - 30],
      [centerX, patchY + 30, centerX, embeddingY - 30],
      [centerX, embeddingY + 30, centerX, encoderStartY],
      [centerX, encoderStartY + numLayers * layerHeight, centerX, clsY - 30],
      [centerX, clsY + 30, centerX, outputY - 30]
    ];

    arrows.forEach(([x1, y1, x2, y2]) => {
      g.append('path')
        .attr('d', `M ${x1} ${y1} L ${x2} ${y2}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-vit)');
    });

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('ViT架构：将图像分割为patches，使用Transformer Encoder处理，[CLS] token用于分类');
  }

  // 渲染Patch Embedding
  function renderPatch(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 原始图像（网格表示）
    const imageSize = 200;
    const patchSize = 50;
    const numPatchesPerSide = 4;
    const imageX = centerX - imageSize / 2;
    const imageY = centerY - 150;

    g.append('rect')
      .attr('x', imageX)
      .attr('y', imageY)
      .attr('width', imageSize)
      .attr('height', imageSize)
      .attr('fill', colors.image)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    // 绘制patch网格
    for (let i = 0; i < numPatchesPerSide; i++) {
      for (let j = 0; j < numPatchesPerSide; j++) {
        const patchX = imageX + j * patchSize;
        const patchY = imageY + i * patchSize;
        g.append('rect')
          .attr('x', patchX)
          .attr('y', patchY)
          .attr('width', patchSize)
          .attr('height', patchSize)
          .attr('fill', 'none')
          .attr('stroke', 'white')
          .attr('stroke-width', 1);
      }
    }

    g.append('text')
      .attr('x', centerX)
      .attr('y', imageY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('224×224 Image → 16×16 Patches');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-patch')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    g.append('path')
      .attr('d', `M ${centerX} ${imageY + imageSize + 20} L ${centerX} ${centerY + 50}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-patch)');

    // Patch序列
    const patchSequenceY = centerY + 100;
    const patchWidth = 30;
    const patchSpacing = 10;
    const numPatches = 8; // 显示部分patches
    const startX = centerX - (numPatches * (patchWidth + patchSpacing)) / 2;

    for (let i = 0; i < numPatches; i++) {
      const x = startX + i * (patchWidth + patchSpacing);
      g.append('rect')
        .attr('x', x)
        .attr('y', patchSequenceY - 15)
        .attr('width', patchWidth)
        .attr('height', 30)
        .attr('fill', colors.patch)
        .attr('stroke', colors.text)
        .attr('stroke-width', 1)
        .attr('rx', 3);

      g.append('text')
        .attr('x', x + patchWidth / 2)
        .attr('y', patchSequenceY + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', 'white')
        .text(`P${i + 1}`);
    }

    g.append('text')
      .attr('x', centerX)
      .attr('y', patchSequenceY + 50)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .text('196 Patches (14×14) → Linear Projection → Embeddings');
  }

  // 渲染Encoder层
  function renderEncoder(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // Encoder层结构
    const layerY = centerY - 100;
    const layerWidth = 400;
    const layerHeight = 120;

    // Multi-Head Self-Attention
    g.append('rect')
      .attr('x', centerX - layerWidth / 2)
      .attr('y', layerY)
      .attr('width', layerWidth)
      .attr('height', layerHeight / 2)
      .attr('fill', colors.encoder)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX)
      .attr('y', layerY + layerHeight / 4 + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Multi-Head Self-Attention');

    // Feed Forward
    g.append('rect')
      .attr('x', centerX - layerWidth / 2)
      .attr('y', layerY + layerHeight / 2)
      .attr('width', layerWidth)
      .attr('height', layerHeight / 2)
      .attr('fill', colors.encoder)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', centerX)
      .attr('y', layerY + layerHeight * 3 / 4 + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Feed Forward Network');

    // Add & Norm标注
    g.append('text')
      .attr('x', centerX + layerWidth / 2 + 20)
      .attr('y', layerY + layerHeight / 4)
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .text('Add & Norm');

    g.append('text')
      .attr('x', centerX + layerWidth / 2 + 20)
      .attr('y', layerY + layerHeight * 3 / 4)
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .text('Add & Norm');

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 100)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('Transformer Encoder：Self-Attention + FFN，处理patch序列');
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

export default ViTDiagram;
