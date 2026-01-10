import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * CLIP多模态架构图解组件
 * 展示CLIP的双塔架构、对比学习和零样本分类
 */
const CLIPDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture', // architecture, contrastive, zeroshot
  title = 'CLIP架构图',
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
      text: '#667eea',
      encoder: '#f093fb',
      embedding: '#43e97b',
      contrastive: '#764ba2',
      output: '#fa709a',
      text: '#2d3748'
    };

    switch (type) {
      case 'contrastive':
        renderContrastive(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'zeroshot':
        renderZeroShot(g, innerWidth, innerHeight, colors, interactive);
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

  // 渲染CLIP双塔架构
  function renderArchitecture(g, width, height, colors, interactive) {
    const leftX = width / 3;
    const rightX = (width * 2) / 3;
    const imageY = 100;
    const encoderY = 250;
    const embeddingY = 400;
    const contrastiveY = 550;
    const outputY = 680;

    // 左侧：图像塔
    // 输入图像
    g.append('rect')
      .attr('x', leftX - 100)
      .attr('y', imageY - 40)
      .attr('width', 200)
      .attr('height', 80)
      .attr('fill', colors.image)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', leftX)
      .attr('y', imageY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Image');

    // Image Encoder
    g.append('rect')
      .attr('x', leftX - 120)
      .attr('y', encoderY - 50)
      .attr('width', 240)
      .attr('height', 100)
      .attr('fill', colors.encoder)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', leftX)
      .attr('y', encoderY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Image Encoder');

    g.append('text')
      .attr('x', leftX)
      .attr('y', encoderY + 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('ResNet / ViT');

    // Image Embedding
    g.append('rect')
      .attr('x', leftX - 100)
      .attr('y', embeddingY - 30)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.embedding)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', leftX)
      .attr('y', embeddingY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Image Embedding');

    // 右侧：文本塔
    // 输入文本
    g.append('rect')
      .attr('x', rightX - 100)
      .attr('y', imageY - 40)
      .attr('width', 200)
      .attr('height', 80)
      .attr('fill', colors.text)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', rightX)
      .attr('y', imageY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Text');

    // Text Encoder
    g.append('rect')
      .attr('x', rightX - 120)
      .attr('y', encoderY - 50)
      .attr('width', 240)
      .attr('height', 100)
      .attr('fill', colors.encoder)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', rightX)
      .attr('y', encoderY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Text Encoder');

    g.append('text')
      .attr('x', rightX)
      .attr('y', encoderY + 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('Transformer');

    // Text Embedding
    g.append('rect')
      .attr('x', rightX - 100)
      .attr('y', embeddingY - 30)
      .attr('width', 200)
      .attr('height', 60)
      .attr('fill', colors.embedding)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', rightX)
      .attr('y', embeddingY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Text Embedding');

    // 对比学习（中间）
    g.append('rect')
      .attr('x', width / 2 - 150)
      .attr('y', contrastiveY - 40)
      .attr('width', 300)
      .attr('height', 80)
      .attr('fill', colors.contrastive)
      .attr('stroke', colors.text)
      .attr('stroke-width', 3)
      .attr('rx', 8);

    g.append('text')
      .attr('x', width / 2)
      .attr('y', contrastiveY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text('Contrastive Learning');

    g.append('text')
      .attr('x', width / 2)
      .attr('y', contrastiveY + 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text('Cosine Similarity');

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
      .attr('font-weight', 'bold')
      .text('Shared Space');

    // 箭头定义
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-clip')
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
      [leftX, imageY + 40, leftX, encoderY - 50],
      [leftX, encoderY + 50, leftX, embeddingY - 30],
      [leftX, embeddingY + 30, width / 2 - 150, contrastiveY],
      [rightX, imageY + 40, rightX, encoderY - 50],
      [rightX, encoderY + 50, rightX, embeddingY - 30],
      [rightX, embeddingY + 30, width / 2 + 150, contrastiveY],
      [width / 2, contrastiveY + 40, width / 2, outputY - 30]
    ];

    arrows.forEach(([x1, y1, x2, y2]) => {
      g.append('path')
        .attr('d', `M ${x1} ${y1} L ${x2} ${y2}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead-clip)');
    });

    // 说明
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('CLIP双塔架构：图像和文本分别编码，通过对比学习映射到共享空间');
  }

  // 渲染对比学习
  function renderContrastive(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 相似度矩阵
    const matrixSize = 4;
    const cellSize = 60;
    const matrixStartX = centerX - (matrixSize * cellSize) / 2;
    const matrixStartY = centerY - (matrixSize * cellSize) / 2;

    // 图像标签
    const imageLabels = ['I1', 'I2', 'I3', 'I4'];
    imageLabels.forEach((label, i) => {
      g.append('text')
        .attr('x', matrixStartX - 30)
        .attr('y', matrixStartY + i * cellSize + cellSize / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', colors.text)
        .attr('font-weight', 'bold')
        .text(label);
    });

    // 文本标签
    const textLabels = ['T1', 'T2', 'T3', 'T4'];
    textLabels.forEach((label, i) => {
      g.append('text')
        .attr('x', matrixStartX + i * cellSize + cellSize / 2)
        .attr('y', matrixStartY - 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', colors.text)
        .attr('font-weight', 'bold')
        .text(label);
    });

    // 相似度矩阵（对角线高，其他低）
    for (let i = 0; i < matrixSize; i++) {
      for (let j = 0; j < matrixSize; j++) {
        const x = matrixStartX + j * cellSize;
        const y = matrixStartY + i * cellSize;
        const isMatch = i === j;
        const similarity = isMatch ? 0.95 : 0.1;

        g.append('rect')
          .attr('x', x)
          .attr('y', y)
          .attr('width', cellSize - 2)
          .attr('height', cellSize - 2)
          .attr('fill', isMatch ? colors.embedding : colors.contrastive)
          .attr('stroke', colors.text)
          .attr('stroke-width', 1)
          .attr('opacity', isMatch ? 1 : 0.5);

        g.append('text')
          .attr('x', x + cellSize / 2)
          .attr('y', y + cellSize / 2 + 5)
          .attr('text-anchor', 'middle')
          .attr('font-size', '12px')
          .attr('fill', isMatch ? 'white' : colors.text)
          .text(similarity.toFixed(2));
      }
    }

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + (matrixSize * cellSize) / 2 + 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('对比学习：最大化匹配对(I_i, T_i)的相似度，最小化不匹配对的相似度');
  }

  // 渲染零样本分类
  function renderZeroShot(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const imageY = 100;
    const embeddingY = 250;
    const textY = 400;
    const similarityY = 550;
    const outputY = 680;

    // 输入图像
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', imageY - 40)
      .attr('width', 200)
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
      .text('Query Image');

    // 图像嵌入
    g.append('rect')
      .attr('x', centerX - 100)
      .attr('y', embeddingY - 30)
      .attr('width', 200)
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
      .text('Image Embedding');

    // 类别文本（多个）
    const categories = ['cat', 'dog', 'bird', 'car'];
    const categorySpacing = 150;
    const startX = centerX - (categories.length - 1) * categorySpacing / 2;

    categories.forEach((cat, i) => {
      const x = startX + i * categorySpacing;
      g.append('rect')
        .attr('x', x - 50)
        .attr('y', textY - 30)
        .attr('width', 100)
        .attr('height', 60)
        .attr('fill', colors.text)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 8);

      g.append('text')
        .attr('x', x)
        .attr('y', textY + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', 'white')
        .text(`"a photo of a ${cat}"`);
    });

    // 相似度计算
    g.append('rect')
      .attr('x', centerX - 150)
      .attr('y', similarityY - 40)
      .attr('width', 300)
      .attr('height', 80)
      .attr('fill', colors.contrastive)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 8);

    g.append('text')
      .attr('x', centerX)
      .attr('y', similarityY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Compute Cosine Similarity');

    // 输出（最高相似度）
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
      .text('Prediction: "cat"');

    // 箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-zeroshot')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    g.append('path')
      .attr('d', `M ${centerX} ${imageY + 40} L ${centerX} ${embeddingY - 30}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-zeroshot)');

    g.append('path')
      .attr('d', `M ${centerX} ${embeddingY + 30} L ${centerX} ${similarityY - 40}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-zeroshot)');

    g.append('path')
      .attr('d', `M ${centerX} ${similarityY + 40} L ${centerX} ${outputY - 30}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-zeroshot)');

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('零样本分类：无需训练，只需提供类别文本，计算与图像嵌入的相似度');
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

export default CLIPDiagram;
