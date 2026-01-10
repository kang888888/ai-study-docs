import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * BERT架构图解组件
 * 展示BERT的Encoder架构、MLM机制和双向注意力
 */
const BERTDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture', // architecture, mlm, attention
  title = 'BERT架构图',
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
      encoder: '#667eea',
      attention: '#f093fb',
      ffn: '#4facfe',
      norm: '#43e97b',
      embedding: '#fa709a',
      mask: '#fee140',
      text: '#2d3748'
    };

    // 根据类型渲染不同的图表
    switch (type) {
      case 'mlm':
        renderMLM(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'attention':
        renderAttention(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'architecture':
      default:
        renderBERTArchitecture(g, innerWidth, innerHeight, colors, interactive);
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

  // 渲染BERT整体架构（只有Encoder）
  function renderBERTArchitecture(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const encoderX = centerX - 150;
    const encoderY = 100;
    const encoderWidth = 300;
    const encoderHeight = 500;
    const numLayers = 6;

    // Encoder外框
    const encoderBox = g.append('rect')
      .attr('x', encoderX)
      .attr('y', encoderY)
      .attr('width', encoderWidth)
      .attr('height', encoderHeight)
      .attr('fill', 'none')
      .attr('stroke', colors.encoder)
      .attr('stroke-width', 3)
      .attr('rx', 8);

    g.append('text')
      .attr('x', encoderX + encoderWidth / 2)
      .attr('y', encoderY - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '20px')
      .attr('font-weight', 'bold')
      .attr('fill', colors.encoder)
      .text('BERT Encoder (6 Layers)');

    // 输入嵌入层
    const embedY = encoderY - 60;
    g.append('rect')
      .attr('x', encoderX)
      .attr('y', embedY)
      .attr('width', encoderWidth)
      .attr('height', 50)
      .attr('fill', colors.embedding)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', encoderX + encoderWidth / 2)
      .attr('y', embedY + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Token Embedding + Position Embedding + Segment Embedding');

    // Encoder层
    const layerHeight = (encoderHeight - 20) / numLayers;
    for (let i = 0; i < numLayers; i++) {
      const layerY = encoderY + 10 + i * layerHeight;
      
      // Multi-Head Attention
      g.append('rect')
        .attr('x', encoderX + 10)
        .attr('y', layerY)
        .attr('width', encoderWidth - 20)
        .attr('height', layerHeight / 2 - 5)
        .attr('fill', colors.attention)
        .attr('stroke', colors.text)
        .attr('stroke-width', 1)
        .attr('rx', 3)
        .attr('opacity', 0.8);

      g.append('text')
        .attr('x', encoderX + encoderWidth / 2)
        .attr('y', layerY + (layerHeight / 2 - 5) / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .attr('fill', 'white')
        .text('Multi-Head Self-Attention');

      // Feed Forward
      g.append('rect')
        .attr('x', encoderX + 10)
        .attr('y', layerY + layerHeight / 2)
        .attr('width', encoderWidth - 20)
        .attr('height', layerHeight / 2 - 5)
        .attr('fill', colors.ffn)
        .attr('stroke', colors.text)
        .attr('stroke-width', 1)
        .attr('rx', 3)
        .attr('opacity', 0.8);

      g.append('text')
        .attr('x', encoderX + encoderWidth / 2)
        .attr('y', layerY + layerHeight / 2 + (layerHeight / 2 - 5) / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .attr('fill', 'white')
        .text('Feed Forward');

      // 层间连接箭头
      if (i < numLayers - 1) {
        g.append('path')
          .attr('d', `M ${encoderX + encoderWidth / 2} ${layerY + layerHeight - 5} L ${encoderX + encoderWidth / 2} ${layerY + layerHeight + 5}`)
          .attr('stroke', colors.text)
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead-bert)');
      }
    }

    // 输出
    const outputY = encoderY + encoderHeight + 20;
    g.append('rect')
      .attr('x', encoderX)
      .attr('y', outputY)
      .attr('width', encoderWidth)
      .attr('height', 50)
      .attr('fill', colors.norm)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);

    g.append('text')
      .attr('x', encoderX + encoderWidth / 2)
      .attr('y', outputY + 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('Contextualized Embeddings');

    // 定义箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-bert')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);

    // 连接箭头
    g.append('path')
      .attr('d', `M ${encoderX + encoderWidth / 2} ${embedY + 50} L ${encoderX + encoderWidth / 2} ${encoderY}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-bert)');

    g.append('path')
      .attr('d', `M ${encoderX + encoderWidth / 2} ${encoderY + encoderHeight} L ${encoderX + encoderWidth / 2} ${outputY}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-bert)');
  }

  // 渲染MLM机制
  function renderMLM(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    const tokenWidth = 80;
    const tokenHeight = 40;
    const tokens = ['[CLS]', 'The', '[MASK]', 'sat', 'on', 'the', 'mat', '[SEP]'];
    const startX = centerX - (tokens.length * tokenWidth) / 2;

    tokens.forEach((token, i) => {
      const x = startX + i * tokenWidth;
      const isMask = token === '[MASK]';
      
      const box = g.append('rect')
        .attr('x', x)
        .attr('y', centerY - tokenHeight / 2)
        .attr('width', tokenWidth - 10)
        .attr('height', tokenHeight)
        .attr('fill', isMask ? colors.mask : colors.embedding)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5)
        .attr('opacity', isMask ? 1 : 0.8);

      g.append('text')
        .attr('x', x + (tokenWidth - 10) / 2)
        .attr('y', centerY + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(token);

      // 如果是MASK，添加预测箭头
      if (isMask) {
        g.append('path')
          .attr('d', `M ${x + (tokenWidth - 10) / 2} ${centerY + tokenHeight / 2 + 10} L ${x + (tokenWidth - 10) / 2} ${centerY + 100}`)
          .attr('stroke', colors.mask)
          .attr('stroke-width', 3)
          .attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead-mlm)');

        g.append('text')
          .attr('x', x + (tokenWidth - 10) / 2)
          .attr('y', centerY + 130)
          .attr('text-anchor', 'middle')
          .attr('font-size', '14px')
          .attr('fill', colors.text)
          .attr('font-weight', 'bold')
          .text('预测: cat');
      }
    });

    // 定义箭头
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead-mlm')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.mask);

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('掩码语言模型（MLM）：随机遮盖15%的词，预测被遮盖的词');
  }

  // 渲染双向注意力
  function renderAttention(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    const tokenSize = 60;
    const tokens = ['The', 'cat', 'sat'];
    const startX = centerX - (tokens.length * tokenSize) / 2;

    tokens.forEach((token, i) => {
      const x = startX + i * tokenSize;
      
      // Token框
      const box = g.append('rect')
        .attr('x', x)
        .attr('y', centerY - tokenSize / 2)
        .attr('width', tokenSize - 10)
        .attr('height', tokenSize)
        .attr('fill', colors.embedding)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5);

      g.append('text')
        .attr('x', x + (tokenSize - 10) / 2)
        .attr('y', centerY + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .text(token);

      // 双向连接
      tokens.forEach((otherToken, j) => {
        if (i !== j) {
          g.append('path')
            .attr('d', `M ${x + (tokenSize - 10) / 2} ${centerY} L ${startX + j * tokenSize + (tokenSize - 10) / 2} ${centerY}`)
            .attr('stroke', colors.attention)
            .attr('stroke-width', 1)
            .attr('fill', 'none')
            .attr('opacity', 0.3);
        }
      });
    });

    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 80)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', colors.text)
      .attr('font-weight', 'bold')
      .text('双向自注意力：每个词可以看到所有其他词');

    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('与GPT的单向注意力不同，BERT可以同时利用前后文信息');
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

export default BERTDiagram;
