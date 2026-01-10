import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import './TransformerDiagram.css';

const TransformerDiagram = ({ width = 1200, height = 900, interactive = true }) => {
  const svgRef = useRef(null);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [hoveredComponent, setHoveredComponent] = useState(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // 清除之前的内容

    const margin = { top: 60, right: 80, bottom: 80, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // 定义颜色方案
    const colors = {
      encoder: '#667eea',
      decoder: '#764ba2',
      attention: '#f093fb',
      ffn: '#4facfe',
      norm: '#43e97b',
      embedding: '#fa709a',
      output: '#fee140',
      connection: '#d3d3d3'
    };

    // Transformer 架构数据
    const architecture = {
      encoder: {
        x: innerWidth / 2 - 250,
        y: 120,
        width: 220,
        height: 580,
        layers: 6,
        label: 'Encoder'
      },
      decoder: {
        x: innerWidth / 2 + 30,
        y: 120,
        width: 220,
        height: 580,
        layers: 6,
        label: 'Decoder'
      }
    };

    // 绘制 Encoder
    const drawEncoder = () => {
      const encoderGroup = g.append('g').attr('class', 'encoder-group');
      
      // Encoder 外框
      encoderGroup.append('rect')
        .attr('x', architecture.encoder.x)
        .attr('y', architecture.encoder.y)
        .attr('width', architecture.encoder.width)
        .attr('height', architecture.encoder.height)
        .attr('fill', 'none')
        .attr('stroke', colors.encoder)
        .attr('stroke-width', 3)
        .attr('rx', 8)
        .attr('opacity', selectedLayer === 'encoder' ? 1 : 0.7);

      // Encoder 标题
      encoderGroup.append('text')
        .attr('x', architecture.encoder.x + architecture.encoder.width / 2)
        .attr('y', architecture.encoder.y - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', 'bold')
        .attr('fill', colors.encoder)
        .text('Encoder');

      // Encoder 层
      const layerHeight = (architecture.encoder.height - 60) / architecture.encoder.layers;
      for (let i = 0; i < architecture.encoder.layers; i++) {
        const layerY = architecture.encoder.y + 40 + i * layerHeight;
        const layerGroup = encoderGroup.append('g')
          .attr('class', `encoder-layer layer-${i}`)
          .attr('data-layer', i);
        
        // 添加层间垂直数据流动（从上一层到当前层）
        if (interactive && i > 0) {
          const prevLayerY = architecture.encoder.y + 40 + (i - 1) * layerHeight;
          const centerX = architecture.encoder.x + architecture.encoder.width / 2;
          
          // 创建垂直流动粒子
          const verticalParticle = encoderGroup.append('circle')
            .attr('r', 3)
            .attr('fill', colors.encoder)
            .attr('opacity', 0.7)
            .attr('class', 'vertical-particle');
          
          const animateVertical = () => {
            verticalParticle
              .attr('cx', centerX)
              .attr('cy', prevLayerY + layerHeight - 5)
              .transition()
              .duration(1500)
              .ease(d3.easeLinear)
              .attr('cy', layerY)
              .on('end', animateVertical);
          };
          
          setTimeout(() => {
            animateVertical();
          }, i * 200);
        }

        // 层背景
        layerGroup.append('rect')
          .attr('x', architecture.encoder.x + 10)
          .attr('y', layerY)
          .attr('width', architecture.encoder.width - 20)
          .attr('height', layerHeight - 5)
          .attr('fill', colors.encoder)
          .attr('opacity', 0.1)
          .attr('rx', 4)
          .on('mouseenter', function() {
            if (interactive) {
              d3.select(this).attr('opacity', 0.3);
              setHoveredComponent(`encoder-layer-${i}`);
            }
          })
          .on('mouseleave', function() {
            if (interactive) {
              d3.select(this).attr('opacity', 0.1);
              setHoveredComponent(null);
            }
          })
          .on('click', function() {
            if (interactive) {
              setSelectedLayer(selectedLayer === `encoder-layer-${i}` ? null : `encoder-layer-${i}`);
            }
          });

        // Multi-Head Attention
        layerGroup.append('rect')
          .attr('x', architecture.encoder.x + 20)
          .attr('y', layerY + 5)
          .attr('width', architecture.encoder.width - 40)
          .attr('height', (layerHeight - 15) / 2)
          .attr('fill', colors.attention)
          .attr('opacity', 0.6)
          .attr('rx', 3)
          .attr('stroke', colors.attention)
          .attr('stroke-width', 1);

        layerGroup.append('text')
          .attr('x', architecture.encoder.x + architecture.encoder.width / 2)
          .attr('y', layerY + (layerHeight - 15) / 4 + 5)
          .attr('text-anchor', 'middle')
          .attr('font-size', '11px')
          .attr('fill', '#333')
          .text('Multi-Head Attention');

        // Add & Norm
        layerGroup.append('rect')
          .attr('x', architecture.encoder.x + 20)
          .attr('y', layerY + (layerHeight - 15) / 2 + 5)
          .attr('width', (architecture.encoder.width - 50) / 2)
          .attr('height', (layerHeight - 15) / 2)
          .attr('fill', colors.norm)
          .attr('opacity', 0.5)
          .attr('rx', 3);

        layerGroup.append('text')
          .attr('x', architecture.encoder.x + 20 + (architecture.encoder.width - 50) / 4)
          .attr('y', layerY + (layerHeight - 15) / 2 + (layerHeight - 15) / 4 + 5)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', '#333')
          .text('Add & Norm');

        // Feed Forward
        layerGroup.append('rect')
          .attr('x', architecture.encoder.x + 30 + (architecture.encoder.width - 50) / 2)
          .attr('y', layerY + (layerHeight - 15) / 2 + 5)
          .attr('width', (architecture.encoder.width - 50) / 2)
          .attr('height', (layerHeight - 15) / 2)
          .attr('fill', colors.ffn)
          .attr('opacity', 0.5)
          .attr('rx', 3);

        layerGroup.append('text')
          .attr('x', architecture.encoder.x + 30 + (architecture.encoder.width - 50) / 2 + (architecture.encoder.width - 50) / 4)
          .attr('y', layerY + (layerHeight - 15) / 2 + (layerHeight - 15) / 4 + 5)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', '#333')
          .text('FFN');

        // Add & Norm (第二个)
        layerGroup.append('rect')
          .attr('x', architecture.encoder.x + 20)
          .attr('y', layerY + layerHeight - 10)
          .attr('width', architecture.encoder.width - 40)
          .attr('height', 5)
          .attr('fill', colors.norm)
          .attr('opacity', 0.5)
          .attr('rx', 2);
      }

      // 输入嵌入
      const inputRect = encoderGroup.append('rect')
        .attr('x', architecture.encoder.x + 10)
        .attr('y', architecture.encoder.y - 30)
        .attr('width', architecture.encoder.width - 20)
        .attr('height', 25)
        .attr('fill', colors.embedding)
        .attr('opacity', 0.7)
        .attr('rx', 4)
        .attr('stroke', colors.embedding)
        .attr('stroke-width', 2);

      encoderGroup.append('text')
        .attr('x', architecture.encoder.x + architecture.encoder.width / 2)
        .attr('y', architecture.encoder.y - 15)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('fill', '#fff')
        .text('Input Embeddings');
      
      // 从输入到第一层的数据流动
      if (interactive) {
        const inputY = architecture.encoder.y - 5;
        const firstLayerY = architecture.encoder.y + 40;
        const centerX = architecture.encoder.x + architecture.encoder.width / 2;
        
        const inputParticle = encoderGroup.append('circle')
          .attr('r', 4)
          .attr('fill', colors.embedding)
          .attr('opacity', 0.9)
          .attr('class', 'input-particle');
        
        const animateInput = () => {
          inputParticle
            .attr('cx', centerX)
            .attr('cy', inputY)
            .transition()
            .duration(1000)
            .ease(d3.easeLinear)
            .attr('cy', firstLayerY)
            .on('end', animateInput);
        };
        
        setTimeout(() => {
          animateInput();
        }, 500);
      }

      // 位置编码
      encoderGroup.append('rect')
        .attr('x', architecture.encoder.x + 10)
        .attr('y', architecture.encoder.y - 5)
        .attr('width', architecture.encoder.width - 20)
        .attr('height', 25)
        .attr('fill', colors.embedding)
        .attr('opacity', 0.5)
        .attr('rx', 4)
        .attr('stroke', colors.embedding)
        .attr('stroke-width', 1);

      encoderGroup.append('text')
        .attr('x', architecture.encoder.x + architecture.encoder.width / 2)
        .attr('y', architecture.encoder.y + 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .attr('fill', '#333')
        .text('Positional Encoding');
    };

    // 绘制 Decoder
    const drawDecoder = () => {
      const decoderGroup = g.append('g').attr('class', 'decoder-group');
      
      // Decoder 外框
      decoderGroup.append('rect')
        .attr('x', architecture.decoder.x)
        .attr('y', architecture.decoder.y)
        .attr('width', architecture.decoder.width)
        .attr('height', architecture.decoder.height)
        .attr('fill', 'none')
        .attr('stroke', colors.decoder)
        .attr('stroke-width', 3)
        .attr('rx', 8)
        .attr('opacity', selectedLayer === 'decoder' ? 1 : 0.7);

      // Decoder 标题
      decoderGroup.append('text')
        .attr('x', architecture.decoder.x + architecture.decoder.width / 2)
        .attr('y', architecture.decoder.y - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', 'bold')
        .attr('fill', colors.decoder)
        .text('Decoder');

      // Decoder 层
      const layerHeight = (architecture.decoder.height - 60) / architecture.decoder.layers;
      for (let i = 0; i < architecture.decoder.layers; i++) {
        const layerY = architecture.decoder.y + 40 + i * layerHeight;
        const layerGroup = decoderGroup.append('g')
          .attr('class', `decoder-layer layer-${i}`);
        
        // 添加层间垂直数据流动（从上一层到当前层）
        if (interactive && i > 0) {
          const prevLayerY = architecture.decoder.y + 40 + (i - 1) * layerHeight;
          const centerX = architecture.decoder.x + architecture.decoder.width / 2;
          
          // 创建垂直流动粒子
          const verticalParticle = decoderGroup.append('circle')
            .attr('r', 3)
            .attr('fill', colors.decoder)
            .attr('opacity', 0.7)
            .attr('class', 'vertical-particle');
          
          const animateVertical = () => {
            verticalParticle
              .attr('cx', centerX)
              .attr('cy', prevLayerY + layerHeight - 5)
              .transition()
              .duration(1500)
              .ease(d3.easeLinear)
              .attr('cy', layerY)
              .on('end', animateVertical);
          };
          
          setTimeout(() => {
            animateVertical();
          }, i * 200);
        }

        // 层背景
        layerGroup.append('rect')
          .attr('x', architecture.decoder.x + 10)
          .attr('y', layerY)
          .attr('width', architecture.decoder.width - 20)
          .attr('height', layerHeight - 5)
          .attr('fill', colors.decoder)
          .attr('opacity', 0.1)
          .attr('rx', 4);

        // Masked Multi-Head Attention
        layerGroup.append('rect')
          .attr('x', architecture.decoder.x + 20)
          .attr('y', layerY + 5)
          .attr('width', architecture.decoder.width - 40)
          .attr('height', (layerHeight - 25) / 3)
          .attr('fill', colors.attention)
          .attr('opacity', 0.6)
          .attr('rx', 3);

        layerGroup.append('text')
          .attr('x', architecture.decoder.x + architecture.decoder.width / 2)
          .attr('y', layerY + (layerHeight - 25) / 6 + 5)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', '#333')
          .text('Masked MHA');

        // Encoder-Decoder Attention
        layerGroup.append('rect')
          .attr('x', architecture.decoder.x + 20)
          .attr('y', layerY + (layerHeight - 25) / 3 + 5)
          .attr('width', architecture.decoder.width - 40)
          .attr('height', (layerHeight - 25) / 3)
          .attr('fill', colors.attention)
          .attr('opacity', 0.5)
          .attr('rx', 3);

        layerGroup.append('text')
          .attr('x', architecture.decoder.x + architecture.decoder.width / 2)
          .attr('y', layerY + (layerHeight - 25) / 2 + 5)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', '#333')
          .text('Encoder-Decoder MHA');

        // Feed Forward
        layerGroup.append('rect')
          .attr('x', architecture.decoder.x + 20)
          .attr('y', layerY + 2 * (layerHeight - 25) / 3 + 5)
          .attr('width', architecture.decoder.width - 40)
          .attr('height', (layerHeight - 25) / 3)
          .attr('fill', colors.ffn)
          .attr('opacity', 0.5)
          .attr('rx', 3);

        layerGroup.append('text')
          .attr('x', architecture.decoder.x + architecture.decoder.width / 2)
          .attr('y', layerY + 2.5 * (layerHeight - 25) / 3 + 5)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', '#333')
          .text('FFN');
      }

      // 输出嵌入
      decoderGroup.append('rect')
        .attr('x', architecture.decoder.x + 10)
        .attr('y', architecture.decoder.y - 30)
        .attr('width', architecture.decoder.width - 20)
        .attr('height', 25)
        .attr('fill', colors.embedding)
        .attr('opacity', 0.7)
        .attr('rx', 4)
        .attr('stroke', colors.embedding)
        .attr('stroke-width', 2);

      decoderGroup.append('text')
        .attr('x', architecture.decoder.x + architecture.decoder.width / 2)
        .attr('y', architecture.decoder.y - 15)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('fill', '#fff')
        .text('Output Embeddings');
    };

    // 绘制连接线
    const drawConnections = () => {
      const connectionGroup = g.append('g').attr('class', 'connection-group');
      
      // Encoder 到 Decoder 的连接
      for (let i = 0; i < architecture.encoder.layers; i++) {
        const encoderY = architecture.encoder.y + 40 + i * (architecture.encoder.height - 60) / architecture.encoder.layers + (architecture.encoder.height - 60) / architecture.encoder.layers / 2;
        const decoderY = architecture.decoder.y + 40 + i * (architecture.decoder.height - 60) / architecture.decoder.layers + (architecture.decoder.height - 60) / architecture.decoder.layers / 2;
        
        const x1 = architecture.encoder.x + architecture.encoder.width;
        const y1 = encoderY;
        const x2 = architecture.decoder.x;
        const y2 = decoderY;
        
        // 绘制连接线
        const line = connectionGroup.append('line')
          .attr('x1', x1)
          .attr('y1', y1)
          .attr('x2', x2)
          .attr('y2', y2)
          .attr('stroke', colors.connection)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.5)
          .attr('marker-end', 'url(#arrowhead)');
        
        // 添加数据流动粒子
        if (interactive) {
          const lineLength = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
          const numParticles = 3;
          
          for (let j = 0; j < numParticles; j++) {
            const particle = connectionGroup.append('circle')
              .attr('r', 4)
              .attr('fill', '#667eea')
              .attr('opacity', 0.8)
              .attr('class', 'data-particle');
            
            // 创建动画
            const animateParticle = () => {
              particle
                .attr('cx', x1)
                .attr('cy', y1)
                .transition()
                .duration(2000 + j * 300) // 错开时间
                .ease(d3.easeLinear)
                .attr('cx', x2)
                .attr('cy', y2)
                .on('end', animateParticle); // 循环动画
            };
            
            // 延迟启动，错开粒子
            setTimeout(() => {
              animateParticle();
            }, j * 600);
          }
        }
      }
    };

    // 绘制箭头标记
    const defs = svg.append('defs');
    const arrowMarker = defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    
    arrowMarker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.connection);

    // 绘制输出层
    const outputGroup = g.append('g').attr('class', 'output-group');
    outputGroup.append('rect')
      .attr('x', architecture.decoder.x + 10)
      .attr('y', architecture.decoder.y + architecture.decoder.height + 10)
      .attr('width', architecture.decoder.width - 20)
      .attr('height', 40)
      .attr('fill', colors.output)
      .attr('opacity', 0.8)
      .attr('rx', 4)
      .attr('stroke', colors.output)
      .attr('stroke-width', 2);

    outputGroup.append('text')
      .attr('x', architecture.decoder.x + architecture.decoder.width / 2)
      .attr('y', architecture.decoder.y + architecture.decoder.height + 35)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text('Linear & Softmax');
    
    // 从最后一层到输出的数据流动
    if (interactive) {
      const lastLayerY = architecture.decoder.y + 40 + (architecture.decoder.layers - 1) * (architecture.decoder.height - 60) / architecture.decoder.layers + (architecture.decoder.height - 60) / architecture.decoder.layers;
      const outputY = architecture.decoder.y + architecture.decoder.height + 10;
      const centerX = architecture.decoder.x + architecture.decoder.width / 2;
      
      const outputParticle = outputGroup.append('circle')
        .attr('r', 4)
        .attr('fill', colors.output)
        .attr('opacity', 0.9)
        .attr('class', 'output-particle');
      
      const animateOutput = () => {
        outputParticle
          .attr('cx', centerX)
          .attr('cy', lastLayerY)
          .transition()
          .duration(1000)
          .ease(d3.easeLinear)
          .attr('cy', outputY)
          .on('end', animateOutput);
      };
      
      setTimeout(() => {
        animateOutput();
      }, 1000);
    }

    // 绘制图例
    const legendGroup = g.append('g').attr('class', 'legend-group');
    const legendItems = [
      { label: 'Multi-Head Attention', color: colors.attention },
      { label: 'Feed Forward', color: colors.ffn },
      { label: 'Add & Norm', color: colors.norm },
      { label: 'Embeddings', color: colors.embedding }
    ];

    legendGroup.selectAll('.legend-item')
      .data(legendItems)
      .enter()
      .append('g')
      .attr('class', 'legend-item')
      .attr('transform', (d, i) => `translate(${innerWidth - 180}, ${i * 25 + 30})`)
      .each(function(d) {
        const item = d3.select(this);
        item.append('rect')
          .attr('width', 15)
          .attr('height', 15)
          .attr('fill', d.color)
          .attr('opacity', 0.7);
        item.append('text')
          .attr('x', 20)
          .attr('y', 12)
          .attr('font-size', '12px')
          .attr('fill', '#333')
          .text(d.label);
      });

    // 执行绘制
    drawEncoder();
    drawDecoder();
    drawConnections();

    // 清理函数
    return () => {
      if (svgRef.current) {
        d3.select(svgRef.current).selectAll('*').remove();
      }
    };
  }, [width, height, interactive, selectedLayer, hoveredComponent]);

  return (
    <div className="transformer-diagram-container">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="transformer-diagram-svg"
      />
    </div>
  );
};

export default TransformerDiagram;
