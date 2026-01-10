import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * 通用交互式SVG图表组件
 * 可以根据配置渲染不同类型的架构图
 */
const GenericDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'architecture', // architecture, flow, comparison, concept
  data = null,
  title = '',
  ...props 
}) => {
  const svgRef = useRef(null);
  const [selectedElement, setSelectedElement] = useState(null);
  const [hoveredElement, setHoveredElement] = useState(null);

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
      primary: '#667eea',
      secondary: '#764ba2',
      accent: '#f093fb',
      highlight: '#4facfe',
      success: '#43e97b',
      warning: '#fee140',
      error: '#fa709a',
      text: '#2d3748',
      background: '#f7fafc',
      border: '#e2e8f0'
    };

    // 根据类型渲染不同的图表
    switch (type) {
      case 'architecture':
        renderArchitecture(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'flow':
        renderFlow(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'comparison':
        renderComparison(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'concept':
        renderConcept(g, innerWidth, innerHeight, colors, interactive);
        break;
      default:
        renderPlaceholder(g, innerWidth, innerHeight, colors, title);
    }

  }, [width, height, type, interactive, title, data]);

  // 渲染架构图
  const renderArchitecture = (g, width, height, colors, interactive) => {
    // 添加标题
    if (title) {
      g.append('text')
        .attr('x', width / 2)
        .attr('y', -20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '24px')
        .attr('font-weight', 'bold')
        .attr('fill', colors.text)
        .text(title);
    }

    // 示例：绘制一个简单的架构框
    const box = g.append('rect')
      .attr('x', width / 2 - 150)
      .attr('y', height / 2 - 100)
      .attr('width', 300)
      .attr('height', 200)
      .attr('fill', colors.background)
      .attr('stroke', colors.primary)
      .attr('stroke-width', 3)
      .attr('rx', 8)
      .attr('opacity', 0.8);

    if (interactive) {
      box
        .on('mouseenter', function() {
          d3.select(this).attr('opacity', 1).attr('stroke-width', 4);
          setHoveredElement('main-box');
        })
        .on('mouseleave', function() {
          d3.select(this).attr('opacity', 0.8).attr('stroke-width', 3);
          setHoveredElement(null);
        })
        .on('click', function() {
          setSelectedElement(selectedElement === 'main-box' ? null : 'main-box');
        });
    }

    // 添加说明文字
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height / 2)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', colors.text)
      .text('架构图解（交互式 SVG）');
  };

  // 渲染流程图
  const renderFlow = (g, width, height, colors, interactive) => {
    // 流程图实现
    const steps = ['输入', '处理', '输出'];
    const stepWidth = width / (steps.length + 1);
    
    steps.forEach((step, i) => {
      const x = stepWidth * (i + 1);
      const y = height / 2;
      
      const circle = g.append('circle')
        .attr('cx', x)
        .attr('cy', y)
        .attr('r', 50)
        .attr('fill', colors.primary)
        .attr('opacity', 0.8);

      if (interactive) {
        circle
          .on('mouseenter', function() {
            d3.select(this).attr('opacity', 1).attr('r', 55);
          })
          .on('mouseleave', function() {
            d3.select(this).attr('opacity', 0.8).attr('r', 50);
          });
      }

      g.append('text')
        .attr('x', x)
        .attr('y', y)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', 'white')
        .text(step);

      // 箭头
      if (i < steps.length - 1) {
        g.append('path')
          .attr('d', `M ${x + 50} ${y} L ${x + stepWidth - 50} ${y}`)
          .attr('stroke', colors.primary)
          .attr('stroke-width', 2)
          .attr('marker-end', 'url(#arrowhead)');
      }
    });

    // 定义箭头标记
    const defs = g.append('defs');
    const marker = defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto');
    marker.append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.primary);
  };

  // 渲染对比图
  const renderComparison = (g, width, height, colors, interactive) => {
    // 对比图实现
    const items = ['方法A', '方法B', '方法C'];
    const itemWidth = width / items.length;
    
    items.forEach((item, i) => {
      const x = itemWidth * i + itemWidth / 2;
      const barHeight = 100 + Math.random() * 200;
      
      const bar = g.append('rect')
        .attr('x', x - 40)
        .attr('y', height - barHeight)
        .attr('width', 80)
        .attr('height', barHeight)
        .attr('fill', colors.primary)
        .attr('opacity', 0.8);

      if (interactive) {
        bar
          .on('mouseenter', function() {
            d3.select(this).attr('opacity', 1);
          })
          .on('mouseleave', function() {
            d3.select(this).attr('opacity', 0.8);
          });
      }

      g.append('text')
        .attr('x', x)
        .attr('y', height + 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', colors.text)
        .text(item);
    });
  };

  // 渲染概念图
  const renderConcept = (g, width, height, colors, interactive) => {
    // 概念图实现
    const centerX = width / 2;
    const centerY = height / 2;
    
    const circle = g.append('circle')
      .attr('cx', centerX)
      .attr('cy', centerY)
      .attr('r', 100)
      .attr('fill', colors.primary)
      .attr('opacity', 0.8);

    if (interactive) {
      circle
        .on('mouseenter', function() {
          d3.select(this).attr('opacity', 1).attr('r', 110);
        })
        .on('mouseleave', function() {
          d3.select(this).attr('opacity', 0.8).attr('r', 100);
        });
    }

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY)
      .attr('text-anchor', 'middle')
      .attr('font-size', '18px')
      .attr('fill', 'white')
      .text(title || '概念图解');
  };

  // 渲染占位符
  const renderPlaceholder = (g, width, height, colors, title) => {
    g.append('rect')
      .attr('x', width / 2 - 200)
      .attr('y', height / 2 - 100)
      .attr('width', 400)
      .attr('height', 200)
      .attr('fill', colors.background)
      .attr('stroke', colors.border)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .attr('rx', 8);

    g.append('text')
      .attr('x', width / 2)
      .attr('y', height / 2)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', colors.text)
      .text(title || '交互式架构图解');
  };

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
      {selectedElement && (
        <div className="diagram-tooltip">
          已选择: {selectedElement}
        </div>
      )}
    </div>
  );
};

export default GenericDiagram;
