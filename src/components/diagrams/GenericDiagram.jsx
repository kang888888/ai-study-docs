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
    const titleLower = (title || '').toLowerCase();
    
    // 根据标题识别不同的概念并绘制相应的可视化
    // 注意：顺序很重要，更具体的匹配应该放在前面
    if (titleLower.includes('梯度下降') || titleLower.includes('gradient descent')) {
      renderGradientDescentConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('梯度') || titleLower.includes('gradient')) {
      renderGradientConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('反向传播') || titleLower.includes('backpropagation')) {
      renderBackpropagationConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('优化器原理') || titleLower.includes('optimizer theory')) {
      // 优化器原理显示更详细的内容
      renderOptimizerTheoryConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('优化器') || titleLower.includes('optimizer')) {
      renderOptimizerConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('学习率调度') || titleLower.includes('lr scheduler') || titleLower.includes('learning rate')) {
      renderLRSchedulerConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('sam') || titleLower.includes('sharpness')) {
      renderSAMConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('二阶优化') || titleLower.includes('second order') || titleLower.includes('newton')) {
      renderSecondOrderConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('数据清洗') || titleLower.includes('data cleaning')) {
      renderDataCleaningConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('数据增强') || titleLower.includes('data augmentation')) {
      renderDataAugmentationConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('正则化') || titleLower.includes('regularization')) {
      renderRegularizationConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('残差') || titleLower.includes('residual')) {
      renderResidualConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('位置编码') || titleLower.includes('position')) {
      renderPositionEncodingConcept(g, width, height, colors, interactive);
    } else if (titleLower.includes('归一化') || titleLower.includes('normalization')) {
      renderNormalizationConcept(g, width, height, colors, interactive);
    } else {
      // 默认显示简单的概念图
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
    }
  };

  // 渲染梯度概念图
  function renderGradientConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制损失函数曲线（简化的2D示例）
    const xScale = d3.scaleLinear().domain([-3, 3]).range([100, width - 100]);
    const yScale = d3.scaleLinear().domain([0, 10]).range([height - 100, 100]);
    
    // 损失函数曲线 y = x^2 + 2
    const curveData = d3.range(-3, 3, 0.1).map(x => ({
      x: x,
      y: x * x + 2
    }));
    
    const line = d3.line()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveBasis);
    
    g.append('path')
      .datum(curveData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', colors.primary)
      .attr('stroke-width', 3);
    
    // 绘制梯度箭头（在x=1处）
    const pointX = 1;
    const pointY = pointX * pointX + 2;
    const gradient = 2 * pointX; // dy/dx = 2x
    
    g.append('circle')
      .attr('cx', xScale(pointX))
      .attr('cy', yScale(pointY))
      .attr('r', 5)
      .attr('fill', colors.error);
    
    // 梯度箭头
    const arrowLength = 50;
    const arrowX = xScale(pointX);
    const arrowY = yScale(pointY);
    const angle = Math.atan(gradient);
    
    g.append('path')
      .attr('d', `M ${arrowX} ${arrowY} L ${arrowX + arrowLength * Math.cos(angle)} ${arrowY - arrowLength * Math.sin(angle)}`)
      .attr('stroke', colors.error)
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#gradient-arrow)');
    
    // 箭头标记
    const defs = g.append('defs');
    defs.append('marker')
      .attr('id', 'gradient-arrow')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto')
      .append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.error);
    
    // 标签
    g.append('text')
      .attr('x', xScale(pointX) + 60)
      .attr('y', yScale(pointY) - 20)
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('梯度方向');
  }

  // 渲染反向传播概念图
  function renderBackpropagationConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制简单的三层网络
    const layers = [
      { x: 150, nodes: 3, label: '输入层' },
      { x: centerX, nodes: 4, label: '隐藏层' },
      { x: width - 150, nodes: 2, label: '输出层' }
    ];
    
    layers.forEach((layer, i) => {
      const nodeSpacing = 80;
      const startY = centerY - (layer.nodes - 1) * nodeSpacing / 2;
      
      for (let j = 0; j < layer.nodes; j++) {
        const y = startY + j * nodeSpacing;
        
        g.append('circle')
          .attr('cx', layer.x)
          .attr('cy', y)
          .attr('r', 20)
          .attr('fill', i === 0 ? colors.primary : i === 1 ? colors.secondary : colors.success)
          .attr('opacity', 0.8);
        
        // 连接线（仅显示反向传播方向）
        if (i > 0) {
          const prevLayer = layers[i - 1];
          const prevStartY = centerY - (prevLayer.nodes - 1) * nodeSpacing / 2;
          
          for (let k = 0; k < prevLayer.nodes; k++) {
            const prevY = prevStartY + k * nodeSpacing;
            g.append('path')
              .attr('d', `M ${layer.x - 20} ${y} L ${prevLayer.x + 20} ${prevY}`)
              .attr('stroke', colors.error)
              .attr('stroke-width', 1.5)
              .attr('opacity', 0.3)
              .attr('marker-end', 'url(#backward-arrow)');
          }
        }
      }
      
      // 层标签
      g.append('text')
        .attr('x', layer.x)
        .attr('y', centerY + (layer.nodes - 1) * nodeSpacing / 2 + 40)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', colors.text)
        .text(layer.label);
    });
    
    // 反向箭头标记
    const defs = g.append('defs');
    defs.append('marker')
      .attr('id', 'backward-arrow')
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('refX', 7)
      .attr('refY', 2)
      .attr('orient', 'auto')
      .append('polygon')
      .attr('points', '0 0, 8 2, 0 4')
      .attr('fill', colors.error);
    
    // 说明文字
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('反向传播：从输出层向输入层传播梯度');
  }

  // 渲染优化器概念图
  function renderOptimizerConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制优化器对比（简化的收敛曲线）
    const optimizers = [
      { name: 'SGD', color: colors.primary, path: [] },
      { name: 'Adam', color: colors.success, path: [] },
      { name: 'AdamW', color: colors.accent, path: [] }
    ];
    
    const xScale = d3.scaleLinear().domain([0, 100]).range([100, width - 100]);
    const yScale = d3.scaleLinear().domain([0, 5]).range([100, height - 200]);
    
    optimizers.forEach((opt, idx) => {
      // 生成不同的收敛曲线
      const pathData = [];
      for (let i = 0; i <= 100; i += 2) {
        let y;
        if (opt.name === 'SGD') {
          y = 5 * Math.exp(-i / 30) + 0.5 + Math.random() * 0.3;
        } else if (opt.name === 'Adam') {
          y = 5 * Math.exp(-i / 20) + 0.2 + Math.random() * 0.2;
        } else {
          y = 5 * Math.exp(-i / 18) + 0.1 + Math.random() * 0.15;
        }
        pathData.push({ x: i, y: Math.max(0, y) });
      }
      
      const line = d3.line()
        .x(d => xScale(d.x))
        .y(d => yScale(d.y))
        .curve(d3.curveBasis);
      
      g.append('path')
        .datum(pathData)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', opt.color)
        .attr('stroke-width', 2);
      
      // 图例
      g.append('circle')
        .attr('cx', 120)
        .attr('cy', 150 + idx * 30)
        .attr('r', 5)
        .attr('fill', opt.color);
      
      g.append('text')
        .attr('x', 135)
        .attr('y', 155 + idx * 30)
        .attr('font-size', '12px')
        .attr('fill', colors.text)
        .text(opt.name);
    });
    
    // 坐标轴标签
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 50)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .text('迭代次数');
    
    g.append('text')
      .attr('x', 50)
      .attr('y', centerY)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .attr('transform', `rotate(-90, 50, ${centerY})`)
      .text('损失值');
  }

  // 渲染正则化概念图
  function renderRegularizationConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制过拟合 vs 正则化对比
    const xScale = d3.scaleLinear().domain([0, 10]).range([100, width - 100]);
    const yScale = d3.scaleLinear().domain([0, 10]).range([height - 100, 100]);
    
    // 过拟合曲线（复杂）
    const overfitData = d3.range(0, 10, 0.1).map(x => ({
      x: x,
      y: 5 + 2 * Math.sin(x * 2) + 1.5 * Math.cos(x * 3)
    }));
    
    // 正则化曲线（平滑）
    const regularizedData = d3.range(0, 10, 0.1).map(x => ({
      x: x,
      y: 5 + 0.5 * x
    }));
    
    const line = d3.line()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveBasis);
    
    g.append('path')
      .datum(overfitData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', colors.error)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5');
    
    g.append('path')
      .datum(regularizedData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', colors.success)
      .attr('stroke-width', 2);
    
    // 图例
    g.append('text')
      .attr('x', 120)
      .attr('y', 150)
      .attr('font-size', '12px')
      .attr('fill', colors.error)
      .text('过拟合（无正则化）');
    
    g.append('text')
      .attr('x', 120)
      .attr('y', 180)
      .attr('font-size', '12px')
      .attr('fill', colors.success)
      .text('正则化后（平滑）');
  }

  // 渲染残差链接概念图
  function renderResidualConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制残差块
    const blockWidth = 200;
    const blockHeight = 80;
    
    // 输入
    g.append('rect')
      .attr('x', centerX - blockWidth / 2)
      .attr('y', centerY - blockHeight / 2 - 100)
      .attr('width', blockWidth)
      .attr('height', blockHeight)
      .attr('fill', colors.primary)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);
    
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 60)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('x');
    
    // 残差块
    g.append('rect')
      .attr('x', centerX - blockWidth / 2)
      .attr('y', centerY - blockHeight / 2)
      .attr('width', blockWidth)
      .attr('height', blockHeight)
      .attr('fill', colors.secondary)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);
    
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('F(x)');
    
    // 输出
    g.append('rect')
      .attr('x', centerX - blockWidth / 2)
      .attr('y', centerY - blockHeight / 2 + 100)
      .attr('width', blockWidth)
      .attr('height', blockHeight)
      .attr('fill', colors.success)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('rx', 5);
    
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 160)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', 'white')
      .text('x + F(x)');
    
    // 前向箭头
    g.append('path')
      .attr('d', `M ${centerX} ${centerY - 60} L ${centerX} ${centerY - 40}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#residual-arrow)');
    
    g.append('path')
      .attr('d', `M ${centerX} ${centerY + 40} L ${centerX} ${centerY + 100}`)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#residual-arrow)');
    
    // 跳跃连接
    g.append('path')
      .attr('d', `M ${centerX - blockWidth / 2} ${centerY - 60} Q ${centerX - blockWidth / 2 - 30} ${centerY} ${centerX - blockWidth / 2} ${centerY + 60}`)
      .attr('fill', 'none')
      .attr('stroke', colors.accent)
      .attr('stroke-width', 3)
      .attr('marker-end', 'url(#residual-arrow)');
    
    // 加号
    g.append('circle')
      .attr('cx', centerX - blockWidth / 2 - 15)
      .attr('cy', centerY + 60)
      .attr('r', 15)
      .attr('fill', colors.accent);
    
    g.append('text')
      .attr('x', centerX - blockWidth / 2 - 15)
      .attr('y', centerY + 65)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', 'white')
      .text('+');
    
    // 箭头标记
    const defs = g.append('defs');
    defs.append('marker')
      .attr('id', 'residual-arrow')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto')
      .append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);
  }

  // 渲染位置编码概念图
  function renderPositionEncodingConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制位置编码矩阵（简化）
    const seqLength = 5;
    const embedDim = 4;
    const cellWidth = 40;
    const cellHeight = 40;
    const startX = centerX - (seqLength * cellWidth) / 2;
    const startY = centerY - (embedDim * cellHeight) / 2;
    
    for (let i = 0; i < seqLength; i++) {
      for (let j = 0; j < embedDim; j++) {
        const x = startX + i * cellWidth;
        const y = startY + j * cellHeight;
        const value = Math.sin(i / Math.pow(10000, 2 * j / embedDim));
        const intensity = (value + 1) / 2;
        
        g.append('rect')
          .attr('x', x)
          .attr('y', y)
          .attr('width', cellWidth - 2)
          .attr('height', cellHeight - 2)
          .attr('fill', d3.interpolateRdYlBu(intensity))
          .attr('stroke', colors.border)
          .attr('stroke-width', 1);
      }
      
      // 位置标签
      g.append('text')
        .attr('x', startX + i * cellWidth + cellWidth / 2)
        .attr('y', startY - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', colors.text)
        .text(`pos ${i}`);
    }
    
    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('位置编码：为每个位置添加唯一的位置信息');
  }

  // 渲染归一化概念图
  function renderNormalizationConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制归一化前后对比
    const methods = [
      { name: 'BatchNorm', x: 200, color: colors.primary },
      { name: 'LayerNorm', x: centerX, color: colors.secondary },
      { name: 'GroupNorm', x: width - 200, color: colors.accent }
    ];
    
    methods.forEach((method, idx) => {
      // 归一化前的数据分布（分散）
      for (let i = 0; i < 20; i++) {
        const angle = (i / 20) * Math.PI * 2;
        const radius = 30 + Math.random() * 20;
        const x = method.x - 60 + radius * Math.cos(angle);
        const y = centerY - 100 + radius * Math.sin(angle);
        
        g.append('circle')
          .attr('cx', x)
          .attr('cy', y)
          .attr('r', 3)
          .attr('fill', colors.error)
          .attr('opacity', 0.6);
      }
      
      // 归一化后的数据分布（集中）
      for (let i = 0; i < 20; i++) {
        const angle = (i / 20) * Math.PI * 2;
        const radius = 10 + Math.random() * 10;
        const x = method.x - 60 + radius * Math.cos(angle);
        const y = centerY + 50 + radius * Math.sin(angle);
        
        g.append('circle')
          .attr('cx', x)
          .attr('cy', y)
          .attr('r', 3)
          .attr('fill', colors.success)
          .attr('opacity', 0.6);
      }
      
      // 箭头
      g.append('path')
        .attr('d', `M ${method.x - 60} ${centerY - 80} L ${method.x - 60} ${centerY + 30}`)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#norm-arrow)');
      
      // 方法名称
      g.append('text')
        .attr('x', method.x - 60)
        .attr('y', centerY + 100)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', colors.text)
        .text(method.name);
    });
    
    // 箭头标记
    const defs = g.append('defs');
    defs.append('marker')
      .attr('id', 'norm-arrow')
      .attr('markerWidth', 10)
      .attr('markerHeight', 10)
      .attr('refX', 9)
      .attr('refY', 3)
      .attr('orient', 'auto')
      .append('polygon')
      .attr('points', '0 0, 10 3, 0 6')
      .attr('fill', colors.text);
    
    // 标签
    g.append('text')
      .attr('x', 100)
      .attr('y', centerY - 100)
      .attr('font-size', '12px')
      .attr('fill', colors.error)
      .text('归一化前');
    
    g.append('text')
      .attr('x', 100)
      .attr('y', centerY + 50)
      .attr('font-size', '12px')
      .attr('fill', colors.success)
      .text('归一化后');
  }

  // 渲染梯度下降概念图
  function renderGradientDescentConcept(g, width, height, colors, interactive) {
    // 与梯度概念图类似，但更强调下降过程
    renderGradientConcept(g, width, height, colors, interactive);
    
    // 添加下降路径示意
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制下降路径
    const pathData = [
      { x: centerX + 200, y: centerY - 100 },
      { x: centerX + 100, y: centerY - 50 },
      { x: centerX, y: centerY },
      { x: centerX - 50, y: centerY + 30 }
    ];
    
    const line = d3.line()
      .x(d => d.x)
      .y(d => d.y)
      .curve(d3.curveBasis);
    
    g.append('path')
      .datum(pathData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', colors.success)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .attr('opacity', 0.7);
    
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('梯度下降：沿着负梯度方向更新参数');
  }

  // 渲染学习率调度概念图
  function renderLRSchedulerConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制不同的学习率调度曲线
    const xScale = d3.scaleLinear().domain([0, 100]).range([100, width - 100]);
    const yScale = d3.scaleLinear().domain([0, 0.01]).range([height - 100, 100]);
    
    // StepLR
    const stepData = d3.range(0, 100, 1).map(i => ({
      x: i,
      y: 0.01 * Math.pow(0.1, Math.floor(i / 30))
    }));
    
    // CosineAnnealing
    const cosineData = d3.range(0, 100, 1).map(i => ({
      x: i,
      y: 0.001 + (0.009 - 0.001) * 0.5 * (1 + Math.cos(Math.PI * i / 100))
    }));
    
    const line = d3.line()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveBasis);
    
    g.append('path')
      .datum(stepData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('opacity', 0.8);
    
    g.append('path')
      .datum(cosineData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', colors.success)
      .attr('stroke-width', 2)
      .attr('opacity', 0.8);
    
    // 图例
    g.append('circle')
      .attr('cx', 120)
      .attr('cy', 150)
      .attr('r', 5)
      .attr('fill', colors.primary);
    
    g.append('text')
      .attr('x', 135)
      .attr('y', 155)
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .text('StepLR');
    
    g.append('circle')
      .attr('cx', 120)
      .attr('cy', 180)
      .attr('r', 5)
      .attr('fill', colors.success);
    
    g.append('text')
      .attr('x', 135)
      .attr('y', 185)
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .text('CosineAnnealing');
    
    // 坐标轴标签
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 50)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .text('Epoch');
    
    g.append('text')
      .attr('x', 50)
      .attr('y', centerY)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', colors.text)
      .attr('transform', `rotate(-90, 50, ${centerY})`)
      .text('Learning Rate');
  }

  // 渲染SAM概念图
  function renderSAMConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制损失景观（简化）
    const xScale = d3.scaleLinear().domain([-2, 2]).range([100, width - 100]);
    const yScale = d3.scaleLinear().domain([0, 5]).range([height - 100, 100]);
    
    // 尖锐区域（高损失）
    const sharpData = d3.range(-2, 2, 0.1).map(x => ({
      x: x,
      y: 2 + Math.abs(x) * 1.5
    }));
    
    // 平坦区域（低损失）
    const flatData = d3.range(-2, 2, 0.1).map(x => ({
      x: x,
      y: 1 + 0.3 * x * x
    }));
    
    const line = d3.line()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveBasis);
    
    g.append('path')
      .datum(sharpData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', colors.error)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .attr('opacity', 0.7);
    
    g.append('path')
      .datum(flatData)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', colors.success)
      .attr('stroke-width', 3);
    
    // 标记最优解
    g.append('circle')
      .attr('cx', xScale(0))
      .attr('cy', yScale(1))
      .attr('r', 8)
      .attr('fill', colors.success);
    
    // 图例
    g.append('text')
      .attr('x', 120)
      .attr('y', 150)
      .attr('font-size', '12px')
      .attr('fill', colors.error)
      .text('尖锐区域（标准SGD）');
    
    g.append('text')
      .attr('x', 120)
      .attr('y', 180)
      .attr('font-size', '12px')
      .attr('fill', colors.success)
      .text('平坦区域（SAM）');
    
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('SAM：在平坦区域寻找最优解');
  }

  // 渲染二阶优化算法概念图
  function renderSecondOrderConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 绘制优化路径对比
    const xScale = d3.scaleLinear().domain([-2, 2]).range([100, width - 100]);
    const yScale = d3.scaleLinear().domain([-2, 2]).range([height - 100, 100]);
    
    // 损失函数等高线（简化）
    for (let r = 0.5; r <= 2; r += 0.5) {
      const circle = g.append('ellipse')
        .attr('cx', centerX)
        .attr('cy', centerY)
        .attr('rx', r * 80)
        .attr('ry', r * 60)
        .attr('fill', 'none')
        .attr('stroke', colors.border)
        .attr('stroke-width', 1)
        .attr('opacity', 0.3);
    }
    
    // 一阶方法路径（SGD，曲折）
    const sgdPath = [
      { x: -1.5, y: 1.5 },
      { x: -1.0, y: 1.0 },
      { x: -0.5, y: 0.5 },
      { x: 0, y: 0 }
    ];
    
    // 二阶方法路径（牛顿法，直接）
    const newtonPath = [
      { x: -1.5, y: 1.5 },
      { x: 0, y: 0 }
    ];
    
    const line = d3.line()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .curve(d3.curveBasis);
    
    g.append('path')
      .datum(sgdPath)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', colors.primary)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5');
    
    g.append('path')
      .datum(newtonPath)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', colors.success)
      .attr('stroke-width', 3);
    
    // 图例
    g.append('text')
      .attr('x', 120)
      .attr('y', 150)
      .attr('font-size', '12px')
      .attr('fill', colors.primary)
      .text('一阶方法（SGD）');
    
    g.append('text')
      .attr('x', 120)
      .attr('y', 180)
      .attr('font-size', '12px')
      .attr('fill', colors.success)
      .text('二阶方法（牛顿法）');
    
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('二阶方法：利用Hessian矩阵信息，收敛更快');
  }

  // 渲染数据清洗概念图
  function renderDataCleaningConcept(g, width, height, colors, interactive) {
    renderFlow(g, width, height, colors, interactive);
  }

  // 渲染数据增强概念图
  function renderDataAugmentationConcept(g, width, height, colors, interactive) {
    renderFlow(g, width, height, colors, interactive);
  }

  // 渲染优化器原理概念图（更详细）
  function renderOptimizerTheoryConcept(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 显示优化器算法的详细对比
    const optimizers = [
      { name: 'SGD', x: 200, color: colors.primary, formula: 'θ = θ - η∇L' },
      { name: 'Momentum', x: centerX, color: colors.secondary, formula: 'v = βv + ∇L\nθ = θ - ηv' },
      { name: 'Adam', x: width - 200, color: colors.success, formula: 'm = β₁m + (1-β₁)∇L\nv = β₂v + (1-β₂)(∇L)²\nθ = θ - ηm̂/√v̂' }
    ];
    
    optimizers.forEach((opt, idx) => {
      // 优化器框
      g.append('rect')
        .attr('x', opt.x - 80)
        .attr('y', centerY - 60)
        .attr('width', 160)
        .attr('height', 120)
        .attr('fill', opt.color)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2)
        .attr('rx', 5)
        .attr('opacity', 0.8);
      
      // 优化器名称
      g.append('text')
        .attr('x', opt.x)
        .attr('y', centerY - 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('font-weight', 'bold')
        .attr('fill', 'white')
        .text(opt.name);
      
      // 公式（简化显示）
      const formulaLines = opt.formula.split('\n');
      formulaLines.forEach((line, i) => {
        g.append('text')
          .attr('x', opt.x)
          .attr('y', centerY + i * 20)
          .attr('text-anchor', 'middle')
          .attr('font-size', '11px')
          .attr('fill', 'white')
          .text(line);
      });
    });
    
    // 说明
    g.append('text')
      .attr('x', centerX)
      .attr('y', height - 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('优化器原理：不同优化算法的数学公式对比');
  }

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
