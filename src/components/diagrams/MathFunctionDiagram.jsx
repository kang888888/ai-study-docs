import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

/**
 * 数学函数可视化组件
 * 展示各种数学函数的图像和数学表达式
 */
const MathFunctionDiagram = ({ 
  width = 1000, 
  height = 800, 
  interactive = true,
  type = 'activation', // activation, loss, normalization, distance
  title = '数学函数图',
  function: highlightFunction = null, // 要突出显示的函数名称（如 'relu', 'sigmoid'等）
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
      relu: '#4facfe',
      sigmoid: '#667eea',
      tanh: '#f093fb',
      gelu: '#43e97b',
      swish: '#764ba2',
      leaky: '#fa709a',
      elu: '#f093fb',
      mish: '#43e97b',
      mse: '#4facfe',
      bce: '#667eea',
      huber: '#f093fb',
      text: '#2d3748',
      grid: '#e2e8f0'
    };

    switch (type) {
      case 'loss':
        renderLossFunctions(g, innerWidth, innerHeight, colors, interactive, highlightFunction);
        break;
      case 'normalization':
        renderNormalizationFunctions(g, innerWidth, innerHeight, colors, interactive, highlightFunction);
        break;
      case 'distance':
        renderDistanceFunctions(g, innerWidth, innerHeight, colors, interactive);
        break;
      case 'activation':
      default:
        renderActivationFunctions(g, innerWidth, innerHeight, colors, interactive, highlightFunction);
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
  }, [width, height, type, interactive, title, highlightFunction]);

  // 渲染激活函数
  function renderActivationFunctions(g, width, height, colors, interactive, highlightFunction) {
    const xRange = [-5, 5];
    const yRange = [-2, 2];
    const numPoints = 200;
    
    // 创建x值数组
    const xScale = d3.scaleLinear()
      .domain(xRange)
      .range([0, width]);
    
    const yScale = d3.scaleLinear()
      .domain(yRange)
      .range([height, 0]);

    // 绘制坐标轴
    drawAxes(g, width, height, xScale, yScale, colors);

    // 函数名称映射（将小写key映射到显示名称）
    const functionNameMap = {
      'relu': 'ReLU',
      'sigmoid': 'Sigmoid',
      'tanh': 'Tanh',
      'gelu': 'GELU',
      'swish': 'Swish',
      'leaky': 'Leaky ReLU',
      'leakyrelu': 'Leaky ReLU',
      'elu': 'ELU',
      'mish': 'Mish'
    };

    // 定义函数
    const functions = [
      { key: 'relu', name: 'ReLU', color: colors.relu, func: x => Math.max(0, x) },
      { key: 'sigmoid', name: 'Sigmoid', color: colors.sigmoid, func: x => 1 / (1 + Math.exp(-x)) },
      { key: 'tanh', name: 'Tanh', color: colors.tanh, func: x => Math.tanh(x) },
      { key: 'gelu', name: 'GELU', color: colors.gelu, func: x => {
        const c = Math.sqrt(2 / Math.PI);
        return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
      }},
      { key: 'swish', name: 'Swish', color: colors.swish, func: x => x / (1 + Math.exp(-x)) },
      { key: 'leaky', name: 'Leaky ReLU', color: colors.leaky, func: x => x > 0 ? x : 0.01 * x },
      { key: 'elu', name: 'ELU', color: colors.elu, func: x => x > 0 ? x : 1.0 * (Math.exp(x) - 1) },
      { key: 'mish', name: 'Mish', color: colors.mish, func: x => {
        const softplus = Math.log(1 + Math.exp(x));
        return x * Math.tanh(softplus);
      }}
    ];

    // 确定要突出显示的函数key
    const highlightKey = highlightFunction ? highlightFunction.toLowerCase() : null;

    // 如果指定了要显示的函数，只显示该函数；否则显示所有函数
    const functionsToRender = highlightKey 
      ? functions.filter(funcObj => funcObj.key === highlightKey || funcObj.name.toLowerCase() === highlightKey)
      : functions;

    // 添加 y=1 和 y=-1 的虚线（仅当有指定函数时）
    if (highlightKey) {
      // y=1 虚线
      if (yRange[0] <= 1 && yRange[1] >= 1) {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', width)
          .attr('y1', yScale(1))
          .attr('y2', yScale(1))
          .attr('stroke', '#666666')
          .attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.7);

        g.append('text')
          .attr('x', width - 10)
          .attr('y', yScale(1) - 5)
          .attr('text-anchor', 'end')
          .attr('font-size', '12px')
          .attr('fill', '#666666')
          .text('y = 1');
      }

      // y=-1 虚线
      if (yRange[0] <= -1 && yRange[1] >= -1) {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', width)
          .attr('y1', yScale(-1))
          .attr('y2', yScale(-1))
          .attr('stroke', '#666666')
          .attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.7);

        g.append('text')
          .attr('x', width - 10)
          .attr('y', yScale(-1) + 15)
          .attr('text-anchor', 'end')
          .attr('font-size', '12px')
          .attr('fill', '#666666')
          .text('y = -1');
      }
    }

    // 绘制函数曲线
    functionsToRender.forEach((funcObj, idx) => {
      const data = [];
      for (let i = 0; i <= numPoints; i++) {
        const x = xRange[0] + (xRange[1] - xRange[0]) * i / numPoints;
        const y = funcObj.func(x);
        data.push([x, y]);
      }

      const line = d3.line()
        .x(d => xScale(d[0]))
        .y(d => yScale(d[1]))
        .curve(d3.curveMonotoneX);

      // 如果只显示一个函数，使用黑色粗线
      const strokeWidth = highlightKey ? 5 : 2.5;
      const strokeColor = highlightKey ? '#000000' : funcObj.color;
      const opacity = 1.0;

      g.append('path')
        .datum(data)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', strokeColor)
        .attr('stroke-width', strokeWidth)
        .attr('opacity', opacity);

      // 添加图例（仅当显示所有函数时）
      if (!highlightKey) {
        const legendY = 20 + idx * 25;
        g.append('rect')
          .attr('x', width - 150)
          .attr('y', legendY - 8)
          .attr('width', 15)
          .attr('height', 3)
          .attr('fill', funcObj.color);

        g.append('text')
          .attr('x', width - 130)
          .attr('y', legendY)
          .attr('font-size', '12px')
          .attr('fill', colors.text)
          .text(funcObj.name);
      }
    });

    // 添加数学公式说明
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 50)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text('激活函数图像：ReLU=max(0,x), Sigmoid=1/(1+e^(-x)), Tanh=(e^x-e^(-x))/(e^x+e^(-x))');
  }

  // 渲染损失函数
  function renderLossFunctions(g, width, height, colors, interactive, highlightFunction) {
    const yTrue = 0.5;
    const yPredRange = [0, 1];
    const numPoints = 200;
    
    const xScale = d3.scaleLinear()
      .domain(yPredRange)
      .range([0, width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    // 绘制坐标轴
    drawAxes(g, width, height, xScale, yScale, colors, 'Predicted Value', 'Loss');

    // 定义损失函数
    const functions = [
      { 
        key: 'mse',
        name: 'MSE', 
        color: colors.mse, 
        func: yPred => Math.pow(yTrue - yPred, 2)
      },
      { 
        key: 'bce',
        name: 'BCE', 
        color: colors.bce, 
        func: yPred => {
          const eps = 1e-8;
          return -(yTrue * Math.log(yPred + eps) + (1 - yTrue) * Math.log(1 - yPred + eps));
        }
      },
      { 
        key: 'huber',
        name: 'Huber (δ=0.5)', 
        color: colors.huber, 
        func: yPred => {
          const delta = 0.5;
          const error = Math.abs(yTrue - yPred);
          return error <= delta 
            ? 0.5 * error * error 
            : delta * error - 0.5 * delta * delta;
        }
      }
    ];

    // 确定要突出显示的函数key
    const highlightKey = highlightFunction ? highlightFunction.toLowerCase() : null;

    // 如果指定了要显示的函数，只显示该函数；否则显示所有函数
    const functionsToRender = highlightKey 
      ? functions.filter(funcObj => funcObj.key === highlightKey || funcObj.name.toLowerCase() === highlightKey)
      : functions;

    // 添加 y=1 和 y=-1 的虚线（仅当有指定函数时）
    if (highlightKey) {
      // y=1 虚线
      const yMax = 1;
      if (yScale.domain()[0] <= yMax && yScale.domain()[1] >= yMax) {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', width)
          .attr('y1', yScale(yMax))
          .attr('y2', yScale(yMax))
          .attr('stroke', '#666666')
          .attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.7);

        g.append('text')
          .attr('x', width - 10)
          .attr('y', yScale(yMax) - 5)
          .attr('text-anchor', 'end')
          .attr('font-size', '12px')
          .attr('fill', '#666666')
          .text('y = 1');
      }

      // y=-1 虚线（损失函数通常不包含-1，但为了一致性也检查）
      const yMin = -1;
      if (yScale.domain()[0] <= yMin && yScale.domain()[1] >= yMin) {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', width)
          .attr('y1', yScale(yMin))
          .attr('y2', yScale(yMin))
          .attr('stroke', '#666666')
          .attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.7);

        g.append('text')
          .attr('x', width - 10)
          .attr('y', yScale(yMin) + 15)
          .attr('text-anchor', 'end')
          .attr('font-size', '12px')
          .attr('fill', '#666666')
          .text('y = -1');
      }
    }

    // 绘制函数曲线
    functionsToRender.forEach((funcObj, idx) => {
      const data = [];
      for (let i = 0; i <= numPoints; i++) {
        const yPred = yPredRange[0] + (yPredRange[1] - yPredRange[0]) * i / numPoints;
        const loss = funcObj.func(yPred);
        data.push([yPred, loss]);
      }

      const line = d3.line()
        .x(d => xScale(d[0]))
        .y(d => yScale(d[1]))
        .curve(d3.curveMonotoneX);

      // 如果只显示一个函数，使用黑色粗线
      const strokeWidth = highlightKey ? 5 : 2.5;
      const strokeColor = highlightKey ? '#000000' : funcObj.color;
      const opacity = 1.0;

      g.append('path')
        .datum(data)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', strokeColor)
        .attr('stroke-width', strokeWidth)
        .attr('opacity', opacity);

      // 图例（仅当显示所有函数时）
      if (!highlightKey) {
        const legendY = 20 + idx * 25;
        g.append('rect')
          .attr('x', width - 150)
          .attr('y', legendY - 8)
          .attr('width', 15)
          .attr('height', 3)
          .attr('fill', funcObj.color);

        g.append('text')
          .attr('x', width - 130)
          .attr('y', legendY)
          .attr('font-size', '12px')
          .attr('fill', colors.text)
          .text(funcObj.name);
      }
    });

    // 标记真实值
    g.append('line')
      .attr('x1', xScale(yTrue))
      .attr('x2', xScale(yTrue))
      .attr('y1', 0)
      .attr('y2', height)
      .attr('stroke', '#e53e3e')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .attr('opacity', 0.7);

    g.append('text')
      .attr('x', xScale(yTrue) + 5)
      .attr('y', 15)
      .attr('font-size', '12px')
      .attr('fill', '#e53e3e')
      .text(`True: ${yTrue}`);
  }

  // 渲染归一化函数
  function renderNormalizationFunctions(g, width, height, colors, interactive, highlightFunction) {
    const highlightKey = highlightFunction ? highlightFunction.toLowerCase() : null;
    
    // 如果指定了Softmax，显示函数图像；否则显示示例
    if (highlightKey === 'softmax') {
      // 显示Softmax函数图像
      const xRange = [-5, 5];
      const yRange = [0, 1];
      const numPoints = 200;
      
      const xScale = d3.scaleLinear()
        .domain(xRange)
        .range([0, width]);
      
      const yScale = d3.scaleLinear()
        .domain(yRange)
        .range([height, 0]);

      // 绘制坐标轴
      drawAxes(g, width, height, xScale, yScale, colors, 'x', 'softmax(x)');

      // 添加 y=1 和 y=-1 的虚线
      // y=1 虚线
      if (yRange[0] <= 1 && yRange[1] >= 1) {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', width)
          .attr('y1', yScale(1))
          .attr('y2', yScale(1))
          .attr('stroke', '#666666')
          .attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.7);

        g.append('text')
          .attr('x', width - 10)
          .attr('y', yScale(1) - 5)
          .attr('text-anchor', 'end')
          .attr('font-size', '12px')
          .attr('fill', '#666666')
          .text('y = 1');
      }

      // y=-1 虚线
      if (yRange[0] <= -1 && yRange[1] >= -1) {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', width)
          .attr('y1', yScale(-1))
          .attr('y2', yScale(-1))
          .attr('stroke', '#666666')
          .attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.7);

        g.append('text')
          .attr('x', width - 10)
          .attr('y', yScale(-1) + 15)
          .attr('text-anchor', 'end')
          .attr('font-size', '12px')
          .attr('fill', '#666666')
          .text('y = -1');
      }

      // Softmax函数图像展示
      // 假设有3个输入：[x, 0, 0]，计算第一个输入的softmax值
      // softmax(x) = e^x / (e^x + e^0 + e^0) = e^x / (e^x + 2)
      // 这样展示当第一个输入变化时，其softmax概率的变化
      const softmaxFunc = x => {
        const expX = Math.exp(x);
        const sumExp = expX + 2; // e^0 + e^0 = 1 + 1 = 2
        return expX / sumExp;
      };

      const data = [];
      for (let i = 0; i <= numPoints; i++) {
        const x = xRange[0] + (xRange[1] - xRange[0]) * i / numPoints;
        const y = softmaxFunc(x);
        data.push([x, y]);
      }

      const line = d3.line()
        .x(d => xScale(d[0]))
        .y(d => yScale(d[1]))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(data)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', '#000000')
        .attr('stroke-width', 5)
        .attr('opacity', 1.0);

      // 添加说明
      g.append('text')
        .attr('x', width / 2)
        .attr('y', height + 50)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', colors.text)
        .text('Softmax函数：softmax(x_i) = e^(x_i) / Σ e^(x_j)，输出范围(0,1)');
    } else {
      // 默认显示示例（条形图）
      const centerX = width / 2;
      const centerY = height / 2;

      // Softmax示例
      g.append('text')
        .attr('x', centerX)
        .attr('y', centerY - 100)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('fill', colors.text)
        .attr('font-weight', 'bold')
        .text('Softmax函数');

      g.append('text')
        .attr('x', centerX)
        .attr('y', centerY - 60)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('fill', colors.text)
        .text('softmax(x_i) = e^(x_i) / Σ e^(x_j)');

      // 可视化Softmax输出（条形图）
      const logits = [2.0, 1.0, 0.1];
      const softmaxProbs = logits.map(x => Math.exp(x) / logits.reduce((sum, y) => sum + Math.exp(y), 0));
      
      const barWidth = 80;
      const barSpacing = 20;
      const maxBarHeight = 200;
      const startX = centerX - (logits.length * (barWidth + barSpacing)) / 2;

      logits.forEach((logit, i) => {
        const x = startX + i * (barWidth + barSpacing);
        const barHeight = softmaxProbs[i] * maxBarHeight;

        g.append('rect')
          .attr('x', x)
          .attr('y', centerY - barHeight)
          .attr('width', barWidth)
          .attr('height', barHeight)
          .attr('fill', colors.gelu)
          .attr('stroke', colors.text)
          .attr('stroke-width', 1);

        g.append('text')
          .attr('x', x + barWidth / 2)
          .attr('y', centerY - barHeight - 10)
          .attr('text-anchor', 'middle')
          .attr('font-size', '12px')
          .attr('fill', colors.text)
          .text(softmaxProbs[i].toFixed(3));

        g.append('text')
          .attr('x', x + barWidth / 2)
          .attr('y', centerY + 20)
          .attr('text-anchor', 'middle')
          .attr('font-size', '12px')
          .attr('fill', colors.text)
          .text(`logit: ${logit}`);
      });

      g.append('text')
        .attr('x', centerX)
        .attr('y', centerY + 60)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', colors.text)
        .text(`总和: ${softmaxProbs.reduce((a, b) => a + b, 0).toFixed(3)}`);
    }
  }

  // 渲染距离函数
  function renderDistanceFunctions(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const centerY = height / 2;

    // 定义两个点
    const point1 = { x: centerX - 150, y: centerY - 100, label: 'P1' };
    const point2 = { x: centerX + 150, y: centerY + 100, label: 'P2' };

    // 绘制点
    [point1, point2].forEach(point => {
      g.append('circle')
        .attr('cx', point.x)
        .attr('cy', point.y)
        .attr('r', 8)
        .attr('fill', colors.relu)
        .attr('stroke', colors.text)
        .attr('stroke-width', 2);

      g.append('text')
        .attr('x', point.x)
        .attr('y', point.y - 15)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', colors.text)
        .attr('font-weight', 'bold')
        .text(point.label);
    });

    // 欧氏距离（直线）
    g.append('line')
      .attr('x1', point1.x)
      .attr('y1', point1.y)
      .attr('x2', point2.x)
      .attr('y2', point2.y)
      .attr('stroke', colors.mse)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5');

    const euclideanDist = Math.sqrt(
      Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2)
    );

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.mse)
      .text(`欧氏距离: ${euclideanDist.toFixed(1)}`);

    // 曼哈顿距离（L形路径）
    g.append('path')
      .attr('d', `M ${point1.x} ${point1.y} L ${point2.x} ${point1.y} L ${point2.x} ${point2.y}`)
      .attr('stroke', colors.bce)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('stroke-dasharray', '3,3')
      .attr('opacity', 0.7);

    const manhattanDist = Math.abs(point2.x - point1.x) + Math.abs(point2.y - point1.y);
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.bce)
      .text(`曼哈顿距离: ${manhattanDist.toFixed(1)}`);

    // 余弦相似度
    const vec1 = { x: point1.x - centerX, y: point1.y - centerY };
    const vec2 = { x: point2.x - centerX, y: point2.y - centerY };
    const dot = vec1.x * vec2.x + vec1.y * vec2.y;
    const norm1 = Math.sqrt(vec1.x * vec1.x + vec1.y * vec1.y);
    const norm2 = Math.sqrt(vec2.x * vec2.x + vec2.y * vec2.y);
    const cosine = dot / (norm1 * norm2);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 60)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.tanh)
      .text(`余弦相似度: ${cosine.toFixed(3)}`);
  }

  // 绘制坐标轴辅助函数
  function drawAxes(g, width, height, xScale, yScale, colors, xLabel = 'x', yLabel = 'f(x)') {
    // 网格线
    const xTicks = xScale.ticks(10);
    const yTicks = yScale.ticks(10);

    xTicks.forEach(tick => {
      g.append('line')
        .attr('x1', xScale(tick))
        .attr('x2', xScale(tick))
        .attr('y1', 0)
        .attr('y2', height)
        .attr('stroke', colors.grid)
        .attr('stroke-width', 0.5)
        .attr('opacity', 0.5);
    });

    yTicks.forEach(tick => {
      g.append('line')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', yScale(tick))
        .attr('y2', yScale(tick))
        .attr('stroke', colors.grid)
        .attr('stroke-width', 0.5)
        .attr('opacity', 0.5);
    });

    // 坐标轴
    g.append('line')
      .attr('x1', 0)
      .attr('x2', width)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', colors.text)
      .attr('stroke-width', 2);

    g.append('line')
      .attr('x1', xScale(0))
      .attr('x2', xScale(0))
      .attr('y1', 0)
      .attr('y2', height)
      .attr('stroke', colors.text)
      .attr('stroke-width', 2);

    // 坐标轴标签
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .text(xLabel);

    g.append('text')
      .attr('x', -40)
      .attr('y', height / 2)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', colors.text)
      .attr('transform', `rotate(-90, -40, ${height / 2})`)
      .text(yLabel);
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

export default MathFunctionDiagram;
