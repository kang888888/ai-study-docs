import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './GenericDiagram.css';

const YOLODiagram = ({ width = 1000, height = 800, interactive = true, type = 'architecture', title = 'YOLO架构图', ...props }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const margin = { top: 60, right: 80, bottom: 80, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const colors = { backbone: '#667eea', neck: '#764ba2', head: '#43e97b', grid: '#f093fb', text: '#2d3748' };

    if (type === 'process') {
      renderProcess(g, innerWidth, innerHeight, colors);
    } else if (type === 'iou') {
      renderIoU(g, innerWidth, innerHeight, colors);
    } else {
      renderArchitecture(g, innerWidth, innerHeight, colors, interactive);
    }

    if (title) {
      g.append('text').attr('x', innerWidth / 2).attr('y', -30).attr('text-anchor', 'middle')
        .attr('font-size', '24px').attr('font-weight', 'bold').attr('fill', colors.text).text(title);
    }
  }, [width, height, type, interactive, title]);

  function renderArchitecture(g, width, height, colors, interactive) {
    const centerX = width / 2;
    const layers = [
      { name: 'Input Image', y: 50, color: colors.backbone },
      { name: 'Backbone (CNN)', y: 150, color: colors.backbone },
      { name: 'Neck (FPN)', y: 300, color: colors.neck },
      { name: 'Detection Head', y: 450, color: colors.head },
      { name: 'Output (BBoxes)', y: 600, color: colors.grid }
    ];

    layers.forEach((layer, i) => {
      g.append('rect').attr('x', centerX - 150).attr('y', layer.y - 30).attr('width', 300).attr('height', 60)
        .attr('fill', layer.color).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 8);
      g.append('text').attr('x', centerX).attr('y', layer.y + 5).attr('text-anchor', 'middle')
        .attr('font-size', '16px').attr('fill', 'white').attr('font-weight', 'bold').text(layer.name);
      if (i < layers.length - 1) {
        g.append('path').attr('d', `M ${centerX} ${layer.y + 30} L ${centerX} ${layers[i + 1].y - 30}`)
          .attr('stroke', colors.text).attr('stroke-width', 2).attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead-yolo)');
      }
    });

    const defs = g.append('defs');
    const marker = defs.append('marker').attr('id', 'arrowhead-yolo').attr('markerWidth', 10)
      .attr('markerHeight', 10).attr('refX', 9).attr('refY', 3).attr('orient', 'auto');
    marker.append('polygon').attr('points', '0 0, 10 3, 0 6').attr('fill', colors.text);
  }

  function renderProcess(g, width, height, colors) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 输入图像
    g.append('rect').attr('x', centerX - 200).attr('y', centerY - 100).attr('width', 150).attr('height', 100)
      .attr('fill', colors.backbone).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('text').attr('x', centerX - 125).attr('y', centerY - 40).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', 'white').text('Input Image');

    // 网格划分
    g.append('rect').attr('x', centerX - 50).attr('y', centerY - 100).attr('width', 150).attr('height', 100)
      .attr('fill', colors.grid).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        g.append('rect').attr('x', centerX - 50 + i * 50).attr('y', centerY - 100 + j * 33)
          .attr('width', 48).attr('height', 31).attr('fill', 'none').attr('stroke', colors.text)
          .attr('stroke-width', 1);
      }
    }
    g.append('text').attr('x', centerX + 25).attr('y', centerY - 40).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', 'white').text('Grid 7×7');

    // 预测框
    g.append('rect').attr('x', centerX + 150).attr('y', centerY - 100).attr('width', 150).attr('height', 100)
      .attr('fill', colors.head).attr('stroke', colors.text).attr('stroke-width', 2).attr('rx', 5);
    g.append('rect').attr('x', centerX + 170).attr('y', centerY - 80).attr('width', 60).attr('height', 40)
      .attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 2);
    g.append('text').attr('x', centerX + 225).attr('y', centerY - 40).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', 'white').text('BBoxes');

    const defs = g.append('defs');
    const marker = defs.append('marker').attr('id', 'arrowhead-process').attr('markerWidth', 10)
      .attr('markerHeight', 10).attr('refX', 9).attr('refY', 3).attr('orient', 'auto');
    marker.append('polygon').attr('points', '0 0, 10 3, 0 6').attr('fill', colors.text);

    g.append('path').attr('d', `M ${centerX - 50} ${centerY - 50} L ${centerX - 50} ${centerY - 50}`)
      .attr('stroke', colors.text).attr('stroke-width', 2).attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-process)');
    g.append('path').attr('d', `M ${centerX + 100} ${centerY - 50} L ${centerX + 150} ${centerY - 50}`)
      .attr('stroke', colors.text).attr('stroke-width', 2).attr('fill', 'none')
      .attr('marker-end', 'url(#arrowhead-process)');
  }

  function renderIoU(g, width, height, colors) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 真实框
    g.append('rect').attr('x', centerX - 150).attr('y', centerY - 100).attr('width', 100).attr('height', 80)
      .attr('fill', 'none').attr('stroke', colors.head).attr('stroke-width', 3).attr('rx', 5);
    g.append('text').attr('x', centerX - 100).attr('y', centerY - 110).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', colors.text).text('Ground Truth');

    // 预测框
    g.append('rect').attr('x', centerX - 50).attr('y', centerY - 80).attr('width', 100).attr('height', 80)
      .attr('fill', 'none').attr('stroke', colors.backbone).attr('stroke-width', 3).attr('rx', 5);
    g.append('text').attr('x', centerX).attr('y', centerY - 90).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', colors.text).text('Prediction');

    // 交集
    g.append('rect').attr('x', centerX - 50).attr('y', centerY - 80).attr('width', 50).attr('height', 60)
      .attr('fill', colors.neck).attr('opacity', 0.5).attr('rx', 5);
    g.append('text').attr('x', centerX - 25).attr('y', centerY - 45).attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('fill', 'white').text('Intersection');

    g.append('text').attr('x', centerX).attr('y', height - 40).attr('text-anchor', 'middle')
      .attr('font-size', '14px').attr('fill', colors.text)
      .text('IoU = Area(Intersection) / Area(Union)');
  }

  return (
    <div className="generic-diagram-container">
      <svg ref={svgRef} width={width} height={height}
        style={{ border: '1px solid #e2e8f0', borderRadius: '8px', backgroundColor: '#ffffff' }} />
    </div>
  );
};

export default YOLODiagram;
