import { useRef, useEffect } from 'react';
import * as echarts from 'echarts';
import { getNodeColor, getLineColor, addLevelToTree } from '../utils/colorUtils';
import { hierarchy } from '../data/hierarchy';
import { hasDetailPage } from '../data/techPages';

// 从 hierarchy 中查找节点
function findNodeInHierarchy(nodeName, node = hierarchy) {
  if (node.name === nodeName) {
    return node;
  }
  if (node.children && node.children.length > 0) {
    for (const child of node.children) {
      const found = findNodeInHierarchy(nodeName, child);
      if (found) return found;
    }
  }
  return null;
}

const KnowledgeGraph = ({ onNodeClick }) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const onNodeClickRef = useRef(onNodeClick);
  const originalTreeDataRef = useRef(null); // 保存原始树形数据

  // 保持回调函数引用最新
  useEffect(() => {
    onNodeClickRef.current = onNodeClick;
  }, [onNodeClick]);

  useEffect(() => {
    if (!chartRef.current) return;

    // 初始化图表
    chartInstance.current = echarts.init(chartRef.current);
    
    // 处理树形数据
    const treeData = addLevelToTree(JSON.parse(JSON.stringify(hierarchy)));
    originalTreeDataRef.current = JSON.parse(JSON.stringify(treeData)); // 保存原始数据

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        backgroundColor: 'rgba(15,23,42,0.92)',
        borderColor: 'rgba(148,163,184,0.4)',
        textStyle: { color: '#f8fafc' },
        formatter: function (params) {
          if (params.dataType === 'node') {
            const nodeName = params.data.name;
            const hasPage = hasDetailPage(nodeName);
            // 从原始hierarchy中查找节点，判断是否有子节点
            const fullNodeData = findNodeInHierarchy(nodeName);
            const hasChildren = fullNodeData && fullNodeData.children && fullNodeData.children.length > 0;
            let hint = hasPage 
              ? '<br/><span style="color:#60a5fa">📄 单击加载详细技术文档</span>' 
              : '<br/><span style="color:#94a3b8">单击查看节点信息</span>';
            if (hasChildren) {
              hint += '<br/><span style="color:#facc15">🖱️ 双击展开/收起子节点</span>';
            }
            return `<strong>${nodeName}</strong>${hint}`;
          }
          return '';
        }
      },
      animationDuration: 1200,
      animationEasing: function (k) {
        return k * (2 - k);
      },
      animationDelay: function (idx) {
        return Math.min(idx * 20, 600);
      },
      series: [
        {
          type: 'tree',
          data: [treeData],
          layout: 'radial',
          symbolRotate: 0,
          left: '5%',
          right: '5%',
          top: '5%',
          bottom: '5%',
          roam: true,
          symbolSize: function (value, params) {
            const data = params.data || params;
            const level = data.level !== undefined ? data.level : 0;
            return level === 0 ? 80 : level === 1 ? 55 : level === 2 ? 40 : 32;
          },
          symbol: 'circle',
          initialTreeDepth: -1,
          expandAndCollapse: false,
          symbolOffset: [0, 0],
          symbolKeepAspect: true,
          animationDuration: 1200,
          animationEasing: function (k) {
            return k * (2 - k);
          },
          animationDelay: function (idx) {
            return Math.min(idx * 25, 800);
          },
          label: {
            show: true,
            position: 'inside',
            verticalAlign: 'middle',
            align: 'center',
            color: '#f8fafc',
            distance: 0,
            rotate: 0
          },
          leaves: {
            label: {
              position: 'inside',
              verticalAlign: 'middle',
              align: 'center',
              distance: 0
            }
          },
          lineStyle: {
            color: function (params) {
              let sourceNode = null;
              if (params && params.source) {
                sourceNode = params.source;
              } else if (params && params.data && params.data.source) {
                sourceNode = params.data.source;
              }
              
              if (sourceNode) {
                const tag = sourceNode.tag || '';
                const level = sourceNode.level !== undefined ? sourceNode.level : 0;
                return getLineColor(tag, level);
              }
              
              const defaultLineColors = [
                'rgba(56,189,248,0.75)',
                'rgba(244,114,182,0.75)',
                'rgba(250,204,21,0.75)',
                'rgba(249,115,22,0.75)',
                'rgba(16,185,129,0.75)',
                'rgba(99,102,241,0.75)'
              ];
              return defaultLineColors[0];
            },
            width: 3.5,
            curveness: 0,
            opacity: 0.85,
            type: 'solid'
          },
          edgeLabel: {
            show: false
          },
          itemStyle: {
            color: function (params) {
              const data = params.data || params;
              const tag = data.tag || '';
              const level = data.level !== undefined ? data.level : 0;
              return getNodeColor(tag, level) || '#cbd5e1';
            },
            borderColor: '#f8fafc',
            borderWidth: function (params) {
              const data = params.data || params;
              const level = data.level !== undefined ? data.level : 0;
              return level === 0 ? 4 : level === 1 ? 3 : 2;
            },
            opacity: 1,
            shadowBlur: function (params) {
              const data = params.data || params;
              const level = data.level !== undefined ? data.level : 0;
              return level === 0 ? 25 : level === 1 ? 18 : level === 2 ? 12 : 8;
            },
            shadowColor: function (params) {
              const data = params.data || params;
              const tag = data.tag || '';
              const level = data.level !== undefined ? data.level : 0;
              const nodeColor = getNodeColor(tag, level);
              if (nodeColor && nodeColor.startsWith('#')) {
                const r = parseInt(nodeColor.slice(1, 3), 16);
                const g = parseInt(nodeColor.slice(3, 5), 16);
                const b = parseInt(nodeColor.slice(5, 7), 16);
                return `rgba(${r}, ${g}, ${b}, 0.6)`;
              }
              return 'rgba(148,163,184,0.6)';
            }
          },
          emphasis: {
            focus: 'ancestor',
            scale: true,
            blurScope: 'coordinateSystem',
            lineStyle: {
              width: 6,
              opacity: 1,
              curveness: 0.45,
              shadowBlur: 15
            },
            label: {
              fontSize: 18,
              textShadowBlur: 8,
              fontWeight: 'bold',
              color: '#FF4500'
            },
            itemStyle: {
              color: function (params) {
                const data = params.data || params;
                const tag = data.tag || '';
                const level = data.level !== undefined ? data.level : 0;
                const nodeColor = getNodeColor(tag, level);
                if (nodeColor && nodeColor.startsWith('#')) {
                  const r = parseInt(nodeColor.slice(1, 3), 16);
                  const g = parseInt(nodeColor.slice(3, 5), 16);
                  const b = parseInt(nodeColor.slice(5, 7), 16);
                  const brighten = (val) => Math.min(255, Math.round(val * 1.2));
                  return `rgb(${brighten(r)}, ${brighten(g)}, ${brighten(b)})`;
                }
                return nodeColor || '#cbd5e1';
              },
              opacity: 1,
              shadowBlur: 45,
              borderWidth: function (params) {
                const data = params.data || params;
                const level = data.level !== undefined ? data.level : 0;
                return level === 0 ? 5 : 3;
              },
              borderColor: '#FF4500'
            }
          }
        }
      ]
    };

    chartInstance.current.setOption(option);

    // 单击/双击检测
    let clickTimer = null;
    let lastClickTime = 0;
    const DOUBLE_CLICK_DELAY = 300; // 双击间隔时间（毫秒）
    
    // 单击事件 - 显示侧边栏或检测双击
    chartInstance.current.on('click', function (params) {
      if (params.data && params.data.name) {
        const currentTime = Date.now();
        const timeSinceLastClick = currentTime - lastClickTime;
        
        if (timeSinceLastClick < DOUBLE_CLICK_DELAY && clickTimer) {
          // 检测到双击
          clearTimeout(clickTimer);
          clickTimer = null;
          lastClickTime = 0;
          
          // 触发展开/收起
          handleNodeToggle(params);
        } else {
          // 可能是单击，设置延迟执行
          lastClickTime = currentTime;
          clickTimer = setTimeout(() => {
            // 从原始 hierarchy 中查找完整的节点信息（包括 children）
            const fullNodeData = findNodeInHierarchy(params.data.name);
            // 如果找到了完整节点，使用它；否则使用 ECharts 提供的数据
            const nodeDataToPass = fullNodeData || params.data;
            onNodeClickRef.current && onNodeClickRef.current(nodeDataToPass);
            clickTimer = null;
          }, DOUBLE_CLICK_DELAY);
        }
      }
    });
    
    // 处理节点展开/收起
    function handleNodeToggle(params) {
      const nodeName = params.data.name;
      console.log('双击节点:', nodeName); // 调试信息
      
      // 从原始数据中查找节点
      function findNodeInOriginalTree(tree, targetName) {
        if (tree.name === targetName) {
          return tree;
        }
        if (tree.children) {
          for (let i = 0; i < tree.children.length; i++) {
            const found = findNodeInOriginalTree(tree.children[i], targetName);
            if (found) return found;
          }
        }
        return null;
      }
      
      // 递归查找节点并切换展开/收起状态
      function toggleNodeCollapse(node, targetName, originalNode) {
        if (node.name === targetName) {
          // 找到目标节点
          if (node.children && node.children.length > 0) {
            // 当前是展开状态，需要收起：保存children到_children，然后清空children
            node._children = JSON.parse(JSON.stringify(node.children));
            node.children = [];
            console.log('收起节点:', nodeName); // 调试信息
            return true;
          } else {
            // 当前是收起状态，需要展开：从_children或原始数据恢复children
            if (node._children && node._children.length > 0) {
              node.children = JSON.parse(JSON.stringify(node._children));
              node._children = null;
              console.log('展开节点（从_children）:', nodeName); // 调试信息
            } else if (originalNode && originalNode.children && originalNode.children.length > 0) {
              // 从原始数据恢复
              node.children = JSON.parse(JSON.stringify(originalNode.children));
              console.log('展开节点（从原始数据）:', nodeName); // 调试信息
            }
            return node.children && node.children.length > 0;
          }
        }
        
        // 递归查找子节点
        if (node.children) {
          for (let i = 0; i < node.children.length; i++) {
            if (toggleNodeCollapse(node.children[i], targetName, originalNode)) {
              return true;
            }
          }
        }
        // 也检查_children（已收起的节点）
        if (node._children) {
          for (let i = 0; i < node._children.length; i++) {
            if (toggleNodeCollapse(node._children[i], targetName, originalNode)) {
              return true;
            }
          }
        }
        return false;
      }
      
      // 从原始数据中查找节点
      const originalNode = findNodeInOriginalTree(originalTreeDataRef.current, nodeName);
      
      // 获取当前option和数据
      const currentOption = chartInstance.current.getOption();
      const series = currentOption.series[0];
      const currentTreeData = series.data[0];
      
      // 切换节点状态
      const hasChildren = toggleNodeCollapse(currentTreeData, nodeName, originalNode);
      
      if (hasChildren || (originalNode && originalNode.children && originalNode.children.length > 0)) {
        // 深度克隆数据以确保更新
        const newTreeData = JSON.parse(JSON.stringify(currentTreeData));
        
        // 重新设置option以更新视图
        chartInstance.current.setOption({
          series: [{
            ...series,
            data: [newTreeData]
          }]
        }, { notMerge: false });
        
        console.log('图表已更新'); // 调试信息
      }
    }


    // 响应式调整
    const handleResize = () => {
      chartInstance.current?.resize();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, []); // 移除 onNodeClick 依赖，只在组件挂载时初始化一次

  // 暴露图表控制方法
  useEffect(() => {
    if (chartInstance.current) {
      window.chartControls = {
        reset: () => {
          const treeData = addLevelToTree(JSON.parse(JSON.stringify(hierarchy)));
          const option = chartInstance.current.getOption();
          option.series[0].data = [treeData];
          chartInstance.current.clear();
          chartInstance.current.setOption(option);
        },
        expand: () => {
          const option = chartInstance.current.getOption();
          option.series[0].initialTreeDepth = 10;
          chartInstance.current.setOption(option, { notMerge: false });
        },
        collapse: () => {
          const option = chartInstance.current.getOption();
          option.series[0].initialTreeDepth = 1;
          chartInstance.current.setOption(option, { notMerge: false });
        }
      };
    }
  }, []);

  return (
    <div 
      ref={chartRef} 
      style={{ 
        width: '100%', 
        height: '100%',
        minHeight: '600px'
      }} 
    />
  );
};

export default KnowledgeGraph;
