/**
 * 知识图谱通用逻辑
 * 提供ECharts知识图谱的初始化、交互等功能
 */

// 全局变量
let chart = null;
let isAnimationPaused = false;
let currentConfig = null;

/**
 * 初始化知识图谱
 * @param {Object} config - 配置对象
 * @param {Object} config.data - 知识图谱数据
 * @param {Object} config.techPages - 技术页面映射
 * @param {Object} config.categoryDescriptions - 分类描述
 * @param {string} config.containerId - 容器ID，默认为 'main' 或 'chart'
 */
function initKnowledgeGraph(config) {
    const containerId = config.containerId || 'main' || 'chart';
    const container = document.getElementById(containerId);
    
    if (!container) {
        console.error(`容器 ${containerId} 不存在`);
        return;
    }
    
    // 保存配置供其他函数使用
    currentConfig = config;
    
    chart = echarts.init(container);
    
    // 数据预处理，转换为ECharts Graph需要的格式
    const nodes = [];
    const links = [];
    const nameToIndex = {};
    let nodeId = 0;
    
    function traverse(node, parentIndex, level = 0) {
        const currentId = nodeId++;
        const nodeName = node.name;
        nameToIndex[nodeName] = currentId;
        
        // 支持自定义symbolSize，如果没有则根据level计算
        let symbolSize = node.symbolSize;
        if (!symbolSize) {
            symbolSize = level === 0 ? 70 : level === 1 ? 46 : 34;
        }
        
        // 支持自定义itemStyle，如果没有则根据level计算
        let itemStyle = node.itemStyle;
        if (!itemStyle) {
            if (level === 0) {
                itemStyle = { 
                    color: '#5470c6', 
                    borderColor: '#ffd700',
                    borderWidth: 4,
                    shadowBlur: 20,
                    shadowColor: 'rgba(255, 215, 0, 0.8)'
                };
            } else if (level === 1) {
                itemStyle = { 
                    color: '#06b6d4', 
                    borderColor: '#f8fafc',
                    borderWidth: 3,
                    shadowBlur: 15,
                    shadowColor: 'rgba(255, 215, 0, 0.6)'
                };
            } else {
                itemStyle = { 
                    color: level === 2 ? '#8b5cf6' : '#a855f7',
                    borderColor: '#f8fafc',
                    borderWidth: 2,
                    shadowBlur: 8,
                    shadowColor: 'rgba(0,0,0,0.3)'
                };
            }
        }
        
        // 支持自定义label，如果没有则根据level计算
        let label = node.label;
        if (!label) {
            label = {
                show: true,
                fontSize: level === 0 ? 20 : (level === 1 ? 15 : 12),
                fontWeight: level <= 1 ? 'bold' : 'normal',
                color: level <= 1 ? '#f8fafc' : '#f8fafc'
            };
        }
        
        nodes.push({
            id: currentId,
            name: nodeName,
            value: node.value || '',
            tag: node.tag || '',
            symbolSize: symbolSize,
            itemStyle: itemStyle,
            label: label,
            category: parentIndex !== null ? nodes[parentIndex].name : null,
            level: level
        });
        
        if (parentIndex !== null) {
            // 支持自定义lineStyle，如果没有则根据level计算
            let lineStyle = node.lineStyle;
            if (!lineStyle) {
                if (level === 1) {
                    lineStyle = {
                        width: 14,
                        opacity: 1.0,
                        color: '#06b6d4',
                        curveness: 0.08
                    };
                } else if (level === 2) {
                    const colors = ['#06b6d4', '#8b5cf6', '#06b6d4', '#8b5cf6'];
                    lineStyle = {
                        width: 10,
                        opacity: 1.0,
                        color: colors[(currentId - 1) % colors.length],
                        curveness: 0.12
                    };
                } else {
                    lineStyle = {
                        width: 6,
                        opacity: 0.9,
                        color: '#f8fafc',
                        curveness: 0.18
                    };
                }
            }
            
            links.push({
                source: parentIndex,
                target: currentId,
                lineStyle: lineStyle
            });
        }
        
        if (node.children) {
            node.children.forEach(child => traverse(child, currentId, level + 1));
        }
    }
    
    traverse(config.data, null);
    
    // 支持自定义option，如果没有则使用默认配置
    let option = config.option;
    if (!option) {
        option = {
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'item',
                formatter: function(params) {
                    if (params.dataType === 'node') {
                        const nodeName = params.data.name;
                        const hasPage = config.techPages && config.techPages[nodeName];
                        const hint = hasPage ? '<br/><span style="color:#94a3b8">点击加载详细页面</span>' : '';
                        return `<strong>${nodeName}</strong>${hint}`;
                    }
                    return '';
                },
                backgroundColor: 'rgba(15,23,42,0.9)',
                borderColor: 'rgba(148,163,184,0.4)',
                textStyle: {
                    color: '#f8fafc'
                }
            },
            animationDuration: 1200,
            animationDurationUpdate: 1500,
            animationEasingUpdate: 'quinticInOut',
            series: [
                {
                    type: 'graph',
                    layout: 'force',
                    force: {
                        repulsion: config.force?.repulsion || 1800,
                        gravity: config.force?.gravity || 0.15,
                        edgeLength: config.force?.edgeLength || [80, 240],
                        friction: config.force?.friction || 0.2,
                        layoutAnimation: !isAnimationPaused
                    },
                    data: nodes,
                    links: links,
                    roam: true,
                    draggable: true,
                    focusNodeAdjacency: true,
                    lineStyle: {
                        opacity: 1
                    },
                    emphasis: {
                        focus: 'adjacency',
                        scale: 1.25,
                        lineStyle: {
                            width: 20,
                            opacity: 1
                        },
                        label: {
                            fontSize: 18,
                            textShadowBlur: 6
                        },
                        itemStyle: {
                            shadowBlur: 25,
                            shadowColor: 'rgba(255,255,255,0.5)'
                        }
                    }
                }
            ]
        };
    }
    
    chart.setOption(option);
    
    // 保存option供resetView使用
    currentConfig.option = option;
    
    // 节点点击事件
    chart.on('click', function (params) {
        if (params.dataType === 'node') {
            const nodeName = params.data.name;
            
            // 检查是否有对应的HTML页面
            if (config.techPages && config.techPages[nodeName]) {
                const pagePath = config.techPages[nodeName];
                showSidebarWithPage(nodeName, pagePath);
                return;
            }
            
            // 检查是否有分类描述
            if (config.categoryDescriptions && config.categoryDescriptions[nodeName]) {
                showSidebar({
                    name: nodeName,
                    description: config.categoryDescriptions[nodeName],
                    tag: params.data.tag || '分类'
                });
                return;
            }
            
            // 使用节点的value字段作为描述
            const nodeValue = params.data.value || '';
            if (nodeValue) {
                showSidebar({
                    name: nodeName,
                    description: nodeValue,
                    tag: params.data.tag || '概览'
                });
                return;
            }
            
            // 默认显示节点信息
            showSidebar({
                name: nodeName,
                description: '该节点暂无详细描述，可点击子节点查看具体技术。',
                tag: params.data.tag || '节点'
            });
        }
    });
    
    // 窗口大小改变时重新调整
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

/**
 * 显示侧边栏（加载HTML页面）
 */
function showSidebarWithPage(nodeName, pagePath) {
    const sidebar = document.getElementById('sidebar');
    const content = document.getElementById('sidebar-content');
    const title = document.getElementById('sidebar-title');
    const tag = document.getElementById('sidebar-tag');
    
    if (!sidebar || !content) return;
    
    // 更新标题和标签
    if (title) title.textContent = nodeName;
    if (tag) tag.textContent = '技术详解';
    
    // 创建iframe加载页面
    content.innerHTML = `<iframe src="${pagePath}" style="width:100%;height:100%;border:none;"></iframe>`;
    
    // 显示侧边栏
    sidebar.classList.add('active');
}

/**
 * 显示侧边栏（显示文本内容）
 */
function showSidebar(data) {
    const sidebar = document.getElementById('sidebar');
    const content = document.getElementById('sidebar-content');
    const title = document.getElementById('sidebar-title');
    const tag = document.getElementById('sidebar-tag');
    
    if (!sidebar || !content) return;
    
    // 更新标题和标签
    if (title) title.textContent = data.name;
    if (tag) tag.textContent = data.tag || '概览';
    
    // 显示描述内容
    content.innerHTML = `
        <div style="padding: 30px;">
            <div class="desc-box">
                <p>${data.description || '该节点暂无详细描述'}</p>
            </div>
        </div>
    `;
    
    // 显示侧边栏
    sidebar.classList.add('active');
}

/**
 * 关闭侧边栏
 */
function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
        sidebar.classList.remove('active');
    }
}

/**
 * 重置视图
 */
function resetView() {
    if (chart && currentConfig && currentConfig.option) {
        chart.clear();
        // 重新初始化数据
        const nodes = [];
        const links = [];
        const nameToIndex = {};
        let nodeId = 0;
        
        function traverse(node, parentIndex, level = 0) {
            const currentId = nodeId++;
            const nodeName = node.name;
            nameToIndex[nodeName] = currentId;
            
            let symbolSize = node.symbolSize;
            if (!symbolSize) {
                symbolSize = level === 0 ? 70 : level === 1 ? 46 : 34;
            }
            
            let itemStyle = node.itemStyle;
            if (!itemStyle) {
                if (level === 0) {
                    itemStyle = { 
                        color: '#5470c6', 
                        borderColor: '#ffd700',
                        borderWidth: 4,
                        shadowBlur: 20,
                        shadowColor: 'rgba(255, 215, 0, 0.8)'
                    };
                } else if (level === 1) {
                    itemStyle = { 
                        color: '#06b6d4', 
                        borderColor: '#f8fafc',
                        borderWidth: 3,
                        shadowBlur: 15,
                        shadowColor: 'rgba(255, 215, 0, 0.6)'
                    };
                } else {
                    itemStyle = { 
                        color: level === 2 ? '#8b5cf6' : '#a855f7',
                        borderColor: '#f8fafc',
                        borderWidth: 2,
                        shadowBlur: 8,
                        shadowColor: 'rgba(0,0,0,0.3)'
                    };
                }
            }
            
            let label = node.label;
            if (!label) {
                label = {
                    show: true,
                    fontSize: level === 0 ? 20 : (level === 1 ? 15 : 12),
                    fontWeight: level <= 1 ? 'bold' : 'normal',
                    color: level <= 1 ? '#f8fafc' : '#f8fafc'
                };
            }
            
            nodes.push({
                id: currentId,
                name: nodeName,
                value: node.value || '',
                tag: node.tag || '',
                symbolSize: symbolSize,
                itemStyle: itemStyle,
                label: label,
                category: parentIndex !== null ? nodes[parentIndex].name : null,
                level: level
            });
            
            if (parentIndex !== null) {
                let lineStyle = node.lineStyle;
                if (!lineStyle) {
                    if (level === 1) {
                        lineStyle = {
                            width: 14,
                            opacity: 1.0,
                            color: '#06b6d4',
                            curveness: 0.08
                        };
                    } else if (level === 2) {
                        const colors = ['#06b6d4', '#8b5cf6', '#06b6d4', '#8b5cf6'];
                        lineStyle = {
                            width: 10,
                            opacity: 1.0,
                            color: colors[(currentId - 1) % colors.length],
                            curveness: 0.12
                        };
                    } else {
                        lineStyle = {
                            width: 6,
                            opacity: 0.9,
                            color: '#f8fafc',
                            curveness: 0.18
                        };
                    }
                }
                
                links.push({
                    source: parentIndex,
                    target: currentId,
                    lineStyle: lineStyle
                });
            }
            
            if (node.children) {
                node.children.forEach(child => traverse(child, currentId, level + 1));
            }
        }
        
        traverse(currentConfig.data, null);
        
        const newOption = Object.assign({}, currentConfig.option, {
            series: [{
                ...currentConfig.option.series[0],
                data: nodes,
                links: links
            }]
        });
        
        chart.setOption(newOption, true);
    }
}

/**
 * 切换动画
 */
function toggleAnimation() {
    isAnimationPaused = !isAnimationPaused;
    if (chart && currentConfig) {
        const currentOption = chart.getOption();
        const newForce = Object.assign({}, currentOption.series[0].force, { 
            repulsion: isAnimationPaused ? 60 : (currentConfig.force?.repulsion || 1800),
            friction: isAnimationPaused ? 0.8 : (currentConfig.force?.friction || 0.2),
            layoutAnimation: !isAnimationPaused
        });
        chart.setOption({ series: [{ force: newForce }] });
    }
}

// 导出函数和变量供外部调用
if (typeof window !== 'undefined') {
    window.initKnowledgeGraph = initKnowledgeGraph;
    window.showSidebar = showSidebar;
    window.showSidebarWithPage = showSidebarWithPage;
    window.closeSidebar = closeSidebar;
    window.resetView = resetView;
    window.toggleAnimation = toggleAnimation;
    // 导出chart变量供外部访问（使用getter/setter）
    Object.defineProperty(window, 'chart', {
        get: function() { return chart; },
        set: function(value) { chart = value; },
        configurable: true
    });
}
