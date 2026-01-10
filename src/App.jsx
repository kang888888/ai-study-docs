import { useState, useCallback, useEffect, useRef } from 'react';
import KnowledgeGraph from './components/KnowledgeGraph';
import Sidebar from './components/Sidebar';
import ControlBar from './components/ControlBar';
import TitleBar from './components/TitleBar';
import './styles/index.css';

function App() {
  const [sidebarActive, setSidebarActive] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const appRef = useRef(null);
  const sidebarRef = useRef(null);

  const handleNodeClick = useCallback((nodeData) => {
    setSelectedNode(nodeData);
    setSidebarActive(true);
  }, []);

  const handleCloseSidebar = () => {
    setSidebarActive(false);
  };

  // 点击空白处关闭侧边栏
  useEffect(() => {
    const handleClickOutside = (event) => {
      // 如果侧边栏未激活，不需要处理
      if (!sidebarActive) return;

      // 检查点击是否在侧边栏内部
      if (sidebarRef.current && sidebarRef.current.contains(event.target)) {
        return;
      }

      // 检查点击是否在控制栏或标题栏上（这些区域不应该关闭侧边栏）
      const controlBar = document.querySelector('.control-bar');
      const titleBar = document.querySelector('.title-bar');
      if (
        (controlBar && controlBar.contains(event.target)) ||
        (titleBar && titleBar.contains(event.target))
      ) {
        return;
      }

      // 点击空白处，关闭侧边栏
      setSidebarActive(false);
    };

    // 添加点击事件监听
    document.addEventListener('mousedown', handleClickOutside);

    return () => {
      // 清理事件监听
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [sidebarActive]);

  return (
    <div className="app" ref={appRef}>
      <div id="chart">
        <KnowledgeGraph onNodeClick={handleNodeClick} />
      </div>
      
      <TitleBar />
      <ControlBar />
      
      <Sidebar 
        ref={sidebarRef}
        isActive={sidebarActive}
        nodeData={selectedNode}
        onClose={handleCloseSidebar}
      />
    </div>
  );
}

export default App;
