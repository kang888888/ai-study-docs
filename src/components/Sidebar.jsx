import { useState, useEffect, forwardRef } from 'react';
import { hasDetailPage, getDetailPagePath } from '../data/techPages';
import { hasTechDocument, getTechDocument } from '../data/techDocuments';
import { hasKnowledgeDocument, getKnowledgeDocument } from '../data/knowledge/knowledgeMap';
import TechDocument from './TechDocument';

const Sidebar = forwardRef(({ isActive, nodeData, onClose }, ref) => {
  const [content, setContent] = useState(null);
  const [showIframe, setShowIframe] = useState(false);
  const [iframeSrc, setIframeSrc] = useState('');

  useEffect(() => {
    if (nodeData) {
      // 优先级1: 检查是否有React技术文档（已手动重构的）
      const hasReactDoc = hasTechDocument(nodeData.name);
      
      if (hasReactDoc) {
        // 使用React组件展示
        setShowIframe(false);
        setContent({
          title: nodeData.name,
          tag: '技术详解',
          description: nodeData.value || '',
          document: getTechDocument(nodeData.name)
        });
        return;
      }

      // 优先级2: 检查是否有提取的知识文档（从HTML提取的）
      const hasKnowledgeDoc = hasKnowledgeDocument(nodeData.name);
      
      if (hasKnowledgeDoc) {
        const knowledgeDoc = getKnowledgeDocument(nodeData.name);
        setShowIframe(false);
        setContent({
          title: knowledgeDoc.title || nodeData.name,
          tag: '技术详解',
          description: knowledgeDoc.subtitle || nodeData.value || '',
          document: knowledgeDoc
        });
        return;
      }

      // 优先级3: 检查是否有HTML页面（iframe方式，兼容旧方式）
      // 注意：以下情况不显示iframe，而是显示节点信息和子节点列表：
      // 1. 以"总览"结尾的节点
      // 2. 有子节点的父节点（如"神经网络架构"、"应用开发与框架"等）
      const isOverviewNode = nodeData.name.endsWith('总览');
      const hasChildren = nodeData.children && nodeData.children.length > 0;
      const hasPage = hasDetailPage(nodeData.name);
      
      if (hasPage && !isOverviewNode && !hasChildren) {
        const pagePath = getDetailPagePath(nodeData.name);
        setShowIframe(true);
        setIframeSrc(pagePath);
        setContent({
          title: nodeData.name,
          tag: '技术详解',
          description: nodeData.value || ''
        });
        return;
      }

      // 默认: 显示节点信息和子节点列表（包括总览节点）
      setShowIframe(false);
      setContent({
        title: nodeData.name,
        tag: nodeData.tag || '概览',
        description: nodeData.value || '该节点暂无额外描述。'
      });
    }
  }, [nodeData]);

  if (!content) {
    return null;
  }

  return (
    <div ref={ref} className={`sidebar ${isActive ? 'active' : ''}`}>
      <button className="close-btn" onClick={onClose}>
        关闭
      </button>
      <div className="sidebar-header">
        <h2 className="sidebar-title">{content.title}</h2>
        <div className="sidebar-tag">{content.tag}</div>
      </div>
      <div className="sidebar-body">
        {content.document ? (
          // 使用React组件展示技术文档
          <TechDocument document={content.document} />
        ) : showIframe ? (
          // 使用iframe加载HTML页面
          <iframe
            src={iframeSrc}
            style={{
              width: '100%',
              height: '100%',
              border: 'none',
              display: 'block'
            }}
            title={content.title}
            frameBorder="0"
            scrolling="auto"
            allowFullScreen
          />
        ) : (
          // 显示节点信息和子节点列表
          <div className="sidebar-placeholder">
            <p>{content.description}</p>
            {nodeData.children && nodeData.children.length > 0 && (
              <>
                <h3 style={{ marginTop: '24px', color: '#f8fafc' }}>子节点</h3>
                <ul style={{ listStyle: 'none', padding: 0 }}>
                  {nodeData.children.map((child, index) => (
                    <li 
                      key={index} 
                      style={{ 
                        padding: '8px 12px', 
                        margin: '8px 0',
                        background: 'rgba(59,130,246,0.1)',
                        borderRadius: '8px',
                        borderLeft: '3px solid #3b82f6'
                      }}
                    >
                      <strong style={{ color: '#60a5fa' }}>{child.name}</strong>
                      <br />
                      <span style={{ fontSize: '14px', color: '#cbd5e1' }}>
                        {child.value}
                      </span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

Sidebar.displayName = 'Sidebar';

export default Sidebar;
