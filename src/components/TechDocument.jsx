import { useState } from 'react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { getDiagramComponent, hasDiagramComponent } from './diagrams';
import './TechDocument.css';

const TechDocument = ({ document }) => {
  const [selectedImage, setSelectedImage] = useState(null);

  if (!document) {
    return (
      <div className="tech-document-loading">
        <p>加载中...</p>
      </div>
    );
  }

  const openImageModal = (imageSrc, alt) => {
    setSelectedImage({ src: imageSrc, alt });
  };

  const closeImageModal = () => {
    setSelectedImage(null);
  };

  const renderContent = (content, parentKey = '') => {
    if (!content) return null;

    return content.map((section, index) => {
      const uniqueKey = parentKey ? `${parentKey}-${index}` : `section-${index}`;
      
      switch (section.type) {
        case 'header':
          return (
            <div key={uniqueKey} className="tech-doc-header">
              <h1>{section.title}</h1>
              {section.subtitle && <p>{section.subtitle}</p>}
            </div>
          );

        case 'section':
          return (
            <div key={uniqueKey} className="tech-doc-section">
              <h2>{section.title}</h2>
              {section.content && renderSectionContent(section.content, uniqueKey)}
            </div>
          );

        case 'desc-box':
          return (
            <div key={uniqueKey} className="tech-doc-desc-box">
              {section.content.map((item, i) => (
                <p key={`${uniqueKey}-desc-${i}`}>{item}</p>
              ))}
            </div>
          );

        case 'features':
          return (
            <div key={uniqueKey} className="tech-doc-features">
              <ul>
                {section.items.map((item, i) => (
                  <li key={`${uniqueKey}-feature-${i}`}>{item}</li>
                ))}
              </ul>
            </div>
          );

        case 'tech-box':
          return (
            <div key={uniqueKey} className="tech-doc-tech-box">
              <p><strong>{section.content}</strong></p>
            </div>
          );

        case 'app-box':
          return (
            <div key={uniqueKey} className="tech-doc-app-box">
              <p>{section.content}</p>
            </div>
          );

        case 'math-box':
          return (
            <div key={uniqueKey} className="tech-doc-math-box">
              <h3>{section.title}</h3>
              <div className="tech-doc-math-formula">
                {section.formulas.map((formula, i) => (
                  <div key={`${uniqueKey}-formula-${i}`}>
                    {formula.text && (
                      <p>
                        {formula.text.split(/(\$[^$]+\$)/g).map((part, j) => {
                          if (part.startsWith('$') && part.endsWith('$')) {
                            const math = part.slice(1, -1);
                            return <InlineMath key={`${uniqueKey}-formula-${i}-math-${j}`} math={math} />;
                          }
                          return <span key={`${uniqueKey}-formula-${i}-text-${j}`}>{part}</span>;
                        })}
                      </p>
                    )}
                    {formula.inline && !formula.text && <InlineMath math={formula.inline} />}
                    {formula.display && <BlockMath math={formula.display} />}
                  </div>
                ))}
              </div>
            </div>
          );

        case 'code-box':
          return (
            <div key={uniqueKey} className="tech-doc-code-box">
              <h3>{section.title}</h3>
              <SyntaxHighlighter
                language={section.language || 'python'}
                style={tomorrow}
                showLineNumbers={section.showLineNumbers !== false}
              >
                {section.code}
              </SyntaxHighlighter>
            </div>
          );

        case 'diagram-gallery':
          return (
            <div key={uniqueKey} className="tech-doc-diagram-gallery">
              {section.images && section.images.map((image, i) => {
                // 支持动态 SVG 组件
                if (image && image.type === 'svg-d3' && image.component) {
                  const DiagramComponent = getDiagramComponent(image.component);
                  if (DiagramComponent) {
                    return (
                      <div 
                        key={`${uniqueKey}-diagram-${i}`} 
                        className="tech-doc-diagram-item tech-doc-diagram-svg"
                        style={{ display: 'block' }}
                      >
                        {image.caption && <div className="tech-doc-diagram-title">{image.caption}</div>}
                        <DiagramComponent
                          width={image.width || 1000}
                          height={image.height || 800}
                          interactive={image.interactive !== false}
                          {...(image.props || {})}
                        />
                      </div>
                    );
                  } else {
                    // 如果组件不存在，显示错误提示
                    console.warn(`Diagram component "${image.component}" not found`);
                    return (
                      <div 
                        key={`${uniqueKey}-diagram-${i}`} 
                        className="tech-doc-diagram-item"
                        style={{ 
                          display: 'block',
                          padding: '20px',
                          textAlign: 'center',
                          color: '#999'
                        }}
                      >
                        <p>架构图解组件加载失败: {image.component}</p>
                      </div>
                    );
                  }
                }
                // 静态图片（需要 src 属性）
                if (image && image.src) {
                  return (
                    <div key={`${uniqueKey}-image-${i}`} className="tech-doc-diagram-item">
                      <img
                        src={image.src}
                        alt={image.alt}
                        onClick={() => openImageModal(image.src, image.alt)}
                        onError={(e) => {
                          e.target.parentElement.style.display = 'none';
                        }}
                      />
                      {image.caption && <div className="tech-doc-caption">{image.caption}</div>}
                    </div>
                  );
                }
                // 如果既不是 SVG 也不是静态图片，返回 null
                return null;
              })}
            </div>
          );

        case 'meta-grid':
          return (
            <div key={uniqueKey} className="tech-doc-meta-grid">
              {section.cards && section.cards.map((card, i) => (
                <div key={`${uniqueKey}-card-${i}`} className="tech-doc-meta-card">
                  <h3>{card.title}</h3>
                  <p>{card.content}</p>
                </div>
              ))}
            </div>
          );

        default:
          return null;
      }
    });
  };

  const renderSectionContent = (content, parentKey = '') => {
    if (typeof content === 'string') {
      return <p key={`${parentKey}-text`}>{content}</p>;
    }
    if (Array.isArray(content)) {
      return content.map((item, i) => {
        const itemKey = `${parentKey}-item-${i}`;
        if (typeof item === 'string') {
          return <p key={itemKey}>{item}</p>;
        }
        // 对于对象类型的 item，递归调用 renderContent
        if (item && typeof item === 'object' && item.type) {
          const rendered = renderContent([item], itemKey);
          return rendered && rendered.length > 0 ? rendered[0] : null;
        }
        return null;
      }).filter(Boolean);
    }
    return null;
  };

  return (
    <div className="tech-document">
      <div className="tech-doc-container">
        {document.title && (
          <div className="tech-doc-header">
            <h1>{document.title}</h1>
            {document.subtitle && <p>{document.subtitle}</p>}
          </div>
        )}
        {renderContent(document.content)}
      </div>

      {/* 图片放大弹窗 */}
      {selectedImage && (
        <div className="tech-doc-image-modal active" onClick={closeImageModal}>
          <span className="tech-doc-image-modal-close" onClick={(e) => {
            e.stopPropagation();
            closeImageModal();
          }}>
            &times;
          </span>
          <img
            className="tech-doc-image-modal-content"
            src={selectedImage.src}
            alt={selectedImage.alt}
            onClick={(e) => e.stopPropagation()}
          />
          <div className="tech-doc-image-modal-caption">{selectedImage.alt}</div>
        </div>
      )}
    </div>
  );
};

export default TechDocument;
