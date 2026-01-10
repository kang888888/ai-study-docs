/**
 * 技术文档通用逻辑
 * 提供图片放大、代码复制等功能
 */

// 图片放大功能
function openImageModal(img) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    
    if (!modal || !modalImg) {
        // 如果模态框不存在，创建它
        createImageModal();
        return openImageModal(img);
    }
    
    modal.classList.add('active');
    modalImg.src = img.src;
    modalImg.alt = img.alt;
    
    if (modalCaption) {
        modalCaption.textContent = img.alt || '';
    }
    
    // 阻止背景滚动
    document.body.style.overflow = 'hidden';
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

function createImageModal() {
    const modal = document.createElement('div');
    modal.id = 'imageModal';
    modal.className = 'image-modal';
    
    const closeBtn = document.createElement('span');
    closeBtn.className = 'image-modal-close';
    closeBtn.innerHTML = '&times;';
    closeBtn.onclick = closeImageModal;
    
    const img = document.createElement('img');
    img.id = 'modalImage';
    img.className = 'image-modal-content';
    
    const caption = document.createElement('div');
    caption.id = 'modalCaption';
    caption.className = 'image-modal-caption';
    
    modal.appendChild(closeBtn);
    modal.appendChild(img);
    modal.appendChild(caption);
    
    // 点击背景关闭
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeImageModal();
        }
    });
    
    document.body.appendChild(modal);
}

// 代码复制功能
function initCodeCopy() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach((codeBlock) => {
        const pre = codeBlock.parentElement;
        if (pre.querySelector('.copy-btn')) return; // 已添加过按钮
        
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.textContent = '复制';
        copyBtn.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        `;
        
        copyBtn.onmouseover = () => {
            copyBtn.style.background = 'rgba(255, 255, 255, 0.3)';
        };
        copyBtn.onmouseout = () => {
            copyBtn.style.background = 'rgba(255, 255, 255, 0.2)';
        };
        
        copyBtn.onclick = async () => {
            try {
                await navigator.clipboard.writeText(codeBlock.textContent);
                copyBtn.textContent = '已复制!';
                setTimeout(() => {
                    copyBtn.textContent = '复制';
                }, 2000);
            } catch (err) {
                console.error('复制失败:', err);
                copyBtn.textContent = '复制失败';
            }
        };
        
        pre.style.position = 'relative';
        pre.appendChild(copyBtn);
    });
}

// 目录导航（可选）
function initTableOfContents() {
    const sections = document.querySelectorAll('.section h2');
    if (sections.length === 0) return;
    
    const toc = document.createElement('div');
    toc.className = 'table-of-contents';
    toc.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 20px;
        max-width: 250px;
        max-height: 70vh;
        overflow-y: auto;
        z-index: 100;
    `;
    
    const tocTitle = document.createElement('h3');
    tocTitle.textContent = '目录';
    tocTitle.style.cssText = 'margin: 0 0 15px 0; color: #f8fafc; font-size: 16px;';
    toc.appendChild(tocTitle);
    
    const tocList = document.createElement('ul');
    tocList.style.cssText = 'list-style: none; padding: 0; margin: 0;';
    
    sections.forEach((section, index) => {
        const id = `section-${index}`;
        section.id = id;
        
        const li = document.createElement('li');
        li.style.cssText = 'margin-bottom: 8px;';
        
        const a = document.createElement('a');
        a.href = `#${id}`;
        a.textContent = section.textContent;
        a.style.cssText = 'color: #cbd5e1; text-decoration: none; font-size: 14px; transition: color 0.2s;';
        
        a.onmouseover = () => {
            a.style.color = '#60a5fa';
        };
        a.onmouseout = () => {
            a.style.color = '#cbd5e1';
        };
        
        li.appendChild(a);
        tocList.appendChild(li);
    });
    
    toc.appendChild(tocList);
    document.body.appendChild(toc);
}

// 初始化所有功能
document.addEventListener('DOMContentLoaded', function() {
    // 为所有图片添加点击事件
    document.querySelectorAll('img').forEach(img => {
        if (img.parentElement.classList.contains('diagram-item') || 
            img.closest('.diagram-gallery')) {
            img.onclick = () => openImageModal(img);
            img.style.cursor = 'pointer';
        }
    });
    
    // 初始化代码复制
    initCodeCopy();
    
    // 可选：初始化目录导航
    // initTableOfContents();
});

// 导出函数供外部调用
if (typeof window !== 'undefined') {
    window.openImageModal = openImageModal;
    window.closeImageModal = closeImageModal;
}
