const TitleBar = () => {
  return (
    <div className="title-bar">
      <h1>AI大模型学习 · 知识图谱总览</h1>
      <p>
        一站式查看各模块知识图谱。点击节点即可查看详细信息，快速定位微调、量化、RAG、评估、安全、部署到算力基础等内容。
      </p>
      <div className="legend">
        <span>
          <i style={{ background: '#38bdf8' }}></i>基础与数据
        </span>
        <span>
          <i style={{ background: '#f472b6' }}></i>核心技术
        </span>
        <span>
          <i style={{ background: '#facc15' }}></i>应用与交付
        </span>
      </div>
    </div>
  );
};

export default TitleBar;
