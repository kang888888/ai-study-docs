const ControlBar = () => {
  const handleReset = () => {
    if (window.chartControls) {
      window.chartControls.reset();
    }
  };

  return (
    <div className="control-bar">
      <button className="control-btn" onClick={handleReset}>
        重新布局
      </button>
    </div>
  );
};

export default ControlBar;
