const SliderComponent = ({ kValue, setKValue }) => {
  return (
    <div style={{ width: '100%', maxWidth: '500px', margin: '16px auto 0 auto' }}>
      <div style={{ 
        backgroundColor: 'white', 
        padding: '16px 20px', 
        borderRadius: '8px', 
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
        border: '1px solid #ddd'
      }}>
        <div style={{ marginBottom: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ fontSize: '0.9rem', fontWeight: '500', color: '#555' }}>
              Top results:
            </label>
            <span style={{ fontSize: '1.25rem', fontWeight: '600', color: '#333' }}>{kValue}</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: '500', color: '#888' }}>1</span>
            <input
              type="range"
              min="1"
              max="10"
              value={kValue}
              onChange={(e) => setKValue(Number(e.target.value))}
              style={{ 
                flex: 1, 
                height: '6px', 
                borderRadius: '4px', 
                cursor: 'pointer',
                accentColor: '#333'
              }}
            />
            <span style={{ fontSize: '0.85rem', fontWeight: '500', color: '#888' }}>10</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SliderComponent;
