const SliderComponent = ({ kValue, setKValue }) => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="bg-white p-8 rounded-2xl shadow-lg w-96">
        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium text-gray-700">
              Selected top n nearest vectors:{" "}
            </label>
            <span className="text-3xl font-bold text-indigo-600">{kValue}</span>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-600">1</span>
            <input
              type="range"
              min="1"
              max="10"
              value={kValue}
              onChange={(e) => setKValue(Number(e.target.value))}
              className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <span className="text-sm font-medium text-gray-600">10</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SliderComponent;
