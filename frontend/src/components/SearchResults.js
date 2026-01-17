import React from 'react';
import './SearchResults.css';

const SearchResults = ({ results, isLoading }) => {
  if (isLoading) {
    return (
      <div className="results-container">
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Finding similar foods...</p>
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="results-container">
        <div className="no-results">
          <span className="no-results-icon">🍽️</span>
          <p>No foods found. Try a different search!</p>
        </div>
      </div>
    );
  }

  const getCategoryEmoji = (category) => {
    const emojis = {
      Fruits: '🍎',
      Vegetables: '🥬',
      Desserts: '🍰',
      Proteins: '🍗',
      Other: '🍴',
    };
    return emojis[category] || '🍴';
  };

  const getScoreColor = (score) => {
    if (score >= 0.9) return '#10b981';
    if (score >= 0.7) return '#f59e0b';
    return '#6366f1';
  };

  return (
    <div className="results-container">
      <h2 className="results-title">Search Results</h2>
      <div className="results-grid">
        {results.map((result, index) => (
          <div
            key={`${result.name}-${index}`}
            className="result-card"
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <div className="result-header">
              <span className="result-emoji">
                {getCategoryEmoji(result.category)}
              </span>
              <div className="result-score">
                <div
                  className="score-bar"
                  style={{
                    width: `${result.score * 100}%`,
                    backgroundColor: getScoreColor(result.score),
                  }}
                ></div>
                <span className="score-text">
                  {(result.score * 100).toFixed(0)}% match
                </span>
              </div>
            </div>
            <h3 className="result-name">{result.name}</h3>
            <span className="result-category">{result.category}</span>
            <p className="result-description">{result.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SearchResults;
