import React, { useState } from 'react';
import RecipeGraph from './RecipeGraph';
import './SearchResults.css';

const SearchResults = ({ results, isLoading }) => {
  const [expandedRecipe, setExpandedRecipe] = useState(null);
  const [showGraph, setShowGraph] = useState(null);

  if (isLoading) {
    return (
      <div className="results-container">
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Finding recipes...</p>
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="results-container">
        <div className="no-results">
          <span className="no-results-icon">🍽️</span>
          <p>No recipes found. Try a different search!</p>
        </div>
      </div>
    );
  }

  const getCategoryEmoji = (category) => {
    const emojis = {
      'Desserts': '🍰',
      'Main Course': '🍖',
      'Salads & Sides': '🥗',
      'Appetizers': '🍴',
      'Other': '🍳',
    };
    return emojis[category] || '🍳';
  };

  const getScoreColor = (score) => {
    if (score >= 0.9) return '#10b981';
    if (score >= 0.7) return '#f59e0b';
    return '#6366f1';
  };

  const toggleExpand = (recipeId) => {
    setExpandedRecipe(expandedRecipe === recipeId ? null : recipeId);
    if (expandedRecipe !== recipeId) {
      setShowGraph(null);
    }
  };

  const toggleGraph = (recipeId) => {
    setShowGraph(showGraph === recipeId ? null : recipeId);
  };

  return (
    <div className="results-container">
      <h2 className="results-title">
        🍳 Found {results.length} Recipe{results.length !== 1 ? 's' : ''}
      </h2>
      <div className="results-list">
        {results.map((result, index) => (
          <div
            key={`${result.name}-${index}`}
            className={`result-card ${expandedRecipe === result.id ? 'expanded' : ''}`}
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            <div className="result-header" onClick={() => toggleExpand(result.id)}>
              <div className="result-title-section">
                <span className="result-emoji">
                  {getCategoryEmoji(result.category)}
                </span>
                <div className="result-info">
                  <h3 className="result-name">{result.name}</h3>
                  <span className="result-category">{result.category}</span>
                </div>
              </div>
              <div className="result-meta">
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
                <span className="expand-icon">
                  {expandedRecipe === result.id ? '▼' : '▶'}
                </span>
              </div>
            </div>

            {expandedRecipe === result.id && (
              <div className="result-details">
                {/* Ingredients */}
                {result.ingredients && result.ingredients.length > 0 && (
                  <div className="detail-section">
                    <h4>🥘 Ingredients ({result.ingredients.length})</h4>
                    <ul className="ingredients-list">
                      {result.ingredients.map((ing, i) => (
                        <li key={i}>{ing}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Directions */}
                {result.directions && result.directions.length > 0 && (
                  <div className="detail-section">
                    <h4>📝 Directions</h4>
                    <ol className="directions-list">
                      {result.directions.map((step, i) => (
                        <li key={i}>{step}</li>
                      ))}
                    </ol>
                  </div>
                )}

                {/* Graph Toggle Button */}
                {result.graph && result.graph.nodes && result.graph.nodes.length > 0 && (
                  <div className="graph-section">
                    <button 
                      className="graph-toggle-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleGraph(result.id);
                      }}
                    >
                      {showGraph === result.id ? '🔽 Hide Recipe Flow' : '🔗 Show Recipe Flow'}
                    </button>
                    
                    {showGraph === result.id && (
                      <RecipeGraph graph={result.graph} recipeName={result.name} />
                    )}
                  </div>
                )}

                {/* NER Tags */}
                {result.ner && result.ner.length > 0 && (
                  <div className="detail-section">
                    <h4>🏷️ Key Ingredients</h4>
                    <div className="ner-tags">
                      {result.ner.map((tag, i) => (
                        <span key={i} className="ner-tag">{tag}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default SearchResults;
