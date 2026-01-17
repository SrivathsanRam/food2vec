import React, { useState } from 'react';
import SearchBar from './SearchBar';
import SearchResults from './SearchResults';
import './LandingPage.css';

const LandingPage = () => {
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (query) => {
    if (!query.trim()) return;
    
    setIsLoading(true);
    setHasSearched(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, top_k: 10 }),
      });
      
      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="landing-page">
      <div className="hero-section">
        <div className="logo-container">
          <span className="logo-icon">🍕</span>
          <h1 className="logo-text">Food2Vec</h1>
        </div>
        <p className="tagline">Discover recipes using AI-powered vector search</p>
        
        <SearchBar onSearch={handleSearch} />
        
        <div className="features">
          <div className="feature">
            <span className="feature-icon">🔍</span>
            <span>Smart Search</span>
          </div>
          <div className="feature">
            <span className="feature-icon">🤖</span>
            <span>AI Powered</span>
          </div>
          <div className="feature">
            <span className="feature-icon">🔗</span>
            <span>Recipe Flow</span>
          </div>
        </div>
      </div>

      {hasSearched && (
        <SearchResults results={searchResults} isLoading={isLoading} />
      )}

      <footer className="footer">
        <p>Powered by Supabase + pgvector + React Flow</p>
      </footer>
    </div>
  );
};

export default LandingPage;
