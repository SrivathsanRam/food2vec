import React, { useState } from "react";
import SearchBar from "./SearchBar";
import SearchResults from "./SearchResults";
import "./LandingPage.css";
import SliderComponent from "./SliderComponent";

const LandingPage = () => {
  const [searchResults, setSearchResults] = useState([]);
  const [kValue, setKValue] = useState(5);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (query) => {
    if (!query.trim()) return;

    setIsLoading(true);
    setHasSearched(true);

    try {
      const response = await fetch("http://localhost:5000/api/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query, top_k: kValue }),
      });

      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (error) {
      console.error("Search error:", error);
      setSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="landing-page">
      <div className="hero-section">
        <div className="logo-container">
          <h1 className="logo-text">Food2Vec</h1>
        </div>
        <p className="tagline">Discover recipes using AI-powered search</p>

        <SearchBar onSearch={handleSearch} />

        <br />
        <SliderComponent nNearest={kValue} setKValue={setKValue} />
      </div>

      {hasSearched && (
        <SearchResults results={searchResults} isLoading={isLoading} />
      )}

      <footer className="footer">
        <p></p>
      </footer>
    </div>
  );
};

export default LandingPage;
