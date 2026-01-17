import React, { useState, useEffect, useRef } from 'react';
import foodNamesCache from '../services/foodNamesCache';
import './SearchBar.css';

const SearchBar = ({ onSearch }) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [cacheReady, setCacheReady] = useState(false);
  const inputRef = useRef(null);
  const suggestionsRef = useRef(null);

  // Initialize cache on mount
  useEffect(() => {
    foodNamesCache.initialize().then(() => {
      setCacheReady(foodNamesCache.isReady());
    });

    // Subscribe to cache updates
    const unsubscribe = foodNamesCache.subscribe(() => {
      setCacheReady(foodNamesCache.isReady());
    });

    // Cleanup on unmount
    return () => {
      unsubscribe();
    };
  }, []);

  // Filter suggestions from cache (instant, no network delay)
  useEffect(() => {
    if (query.length < 1) {
      setSuggestions([]);
      return;
    }

    // Use cached data for instant filtering
    if (cacheReady) {
      const results = foodNamesCache.search(query, 10);
      setSuggestions(results);
    } else {
      // Fallback to API if cache not ready
      const fetchSuggestions = async () => {
        try {
          const response = await fetch(
            `http://localhost:5000/api/autocomplete?q=${encodeURIComponent(query)}`
          );
          const data = await response.json();
          setSuggestions(data.suggestions || []);
        } catch (error) {
          console.error('Autocomplete error:', error);
          setSuggestions([]);
        }
      };
      
      const debounceTimer = setTimeout(fetchSuggestions, 150);
      return () => clearTimeout(debounceTimer);
    }
  }, [query, cacheReady]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target) &&
        !inputRef.current.contains(event.target)
      ) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleInputChange = (e) => {
    setQuery(e.target.value);
    setShowSuggestions(true);
    setSelectedIndex(-1);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) =>
        prev < suggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0 && suggestions[selectedIndex]) {
        selectSuggestion(suggestions[selectedIndex]);
      } else {
        handleSearch();
      }
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  const selectSuggestion = (suggestion) => {
    setQuery(suggestion);
    setShowSuggestions(false);
    setSuggestions([]);
    onSearch(suggestion);
  };

  const handleSearch = () => {
    if (query.trim()) {
      setShowSuggestions(false);
      onSearch(query);
    }
  };

  return (
    <div className="search-container">
      <div className="search-bar">
        <span className="search-icon">🔍</span>
        <input
          ref={inputRef}
          type="text"
          placeholder="Search for any food..."
          value={query}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowSuggestions(true)}
          className="search-input"
        />
        <button onClick={handleSearch} className="search-button">
          Search
        </button>
      </div>

      {showSuggestions && suggestions.length > 0 && (
        <ul ref={suggestionsRef} className="suggestions-list">
          {suggestions.map((suggestion, index) => (
            <li
              key={suggestion}
              className={`suggestion-item ${
                index === selectedIndex ? 'selected' : ''
              }`}
              onClick={() => selectSuggestion(suggestion)}
              onMouseEnter={() => setSelectedIndex(index)}
            >
              <span className="suggestion-icon">🍴</span>
              {suggestion}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default SearchBar;
