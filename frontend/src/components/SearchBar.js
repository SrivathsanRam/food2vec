import React, { useState, useEffect, useRef } from "react";
import foodNamesCache from "../services/foodNamesCache";
import "./SearchBar.css";

const baseURL = process.env.REACT_APP_API_BASE_URL;

const SearchBar = ({ onSearch }) => {
    const [query, setQuery] = useState("");
    const [suggestions, setSuggestions] = useState([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(-1);
    const [cacheReady, setCacheReady] = useState(false);
    const [isValidSelection, setIsValidSelection] = useState(false); // Track if user selected from autocomplete
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

            // Check if current query exactly matches a valid recipe name
            const allNames = foodNamesCache.getAllNames();
            const exactMatch = allNames.some((name) => name.toLowerCase() === query.toLowerCase());
            setIsValidSelection(exactMatch);
        } else {
            // Fallback to API if cache not ready
            const fetchSuggestions = async () => {
                try {
                    const response = await fetch(`/api/autocomplete?q=${encodeURIComponent(query)}`);
                    const data = await response.json();
                    setSuggestions(data.suggestions || []);
                } catch (error) {
                    console.error("Autocomplete error:", error);
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

        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const handleInputChange = (e) => {
        setQuery(e.target.value);
        setShowSuggestions(true);
        setSelectedIndex(-1);
        setIsValidSelection(false); // Reset valid selection when user types
    };

    const handleKeyDown = (e) => {
        if (e.key === "ArrowDown") {
            e.preventDefault();
            setSelectedIndex((prev) => (prev < suggestions.length - 1 ? prev + 1 : prev));
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
        } else if (e.key === "Enter") {
            e.preventDefault();
            if (selectedIndex >= 0 && suggestions[selectedIndex]) {
                selectSuggestion(suggestions[selectedIndex]);
            } else if (isValidSelection) {
                // Only allow search if it's a valid selection
                handleSearch();
            }
        } else if (e.key === "Escape") {
            setShowSuggestions(false);
        }
    };

    const selectSuggestion = (suggestion) => {
        setQuery(suggestion);
        setShowSuggestions(false);
        setSuggestions([]);
        setIsValidSelection(true); // Mark as valid since user selected from list
        onSearch(suggestion);
    };

    const handleSearch = () => {
        if (query.trim() && isValidSelection) {
            setShowSuggestions(false);
            onSearch(query);
        }
    };

    return (
        <div className="search-container">
            <div className="search-bar">
                <input
                    ref={inputRef}
                    type="text"
                    placeholder="Search recipes..."
                    value={query}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    onFocus={() => setShowSuggestions(true)}
                    className="search-input"
                />
                <button
                    onClick={handleSearch}
                    className={`search-button ${!isValidSelection ? "disabled" : ""}`}
                    disabled={!isValidSelection}
                    title={!isValidSelection ? "Please select a recipe from the suggestions" : ""}
                >
                    Search
                </button>
            </div>

            {showSuggestions && suggestions.length > 0 && (
                <ul ref={suggestionsRef} className="suggestions-list">
                    {suggestions.map((suggestion, index) => (
                        <li
                            key={suggestion}
                            className={`suggestion-item ${index === selectedIndex ? "selected" : ""}`}
                            onClick={() => selectSuggestion(suggestion)}
                            onMouseEnter={() => setSelectedIndex(index)}
                        >
                            {suggestion}
                        </li>
                    ))}
                </ul>
            )}

            {query.length > 0 && !isValidSelection && !showSuggestions && (
                <div className="search-hint">Please select a recipe from the suggestions</div>
            )}
        </div>
    );
};

export default SearchBar;
