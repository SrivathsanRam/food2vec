const API_BASE = 'http://localhost:5000/api';
const CACHE_KEY = 'food_names_cache';
const VERSION_CHECK_INTERVAL = 60000; // Check for updates every 60 seconds

class FoodNamesCache {
  constructor() {
    this.names = [];
    this.version = '';
    this.isLoaded = false;
    this.isLoading = false;
    this.listeners = new Set();
    this.versionCheckTimer = null;
  }

  // Subscribe to cache updates
  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // Notify all listeners of updates
  notifyListeners() {
    this.listeners.forEach(listener => listener(this.names));
  }

  // Load cache from localStorage
  loadFromStorage() {
    try {
      const cached = localStorage.getItem(CACHE_KEY);
      if (cached) {
        const { names, version } = JSON.parse(cached);
        this.names = names || [];
        this.version = version || '';
        this.isLoaded = true;
        return true;
      }
    } catch (e) {
      console.error('Error loading cache from storage:', e);
    }
    return false;
  }

  // Save cache to localStorage
  saveToStorage() {
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify({
        names: this.names,
        version: this.version
      }));
    } catch (e) {
      console.error('Error saving cache to storage:', e);
    }
  }

  // Initialize the cache
  async initialize() {
    if (this.isLoading) return;
    this.isLoading = true;

    // First, try to load from localStorage for instant access
    this.loadFromStorage();
    if (this.isLoaded) {
      this.notifyListeners();
    }

    // Then check if we need to refresh from the server
    await this.checkAndRefresh();

    // Start periodic version checks
    this.startVersionChecks();
    
    this.isLoading = false;
  }

  // Check version and refresh if needed
  async checkAndRefresh() {
    try {
      const response = await fetch(`${API_BASE}/food-names/version`);
      const { version } = await response.json();

      // If version differs or we have no data, refresh the full list
      if (version !== this.version || this.names.length === 0) {
        await this.refreshNames();
      }
    } catch (error) {
      console.error('Error checking cache version:', error);
      // If check fails but we have cached data, continue using it
    }
  }

  // Refresh names from the server
  async refreshNames() {
    try {
      const response = await fetch(`${API_BASE}/food-names`);
      const { names, version, count } = await response.json();
      
      if (names && names.length > 0) {
        this.names = names;
        this.version = version;
        this.isLoaded = true;
        this.saveToStorage();
        this.notifyListeners();
        console.log(`Food names cache updated: ${count} items, version ${version}`);
      }
    } catch (error) {
      console.error('Error refreshing food names:', error);
    }
  }

  // Start periodic version checks
  startVersionChecks() {
    if (this.versionCheckTimer) {
      clearInterval(this.versionCheckTimer);
    }
    this.versionCheckTimer = setInterval(() => {
      this.checkAndRefresh();
    }, VERSION_CHECK_INTERVAL);
  }

  // Stop version checks (call on unmount)
  stopVersionChecks() {
    if (this.versionCheckTimer) {
      clearInterval(this.versionCheckTimer);
      this.versionCheckTimer = null;
    }
  }

  // Search names locally (fast client-side filtering)
  search(query, limit = 10) {
    if (!query || query.length < 1) return [];
    
    const lowerQuery = query.toLowerCase();
    
    // Prioritize names that start with the query
    const startsWithMatches = [];
    const containsMatches = [];
    
    for (const name of this.names) {
      const lowerName = name.toLowerCase();
      if (lowerName.startsWith(lowerQuery)) {
        startsWithMatches.push(name);
      } else if (lowerName.includes(lowerQuery)) {
        containsMatches.push(name);
      }
      
      // Early exit if we have enough matches
      if (startsWithMatches.length + containsMatches.length >= limit * 2) {
        break;
      }
    }
    
    // Combine results, prioritizing startsWith matches
    return [...startsWithMatches, ...containsMatches].slice(0, limit);
  }

  // Force a refresh (useful after adding new items)
  async forceRefresh() {
    await this.refreshNames();
  }

  // Get all names
  getAllNames() {
    return this.names;
  }

  // Check if cache is loaded
  isReady() {
    return this.isLoaded && this.names.length > 0;
  }
}

// Singleton instance
const foodNamesCache = new FoodNamesCache();

export default foodNamesCache;
