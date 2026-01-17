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

  // Clear localStorage cache
  clearStorage() {
    try {
      localStorage.removeItem(CACHE_KEY);
    } catch (e) {
      console.error('Error clearing cache from storage:', e);
    }
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

    // Clear old cache and fetch fresh data from server on page load
    this.clearStorage();
    await this.refreshNames();

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
      console.log('Fetching fresh food names from server...');
      
      // Add cache-busting timestamp to avoid browser caching
      const timestamp = Date.now();
      const response = await fetch(`${API_BASE}/food-names?t=${timestamp}`, {
        method: 'GET',
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      });
      const { names, version, count } = await response.json();
      
      console.log(`Received ${count} food names from server, version: ${version}`);
      
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
    console.log('Force refreshing food names cache...');
    this.clearStorage();
    this.names = [];
    this.version = '';
    this.isLoaded = false;
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
