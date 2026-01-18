import React, { useState, useEffect } from "react";
import SearchBar from "./SearchBar";
import SearchResults from "./SearchResults";
import "./LandingPage.css";
import SliderComponent from "./SliderComponent";
import foodNamesCache from "../services/foodNamesCache";
import Button from "@mui/material/Button";
import Modal from "@mui/material/Modal";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import CloseIcon from "@mui/icons-material/Close";
import TextField from "@mui/material/TextField";
import Snackbar from "@mui/material/Snackbar";
import Stack from "@mui/material/Stack";
import CircularProgress from "@mui/material/CircularProgress";
import Alert from "@mui/material/Alert";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Divider from "@mui/material/Divider";
import Avatar from "@mui/material/Avatar";
import PersonIcon from "@mui/icons-material/Person";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import LogoutIcon from "@mui/icons-material/Logout";
import CompareIcon from "@mui/icons-material/Compare";
import { useNavigate } from "react-router-dom";
import Cookies from "js-cookie";

// Modern, improved modal style with smooth animations and better UI
const style = {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: { xs: "90%", sm: 450, md: 500 }, // Responsive width
    maxWidth: "95vw",
    maxHeight: "90vh",
    overflow: "auto",
    bgcolor: "background.paper",
    borderRadius: 3, // Smooth rounded corners
    boxShadow: "0 20px 60px rgba(0, 0, 0, 0.3)", // Elevated shadow
    p: 0, // Remove padding to control it internally
    outline: "none", // Remove focus outline

    // Smooth scroll behavior
    overflowY: "auto",
    "&::-webkit-scrollbar": {
        width: "8px",
    },
    "&::-webkit-scrollbar-track": {
        background: "transparent",
    },
    "&::-webkit-scrollbar-thumb": {
        background: "#888",
        borderRadius: "4px",
    },
    "&::-webkit-scrollbar-thumb:hover": {
        background: "#555",
    },
};

// Header section style (optional, for modal header)
const modalHeaderStyle = {
    p: 3,
    pb: 2,
    borderBottom: "1px solid",
    borderColor: "divider",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
};

// Content section style
const modalContentStyle = {
    p: 3,
};

// Footer section style (optional, for actions)
const modalFooterStyle = {
    p: 3,
    pt: 2,
    borderTop: "1px solid",
    borderColor: "divider",
    display: "flex",
    gap: 2,
    justifyContent: "flex-end",
};

// Close button style
const closeButtonStyle = {
    color: "text.secondary",
    "&:hover": {
        color: "text.primary",
        bgcolor: "action.hover",
    },
};

// Text input field style
const textInputStyle = {
    width: "100%",
    mb: 2,
    "& .MuiOutlinedInput-root": {
        borderRadius: 2,
        "&:hover fieldset": {
            borderColor: "primary.main",
        },
    },
};

const baseURL = process.env.REACT_APP_API_BASE_URL;

const LandingPage = () => {
  const navigate = useNavigate();
  const [allResults, setAllResults] = useState([]); // Store all 10 results
  const [searchResults, setSearchResults] = useState([]); // Filtered results based on kValue
  const [kValue, setKValue] = useState(5);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [recipeSteps, setRecipeSteps] = useState("");
  const [recipeName, setRecipeName] = useState("");
  const [isSnackbarOpen, setIsSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [createStatus, setCreateStatus] = useState(null); // null, 'success', 'error'
  
  // Profile menu state
  const [anchorEl, setAnchorEl] = useState(null);
  const [palateCode, setPalateCode] = useState("");
  const [codeCopied, setCodeCopied] = useState(false);
  const username = Cookies.get("username") || "User";
  const profileMenuOpen = Boolean(anchorEl);

  // Update displayed results when kValue changes
  useEffect(() => {
    if (allResults.length > 0) {
      setSearchResults(allResults.slice(0, kValue));
    }
  }, [kValue, allResults]);

  // Fetch palate code on mount
  useEffect(() => {
    const fetchPalate = async () => {
      try {
        const response = await fetch(
          `http://localhost:5000/api/palate/check?username=${encodeURIComponent(username)}`
        );
        const data = await response.json();
        if (data.palate_code) {
          setPalateCode(data.palate_code);
        }
      } catch (error) {
        console.error("Error fetching palate:", error);
      }
    };
    if (username) fetchPalate();
  }, [username]);

  const handleProfileClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileClose = () => {
    setAnchorEl(null);
  };

  const handleCopyPalateCode = () => {
    navigator.clipboard.writeText(palateCode);
    setCodeCopied(true);
    setTimeout(() => setCodeCopied(false), 2000);
    setSnackbarMessage("Palate code copied!");
    setIsSnackbarOpen(true);
  };

  const handleLogout = () => {
    Cookies.remove("username");
    Cookies.remove("isLoggedIn");
    Cookies.remove("isOnboarded");
    handleProfileClose();
    navigate("/login");
  };

  const handleCompareClick = () => {
    navigate("/compare");
  };

  const handleSearch = async (query) => {
    if (!query.trim()) return;

    setIsLoading(true);
    setHasSearched(true);

    try {
      // Always fetch top 10, filter in frontend
      const response = await fetch("http://localhost:5000/api/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query, top_k: 10 }),
      });

      const data = await response.json();
      const results = data.results || [];
      setAllResults(results); // Store all results
      setSearchResults(results.slice(0, kValue)); // Display filtered results
    } catch (error) {
      console.error("Search error:", error);
      setAllResults([]);
      setSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModalOpen = () => {
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    if (isCreating) return; // Don't allow closing while creating
    setIsModalOpen(false);
    setRecipeName("");
    setRecipeSteps("");
    setCreateStatus(null);
  };

  const handleSubmit = async () => {
    const nameToSubmit = recipeName;
    const stepsToSubmit = recipeSteps;

    setRecipeName("");
    setRecipeSteps("");
    handleModalClose();
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:5000/api/recipe", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name: recipeName, steps: recipeSteps }),
      });

      if (!response.ok) {
        setCreateStatus('error');
        return;
      }

      setCreateStatus('success');
      
      // Refresh the food names cache with the new recipe
      await foodNamesCache.forceRefresh();
      
      // Clear search results
      setAllResults([]);
      setSearchResults([]);
      setHasSearched(false);
    } catch (error) {
      console.error("Create recipe error:", error);
      setCreateStatus('error');
    } finally {
      setIsCreating(false);
    }
  };

  const handleSnackbarClose = () => {
    setIsSnackbarOpen(false);
  };

  const handleGenerateRecipe = async () => {
    if (!recipeName.trim()) {
      setIsSnackbarOpen(true);
      setSnackbarMessage("Please enter a recipe name first.");
      return;
    }

    setIsGenerating(true);
    try {
      const response = await fetch("http://localhost:5000/api/recipe/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name: recipeName }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate recipe");
      }

      const data = await response.json();
      setRecipeSteps(data.steps || "");
      setIsSnackbarOpen(true);
      setSnackbarMessage("Recipe steps generated!");
    } catch (error) {
      console.error("Generate recipe error:", error);
      setIsSnackbarOpen(true);
      setSnackbarMessage("Failed to generate recipe.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="landing-page">
      {/* Top Navigation Bar */}
      <Box
        sx={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          px: 2,
          py: 1,
          backgroundColor: "rgba(255, 255, 255, 0.95)",
          backdropFilter: "blur(8px)",
          borderBottom: "1px solid #eee",
          zIndex: 1000,
        }}
      >
        {/* Profile Button - Left */}
        <IconButton
          onClick={handleProfileClick}
          size="small"
          sx={{ border: "1px solid #e0e0e0" }}
        >
          <Avatar sx={{ width: 32, height: 32, bgcolor: "primary.main" }}>
            <PersonIcon fontSize="small" />
          </Avatar>
        </IconButton>

        <Menu
          anchorEl={anchorEl}
          open={profileMenuOpen}
          onClose={handleProfileClose}
          PaperProps={{
            sx: { minWidth: 240, mt: 1 },
          }}
        >
          <Box sx={{ px: 2, py: 1 }}>
            <Typography variant="subtitle2">{username}</Typography>
            <Typography variant="caption" color="text.secondary">Your palate code</Typography>
            <Box
              sx={{
                mt: 1,
                p: 1,
                backgroundColor: "#f5f5f5",
                borderRadius: 1,
                fontFamily: "monospace",
                fontSize: "0.75rem",
                wordBreak: "break-all",
                maxWidth: 200,
              }}
            >
              {palateCode ? palateCode.substring(0, 30) + "..." : "Loading..."}
            </Box>
          </Box>
          <MenuItem onClick={handleCopyPalateCode} disabled={!palateCode}>
            <ContentCopyIcon fontSize="small" sx={{ mr: 1 }} />
            {codeCopied ? "Copied!" : "Copy Palate Code"}
          </MenuItem>
          <Divider />
          <MenuItem onClick={handleLogout}>
            <LogoutIcon fontSize="small" sx={{ mr: 1 }} />
            Log Out
          </MenuItem>
        </Menu>

        {/* Compare Button - Right */}
        <Button
          variant="outlined"
          size="small"
          startIcon={<CompareIcon />}
          onClick={handleCompareClick}
          sx={{ textTransform: "none" }}
        >
          Compare
        </Button>
      </Box>

      <div className="hero-section" style={{ paddingTop: "60px" }}>
        <div className="logo-container">
          <h1 className="logo-text">Food2Vec</h1>
        </div>
      </div>

      <SearchBar onSearch={handleSearch} />

      {isLoading && (
        <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {hasSearched && !isLoading && (
        <SearchResults results={searchResults} />
      )}

      <SliderComponent kValue={kValue} setKValue={setKValue} />

      <Button
        variant="contained"
        onClick={handleModalOpen}
        sx={{ mt: 4, mb: 4 }}
      >
        Create Custom Recipe
      </Button>

      <Modal open={isModalOpen} onClose={handleModalClose}>
        <Box sx={style}>
          <Box sx={modalHeaderStyle}>
            <Typography variant="h6">Create Custom Recipe</Typography>
            <IconButton
              onClick={handleModalClose}
              sx={closeButtonStyle}
              disabled={isCreating}
            >
              <CloseIcon />
            </IconButton>
          </Box>
          <Box sx={modalContentStyle}>
            <TextField
              label="Recipe Name"
              value={recipeName}
              onChange={(e) => setRecipeName(e.target.value)}
              sx={textInputStyle}
              disabled={isCreating}
            />
            <TextField
              label="Recipe Steps"
              value={recipeSteps}
              onChange={(e) => setRecipeSteps(e.target.value)}
              multiline
              rows={6}
              sx={textInputStyle}
              disabled={isCreating}
            />
            {createStatus === 'success' && (
              <Alert severity="success">Recipe created successfully!</Alert>
            )}
            {createStatus === 'error' && (
              <Alert severity="error">Failed to create recipe.</Alert>
            )}
          </Box>
          <Box sx={modalFooterStyle}>
            <Button
              onClick={handleGenerateRecipe}
              disabled={isGenerating || isCreating}
              startIcon={isGenerating ? <CircularProgress size={20} /> : null}
            >
              {isGenerating ? "Generating..." : "Generate with AI"}
            </Button>
            <Button
              onClick={handleSubmit}
              variant="contained"
              disabled={isCreating}
            >
              {isCreating ? "Creating..." : "Create Recipe"}
            </Button>
          </Box>
        </Box>
      </Modal>

      <Snackbar
        open={isSnackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
      >
        <Alert onClose={handleSnackbarClose} severity="info">
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </div>
    );
};

export default LandingPage;
