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

    // Update displayed results when kValue changes
    useEffect(() => {
        if (allResults.length > 0) {
            setSearchResults(allResults.slice(0, kValue));
        }
    }, [kValue, allResults]);

    const handleSearch = async (query) => {
        if (!query.trim()) return;

        setIsLoading(true);
        setHasSearched(true);

        try {
            // Always fetch top 10, filter in frontend
            const response = await fetch(`${baseURL}/api/search`, {
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
            const response = await fetch(`${baseURL}/api/recipe`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ name: recipeName, steps: recipeSteps }),
            });

            if (!response.ok) {
                setCreateStatus("error");
                return;
            }

            setCreateStatus("success");

            // Refresh the food names cache with the new recipe
            await foodNamesCache.forceRefresh();

            // Clear search results
            setAllResults([]);
            setSearchResults([]);
            setHasSearched(false);
        } catch (error) {
            console.error("Create recipe error:", error);
            setCreateStatus("error");
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
            const response = await fetch(`${baseURL}/api/recipe/generate`, {
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
            <div className="hero-section">
                <div className="logo-container">
                    <h1 className="logo-text">Food2Vec</h1>
                </div>
                <p className="tagline">Discover recipes using AI-powered search</p>

                <SearchBar onSearch={handleSearch} />

                <br />
                <SliderComponent kValue={kValue} setKValue={setKValue} />

                <br />
                <Stack alignItems="center" gap={2}>
                    <Button onClick={handleModalOpen}>+ Add recipe</Button>
                </Stack>

                <Modal open={isModalOpen} onClose={handleModalClose}>
                    <Box sx={style}>
                        <Box sx={modalHeaderStyle}>
                            <Typography id="modal-title" variant="h6" component="h2">
                                Add recipe
                            </Typography>
                            <IconButton onClick={handleModalClose} sx={closeButtonStyle} size="small">
                                <CloseIcon />
                            </IconButton>
                        </Box>

                        <Box sx={modalContentStyle}>
                            <TextField
                                label="Recipe Name"
                                variant="outlined"
                                value={recipeName}
                                onChange={(e) => setRecipeName(e.target.value)}
                                placeholder="Enter recipe name..."
                                sx={textInputStyle}
                                autoFocus
                            />

                            <Button
                                onClick={handleGenerateRecipe}
                                variant="outlined"
                                color="secondary"
                                disabled={isGenerating || !recipeName.trim()}
                                sx={{ mb: 2, textTransform: "none" }}
                                startIcon={isGenerating ? <CircularProgress size={16} /> : null}
                            >
                                {isGenerating ? "Generating..." : "✨ AI Generate Steps"}
                            </Button>

                            <TextField
                                label="Recipe steps"
                                variant="outlined"
                                value={recipeSteps}
                                onChange={(e) => setRecipeSteps(e.target.value)}
                                placeholder="Enter recipe steps or use AI generate..."
                                multiline
                                rows={6}
                                sx={textInputStyle}
                                disabled={isCreating || createStatus === "success"}
                            />

                            {createStatus === "success" && (
                                <Alert severity="success" sx={{ mb: 2 }}>
                                    Recipe created successfully! The embedding has been generated.
                                </Alert>
                            )}

                            {createStatus === "error" && (
                                <Alert severity="error" sx={{ mb: 2 }}>
                                    Failed to create recipe. Please check your inputs and try again.
                                </Alert>
                            )}

                            {isCreating && (
                                <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
                                    <CircularProgress size={20} />
                                    <Typography variant="body2" color="text.secondary">
                                        Creating recipe and generating embedding...
                                    </Typography>
                                </Box>
                            )}
                        </Box>

                        <Box sx={modalFooterStyle}>
                            <Button onClick={handleModalClose} variant="outlined" disabled={isCreating}>
                                {createStatus === "success" ? "Close" : "Cancel"}
                            </Button>
                            {createStatus !== "success" && (
                                <Button
                                    onClick={handleSubmit}
                                    variant="contained"
                                    disabled={isCreating || isGenerating || !recipeName.trim() || !recipeSteps.trim()}
                                    startIcon={isCreating ? <CircularProgress size={16} color="inherit" /> : null}
                                >
                                    {isCreating ? "Creating..." : "Create"}
                                </Button>
                            )}
                        </Box>
                    </Box>
                </Modal>
            </div>

            {hasSearched && <SearchResults results={searchResults} isLoading={isLoading} />}

            <Snackbar
                open={isSnackbarOpen}
                autoHideDuration={2000}
                onClose={handleSnackbarClose}
                message={snackbarMessage}
            />

            <footer className="footer">
                <p></p>
            </footer>
        </div>
    );
};

export default LandingPage;
