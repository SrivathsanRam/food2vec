import React, { useState, useEffect } from "react";
import {
    Box,
    Typography,
    Container,
    Paper,
    Button,
    Rating,
    CircularProgress,
    LinearProgress,
    Chip,
    IconButton,
    Snackbar,
    Alert,
    TextField,
    InputAdornment,
} from "@mui/material";
import { ContentCopy, Check, Upload } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import Cookies from "js-cookie";

const baseURL = process.env.REACT_APP_API_BASE_URL;

// Dishes from the recipe database for rating
const ONBOARDING_DISHES = [
    { id: 1, name: '"Refried" Beans', category: "Beans" },
    { id: 2, name: "3 Bean Salad", category: "Salad" },
    { id: 3, name: "20 minute seared strip steak with sweet-and-sour carrots", category: "Meat" },
    { id: 4, name: "Almond Shortbread", category: "Cookies" },
    { id: 5, name: "Almond Crescent", category: "Cookies" },
    { id: 6, name: "5-Minute Fudge", category: "Dessert" },
    { id: 7, name: "1950'S Potato Chip Cookies", category: "Cookies" },
    { id: 8, name: "(Web Exclusive) Round 2 Recipe: Edamame with Pasta", category: "Pasta" },
    { id: 9, name: "*Sweet And Sour Carrots", category: "Vegetables" },
    { id: 10, name: '"Pecan Pie" Acorn Squash', category: "Vegetables" },
];

export default function OnboardingPage() {
    const navigate = useNavigate();
    const [ratings, setRatings] = useState({});
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [palateCode, setPalateCode] = useState("");
    const [showResult, setShowResult] = useState(false);
    const [copied, setCopied] = useState(false);
    const [snackbarOpen, setSnackbarOpen] = useState(false);
    const [snackbarMessage, setSnackbarMessage] = useState("");
    const [importCode, setImportCode] = useState("");
    const [showImport, setShowImport] = useState(false);

    const username = Cookies.get("username");

    useEffect(() => {
        // Check if user is logged in
        if (!Cookies.get("isLoggedIn")) {
            navigate("/login");
        }
    }, [navigate]);

    const handleRating = (dishId, value) => {
        setRatings((prev) => ({
            ...prev,
            [dishId]: value,
        }));
    };

    const handleNext = () => {
        if (currentIndex < ONBOARDING_DISHES.length - 1) {
            setCurrentIndex((prev) => prev + 1);
        }
    };

    const handlePrevious = () => {
        if (currentIndex > 0) {
            setCurrentIndex((prev) => prev - 1);
        }
    };

    const handleSkip = () => {
        handleNext();
    };

    const progress = (Object.keys(ratings).length / ONBOARDING_DISHES.length) * 100;

    const handleSubmit = async () => {
        setIsSubmitting(true);

        try {
            const response = await fetch(`${baseURL}/api/palate/create`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    username: username,
                    ratings: ratings,
                    dishes: ONBOARDING_DISHES.map((d) => ({ id: d.id, name: d.name, category: d.category })),
                }),
            });

            if (!response.ok) {
                throw new Error("Failed to create palate");
            }

            const data = await response.json();
            setPalateCode(data.palate_code);
            setShowResult(true);

            // Mark user as onboarded
            Cookies.set("isOnboarded", "true", { expires: 365 });
        } catch (error) {
            console.error("Onboarding error:", error);
            setSnackbarMessage("Failed to save your palate. Please try again.");
            setSnackbarOpen(true);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleCopyCode = () => {
        navigator.clipboard.writeText(palateCode);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleImportPalate = async () => {
        if (!importCode.trim()) {
            setSnackbarMessage("Please enter a palate code");
            setSnackbarOpen(true);
            return;
        }

        setIsSubmitting(true);

        try {
            const response = await fetch(`${baseURL}/api/palate/import`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    username: username,
                    palate_code: importCode.trim(),
                }),
            });

            if (!response.ok) {
                throw new Error("Invalid palate code");
            }

            setPalateCode(importCode.trim());
            setShowResult(true);
            Cookies.set("isOnboarded", "true", { expires: 365 });
            setSnackbarMessage("Palate imported successfully!");
            setSnackbarOpen(true);
        } catch (error) {
            console.error("Import error:", error);
            setSnackbarMessage("Invalid palate code. Please check and try again.");
            setSnackbarOpen(true);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleContinue = () => {
        navigate("/recipes");
    };

    const currentDish = ONBOARDING_DISHES[currentIndex];
    const ratedCount = Object.keys(ratings).length;
    const canSubmit = ratedCount >= 5; // Require at least 5 ratings

    if (showResult) {
        return (
            <Box
                sx={{
                    minHeight: "100vh",
                    background: "#fafafa",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    py: 4,
                }}
            >
                <Container maxWidth="sm">
                    <Paper
                        elevation={0}
                        sx={{
                            p: 4,
                            borderRadius: 2,
                            textAlign: "center",
                            border: "1px solid #e0e0e0",
                        }}
                    >
                        <Typography variant="h5" fontWeight="600" gutterBottom>
                            Your Palate Profile
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                            Share this code with friends to let them try your taste profile.
                        </Typography>

                        <Paper
                            elevation={0}
                            sx={{
                                p: 2,
                                mb: 3,
                                backgroundColor: "#f5f5f5",
                                borderRadius: 1,
                            }}
                        >
                            <Typography variant="body1" fontFamily="monospace" sx={{ wordBreak: "break-all" }}>
                                {palateCode}
                            </Typography>
                            <IconButton
                                onClick={handleCopyCode}
                                size="small"
                                sx={{ mt: 1 }}
                                color={copied ? "success" : "default"}
                            >
                                {copied ? <Check fontSize="small" /> : <ContentCopy fontSize="small" />}
                            </IconButton>
                        </Paper>

                        <Button
                            variant="contained"
                            onClick={handleContinue}
                            disableElevation
                            sx={{
                                py: 1,
                                px: 3,
                                textTransform: "none",
                                fontWeight: 500,
                            }}
                        >
                            Continue
                        </Button>
                    </Paper>
                </Container>
            </Box>
        );
    }

    return (
        <Box
            sx={{
                minHeight: "100vh",
                background: "#fafafa",
                py: 4,
            }}
        >
            <Container maxWidth="sm">
                {/* Header */}
                <Box sx={{ textAlign: "center", mb: 3 }}>
                    <Typography variant="h5" fontWeight="600" gutterBottom>
                        Build Your Palate
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Rate dishes to personalize your experience
                    </Typography>
                </Box>

                {/* Progress */}
                <Box sx={{ mb: 3 }}>
                    <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                        <Typography variant="caption" color="text.secondary">
                            {ratedCount} / {ONBOARDING_DISHES.length} rated
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            {Math.round(progress)}%
                        </Typography>
                    </Box>
                    <LinearProgress
                        variant="determinate"
                        value={progress}
                        sx={{ height: 4, borderRadius: 2, backgroundColor: "#e0e0e0" }}
                    />
                </Box>

                {/* Import Option */}
                <Box sx={{ textAlign: "center", mb: 2 }}>
                    <Button
                        variant="text"
                        size="small"
                        startIcon={<Upload fontSize="small" />}
                        onClick={() => setShowImport(!showImport)}
                        sx={{ textTransform: "none", color: "text.secondary" }}
                    >
                        {showImport ? "Rate dishes instead" : "Import palate code"}
                    </Button>
                </Box>

                {showImport ? (
                    <Paper elevation={0} sx={{ p: 3, borderRadius: 2, mb: 3, border: "1px solid #e0e0e0" }}>
                        <Typography variant="subtitle2" gutterBottom>
                            Import Palate Code
                        </Typography>
                        <TextField
                            fullWidth
                            size="small"
                            value={importCode}
                            onChange={(e) => setImportCode(e.target.value)}
                            placeholder="Paste code here..."
                            sx={{ mb: 2 }}
                            InputProps={{
                                endAdornment: (
                                    <InputAdornment position="end">
                                        <Button
                                            size="small"
                                            onClick={handleImportPalate}
                                            disabled={isSubmitting || !importCode.trim()}
                                            sx={{ textTransform: "none" }}
                                        >
                                            {isSubmitting ? <CircularProgress size={16} /> : "Import"}
                                        </Button>
                                    </InputAdornment>
                                ),
                            }}
                        />
                    </Paper>
                ) : (
                    <>
                        {/* Dish Card */}
                        <Paper
                            elevation={0}
                            sx={{
                                p: 3,
                                borderRadius: 2,
                                mb: 2,
                                textAlign: "center",
                                border: "1px solid #e0e0e0",
                            }}
                        >
                            <Typography variant="h6" fontWeight="500" gutterBottom>
                                {currentDish.name}
                            </Typography>
                            <Chip label={currentDish.category} size="small" variant="outlined" sx={{ mb: 2 }} />

                            <Box sx={{ mb: 2 }}>
                                <Rating
                                    value={ratings[currentDish.id] || 0}
                                    onChange={(_, value) => handleRating(currentDish.id, value)}
                                    size="large"
                                />
                            </Box>

                            {/* Navigation */}
                            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <Button
                                    size="small"
                                    onClick={handlePrevious}
                                    disabled={currentIndex === 0}
                                    sx={{ textTransform: "none" }}
                                >
                                    Back
                                </Button>
                                <Typography variant="caption" color="text.secondary">
                                    {currentIndex + 1} / {ONBOARDING_DISHES.length}
                                </Typography>
                                {currentIndex < ONBOARDING_DISHES.length - 1 ? (
                                    <Box>
                                        <Button size="small" onClick={handleSkip} sx={{ textTransform: "none", mr: 1 }}>
                                            Skip
                                        </Button>
                                        <Button
                                            variant="contained"
                                            size="small"
                                            onClick={handleNext}
                                            disableElevation
                                            sx={{ textTransform: "none" }}
                                        >
                                            Next
                                        </Button>
                                    </Box>
                                ) : (
                                    <Button
                                        variant="contained"
                                        size="small"
                                        onClick={handleSubmit}
                                        disabled={!canSubmit || isSubmitting}
                                        disableElevation
                                        sx={{ textTransform: "none" }}
                                    >
                                        {isSubmitting ? <CircularProgress size={16} color="inherit" /> : "Done"}
                                    </Button>
                                )}
                            </Box>
                        </Paper>

                        {/* Quick Rate Grid */}
                        <Paper elevation={0} sx={{ p: 2, borderRadius: 2, border: "1px solid #e0e0e0" }}>
                            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: "block" }}>
                                Quick jump
                            </Typography>
                            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                                {ONBOARDING_DISHES.map((dish, idx) => (
                                    <Chip
                                        key={dish.id}
                                        label={ratings[dish.id] ? `${idx + 1}` : `${idx + 1}`}
                                        size="small"
                                        onClick={() => setCurrentIndex(idx)}
                                        variant={currentIndex === idx ? "filled" : "outlined"}
                                        color={ratings[dish.id] ? "primary" : "default"}
                                        sx={{
                                            cursor: "pointer",
                                            minWidth: 32,
                                        }}
                                    />
                                ))}
                            </Box>
                        </Paper>

                        {!canSubmit && (
                            <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ textAlign: "center", mt: 2, display: "block" }}
                            >
                                Rate at least 5 dishes to continue
                            </Typography>
                        )}
                    </>
                )}
            </Container>

            <Snackbar open={snackbarOpen} autoHideDuration={3000} onClose={() => setSnackbarOpen(false)}>
                <Alert
                    onClose={() => setSnackbarOpen(false)}
                    severity={snackbarMessage.includes("success") ? "success" : "error"}
                    variant="filled"
                >
                    {snackbarMessage}
                </Alert>
            </Snackbar>
        </Box>
    );
}
