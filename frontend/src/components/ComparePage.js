import React, { useState, useEffect } from "react";
import {
    Box,
    Typography,
    Container,
    Paper,
    Button,
    TextField,
    CircularProgress,
    Chip,
    LinearProgress,
    IconButton,
    Snackbar,
    Alert,
    Divider,
} from "@mui/material";
import { ArrowBack, Compare, Search, ContentCopy, Check } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import Cookies from "js-cookie";

export default function ComparePage() {
    const navigate = useNavigate();
    const [myPalateCode, setMyPalateCode] = useState("");
    const [friendCode, setFriendCode] = useState("");
    const [isComparing, setIsComparing] = useState(false);
    const [comparisonResult, setComparisonResult] = useState(null);
    const [isSearching, setIsSearching] = useState(false);
    const [intersectionRecipes, setIntersectionRecipes] = useState([]);
    const [snackbarOpen, setSnackbarOpen] = useState(false);
    const [snackbarMessage, setSnackbarMessage] = useState("");
    const [copied, setCopied] = useState(false);

    const username = Cookies.get("username");

    useEffect(() => {
        if (!Cookies.get("isLoggedIn")) {
            navigate("/login");
            return;
        }
        const fetchMyPalate = async () => {
            try {
                const response = await fetch(`/api/palate/check?username=${encodeURIComponent(username)}`);
                const data = await response.json();
                if (data.palate_code) {
                    setMyPalateCode(data.palate_code);
                }
            } catch (error) {
                console.error("Error fetching palate:", error);
            }
        };

        fetchMyPalate();
    }, [navigate, username]);

    const handleCopyCode = () => {
        navigator.clipboard.writeText(myPalateCode);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleCompare = async () => {
        if (!friendCode.trim()) {
            setSnackbarMessage("Please enter a friend's palate code");
            setSnackbarOpen(true);
            return;
        }

        setIsComparing(true);
        setComparisonResult(null);

        try {
            const response = await fetch(`/api/palate/compare`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    palate_code_1: myPalateCode,
                    palate_code_2: friendCode.trim(),
                }),
            });

            if (!response.ok) throw new Error("Comparison failed");

            const data = await response.json();
            setComparisonResult(data);
        } catch (error) {
            console.error("Compare error:", error);
            setSnackbarMessage("Invalid palate code or comparison failed");
            setSnackbarOpen(true);
        } finally {
            setIsComparing(false);
        }
    };

    const handleFindIntersection = async () => {
        if (!friendCode.trim()) {
            setSnackbarMessage("Please enter a friend's palate code first");
            setSnackbarOpen(true);
            return;
        }

        setIsSearching(true);
        setIntersectionRecipes([]);

        try {
            const response = await fetch(`/api/palate/intersection`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    palate_code_1: myPalateCode,
                    palate_code_2: friendCode.trim(),
                    limit: 10,
                }),
            });

            if (!response.ok) throw new Error("Search failed");

            const data = await response.json();
            setIntersectionRecipes(data.recipes || []);
        } catch (error) {
            console.error("Intersection error:", error);
            setSnackbarMessage("Failed to find matching recipes");
            setSnackbarOpen(true);
        } finally {
            setIsSearching(false);
        }
    };

    return (
        <Box sx={{ minHeight: "100vh", background: "#fafafa", py: 3 }}>
            <Container maxWidth="md">
                {/* Header */}
                <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
                    <IconButton onClick={() => navigate("/recipes")} sx={{ mr: 2 }}>
                        <ArrowBack />
                    </IconButton>
                    <Typography variant="h5" fontWeight="600">
                        Compare Palates
                    </Typography>
                </Box>

                {/* My Palate Code */}
                <Paper elevation={0} sx={{ p: 3, borderRadius: 2, border: "1px solid #e0e0e0", mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Your Palate Code
                    </Typography>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        <Typography
                            variant="body2"
                            fontFamily="monospace"
                            sx={{
                                flex: 1,
                                p: 1.5,
                                backgroundColor: "#f5f5f5",
                                borderRadius: 1,
                                wordBreak: "break-all",
                            }}
                        >
                            {myPalateCode || "Loading..."}
                        </Typography>
                        <IconButton onClick={handleCopyCode} size="small" color={copied ? "success" : "default"}>
                            {copied ? <Check fontSize="small" /> : <ContentCopy fontSize="small" />}
                        </IconButton>
                    </Box>
                </Paper>

                {/* Friend's Code Input */}
                <Paper elevation={0} sx={{ p: 3, borderRadius: 2, border: "1px solid #e0e0e0", mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Friend's Palate Code
                    </Typography>
                    <TextField
                        fullWidth
                        size="small"
                        value={friendCode}
                        onChange={(e) => setFriendCode(e.target.value)}
                        placeholder="Paste your friend's palate code..."
                        sx={{ mb: 2 }}
                    />
                    <Box sx={{ display: "flex", gap: 1 }}>
                        <Button
                            variant="contained"
                            onClick={handleCompare}
                            disabled={isComparing || !friendCode.trim()}
                            startIcon={isComparing ? <CircularProgress size={16} /> : <Compare />}
                            disableElevation
                            sx={{ textTransform: "none" }}
                        >
                            Compare Similarity
                        </Button>
                        <Button
                            variant="outlined"
                            onClick={handleFindIntersection}
                            disabled={isSearching || !friendCode.trim()}
                            startIcon={isSearching ? <CircularProgress size={16} /> : <Search />}
                            sx={{ textTransform: "none" }}
                        >
                            Find Common Recipes
                        </Button>
                    </Box>
                </Paper>

                {/* Comparison Results */}
                {comparisonResult && (
                    <Paper elevation={0} sx={{ p: 3, borderRadius: 2, border: "1px solid #e0e0e0", mb: 3 }}>
                        <Typography variant="subtitle2" gutterBottom>
                            Similarity Score
                        </Typography>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
                            <Box sx={{ flex: 1 }}>
                                <LinearProgress
                                    variant="determinate"
                                    value={comparisonResult.overall_similarity || 0}
                                    sx={{ height: 8, borderRadius: 4 }}
                                />
                            </Box>
                            <Typography variant="h6" fontWeight="600">
                                {Math.round(comparisonResult.overall_similarity || 0)}%
                            </Typography>
                        </Box>

                        {comparisonResult.common_categories && comparisonResult.common_categories.length > 0 && (
                            <>
                                <Typography variant="caption" color="text.secondary">
                                    Categories you both enjoy
                                </Typography>
                                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 1 }}>
                                    {comparisonResult.common_categories.map((cat) => (
                                        <Chip key={cat} label={cat} size="small" color="primary" variant="outlined" />
                                    ))}
                                </Box>
                            </>
                        )}

                        {comparisonResult.different_categories && comparisonResult.different_categories.length > 0 && (
                            <Box sx={{ mt: 2 }}>
                                <Typography variant="caption" color="text.secondary">
                                    Different preferences
                                </Typography>
                                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 1 }}>
                                    {comparisonResult.different_categories.map((cat) => (
                                        <Chip key={cat} label={cat} size="small" variant="outlined" />
                                    ))}
                                </Box>
                            </Box>
                        )}
                    </Paper>
                )}

                {/* Intersection Recipes */}
                {intersectionRecipes.length > 0 && (
                    <Paper elevation={0} sx={{ p: 3, borderRadius: 2, border: "1px solid #e0e0e0" }}>
                        <Typography variant="subtitle2" gutterBottom>
                            Recipes You Both Might Enjoy
                        </Typography>
                        <Divider sx={{ my: 1 }} />
                        {intersectionRecipes.map((recipe, idx) => (
                            <Box
                                key={recipe.id || idx}
                                sx={{
                                    py: 1.5,
                                    borderBottom: idx < intersectionRecipes.length - 1 ? "1px solid #eee" : "none",
                                }}
                            >
                                <Typography variant="body2" fontWeight="500">
                                    {recipe.name}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    {recipe.category} • Match: {Math.round(recipe.score * 100)}%
                                </Typography>
                            </Box>
                        ))}
                    </Paper>
                )}
            </Container>

            <Snackbar open={snackbarOpen} autoHideDuration={3000} onClose={() => setSnackbarOpen(false)}>
                <Alert severity="error" onClose={() => setSnackbarOpen(false)}>
                    {snackbarMessage}
                </Alert>
            </Snackbar>
        </Box>
    );
}
