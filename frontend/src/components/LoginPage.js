import React, { useState } from "react";
import {
    Box,
    TextField,
    Button,
    Typography,
    Container,
    Paper,
    InputAdornment,
    IconButton,
    Divider,
    Link,
    CircularProgress,
    Avatar,
    Snackbar,
} from "@mui/material";
import { Visibility, VisibilityOff, Lock, Person, Restaurant } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import Cookies from "js-cookie";

const baseURL = process.env.REACT_APP_API_BASE_URL;

export default function LoginPage() {
    const navigate = useNavigate();
    const [isSnackbarOpen, setIsSnackbarOpen] = useState(false);
    const [snackbarMessage, setSnackbarMessage] = useState("");
    const [formData, setFormData] = useState({
        name: "",
        password: "",
    });
    const [showPassword, setShowPassword] = useState(false);
    const [errors, setErrors] = useState({});
    const [isLoading, setIsLoading] = useState(false);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData((prev) => ({
            ...prev,
            [name]: value,
        }));
        if (errors[name]) {
            setErrors((prev) => ({
                ...prev,
                [name]: "",
            }));
        }
    };

    const validateForm = () => {
        const newErrors = {};

        if (!formData.name.trim()) {
            newErrors.name = "Name is required";
        }

        if (!formData.password) {
            newErrors.password = "Password is required";
        } else if (formData.password.length < 8) {
            newErrors.password = "Password must be at least 8 characters";
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const checkOnboardingStatus = async (username) => {
        try {
            const response = await fetch(
                `http://localhost:5000/api/palate/check?username=${encodeURIComponent(username)}`
            );
            const data = await response.json();
            return data.is_onboarded;
        } catch (error) {
            console.error("Check onboarding error:", error);
            return false;
        }
    };

    const handleSubmit = async () => {
        if (!validateForm()) return;

        setIsLoading(true);

        try {
            const response = await fetch(`${baseURL}/auth/login`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    username: formData.name,
                    password: formData.password,
                }),
            });

            if (!response.ok) {
                setIsSnackbarOpen(true);
                setSnackbarMessage("Error logging in");
            } else {
                // Store user info in cookies (expires in 1 day)
                Cookies.set("username", formData.name, { expires: 1 });
                Cookies.set("isLoggedIn", "true", { expires: 1 });

                // Check if user has been onboarded
                const isOnboarded = await checkOnboardingStatus(formData.name);

                setSnackbarMessage("Login successful!");
                setIsSnackbarOpen(true);

                setTimeout(() => {
                    if (isOnboarded) {
                        Cookies.set("isOnboarded", "true", { expires: 365 });
                        navigate("/recipes");
                    } else {
                        navigate("/onboarding");
                    }
                }, 500);
            }
        } catch (error) {
            console.error("Login error:", error);
            setIsSnackbarOpen(true);
            setSnackbarMessage("Network error. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleSnackbarClose = () => {
        setIsSnackbarOpen(false);
    };

    return (
        <Box
            sx={{
                minHeight: "100vh",
                background: "linear-gradient(135deg, #fff5f0 0%, #ffffff 50%, #fffaf0 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                py: 4,
            }}
        >
            <Container maxWidth="sm">
                <Box sx={{ textAlign: "center", mb: 4 }}>
                    <Avatar
                        sx={{
                            width: 64,
                            height: 64,
                            bgcolor: "primary.main",
                            margin: "0 auto",
                            mb: 2,
                        }}
                    >
                        <Restaurant sx={{ fontSize: 32 }} />
                    </Avatar>
                    <Typography variant="h4" component="h1" fontWeight="bold" gutterBottom>
                        Log in to Account
                    </Typography>
                </Box>

                <Paper
                    elevation={3}
                    sx={{
                        p: 4,
                        borderRadius: 3,
                    }}
                >
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                        <TextField
                            fullWidth
                            label="Full Name"
                            name="name"
                            value={formData.name}
                            onChange={handleChange}
                            error={!!errors.name}
                            helperText={errors.name}
                            placeholder="John Doe"
                            InputProps={{
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <Person color="action" />
                                    </InputAdornment>
                                ),
                            }}
                        />

                        <TextField
                            fullWidth
                            label="Password"
                            name="password"
                            type={showPassword ? "text" : "password"}
                            value={formData.password}
                            onChange={handleChange}
                            error={!!errors.password}
                            helperText={errors.password}
                            placeholder="••••••••"
                            InputProps={{
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <Lock color="action" />
                                    </InputAdornment>
                                ),
                                endAdornment: (
                                    <InputAdornment position="end">
                                        <IconButton onClick={() => setShowPassword(!showPassword)} edge="end">
                                            {showPassword ? <VisibilityOff /> : <Visibility />}
                                        </IconButton>
                                    </InputAdornment>
                                ),
                            }}
                        />

                        <Button
                            fullWidth
                            variant="contained"
                            size="large"
                            onClick={handleSubmit}
                            disabled={isLoading}
                            sx={{
                                py: 1.5,
                                textTransform: "none",
                                fontSize: "1rem",
                                fontWeight: 600,
                            }}
                        >
                            {isLoading ? (
                                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                                    <CircularProgress size={20} color="inherit" />
                                    Logging in...
                                </Box>
                            ) : (
                                "Sign In"
                            )}
                        </Button>

                        <Divider sx={{ my: 1 }}>
                            <Typography variant="body2" color="text.secondary">
                                or
                            </Typography>
                        </Divider>

                        <Typography variant="body2" align="center" sx={{ mt: 2 }}>
                            Don't have an account?{" "}
                            <Link href="/" underline="hover" fontWeight={600}>
                                Sign up
                            </Link>
                        </Typography>

                        <Snackbar
                            open={isSnackbarOpen}
                            autoHideDuration={2000}
                            onClose={handleSnackbarClose}
                            message={snackbarMessage}
                        />
                    </Box>
                </Paper>
            </Container>
        </Box>
    );
}
