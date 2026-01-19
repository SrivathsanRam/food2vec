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

export default function SignUpPage() {
    const navigate = useNavigate();
    const [isSnackbarOpen, setIsSnackbarOpen] = useState(false);
    const [snackbarMessage, setSnackbarMessage] = useState("");
    const [formData, setFormData] = useState({
        name: "",
        password: "",
        confirmPassword: "",
    });
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
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

        if (!formData.confirmPassword) {
            newErrors.confirmPassword = "Please confirm your password";
        } else if (formData.password !== formData.confirmPassword) {
            newErrors.confirmPassword = "Passwords do not match";
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async () => {
        if (!validateForm()) return;

        setIsLoading(true);

        try {
            const response = await fetch(`/auth/signup`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ username: formData.name, password: formData.password }),
            });

            if (!response.ok) {
                setIsSnackbarOpen(true);
                setSnackbarMessage("Error signing up user");
            } else {
                // Store user info in cookies
                Cookies.set("username", formData.name, { expires: 1 });
                Cookies.set("isLoggedIn", "true", { expires: 1 });

                setSnackbarMessage("Account created!");
                setIsSnackbarOpen(true);

                // New users always go to onboarding
                setTimeout(() => {
                    navigate("/onboarding");
                }, 500);
            }
        } catch (error) {
            console.error("Signup error:", error);
            setIsSnackbarOpen(true);
            setSnackbarMessage("Network error. Please try again.");
        } finally {
            setIsLoading(false);
            setFormData({
                name: "",
                password: "",
                confirmPassword: "",
            });
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
                        Create Account
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                        Join our recipe community today
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

                        <TextField
                            fullWidth
                            label="Confirm Password"
                            name="confirmPassword"
                            type={showConfirmPassword ? "text" : "password"}
                            value={formData.confirmPassword}
                            onChange={handleChange}
                            error={!!errors.confirmPassword}
                            helperText={errors.confirmPassword}
                            placeholder="••••••••"
                            InputProps={{
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <Lock color="action" />
                                    </InputAdornment>
                                ),
                                endAdornment: (
                                    <InputAdornment position="end">
                                        <IconButton
                                            onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                            edge="end"
                                        >
                                            {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
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
                                    Creating Account...
                                </Box>
                            ) : (
                                "Sign Up"
                            )}
                        </Button>

                        <Divider sx={{ my: 1 }}>
                            <Typography variant="body2" color="text.secondary">
                                or
                            </Typography>
                        </Divider>

                        <Typography variant="body2" align="center" sx={{ mt: 2 }}>
                            Already have an account?{" "}
                            <Link href="/login" underline="hover" fontWeight={600}>
                                Sign in
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
