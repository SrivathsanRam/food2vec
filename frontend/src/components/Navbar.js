import React from "react";
import Cookies from "js-cookie";
import { useNavigate } from "react-router-dom";
import LogoutIcon from "@mui/icons-material/Logout";
import { AppBar, Toolbar, Typography, Button, Box } from "@mui/material";

const Navbar = () => {
    const user = Cookies.get("username");
    console.log("user", user);
    const navigate = useNavigate();

    const handleLogout = () => {
        // Fix: Use the imported js-cookie library for cleaner removal
        Cookies.remove("username");
        Cookies.remove("isLoggedIn");

        localStorage.clear();
        navigate("/login");
    };

    return (
        // AppBar provides the blue header bar background
        <AppBar position="static" sx={{ backgroundColor: "white", color: "black" }}>
            <Toolbar>
                {/* flexGrow: 1 makes this text take up all available space, 
                  pushing everything after it to the far right.
                */}
                <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                    Word2Vec
                </Typography>

                {/* Right side content */}
                <Box display="flex" alignItems="center" gap={2}>
                    {/* Wrapped in a Button to make it clickable and accessible */}
                    {user !== undefined && (
                        <Button onClick={handleLogout} endIcon={<LogoutIcon />}>
                            Logout
                        </Button>
                    )}
                </Box>
            </Toolbar>
        </AppBar>
    );
};

export default Navbar;
