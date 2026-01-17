import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import SignUpPage from "./components/SignUpPage";
import LoginPage from "./components/LoginPage";
import LandingPage from "./components/LandingPage";
import OnboardingPage from "./components/OnboardingPage";
import ComparePage from "./components/ComparePage";
import "./App.css";

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<SignUpPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/recipes" element={<LandingPage />} />
                <Route path="/onboarding" element={<OnboardingPage />} />
                <Route path="/compare" element={<ComparePage />} />
            </Routes>
        </Router>
    );
}

export default App;
