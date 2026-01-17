import { Route, Routes } from "react-router-dom";
import SignUpPage from "../components/SignUpPage";
import LandingPage from "../components/LandingPage";
import LoginPage from "../components/LoginPage";

const AppRoutes = () => {
    return (
        <Routes>
            <Route path="/" element={<SignUpPage />} />
            <Route path="/recipes" element={<LandingPage />} />
            <Route path="/login" element={<LoginPage />} />
        </Routes>
    );
};

export default AppRoutes;
