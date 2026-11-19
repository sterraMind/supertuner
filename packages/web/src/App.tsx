import { Link, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { EmptyState } from "./components/EmptyState";
import { Icon } from "./components/icons";
import { Dashboard } from "./pages/Dashboard";
import { Recipes } from "./pages/Recipes";
import { Planner } from "./pages/Planner";
import { Runs } from "./pages/Runs";
import { RunDetail } from "./pages/RunDetail";
import { Plugins } from "./pages/Plugins";
import { Settings } from "./pages/Settings";

function NotFound() {
  return (
    <EmptyState
      icon="warning"
      title="This route does not exist"
      description="The URL is unknown to the app. Head back to the dashboard or open the planner."
      action={
        <Link className="btn btn-primary btn-sm" to="/">
          <Icon name="dashboard" size={13} />
          Dashboard
        </Link>
      }
    />
  );
}

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="recipes" element={<Recipes />} />
        <Route path="planner" element={<Planner />} />
        <Route path="runs" element={<Runs />} />
        <Route path="runs/:id" element={<RunDetail />} />
        <Route path="plugins" element={<Plugins />} />
        <Route path="settings" element={<Settings />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}