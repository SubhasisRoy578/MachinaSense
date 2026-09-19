import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { DashboardLayout } from './layouts/DashboardLayout';

// Route-based code splitting for optimal production performance
const OverviewPage = lazy(() => import('./features/overview/OverviewPage').then(m => ({ default: m.OverviewPage })));
const MachinesPage = lazy(() => import('./features/machines/MachinesPage').then(m => ({ default: m.MachinesPage })));
const MachineDetailPage = lazy(() => import('./features/machines/MachineDetailPage').then(m => ({ default: m.MachineDetailPage })));
const SensorsPage = lazy(() => import('./features/sensors/SensorsPage').then(m => ({ default: m.SensorsPage })));
const AnomaliesPage = lazy(() => import('./features/anomalies/AnomaliesPage').then(m => ({ default: m.AnomaliesPage })));
const PredictionsPage = lazy(() => import('./features/predictions/PredictionsPage').then(m => ({ default: m.PredictionsPage })));
const MaintenancePage = lazy(() => import('./features/maintenance/MaintenancePage').then(m => ({ default: m.MaintenancePage })));
const DiagnosticsPage = lazy(() => import('./features/diagnostics/DiagnosticsPage').then(m => ({ default: m.DiagnosticsPage })));
const DiagnosticCaseView = lazy(() => import('./features/diagnostics/DiagnosticCaseView').then(m => ({ default: m.DiagnosticCaseView })));
const KnowledgeBasePage = lazy(() => import('./features/knowledge/KnowledgeBasePage').then(m => ({ default: m.KnowledgeBasePage })));
const AnalyticsPage = lazy(() => import('./features/analytics/AnalyticsPage').then(m => ({ default: m.AnalyticsPage })));
const SettingsPage = lazy(() => import('./features/settings/SettingsPage').then(m => ({ default: m.SettingsPage })));

function RouteLoadingFallback() {
  return (
    <div className="flex items-center justify-center min-h-[400px] text-slate-500 font-mono text-sm">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        <span className="tracking-wider text-xs animate-pulse">[LOADING_TELEMETRY_VIEW...]</span>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<RouteLoadingFallback />}>
        <Routes>
          <Route path="/" element={<DashboardLayout />}>
            <Route index element={<OverviewPage />} />
            <Route path="machines" element={<MachinesPage />} />
            <Route path="machines/:id" element={<MachineDetailPage />} />
            <Route path="sensors" element={<SensorsPage />} />
            <Route path="anomalies" element={<AnomaliesPage />} />
            <Route path="predictions" element={<PredictionsPage />} />
            <Route path="maintenance" element={<MaintenancePage />} />
            <Route path="diagnostics" element={<DiagnosticsPage />} />
            <Route path="diagnostics/:id" element={<DiagnosticCaseView />} />
            <Route path="knowledge-base" element={<KnowledgeBasePage />} />
            <Route path="analytics" element={<AnalyticsPage />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
