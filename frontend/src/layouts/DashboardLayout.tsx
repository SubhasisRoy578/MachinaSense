
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';
import SplashCursor from '../components/SplashCursor';

export function DashboardLayout() {
  return (
    <div className="flex h-screen w-full bg-slate-950 text-slate-200 overflow-hidden font-sans relative">
      {/* Background Splash Cursor */}
      <SplashCursor 
        DENSITY_DISSIPATION={9}
        VELOCITY_DISSIPATION={2}
        PRESSURE={0.1}
        CURL={3}
        SPLAT_RADIUS={0.2}
        SPLAT_FORCE={6000}
        COLOR_UPDATE_SPEED={10}
        SHADING={true}
        RAINBOW_MODE={true}
        COLOR="#06b6d4" // Cyan industrial accent, though rainbow overrides it
      />
      
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0 z-10 relative">
        <Topbar />
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 custom-scrollbar">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
