import { useEffect, useState } from 'react';
import { Menu, Bell, User, AlertCircle } from 'lucide-react';
import { useUIStore } from '../store/uiStore';
import { subscribeSystemStatus } from '../services/api';
import type { SystemStatus } from '../services/api';

export function Topbar() {
  const { toggleSidebar } = useUIStore();
  const [status, setStatus] = useState<SystemStatus>({ mode: 'mock', label: 'DEVELOPMENT MOCK' });

  useEffect(() => {
    const unsubscribe = subscribeSystemStatus(setStatus);
    return () => unsubscribe();
  }, []);

  return (
    <header className="h-16 bg-slate-950 border-b border-slate-800 flex items-center justify-between px-4 shrink-0 z-10 relative shadow-sm">
      <div className="flex items-center gap-4">
        <button 
          onClick={toggleSidebar}
          className="p-2 -ml-2 rounded-md text-slate-400 hover:text-primary hover:bg-slate-800 transition-colors"
          aria-label="Toggle sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div className="h-4 w-px bg-slate-800 mx-2 hidden sm:block"></div>
        <div className="hidden sm:flex items-center text-sm font-medium text-slate-400 gap-3">
          <span>Industrial Intelligence Engine</span>
          <span className="text-slate-700">•</span>
          
          {/* Provider System Status Indicator */}
          {status.mode === 'live_rag' && (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              LIVE ML + REAL RAG
            </span>
          )}

          {status.mode === 'live_fallback' && (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-sky-950/80 text-sky-400 border border-sky-800/80 shadow-[0_0_10px_rgba(14,165,233,0.15)]">
              <span className="w-2 h-2 rounded-full bg-sky-500 animate-pulse"></span>
              LIVE ML + GROUNDED FALLBACK
            </span>
          )}

          {status.mode === 'live_no_rag' && (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-violet-950/80 text-violet-400 border border-violet-800/80">
              <span className="w-2 h-2 rounded-full bg-violet-500 animate-pulse"></span>
              LIVE ML (RAG UNAVAILABLE)
            </span>
          )}

          {status.mode === 'mock' && (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-slate-800/90 text-slate-300 border border-slate-700">
              <span className="w-2 h-2 rounded-full bg-blue-400"></span>
              DEVELOPMENT MOCK
            </span>
          )}

          {status.mode === 'error' && (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-amber-950/90 text-amber-400 border border-amber-800" title={status.message}>
              <AlertCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              BACKEND UNREACHABLE
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-4">
        {status.mode === 'error' && (
          <span className="hidden md:inline-block text-xs font-mono text-amber-400 bg-amber-950/40 px-3 py-1 rounded border border-amber-800/50">
            Live ML backend unavailable. Showing development data.
          </span>
        )}

        <button className="relative p-2 rounded-md text-slate-400 hover:text-primary hover:bg-slate-800 transition-colors">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-red-500 border-2 border-slate-950"></span>
        </button>
        <div className="h-8 w-8 rounded bg-slate-800 border border-slate-700 flex items-center justify-center overflow-hidden cursor-pointer hover:border-slate-500 transition-colors">
          <User className="w-4 h-4 text-slate-400" />
        </div>
      </div>
    </header>
  );
}
