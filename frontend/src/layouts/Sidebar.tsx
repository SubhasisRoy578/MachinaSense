
import { NavLink } from 'react-router-dom';
import { 
  Activity, 
  Cpu, 
  AlertTriangle, 
  TrendingDown, 
  Wrench, 
  BrainCircuit, 
  BookOpen, 
  BarChart2, 
  Settings 
} from 'lucide-react';
import { cn } from '../utils/cn';
import { useUIStore } from '../store/uiStore';

const navigation = [
  { name: 'Overview', to: '/', icon: Activity },
  { name: 'Machines', to: '/machines', icon: Cpu },
  { name: 'Sensors', to: '/sensors', icon: TrendingDown },
  { name: 'Anomalies', to: '/anomalies', icon: AlertTriangle },
  { name: 'Predictions', to: '/predictions', icon: BarChart2 },
  { name: 'Maintenance', to: '/maintenance', icon: Wrench },
  { name: 'AI Diagnostics', to: '/diagnostics', icon: BrainCircuit },
  { name: 'Knowledge Base', to: '/knowledge-base', icon: BookOpen },
  { name: 'Analytics', to: '/analytics', icon: BarChart2 },
];

export function Sidebar() {
  const { isSidebarOpen } = useUIStore();

  return (
    <aside 
      className={cn(
        "flex flex-col bg-slate-900 border-r border-slate-800 transition-all duration-300 z-10",
        isSidebarOpen ? "w-64" : "w-16"
      )}
    >
      <div className="flex h-16 shrink-0 items-center px-4 border-b border-slate-800 bg-slate-950/50">
        <div className="flex items-center gap-3 overflow-hidden whitespace-nowrap">
          <div className="w-8 h-8 rounded bg-accent/20 border border-accent/50 flex items-center justify-center shrink-0">
            <Cpu className="w-5 h-5 text-accent" />
          </div>
          <span className={cn(
            "font-bold text-primary tracking-wide transition-opacity duration-300",
            !isSidebarOpen && "opacity-0"
          )}>
            MACHINASENSE
          </span>
        </div>
      </div>

      <nav className="flex flex-1 flex-col overflow-y-auto px-2 py-4 gap-1 custom-scrollbar">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.to}
            className={({ isActive }) =>
              cn(
                "group flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors",
                isActive
                  ? "bg-slate-800/80 text-accent"
                  : "text-slate-400 hover:bg-slate-800/40 hover:text-primary"
              )
            }
            title={!isSidebarOpen ? item.name : undefined}
          >
            <item.icon className={cn("mr-3 h-5 w-5 shrink-0")} aria-hidden="true" />
            <span className={cn(
              "whitespace-nowrap transition-opacity duration-300",
              !isSidebarOpen && "opacity-0 hidden"
            )}>
              {item.name}
            </span>
          </NavLink>
        ))}
        
        <div className="mt-auto">
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              cn(
                "group flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors",
                isActive
                  ? "bg-slate-800/80 text-accent"
                  : "text-slate-400 hover:bg-slate-800/40 hover:text-primary"
              )
            }
            title={!isSidebarOpen ? "Settings" : undefined}
          >
            <Settings className={cn("mr-3 h-5 w-5 shrink-0")} aria-hidden="true" />
            <span className={cn(
              "whitespace-nowrap transition-opacity duration-300",
              !isSidebarOpen && "opacity-0 hidden"
            )}>
              Settings
            </span>
          </NavLink>
        </div>
      </nav>
    </aside>
  );
}
