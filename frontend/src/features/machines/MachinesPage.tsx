import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { Machine } from '../../types/models';
import { Card, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { formatDistanceToNow } from 'date-fns';
import { Search } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export function MachinesPage() {
  const navigate = useNavigate();
  const [machines, setMachines] = useState<Machine[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMachines = async () => {
      setLoading(true);
      try {
        const data = await api.machines.list();
        setMachines(data);
      } catch (error) {
        console.error("Failed to fetch machines", error);
      } finally {
        setLoading(false);
      }
    };
    fetchMachines();
  }, []);

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-primary">Machines</h1>
        <p className="text-sm text-slate-400 mt-1">Fleet overview and real-time status.</p>
      </div>

      <Card>
        <CardContent className="p-0">
          <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
            <div className="relative">
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input 
                type="text" 
                placeholder="Search machines..." 
                className="bg-slate-950 border border-slate-700 rounded-md pl-9 pr-3 py-2 text-sm focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent text-primary w-72 transition-shadow"
              />
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-slate-400 uppercase bg-slate-950/50 border-b border-slate-800">
                <tr>
                  <th className="px-6 py-4 font-medium tracking-wider">Machine</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Location</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Status</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Est. RUL</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Risk Level</th>
                  <th className="px-6 py-4 font-medium tracking-wider text-right">Last Sync</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {loading ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-slate-500 font-mono">
                      [LOADING_FLEET_DATA...]
                    </td>
                  </tr>
                ) : (
                  machines.map((machine) => (
                    <tr 
                      key={machine.id} 
                      onClick={() => navigate(`/machines/${machine.id}`)}
                      className="hover:bg-slate-800/20 transition-colors group cursor-pointer"
                    >
                      <td className="px-6 py-4">
                        <div className="font-medium text-primary group-hover:text-accent transition-colors">{machine.name}</div>
                        <div className="text-xs text-slate-500 mt-0.5">{machine.id} • {machine.type}</div>
                      </td>
                      <td className="px-6 py-4 text-slate-400">{machine.location}</td>
                      <td className="px-6 py-4">
                        <Badge variant={machine.status as any}>{machine.status}</Badge>
                      </td>
                      <td className="px-6 py-4">
                        <div className="font-mono text-primary">{machine.rulDays} <span className="text-xs text-slate-500 font-sans">days</span></div>
                        <div className="text-xs text-slate-500 mt-0.5">± {100 - machine.rulConfidence}% error</div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${
                                machine.failureRisk > 75 ? 'bg-red-500' : 
                                machine.failureRisk > 30 ? 'bg-amber-500' : 'bg-emerald-500'
                              }`}
                              style={{ width: `${machine.failureRisk}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs text-slate-400">{machine.failureRisk}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right text-slate-400 font-mono text-xs">
                        {formatDistanceToNow(new Date(machine.lastUpdated), { addSuffix: true })}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
