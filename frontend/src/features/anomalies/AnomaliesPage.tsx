import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import { Card, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { formatDistanceToNow, format } from 'date-fns';
import { Search, AlertOctagon, Filter, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export function AnomaliesPage() {
  const navigate = useNavigate();
  const [anomalies, setAnomalies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterSeverity, setFilterSeverity] = useState('all');

  useEffect(() => {
    const fetchAnomalies = async () => {
      setLoading(true);
      try {
        const data = await api.diagnostics.listFleetAnomalies();
        setAnomalies(data);
      } catch (error) {
        console.error("Failed to fetch anomalies", error);
      } finally {
        setLoading(false);
      }
    };
    fetchAnomalies();
  }, []);

  const filteredAnomalies = anomalies.filter(a => filterSeverity === 'all' || a.severity === filterSeverity);
  
  const activeCount = anomalies.filter(a => a.status === 'active').length;
  const criticalCount = anomalies.filter(a => a.severity === 'critical').length;

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500 pb-12">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-primary flex items-center gap-2">
          <AlertOctagon className="w-6 h-6 text-red-500" /> Anomaly Detection
        </h1>
        <p className="text-sm text-slate-400 mt-1">Centralized workspace for investigating irregular sensor behavior and structural anomalies.</p>
      </div>

      {/* KPI Header */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-l-4 border-l-slate-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2">Total Detected (30d)</div>
            <div className="text-3xl font-mono text-primary">{anomalies.length}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-amber-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2">Unresolved Anomalies</div>
            <div className="text-3xl font-mono text-amber-400">{activeCount}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-red-500 bg-red-950/10">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 text-red-500">Critical Severity</div>
            <div className="text-3xl font-mono text-red-500">{criticalCount}</div>
          </CardContent>
        </Card>
      </div>

      {/* Anomaly Table */}
      <Card>
        <CardContent className="p-0">
          <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/50 flex-wrap gap-4">
            <div className="relative">
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input 
                type="text" 
                placeholder="Search anomalies or machines..." 
                className="bg-slate-950 border border-slate-700 rounded-md pl-9 pr-3 py-2 text-sm focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent text-primary w-72 transition-shadow"
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-slate-500" />
              <select 
                className="bg-slate-950 border border-slate-700 rounded text-sm px-3 py-1.5 text-primary focus:outline-none focus:border-accent"
                value={filterSeverity}
                onChange={(e) => setFilterSeverity(e.target.value)}
              >
                <option value="all">All Severities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-slate-400 uppercase bg-slate-950/50 border-b border-slate-800">
                <tr>
                  <th className="px-6 py-4 font-medium tracking-wider">Severity</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Anomaly & Context</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Machine Origin</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Score</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Status</th>
                  <th className="px-6 py-4 font-medium tracking-wider text-right">Detected</th>
                  <th className="px-6 py-4 font-medium tracking-wider text-right">Investigate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {loading ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-slate-500 font-mono">
                      [LOADING_ANOMALY_INDEX...]
                    </td>
                  </tr>
                ) : filteredAnomalies.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-slate-500 italic">
                      No anomalies match the current filters.
                    </td>
                  </tr>
                ) : (
                  filteredAnomalies.map((anom) => (
                    <tr key={anom.id} className="hover:bg-slate-800/20 transition-colors group">
                      <td className="px-6 py-4">
                        <div className="flex items-center justify-center w-8 h-8 rounded bg-slate-900 border border-slate-700">
                          <div className={`w-3 h-3 rounded-full ${
                            anom.severity === 'critical' ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]' : 
                            anom.severity === 'high' ? 'bg-amber-500' : 
                            anom.severity === 'medium' ? 'bg-amber-400' : 'bg-slate-400'
                          }`} />
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="font-medium text-primary mb-1">{anom.description}</div>
                        <div className="text-xs text-slate-400 font-mono">Sensor: {anom.sensor} <span className="mx-2">•</span> ID: {anom.id}</div>
                      </td>
                      <td className="px-6 py-4">
                        <div 
                          className="font-medium text-primary hover:text-accent cursor-pointer transition-colors"
                          onClick={() => navigate(`/machines/${anom.machineId}`)}
                        >
                          {anom.machineName}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">{anom.machineId}</div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="font-mono text-primary">{anom.score.toFixed(2)}</div>
                        <div className="w-16 h-1 bg-slate-800 rounded-full mt-1.5 overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${anom.score > 0.8 ? 'bg-red-500' : anom.score > 0.6 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                            style={{ width: `${anom.score * 100}%` }}
                          />
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <Badge variant={anom.status === 'active' ? 'critical' : anom.status === 'investigating' ? 'warning' : 'healthy'}>
                          {anom.status.toUpperCase()}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="text-slate-300">{formatDistanceToNow(new Date(anom.timestamp), { addSuffix: true })}</div>
                        <div className="text-xs text-slate-500 font-mono mt-0.5">{format(new Date(anom.timestamp), 'MMM dd, HH:mm')}</div>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex flex-col gap-1 items-end">
                          <button 
                            onClick={() => navigate(`/diagnostics?machine=${anom.machineId}`)}
                            className="inline-flex items-center justify-center px-2 py-1 rounded text-xs font-medium text-accent border border-accent/20 hover:bg-accent/10 transition-colors"
                          >
                            Diagnostics
                          </button>
                          <button 
                            onClick={() => navigate(`/machines/${anom.machineId}`)}
                            className="inline-flex items-center justify-center p-1 rounded text-slate-400 hover:text-accent hover:bg-accent/10 transition-colors"
                            title="Investigate Machine"
                          >
                            <ArrowRight className="w-4 h-4" />
                          </button>
                        </div>
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
