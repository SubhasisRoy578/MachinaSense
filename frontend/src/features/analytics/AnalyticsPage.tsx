import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, subscribeSystemStatus } from '../../services/api';
import type { SystemStatus } from '../../services/api';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { 
  Cpu, Activity, ShieldCheck, Database, Sliders, Layers, 
  BarChart3, ArrowRight
} from 'lucide-react';

interface FleetAnalytics {
  total_machines: number;
  healthy: number;
  warning: number;
  critical: number;
  offline: number;
  average_rul: number | null;
  total_anomalies: number;
  pending_maintenance: number;
  fleet: any[];
}

export function AnalyticsPage() {
  const navigate = useNavigate();
  const [modelData, setModelData] = useState<any>(null);
  const [fleetData, setFleetData] = useState<FleetAnalytics | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({ mode: 'error', label: 'CONNECTING' });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    const unsubscribe = subscribeSystemStatus(setSystemStatus);
    const fetchData = async () => {
      setLoading(true);
      setError('');
      try {
        const [models, fleet] = await Promise.all([
          api.analytics.getModels().catch(err => {
            console.warn("Could not fetch models:", err);
            return null;
          }),
          api.analytics.getFleet().catch(err => {
            console.warn("Could not fetch fleet analytics:", err);
            return null;
          }),
        ]);
        setModelData(models);
        setFleetData(fleet);
      } catch (err: any) {
        console.error("Failed loading analytics data:", err);
        setError(err.message || 'Analytics data is currently unavailable.');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
    return () => unsubscribe();
  }, []);

  const evalInfo = modelData?.evaluation || {};
  const metaInfo = modelData?.metadata || {};
  const rf = evalInfo.rul_models?.random_forest;
  const lstm = evalInfo.rul_models?.lstm;

  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-500 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-primary flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-accent" /> Fleet Intelligence & ML Model Analytics
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real user operational fleet telemetry combined with NASA C-MAPSS FD001 ML benchmark validation.
          </p>
        </div>
        <Badge 
          variant={systemStatus.mode.startsWith('live') ? 'healthy' : (systemStatus.mode === 'error' ? 'critical' : 'default')}
          className="w-fit text-xs px-3 py-1.5 font-mono uppercase tracking-wider"
        >
          {systemStatus.mode.startsWith('live') ? '⚡ LIVE ML FASTAPI' : '⚠️ BACKEND UNREACHABLE'}
        </Badge>
      </div>

      {systemStatus.mode === 'error' && (
        <div className="bg-amber-950/40 border border-amber-500/50 rounded-lg p-4 text-amber-300 text-sm flex items-center gap-3">
          <Activity className="w-5 h-5 text-amber-400 shrink-0" />
          <div>
            <span className="font-semibold block">Live ML Backend Unavailable</span>
            <p className="text-xs text-amber-400/90">Unable to establish connection to the FastAPI backend service. Live metrics cannot be loaded.</p>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* SECTION 1: USER FLEET OPERATIONAL ANALYTICS */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold text-primary flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-400" /> User Fleet Operational Statistics
            </h2>
            <p className="text-xs text-slate-400">
              Aggregated directly from your authenticated assets, telemetry cycles, predictions, and anomalies in PostgreSQL.
            </p>
          </div>
          <Badge variant="outline" className="text-slate-400 text-xs font-mono">
            Source: Authenticated Fleet Database
          </Badge>
        </div>

        {/* Fleet KPI Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <Card className="border-l-4 border-l-slate-500">
            <CardContent className="py-4">
              <div className="text-[11px] text-slate-400 font-bold uppercase tracking-wider mb-1">Total Assets</div>
              <div className="text-2xl font-mono font-bold text-primary">{loading ? '...' : (fleetData?.total_machines ?? 0)}</div>
            </CardContent>
          </Card>
          <Card className="border-l-4 border-l-emerald-500 bg-emerald-950/10">
            <CardContent className="py-4">
              <div className="text-[11px] text-emerald-400 font-bold uppercase tracking-wider mb-1">Healthy</div>
              <div className="text-2xl font-mono font-bold text-emerald-400">{loading ? '...' : (fleetData?.healthy ?? 0)}</div>
            </CardContent>
          </Card>
          <Card className="border-l-4 border-l-amber-500 bg-amber-950/10">
            <CardContent className="py-4">
              <div className="text-[11px] text-amber-400 font-bold uppercase tracking-wider mb-1">Warning</div>
              <div className="text-2xl font-mono font-bold text-amber-400">{loading ? '...' : (fleetData?.warning ?? 0)}</div>
            </CardContent>
          </Card>
          <Card className="border-l-4 border-l-red-500 bg-red-950/10">
            <CardContent className="py-4">
              <div className="text-[11px] text-red-400 font-bold uppercase tracking-wider mb-1">Critical</div>
              <div className="text-2xl font-mono font-bold text-red-400">{loading ? '...' : (fleetData?.critical ?? 0)}</div>
            </CardContent>
          </Card>
          <Card className="border-l-4 border-l-cyan-500 bg-cyan-950/10">
            <CardContent className="py-4">
              <div className="text-[11px] text-cyan-400 font-bold uppercase tracking-wider mb-1">Mean Est. RUL</div>
              <div className="text-2xl font-mono font-bold text-cyan-400">
                {loading ? '...' : (fleetData?.average_rul !== null && fleetData?.average_rul !== undefined ? `${fleetData.average_rul}` : 'N/A')}
              </div>
              <div className="text-[10px] text-slate-500">Cycles</div>
            </CardContent>
          </Card>
          <Card className="border-l-4 border-l-purple-500 bg-purple-950/10">
            <CardContent className="py-4">
              <div className="text-[11px] text-purple-400 font-bold uppercase tracking-wider mb-1">Active Anomalies</div>
              <div className="text-2xl font-mono font-bold text-purple-400">{loading ? '...' : (fleetData?.total_anomalies ?? 0)}</div>
            </CardContent>
          </Card>
        </div>

        {/* Fleet Asset Breakdown or Empty State */}
        {!loading && (!fleetData || fleetData.total_machines === 0) ? (
          <Card className="p-8 text-center bg-slate-900/40 border-dashed border-slate-800">
            <Activity className="w-8 h-8 text-slate-500 mx-auto mb-3" />
            <h3 className="text-sm font-semibold text-slate-300">No User Assets Registered</h3>
            <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto">
              You do not have any machines in your account yet. Register an engine asset and upload telemetry data to view operational fleet analytics.
            </p>
            <button
              onClick={() => navigate('/machines')}
              className="mt-4 inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-accent text-slate-950 text-xs font-semibold hover:bg-accent/90 transition-colors"
            >
              Go to Machines <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </Card>
        ) : (
          fleetData && fleetData.fleet.length > 0 && (
            <Card className="bg-slate-900/40 border-slate-800">
              <CardHeader className="py-3 px-4 border-b border-slate-800">
                <CardTitle className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Fleet Assets Operational Status
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="text-[11px] text-slate-400 uppercase bg-slate-950/60 border-b border-slate-800">
                      <tr>
                        <th className="px-4 py-3 font-medium">Status</th>
                        <th className="px-4 py-3 font-medium">Machine ID & Name</th>
                        <th className="px-4 py-3 font-medium">Type</th>
                        <th className="px-4 py-3 font-medium">Current Cycle</th>
                        <th className="px-4 py-3 font-medium">Predicted RUL</th>
                        <th className="px-4 py-3 font-medium">Failure Risk</th>
                        <th className="px-4 py-3 font-medium text-right">Inspect</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/40 font-mono">
                      {fleetData.fleet.map((m: any) => (
                        <tr key={m.id} className="hover:bg-slate-800/30 transition-colors">
                          <td className="px-4 py-3">
                            <Badge variant={m.status === 'critical' ? 'critical' : m.status === 'warning' ? 'warning' : m.status === 'healthy' ? 'healthy' : 'outline'}>
                              {m.status.toUpperCase()}
                            </Badge>
                          </td>
                          <td className="px-4 py-3 font-sans">
                            <div className="font-semibold text-primary">{m.name}</div>
                            <div className="text-[10px] text-slate-500 font-mono">{m.id}</div>
                          </td>
                          <td className="px-4 py-3 text-slate-400">{m.type}</td>
                          <td className="px-4 py-3 text-slate-300">{m.currentCycle ?? 0}</td>
                          <td className="px-4 py-3 font-bold text-accent">
                            {m.predictedRul !== null ? `${m.predictedRul} cycles` : 'No telemetry'}
                          </td>
                          <td className="px-4 py-3 text-slate-300">
                            {m.failureRisk !== null ? `${m.failureRisk}%` : 'N/A'}
                          </td>
                          <td className="px-4 py-3 text-right font-sans">
                            <button
                              onClick={() => navigate(`/machines/${m.id}`)}
                              className="text-xs text-accent hover:underline inline-flex items-center gap-1"
                            >
                              View <ArrowRight className="w-3 h-3" />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )
        )}
      </div>

      {/* SECTION 2: NASA C-MAPSS FD001 ML BENCHMARK PERFORMANCE */}
      <div className="space-y-4 pt-4 border-t border-slate-800">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold text-primary flex items-center gap-2">
              <Cpu className="w-5 h-5 text-accent" /> NASA C-MAPSS FD001 ML Benchmark Performance
            </h2>
            <p className="text-xs text-slate-400">
              Evaluation metrics evaluated on the NASA C-MAPSS FD001 test split. Strictly distinct from user operational fleet data.
            </p>
          </div>
          <Badge variant="outline" className="text-accent text-xs font-mono">
            Offline Model Evaluation
          </Badge>
        </div>

        {/* Model Performance Comparison Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* PyTorch LSTM Card */}
          <Card className="border-l-4 border-l-emerald-500 bg-slate-900/80">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-base font-semibold flex items-center gap-2 text-emerald-400">
                <Layers className="w-5 h-5 text-emerald-400" /> PyTorch LSTM (Temporal RUL)
              </CardTitle>
              <Badge variant="healthy">Primary Model</Badge>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-xs text-slate-400">
                2-layer LSTM with 64 hidden units, 30-cycle sliding window, sequence-to-one RUL regression.
              </p>
              <div className="grid grid-cols-3 gap-3 pt-2">
                <div className="bg-slate-950/60 rounded-md p-3 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">MAE</div>
                  <div className="text-2xl font-mono font-bold text-emerald-400">
                    {loading ? '...' : (lstm ? lstm.mae : 'N/A')}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Cycles Error</div>
                </div>
                <div className="bg-slate-950/60 rounded-md p-3 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">RMSE</div>
                  <div className="text-2xl font-mono font-bold text-emerald-400">
                    {loading ? '...' : (lstm ? lstm.rmse : 'N/A')}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Root Mean Sq Error</div>
                </div>
                <div className="bg-slate-950/60 rounded-md p-3 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">R² Score</div>
                  <div className="text-2xl font-mono font-bold text-emerald-400">
                    {loading ? '...' : (lstm ? lstm.r2 : 'N/A')}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Variance Explained</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Random Forest Baseline Card */}
          <Card className="border-l-4 border-l-blue-500 bg-slate-900/80">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-base font-semibold flex items-center gap-2 text-blue-400">
                <Sliders className="w-5 h-5 text-blue-400" /> Random Forest Baseline
              </CardTitle>
              <Badge variant="outline">Baseline</Badge>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-xs text-slate-400">
                100 estimators, max depth 10, evaluated on single-cycle sensor snapshot features.
              </p>
              <div className="grid grid-cols-3 gap-3 pt-2">
                <div className="bg-slate-950/60 rounded-md p-3 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">MAE</div>
                  <div className="text-2xl font-mono font-bold text-blue-400">
                    {loading ? '...' : (rf ? rf.mae : 'N/A')}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Cycles Error</div>
                </div>
                <div className="bg-slate-950/60 rounded-md p-3 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">RMSE</div>
                  <div className="text-2xl font-mono font-bold text-blue-400">
                    {loading ? '...' : (rf ? rf.rmse : 'N/A')}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Root Mean Sq Error</div>
                </div>
                <div className="bg-slate-950/60 rounded-md p-3 border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">R² Score</div>
                  <div className="text-2xl font-mono font-bold text-blue-400">
                    {loading ? '...' : (rf ? rf.r2 : 'N/A')}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Variance Explained</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Anomaly & Dataset Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <ShieldCheck className="w-5 h-5 text-purple-400" /> Isolation Forest Anomaly Detection
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-slate-300">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="bg-slate-950/50 p-3 rounded border border-slate-800">
                  <div className="text-xs text-slate-400 uppercase font-mono mb-1">Methodology</div>
                  <div className="font-medium text-primary">Unsupervised Isolation Forest</div>
                  <div className="text-xs text-slate-500 mt-1">Trained on healthy operational cycles (RUL &gt; 100).</div>
                </div>
                <div className="bg-slate-950/50 p-3 rounded border border-slate-800">
                  <div className="text-xs text-slate-400 uppercase font-mono mb-1">Scoring System</div>
                  <div className="font-medium text-primary">Inverted Decision Function</div>
                  <div className="text-xs text-slate-500 mt-1">Normalized to [0.0 - 1.0] anomaly severity index.</div>
                </div>
              </div>
              <div className="p-3 bg-slate-950/30 rounded border border-slate-800/80 text-xs text-slate-400">
                <strong className="text-slate-300">Data Integrity Notice:</strong> C-MAPSS FD001 dataset provides run-to-failure cycle data without explicit ground-truth anomaly labels. Anomaly detection is strictly unsupervised.
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Database className="w-5 h-5 text-accent" /> Benchmark Dataset Metadata
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Dataset</span>
                <span className="text-primary">{metaInfo.dataset || 'C-MAPSS FD001'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">RUL Capping</span>
                <span className="text-emerald-400">{metaInfo.rul_cap || 125} cycles</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Window Size</span>
                <span className="text-primary">{metaInfo.window_size || 30} cycles</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Random Seed</span>
                <span className="text-accent">{metaInfo.random_seed || 42}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Sensor Features</span>
                <span className="text-primary">15 Selected</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
