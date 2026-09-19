import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { formatDistanceToNow, format } from 'date-fns';
import { BrainCircuit, LineChart as LineChartIcon, ShieldAlert, ArrowRight, ActivitySquare } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Line
} from 'recharts';
import type { RulForecastPoint } from '../../types/models';

export function PredictionsPage() {
  const navigate = useNavigate();
  const [predictions, setPredictions] = useState<any[]>([]);
  const [sampleRulData, setSampleRulData] = useState<RulForecastPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPredictions = async () => {
      setLoading(true);
      try {
        const [fleetPreds, rulSample] = await Promise.all([
          api.diagnostics.listFleetPredictions(),
          api.diagnostics.getRulForecast('M-003') // Load worst machine for chart preview
        ]);
        setPredictions(fleetPreds);
        setSampleRulData(rulSample);
      } catch (error) {
        console.error("Failed to fetch predictions", error);
      } finally {
        setLoading(false);
      }
    };
    fetchPredictions();
  }, []);

  const highRiskCount = predictions.filter(p => p.failureRisk > 75).length;

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500 pb-12">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-primary flex items-center gap-2">
          <BrainCircuit className="w-6 h-6 text-accent" /> Predictive Models
        </h1>
        <p className="text-sm text-slate-400 mt-1">Machine Learning forecasts for Remaining Useful Life (RUL) and failure risk.</p>
      </div>

      {/* Model Transparency Warning */}
      <div className="bg-slate-900/50 border border-slate-700 p-4 rounded-lg flex items-start gap-4">
        <ActivitySquare className="w-5 h-5 text-slate-400 mt-0.5" />
        <div>
          <h3 className="text-sm font-bold text-slate-200">Model Evaluation Status: Data Unavailable</h3>
          <p className="text-xs text-slate-400 mt-1">
            Real-world model validation metrics (MAE, RMSE, R²) for the current predictive horizon are pending the next scheduled maintenance teardowns. Predicted RUL values are derived from historical sensor-degradation baselines using XGBoost survival ensembles.
          </p>
        </div>
      </div>

      {/* Prediction Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-l-4 border-l-slate-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2">Monitored Fleet</div>
            <div className="text-3xl font-mono text-primary">{predictions.length}</div>
          </CardContent>
        </Card>
        <Card className={highRiskCount > 0 ? "border-l-4 border-l-red-500 bg-red-950/10" : "border-l-4 border-l-emerald-500"}>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-red-500" /> High Risk Assets
            </div>
            <div className="text-3xl font-mono text-red-500">{highRiskCount}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-accent">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2">Primary Model</div>
            <div className="text-sm font-mono text-primary mt-1">XGBoost Survival</div>
            <div className="text-xs text-slate-500 mt-0.5">v2.4.1-prod</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Fleet Prediction Table */}
        <Card className="lg:col-span-2 flex flex-col">
          <CardHeader>
            <CardTitle>Fleet Forecast Summary</CardTitle>
          </CardHeader>
          <CardContent className="p-0 overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-slate-400 uppercase bg-slate-950/50 border-b border-slate-800">
                <tr>
                  <th className="px-6 py-4 font-medium tracking-wider">Machine</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Degradation</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Predicted RUL</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Failure Risk</th>
                  <th className="px-6 py-4 font-medium tracking-wider text-right">Updated</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {loading ? (
                  <tr>
                    <td colSpan={5} className="px-6 py-12 text-center text-slate-500 font-mono">
                      [LOADING_PREDICTIONS...]
                    </td>
                  </tr>
                ) : (
                  predictions.map((pred) => (
                    <tr 
                      key={pred.machineId} 
                      className="hover:bg-slate-800/20 transition-colors group cursor-pointer"
                      onClick={() => navigate(`/machines/${pred.machineId}`)}
                    >
                      <td className="px-6 py-4">
                        <div className="font-medium text-primary group-hover:text-accent transition-colors">
                          {pred.machineName}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">{pred.machineId}</div>
                      </td>
                      <td className="px-6 py-4">
                        <Badge variant={
                          pred.degradationState === 'Accelerated' ? 'critical' : 
                          pred.degradationState === 'Linear' ? 'warning' : 'healthy'
                        }>
                          {pred.degradationState.toUpperCase()}
                        </Badge>
                      </td>
                      <td className="px-6 py-4">
                        <div className="font-mono text-primary flex items-end gap-1">
                          <span className="text-lg">{pred.predictedRul}</span> 
                          <span className="text-xs text-slate-500 mb-0.5 font-sans">days</span>
                        </div>
                        <div className="text-xs text-slate-500">Conf: {pred.confidence}%</div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${
                                pred.failureRisk > 75 ? 'bg-red-500' : 
                                pred.failureRisk > 30 ? 'bg-amber-500' : 'bg-emerald-500'
                              }`}
                              style={{ width: `${pred.failureRisk}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs text-slate-400">{pred.failureRisk}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="text-slate-300 font-mono text-xs mb-2">{formatDistanceToNow(new Date(pred.lastPredictionTime), { addSuffix: true })}</div>
                        {pred.failureRisk > 40 && (
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              navigate(`/diagnostics?machine=${pred.machineId}`);
                            }}
                            className="inline-flex items-center justify-center px-2 py-1 rounded text-xs font-medium text-accent border border-accent/20 hover:bg-accent/10 transition-colors"
                          >
                            Open Diagnostic
                          </button>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </CardContent>
        </Card>

        {/* Highlight Chart for Worst Case */}
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <LineChartIcon className="w-4 h-4 text-accent" /> Sample Forecast Trajectory
            </CardTitle>
            <div className="text-sm text-slate-400 font-mono">M-003 (Critical Risk)</div>
          </CardHeader>
          <CardContent className="flex-1 min-h-[300px]">
             {loading ? (
                <div className="flex items-center justify-center h-full text-slate-500 font-mono text-sm">
                  [LOADING_TRAJECTORY...]
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={sampleRulData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2b3038" vertical={false} />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(val) => format(new Date(val), 'MMM dd')}
                      stroke="#475569" 
                      tick={{ fill: '#64748b', fontSize: 10 }} 
                    />
                    <YAxis stroke="#475569" tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f1115', borderColor: '#2b3038', borderRadius: '6px' }}
                      labelFormatter={(val) => format(new Date(val as string | number), 'MMM dd, yyyy')}
                    />
                    <ReferenceLine x={new Date().toISOString()} stroke="#94a3b8" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'NOW', fill: '#94a3b8', fontSize: 10 }} />
                    <Area type="monotone" dataKey="upperBound" stroke="none" fill="#10b981" fillOpacity={0.05} />
                    <Area type="monotone" dataKey="lowerBound" stroke="none" fill="#0f1115" fillOpacity={1} />
                    <Line type="monotone" dataKey="predictedRul" name="RUL (Days)" stroke="#10b981" strokeWidth={2} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              )}
          </CardContent>
          <div className="p-4 border-t border-slate-800 bg-slate-900/50 flex justify-between items-center rounded-b-lg">
             <div className="text-xs text-slate-400">Review M-003 analysis for full RUL data.</div>
             <button 
                onClick={() => navigate(`/machines/M-003`)}
                className="text-xs text-accent hover:text-blue-400 flex items-center gap-1 transition-colors"
              >
                Inspect <ArrowRight className="w-3 h-3" />
             </button>
          </div>
        </Card>

      </div>
    </div>
  );
}
