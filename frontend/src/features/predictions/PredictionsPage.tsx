import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { formatDistanceToNow, format } from 'date-fns';
import { BrainCircuit, LineChart as LineChartIcon, ShieldAlert, ArrowRight, ActivitySquare, Sparkles, History } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Line
} from 'recharts';
import { PredictionExplanationModal } from './PredictionExplanationModal';
import { ModelExplanationView } from '../training/ModelExplanationView';
import type { RulForecastPoint } from '../../types/models';
import type { RegisteredModel } from '../../types/training';
import type { PredictionHistoryRecord } from '../../types/inference';

export function PredictionsPage() {
  const navigate = useNavigate();
  const [predictions, setPredictions] = useState<any[]>([]);
  const [sampleRulData, setSampleRulData] = useState<RulForecastPoint[]>([]);
  const [sampleMachine, setSampleMachine] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeUserModel, setActiveUserModel] = useState<RegisteredModel | null>(null);
  const [historyRecords, setHistoryRecords] = useState<PredictionHistoryRecord[]>([]);
  const [explainPredictionId, setExplainPredictionId] = useState<string | null>(null);
  const [explainBenchmarkId, setExplainBenchmarkId] = useState<string | null>(null);
  const [explainModelId, setExplainModelId] = useState<string | null>(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      setLoading(true);
      try {
        const [fleetPreds, userModel, hist] = await Promise.all([
          api.diagnostics.listFleetPredictions(),
          api.models.getActive('rul_regression').catch(() => null),
          api.inference.history({ task: 'rul_regression', limit: 8 }).catch(() => ({ records: [] })),
        ]);
        const sample = fleetPreds[0] || null;
        const rulSample = sample ? await api.diagnostics.getRulForecast(sample.machineId) : [];
        setPredictions(fleetPreds);
        setSampleRulData(rulSample);
        setSampleMachine(sample);
        setActiveUserModel(userModel || null);
        setHistoryRecords(hist?.records || []);
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

      {/* Explicit ML Source Comparison & Model Provenance Banner */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Source 1: NASA C-MAPSS Benchmark */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4 font-mono text-xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">ML SOURCE</span>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-500/20 text-sky-400 border border-sky-500/30">
              BENCHMARK BASELINE
            </span>
          </div>
          <div className="text-sm font-bold text-slate-100">NASA C-MAPSS Benchmark</div>
          <div className="text-slate-400 text-[11px]">
            Model: <span className="text-slate-200">PyTorch LSTM & RF Regressor</span> | Task: <span className="text-accent">RUL Regression</span>
          </div>
          <div className="flex items-center justify-between pt-1 border-t border-slate-800/80">
            <span className="text-slate-500 text-[10px]">
              Status: <span className="text-emerald-400 font-bold">ACTIVE (FLEET TELEMETRY)</span>
            </span>
            <button
              type="button"
              onClick={() => setExplainBenchmarkId('rf')}
              className="text-[10px] text-purple-400 hover:text-purple-300 font-semibold flex items-center gap-1 border border-purple-500/30 px-2 py-0.5 rounded bg-purple-500/10 transition-colors"
            >
              <Sparkles className="w-3 h-3" />
              <span>Explain Benchmark</span>
            </button>
          </div>
        </div>

        {/* Source 2: User-Trained Model */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4 font-mono text-xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">ML SOURCE</span>
            {activeUserModel ? (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                ACTIVE USER MODEL
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-500 border border-slate-700">
                NO ACTIVE MODEL
              </span>
            )}
          </div>
          <div className="text-sm font-bold text-slate-100">
            {activeUserModel ? activeUserModel.name : 'User-Trained Model Pipeline'}
          </div>
          <div className="text-slate-400 text-[11px]">
            {activeUserModel ? (
              <>
                Version: <span className="text-accent font-bold">v{activeUserModel.version}</span> | Algorithm: <span className="text-slate-200">{activeUserModel.model_type}</span>
              </>
            ) : (
              <span>No custom RUL model activated yet for this tenant.</span>
            )}
          </div>
          <div className="flex items-center justify-between pt-1 border-t border-slate-800/80">
            <span className="text-slate-500 text-[10px]">
              Status: <span className={activeUserModel ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
                {activeUserModel ? 'ACTIVE' : 'INACTIVE'}
              </span>
            </span>
            <div className="flex items-center gap-2">
              {activeUserModel && (
                <button
                  type="button"
                  onClick={() => setExplainModelId(activeUserModel.id)}
                  className="text-[10px] text-purple-400 hover:text-purple-300 font-semibold flex items-center gap-1 border border-purple-500/30 px-2 py-0.5 rounded bg-purple-500/10 transition-colors"
                >
                  <Sparkles className="w-3 h-3" />
                  <span>Explain Model</span>
                </button>
              )}
              <button
                type="button"
                onClick={() => navigate('/model-training')}
                className="text-[11px] text-accent hover:underline flex items-center gap-1"
              >
                <span>{activeUserModel ? 'Run Inference' : 'Train Custom Model'}</span>
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Model Transparency Warning */}
      <div className="bg-slate-900/50 border border-slate-700 p-4 rounded-lg flex items-start gap-4">
        <ActivitySquare className="w-5 h-5 text-slate-400 mt-0.5" />
        <div>
          <h3 className="text-sm font-bold text-slate-200">Industrial Baseline & Custom ML Governance</h3>
          <p className="text-xs text-slate-400 mt-1">
            Fleet forecasts below use the NASA C-MAPSS benchmark architecture. User-trained models registered in ML Studio run in isolated execution environments and can be evaluated on custom telemetry datasets.
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
            <div className="text-sm font-mono text-primary mt-1">PyTorch LSTM</div>
            <div className="text-xs text-slate-500 mt-0.5">C-MAPSS benchmark asset</div>
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
                ) : predictions.length === 0 ? <tr><td colSpan={5} className="px-6 py-12 text-center text-slate-500">No predictions available. Upload telemetry for a machine to run inference.</td></tr> : (
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
              <div className="text-sm text-slate-400 font-mono">{sampleMachine ? `${sampleMachine.machineId} — ${sampleMachine.machineName}` : 'No inference data'}</div>
          </CardHeader>
          <CardContent className="flex-1 min-h-[300px]">
             {loading ? (
                <div className="flex items-center justify-center h-full text-slate-500 font-mono text-sm">
                  [LOADING_TRAJECTORY...]
                </div>
              ) : sampleRulData.length === 0 ? <div className="flex h-full items-center justify-center text-sm text-slate-500">No predictions available.</div> : (
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
             <div className="text-xs text-slate-400">{sampleMachine ? 'Review this machine’s real inference data.' : 'Upload telemetry to create a forecast.'}</div>
             <button 
                disabled={!sampleMachine}
                onClick={() => sampleMachine && navigate(`/machines/${sampleMachine.machineId}`)}
                className="text-xs text-accent hover:text-blue-400 flex items-center gap-1 transition-colors"
              >
                Inspect <ArrowRight className="w-3 h-3" />
             </button>
          </div>
        </Card>

      </div>

      {/* Historical Prediction Audits & Attribution (Phase 7 Explainability) */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <History className="w-4 h-4 text-accent" />
            <span>Historical Prediction Audits & Local Feature Attributions</span>
          </CardTitle>
          <span className="text-xs text-slate-400 font-mono">
            {historyRecords.length} Audited Records
          </span>
        </CardHeader>
        <CardContent className="p-0 overflow-x-auto">
          {historyRecords.length === 0 ? (
            <div className="p-8 text-center text-xs font-mono text-slate-500">
              No persisted inference audit records yet. Run user-trained model inference in ML Studio to inspect tree attributions.
            </div>
          ) : (
            <table className="w-full text-xs font-mono text-left">
              <thead className="bg-slate-950/50 border-b border-slate-800 text-slate-400 uppercase text-[11px]">
                <tr>
                  <th className="px-6 py-3 font-medium">Record ID</th>
                  <th className="px-6 py-3 font-medium">Model & Version</th>
                  <th className="px-6 py-3 font-medium">Type</th>
                  <th className="px-6 py-3 font-medium">Predicted Value</th>
                  <th className="px-6 py-3 font-medium">Recorded At</th>
                  <th className="px-6 py-3 font-medium text-right">Interpretability</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {historyRecords.map((rec) => (
                  <tr key={rec.id} className="hover:bg-slate-800/20 transition-colors">
                    <td className="px-6 py-3 text-slate-400 truncate max-w-[140px]">{rec.id}</td>
                    <td className="px-6 py-3">
                      <span className="font-bold text-slate-200">
                        {rec.model_name || rec.model_id}
                      </span>
                      {rec.model_version && (
                        <span className="ml-1 px-1.5 py-0.2 rounded text-[10px] bg-slate-800 text-accent border border-slate-700">
                          v{rec.model_version}
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-3 text-slate-400 capitalize">
                      {rec.input_metadata?.type || 'Single'}
                    </td>
                    <td className="px-6 py-3 font-bold text-accent">
                      {rec.prediction.rul_cycles !== undefined
                        ? `${rec.prediction.rul_cycles} cycles`
                        : rec.prediction.condition || 'Computed'}
                    </td>
                    <td className="px-6 py-3 text-slate-400">
                      {rec.created_at ? formatDistanceToNow(new Date(rec.created_at), { addSuffix: true }) : 'N/A'}
                    </td>
                    <td className="px-6 py-3 text-right">
                      <button
                        type="button"
                        onClick={() => setExplainPredictionId(rec.id)}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium text-purple-400 bg-purple-500/10 border border-purple-500/30 hover:bg-purple-500/20 transition-colors"
                      >
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Explain Attribution</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>

      {/* BENCHMARK EXPLANATION MODAL */}
      {explainBenchmarkId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <ModelExplanationView
              benchmarkId={explainBenchmarkId}
              modelName="NASA C-MAPSS RF Regressor Benchmark"
              onClose={() => setExplainBenchmarkId(null)}
            />
          </div>
        </div>
      )}

      {/* USER MODEL EXPLANATION MODAL */}
      {explainModelId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <ModelExplanationView
              modelId={explainModelId}
              onClose={() => setExplainModelId(null)}
            />
          </div>
        </div>
      )}

      {/* LOCAL PREDICTION EXPLANATION MODAL */}
      {explainPredictionId && (
        <PredictionExplanationModal
          predictionId={explainPredictionId}
          onClose={() => setExplainPredictionId(null)}
        />
      )}
    </div>
  );
}
