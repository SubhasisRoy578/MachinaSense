import { useEffect, useState, useMemo } from 'react';
import type { ChangeEvent } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { api } from '../../services/api';
import type { Machine, SensorDataPoint, RulForecastPoint, AnomalyEvent, DiagnosticPreview } from '../../types/models';
import type { RegisteredModel } from '../../types/training';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceLine
} from 'recharts';
import { format, formatDistanceToNow } from 'date-fns';
import { ArrowLeft, Activity, Wrench, AlertTriangle, BrainCircuit, Sparkles, Loader2, Cpu, CheckCircle2, ArrowRight } from 'lucide-react';
import { DecisionSupportPanel } from '../diagnostics/DecisionSupportPanel';
import { ModelExplanationView } from '../training/ModelExplanationView';

export function MachineDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [machine, setMachine] = useState<Machine | null>(null);
  const [sensorData, setSensorData] = useState<SensorDataPoint[]>([]);
  const [rulData, setRulData] = useState<RulForecastPoint[]>([]);
  const [anomalies, setAnomalies] = useState<AnomalyEvent[]>([]);
  const [diagnostics, setDiagnostics] = useState<DiagnosticPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');
  const [activeRulModel, setActiveRulModel] = useState<RegisteredModel | null>(null);
  const [userModelPrediction, setUserModelPrediction] = useState<number | null>(null);
  const [userModelRunning, setUserModelRunning] = useState(false);
  const [userModelError, setUserModelError] = useState<string | null>(null);
  const [explainModelId, setExplainModelId] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    const fetchMachineData = async () => {
      setLoading(true);
      try {
        const [m, sensors, rul, anom, diag, userModel] = await Promise.all([
          api.machines.get(id),
          api.sensors.getMachineData(id, 24),
          api.diagnostics.getRulForecast(id),
          api.diagnostics.getMachineAnomalies(id),
          api.diagnostics.getMachineDiagnostics(id),
          api.models.getActive('rul_regression').catch(() => null),
        ]);
        setMachine(m || null);
        setSensorData(sensors);
        setRulData(rul);
        setAnomalies(anom);
        setDiagnostics(diag);
        setActiveRulModel(userModel || null);
      } catch (error) {
        console.error("Failed to fetch machine details", error);
      } finally {
        setLoading(false);
      }
    };
    fetchMachineData();
  }, [id]);

  // Telemetry Feature Compatibility Check
  const featureCompatibility = useMemo(() => {
    if (!activeRulModel) return null;
    const reqCols: string[] = activeRulModel.training_metadata?.feature_columns || [];
    if (reqCols.length === 0) return { compatible: true, missing: [] };
    const latestSensor = sensorData.length > 0 ? sensorData[sensorData.length - 1] : {};
    const available = new Set(Object.keys(latestSensor));
    const missing = reqCols.filter((c) => !available.has(c));
    return {
      compatible: missing.length === 0,
      missing,
      required: reqCols,
    };
  }, [activeRulModel, sensorData]);

  const handleRunUserModelInference = async () => {
    if (!activeRulModel || !featureCompatibility?.compatible) return;
    setUserModelRunning(true);
    setUserModelError(null);
    try {
      const latestSensor = sensorData[sensorData.length - 1] as any;
      const reqCols: string[] = activeRulModel.training_metadata?.feature_columns || [];
      const features: Record<string, number> = {};
      reqCols.forEach((col) => {
        features[col] = Number(latestSensor?.[col] ?? 0);
      });
      const res = await api.inference.predictRul(features);
      setUserModelPrediction(res.prediction.rul_cycles);
    } catch (err) {
      setUserModelError(err instanceof Error ? err.message : 'Inference failed');
    } finally {
      setUserModelRunning(false);
    }
  };

  const handleRunDiagnostic = async () => {
    if (!id || !machine) return;
    setIsGenerating(true);
    try {
      const result = await api.diagnostics.createCase(machine.id);
      if (result && result.id) {
        navigate(`/diagnostics/${result.id}`);
      } else {
        navigate(`/diagnostics?machine=${machine.id}`);
      }
    } catch (err) {
      console.error("Failed to trigger diagnostic", err);
      navigate(`/diagnostics?machine=${machine.id}`);
    } finally {
      setIsGenerating(false);
    }
  };
  const handleTelemetryUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file || !id) return;
    if (!file.name.toLowerCase().endsWith('.csv')) { setUploadMessage('Select a CSV telemetry file.'); return; }
    if (file.size === 0 || file.size > 25 * 1024 * 1024) { setUploadMessage('CSV files must be between 1 byte and 25 MB.'); return; }
    setUploading(true); setUploadMessage('Validating telemetry and running ML inference…');
    try {
      const result = await api.machines.uploadTelemetry(id, file);
      setUploadMessage(`${result.accepted_rows} rows imported. Refreshing machine intelligence…`);
      window.setTimeout(() => window.location.reload(), 500);
    } catch (error) {
      setUploadMessage(error instanceof Error ? error.message : 'Telemetry upload failed.');
    } finally { setUploading(false); }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 font-mono text-sm">
        [LOADING_MACHINE_TELEMETRY...]
      </div>
    );
  }

  if (!machine) {
    return <div className="text-red-500">Machine not found.</div>;
  }

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500 pb-12">
      {/* 1. Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-start justify-between border-b border-slate-800 pb-6">
        <div>
          <Link to="/machines" className="inline-flex items-center text-sm text-slate-400 hover:text-accent mb-4 transition-colors">
            <ArrowLeft className="w-4 h-4 mr-1" /> Back to Machines
          </Link>
          <h1 className="text-3xl font-bold tracking-tight text-primary flex items-center gap-3">
            {machine.name}
            <Badge variant={machine.status as any} className="text-sm px-3 py-1">{machine.status.toUpperCase()}</Badge>
          </h1>
          <p className="text-sm text-slate-400 mt-2 font-mono">
            ID: {machine.id} <span className="mx-2">•</span> TYPE: {machine.type.toUpperCase()} <span className="mx-2">•</span> LOC: {machine.location.toUpperCase()}
          </p>
        </div>
        <div className="text-right flex flex-col items-end gap-2">
          <div className="text-xs text-slate-500 uppercase font-bold tracking-wider">Last Updated</div>
          <div className="text-sm text-slate-300 font-mono">
            {format(new Date(machine.lastUpdated), 'yyyy-MM-dd HH:mm:ss')}
          </div>
          <button 
            onClick={handleRunDiagnostic}
            disabled={isGenerating}
            className="inline-flex items-center justify-center gap-2 mt-1 px-4 py-2 bg-accent hover:bg-accent/90 text-slate-950 font-semibold rounded-md transition-colors text-xs uppercase tracking-wider disabled:opacity-50"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Generating...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" /> Run Grounded AI Diagnostic
              </>
            )}
          </button>
          <label className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-md border border-accent/40 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-accent hover:bg-accent/10">
            {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
            {uploading ? 'Processing telemetry' : 'Upload telemetry CSV'}
            <input type="file" accept=".csv,text/csv" className="hidden" disabled={uploading} onChange={handleTelemetryUpload} />
          </label>
          {uploadMessage && <div className={`max-w-xs text-right text-xs ${uploadMessage.includes('failed') || uploadMessage.includes('Select') || uploadMessage.includes('must') ? 'text-red-400' : 'text-slate-400'}`}>{uploadMessage}</div>}
        </div>
      </div>

      {/* 2. Health & Risk Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-500" /> Overall Health
            </div>
            <div className="text-3xl font-mono text-primary">{100 - machine.failureRisk}<span className="text-lg text-slate-500">%</span></div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-500" /> Failure Risk
            </div>
            <div className="text-3xl font-mono text-amber-500">{machine.failureRisk}<span className="text-lg text-slate-500">%</span></div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <Wrench className="w-4 h-4 text-blue-500" /> Est. RUL
            </div>
            <div className="text-3xl font-mono text-primary">{machine.rulDays || 0} <span className="text-xs text-slate-500">DAYS</span></div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <BrainCircuit className="w-4 h-4 text-accent" /> Active Anomalies
            </div>
            <div className="text-3xl font-mono text-accent">{anomalies.length}</div>
          </CardContent>
        </Card>
      </div>

      {/* Main Grid Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left 2 Columns: Charts */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          
          {/* 3. Sensor Telemetry Stream */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-base font-medium text-slate-300">Live Telemetry (Vibration & Temperature)</CardTitle>
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-accent inline-block"></span> Vib (mm/s)</span>
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block"></span> Temp (°C)</span>
              </div>
            </CardHeader>
            <CardContent className="h-72">
              {sensorData.length === 0 ? <div className="grid h-full place-items-center text-sm text-slate-500">No telemetry data available. Upload a validated CSV to begin inference.</div> : <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={sensorData}>
                  <defs>
                    <linearGradient id="vibColor" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#a855f7" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="tempColor" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="timestamp" stroke="#64748b" tickFormatter={(t) => format(new Date(t), 'HH:mm')} />
                  <YAxis stroke="#64748b" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.375rem', color: '#f8fafc' }} 
                    labelFormatter={(t) => t ? format(new Date(String(t)), 'yyyy-MM-dd HH:mm:ss') : ''}
                  />
                  <Area type="monotone" dataKey="vibration" stroke="#a855f7" fillOpacity={1} fill="url(#vibColor)" strokeWidth={2} />
                  <Area type="monotone" dataKey="temperature" stroke="#f59e0b" fillOpacity={1} fill="url(#tempColor)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>}
            </CardContent>
          </Card>

          {/* 4. RUL Forecast Curve */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-base font-medium text-slate-300">Remaining Useful Life (RUL) Trajectory</CardTitle>
              <div className="text-xs text-slate-500 font-mono">LSTM PyTorch Temporal Degradation Model</div>
            </CardHeader>
            <CardContent className="h-64">
              {rulData.length === 0 ? <div className="grid h-full place-items-center text-sm text-slate-500">No predictions available until telemetry is uploaded.</div> : <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rulData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="timestamp" stroke="#64748b" tickFormatter={(t) => format(new Date(t), 'MM-dd')} />
                  <YAxis stroke="#64748b" label={{ value: 'Hours', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.375rem', color: '#f8fafc' }}
                  />
                  <ReferenceLine y={20} label="Critical Maintenance Threshold" stroke="#ef4444" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="predictedRul" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="lowerBound" stroke="#1d4ed8" strokeDasharray="2 2" dot={false} />
                  <Line type="monotone" dataKey="upperBound" stroke="#1d4ed8" strokeDasharray="2 2" dot={false} />
                </LineChart>
              </ResponsiveContainer>}
            </CardContent>
          </Card>

          {/* 4b. User-Trained Model Telemetry Inference Card */}
          <Card className="border-slate-800 bg-slate-950/60 font-mono text-xs">
            <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-accent" />
                <CardTitle className="text-sm font-mono text-slate-200">
                  User-Trained Model Telemetry Inference
                </CardTitle>
              </div>
              {activeRulModel ? (
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                  ACTIVE: {activeRulModel.name} (v{activeRulModel.version})
                </span>
              ) : (
                <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-900 text-slate-500 border border-slate-800">
                  NO ACTIVE USER MODEL
                </span>
              )}
            </CardHeader>
            <CardContent className="pt-4 space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-slate-400 text-[11px]">
                <div>
                  <span className="text-slate-500 uppercase text-[10px] block">ML SOURCE TYPE</span>
                  <span className="text-slate-200 font-bold">
                    {activeRulModel ? 'User-Trained ML Model' : 'NASA C-MAPSS Benchmark (Active Above)'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 uppercase text-[10px] block">ALGORITHM</span>
                  <span className="text-slate-200 font-bold">
                    {activeRulModel?.model_type || 'PyTorch LSTM (Benchmark Default)'}
                  </span>
                </div>
              </div>

              {activeRulModel ? (
                <div className="space-y-3 pt-2 border-t border-slate-800">
                  {/* Schema Validation Check */}
                  {!featureCompatibility?.compatible ? (
                    <div className="p-3 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 space-y-1.5 text-xs">
                      <div className="flex items-center gap-2 font-bold">
                        <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400" />
                        <span>Active model requires features that are not available in this telemetry dataset.</span>
                      </div>
                      <p className="text-[11px] text-slate-400">
                        Missing required telemetry features:{' '}
                        <span className="text-amber-200 font-mono">
                          {featureCompatibility?.missing.join(', ')}
                        </span>
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="p-2.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-between">
                        <div className="flex items-center gap-2 text-xs">
                          <CheckCircle2 className="w-4 h-4" />
                          <span>Telemetry feature schema matches active model requirements ({featureCompatibility?.required?.length ?? 0} features).</span>
                        </div>
                        <button
                          type="button"
                          onClick={handleRunUserModelInference}
                          disabled={userModelRunning || sensorData.length === 0}
                          className="bg-accent hover:bg-accent/90 text-slate-950 font-bold px-3 py-1.5 rounded text-xs transition-colors flex items-center gap-1 disabled:opacity-50"
                        >
                          {userModelRunning ? (
                            <>
                              <Loader2 className="w-3.5 h-3.5 animate-spin" />
                              <span>Evaluating...</span>
                            </>
                          ) : (
                            <>
                              <Cpu className="w-3.5 h-3.5" />
                              <span>Run Prediction</span>
                            </>
                          )}
                        </button>
                      </div>

                      {userModelPrediction !== null && (
                        <div className="bg-slate-900 border border-slate-800 rounded p-4 text-center space-y-1">
                          <span className="text-slate-500 text-[10px] uppercase">
                            USER-TRAINED MODEL PREDICTED RUL
                          </span>
                          <div className="text-3xl font-bold text-accent">
                            {userModelPrediction}{' '}
                            <span className="text-sm font-normal text-slate-400">cycles</span>
                          </div>
                          <span className="text-[10px] text-slate-500 block">
                            Prediction calculated and persisted to audit history
                          </span>
                          {activeRulModel && (
                            <div className="pt-2">
                              <button
                                type="button"
                                onClick={() => setExplainModelId(activeRulModel.id)}
                                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono font-medium text-purple-400 bg-purple-500/10 border border-purple-500/30 hover:bg-purple-500/20 transition-colors"
                              >
                                <Sparkles className="w-3.5 h-3.5" />
                                <span>Explain Feature Importance</span>
                              </button>
                            </div>
                          )}
                        </div>
                      )}

                      {userModelError && (
                        <div className="p-2.5 rounded bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
                          {userModelError}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-3 rounded bg-slate-900/60 border border-slate-800 text-slate-400 text-xs flex items-center justify-between">
                  <span>
                    Train and activate a customized RUL model in ML Studio to evaluate machine telemetry with user-trained algorithms.
                  </span>
                  <Link
                    to="/model-training"
                    className="text-accent hover:underline flex items-center gap-1 shrink-0 ml-3 font-bold"
                  >
                    <span>ML Studio</span>
                    <ArrowRight className="w-3 h-3" />
                  </Link>
                </div>
              )}
            </CardContent>
          </Card>

        </div>

        {/* Right Column: Context & Diagnostics */}
        <div className="flex flex-col gap-6">
          
          {/* 8. AI Diagnostic Preview */}
          <Card className="border-accent/30 bg-accent/5 overflow-visible relative">
            <div className="absolute -top-3 -right-3 w-6 h-6 rounded-full bg-accent text-slate-950 flex items-center justify-center shadow-[0_0_12px_rgba(59,130,246,0.5)]">
              <BrainCircuit className="w-4 h-4" />
            </div>
            <CardHeader className="border-b-accent/20 flex flex-row items-center justify-between">
              <CardTitle className="text-accent flex items-center gap-2">AI Diagnostic Summary</CardTitle>
              <button
                onClick={handleRunDiagnostic}
                disabled={isGenerating}
                className="text-xs text-accent hover:underline font-medium flex items-center gap-1"
              >
                <Sparkles className="w-3 h-3" /> New Investigation
              </button>
            </CardHeader>
            <CardContent className="pt-4 text-sm">
              <div className="mb-4">
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">ML Finding</div>
                <div className="text-primary font-medium">{diagnostics?.finding}</div>
              </div>
              <div className="mb-4">
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Explanation</div>
                <div className="text-slate-300 leading-relaxed">{diagnostics?.explanation}</div>
              </div>
              <div>
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Supporting Evidence</div>
                <ul className="list-disc pl-4 space-y-1 text-slate-400 font-mono text-xs">
                  {diagnostics?.evidence.map((ev, i) => (
                    <li key={i}>{ev}</li>
                  ))}
                </ul>
              </div>
            </CardContent>
          </Card>

          {/* 9. Evidence-Backed Advisory Decision Support */}
          <DecisionSupportPanel machineId={id} />

          {/* 5. Anomaly Timeline */}
          <Card className="flex-1 min-h-[300px]">
            <CardHeader>
              <CardTitle>Recent Anomalies</CardTitle>
            </CardHeader>
            <CardContent className="pt-4 overflow-y-auto">
              <div className="relative border-l border-slate-700 ml-3 space-y-6">
                {anomalies.map((anom) => (
                  <div key={anom.id} className="relative pl-6">
                    <div className={`absolute -left-1.5 top-1.5 w-3 h-3 rounded-full border-2 border-slate-900 ${
                      anom.severity === 'critical' ? 'bg-red-500' :
                      anom.severity === 'high' ? 'bg-amber-500' :
                      anom.severity === 'medium' ? 'bg-amber-400' : 'bg-slate-400'
                    }`}></div>
                    <div className="text-xs font-mono text-slate-400 mb-1">
                      {formatDistanceToNow(new Date(anom.timestamp), { addSuffix: true })}
                    </div>
                    <div className="text-sm font-medium text-primary">{anom.description}</div>
                    <div className="text-xs text-slate-500 mt-1 flex items-center justify-between">
                      <span>Sensor: {anom.sensor}</span>
                      <span className="font-mono">Score: {anom.score.toFixed(2)}</span>
                    </div>
                  </div>
                ))}
                {anomalies.length === 0 && (
                  <div className="text-sm text-slate-500 pl-4 py-4 italic">No recent anomalies detected.</div>
                )}
              </div>
            </CardContent>
          </Card>
          
        </div>
      </div>

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
    </div>
  );
}
