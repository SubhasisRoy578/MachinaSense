import { useState, useEffect } from 'react';
import { api } from '../../services/api';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { 
  BrainCircuit, 
  Layers, 
  Cpu, 
  ShieldCheck, 
  Activity, 
  CheckCircle2, 
  AlertTriangle, 
  Gauge, 
  Zap, 
  TrendingUp, 
  Clock, 
  Sparkles, 
  RefreshCw,
  HardDrive
} from 'lucide-react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';

export function ExplainabilityDashboard() {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [summaryData, setSummaryData] = useState<any>(null);
  const [engineData, setEngineData] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState<'rf' | 'lstm'>('rf');
  const [selectedEngineId, setSelectedEngineId] = useState<number>(1);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sumRes, engRes] = await Promise.all([
        api.explainability.getPhase10Summary(),
        api.explainability.getPhase10Engines(),
      ]);
      setSummaryData(sumRes);
      setEngineData(engRes);
    } catch (err: any) {
      console.error("Failed to load explainability data:", err);
      setError(err.message || "Failed to load explainability and robustness intelligence.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] gap-4 font-mono text-slate-400">
        <RefreshCw className="w-8 h-8 text-accent animate-spin" />
        <p className="text-sm tracking-wider">[COMPUTING_EXPLAINABILITY_TELEMETRY...]</p>
      </div>
    );
  }

  if (error || !summaryData) {
    return (
      <div className="bg-rose-950/20 border border-rose-800/50 rounded-xl p-6 text-rose-300 font-mono text-sm space-y-3">
        <div className="flex items-center gap-2 text-rose-400 font-bold">
          <AlertTriangle className="w-5 h-5" />
          <span>Error Loading Explainability Telemetry</span>
        </div>
        <p>{error || "No data returned from backend."}</p>
        <button
          onClick={fetchData}
          className="px-4 py-2 bg-rose-900/40 hover:bg-rose-900/60 border border-rose-700/50 rounded-lg text-xs font-semibold text-rose-200 transition"
        >
          Retry Fetch
        </button>
      </div>
    );
  }

  const rfData = summaryData?.models?.random_forest || {};
  const lstmData = summaryData?.models?.lstm || {};
  const robustData = summaryData?.robustness || {};
  const perfData = summaryData?.performance || {};

  // RF Feature chart data
  const rfGini = (rfData?.top_5_global_features_gini || []).map((f: any) => ({
    name: f.feature,
    value: Number(f.importance_percentage?.toFixed(1) || (f.gini_importance * 100).toFixed(1)),
    metric: 'Gini MDI %'
  }));

  const rfPerm = (rfData?.top_5_global_features_permutation || []).map((f: any) => ({
    name: f.feature,
    value: Number(f.mae_increase_when_permuted?.toFixed(2) || 0),
    metric: '+MAE Cycles'
  }));

  // LSTM Feature chart data
  const lstmSensors = (lstmData?.sensor_importance_ranking || []).slice(0, 8).map((s: any) => ({
    name: s.sensor,
    value: Number((s.gradient_saliency * 100).toFixed(2)),
    raw: s.gradient_saliency,
    pct: Number(s.importance_percentage?.toFixed(1) || 0)
  }));

  // LSTM Temporal curve
  const lstmTemporal = (lstmData?.temporal_importance_ranking || []).map((t: any) => ({
    cycle: t.relative_cycle,
    saliency: Number(t.gradient_saliency?.toFixed(4) || 0),
    percentage: Number(t.importance_percentage?.toFixed(1) || 0)
  }));

  // Selected engine
  const currentEngineList = selectedModel === 'rf' 
    ? (engineData?.random_forest_engines || []) 
    : (engineData?.lstm_engines || []);
  const activeEngineExplanation = currentEngineList.find((e: any) => e.engine_id === selectedEngineId) || currentEngineList[0];

  return (
    <div className="space-y-6 pb-12 font-sans">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-accent/10 border border-accent/30 rounded-lg">
              <BrainCircuit className="w-6 h-6 text-accent" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              Model Explainability & Robustness Intelligence
            </h1>
          </div>
          <p className="text-sm text-slate-400 mt-1 max-w-3xl">
            Empirical attributions, gradient saliency, temporal dynamics, and sensor noise degradation benchmarks on NASA C-MAPSS FD001 test engines.
          </p>
        </div>

        {/* Global Controls / Model Switcher */}
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-1 rounded-xl">
          <button
            onClick={() => setSelectedModel('rf')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              selectedModel === 'rf'
                ? 'bg-sky-500/20 text-sky-400 border border-sky-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>Random Forest (19 Feat)</span>
          </button>
          <button
            onClick={() => setSelectedModel('lstm')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              selectedModel === 'lstm'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>LSTM Recurrent (30×15)</span>
          </button>
        </div>
      </div>

      {/* Model Spec Card */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-900/60 border-slate-800">
          <CardContent className="p-4 flex items-center justify-between">
            <div>
              <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">ARCHITECTURE</div>
              <div className="text-sm font-bold text-slate-100 mt-0.5">
                {selectedModel === 'rf' ? 'Random Forest Regressor' : '2-Layer Recurrent LSTM'}
              </div>
              <div className="text-xs text-slate-400">
                {selectedModel === 'rf' ? '200 Decision Trees | Cap: 125' : 'Hidden Size: 64 | Window: 30'}
              </div>
            </div>
            <Badge variant="outline" className={selectedModel === 'rf' ? 'border-sky-500/40 text-sky-400' : 'border-emerald-500/40 text-emerald-400'}>
              {selectedModel === 'rf' ? 'MDI & Permutation' : 'Taylor Saliency'}
            </Badge>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/60 border-slate-800">
          <CardContent className="p-4 flex items-center justify-between">
            <div>
              <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">OFFICIAL TEST ACCURACY</div>
              <div className="text-sm font-bold text-slate-100 mt-0.5">
                {selectedModel === 'rf' ? 'MAE 12.21 | RMSE 16.56' : 'MAE 10.23 | RMSE 13.64'}
              </div>
              <div className="text-xs text-slate-400">
                {selectedModel === 'rf' ? 'R²: 0.8293 | NASA: 465.3' : 'R²: 0.8841 | NASA: 358.2'}
              </div>
            </div>
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          </CardContent>
        </Card>

        <Card className="bg-slate-900/60 border-slate-800">
          <CardContent className="p-4 flex items-center justify-between">
            <div>
              <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">NOISE RESILIENCE (5% NOISE)</div>
              <div className="text-sm font-bold text-slate-100 mt-0.5">
                {selectedModel === 'rf' ? 'Δ 0.93 Cycles | 100% Stable' : 'Δ 0.61 Cycles | 100% Stable'}
              </div>
              <div className="text-xs text-slate-400">
                Under Gaussian Sensor Perturbation
              </div>
            </div>
            <ShieldCheck className="w-5 h-5 text-sky-400" />
          </CardContent>
        </Card>

        <Card className="bg-slate-900/60 border-slate-800">
          <CardContent className="p-4 flex items-center justify-between">
            <div>
              <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">INFERENCE LATENCY (WARM)</div>
              <div className="text-sm font-bold text-slate-100 mt-0.5">
                {selectedModel === 'rf' ? '102.6 ms (P95: 133.6 ms)' : '2.8 ms (P95: 3.4 ms)'}
              </div>
              <div className="text-xs text-slate-400">
                {selectedModel === 'rf' ? 'Throughput: ~10 req/s' : 'Throughput: ~353 req/s'}
              </div>
            </div>
            <Zap className="w-5 h-5 text-amber-400" />
          </CardContent>
        </Card>
      </div>

      {/* Main Section 1: Global Feature Importance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {selectedModel === 'rf' ? (
          <>
            {/* RF Gini Importance */}
            <Card className="bg-slate-900/60 border-slate-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-slate-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-sky-400" />
                    <span>Global Gini Impurity Importance (MDI)</span>
                  </div>
                  <Badge variant="outline" className="text-[10px] text-sky-400 border-sky-500/30">Top 5 Drivers</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-2">
                <p className="text-xs text-slate-400 mb-4 font-mono">
                  Relative contribution of each feature to tree split impurity reduction across 200 estimators.
                </p>
                <div className="h-60 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={rfGini} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                      <XAxis type="number" unit="%" stroke="#64748b" tick={{ fontSize: 11 }} />
                      <YAxis dataKey="name" type="category" stroke="#94a3b8" tick={{ fontSize: 12, fill: '#cbd5e1' }} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc' }}
                        formatter={(val: any) => [`${val}%`, 'Importance']}
                      />
                      <Bar dataKey="value" fill="#38bdf8" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* RF Permutation Importance */}
            <Card className="bg-slate-900/60 border-slate-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-slate-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-emerald-400" />
                    <span>Test-Set Permutation Importance (30 Repeats)</span>
                  </div>
                  <Badge variant="outline" className="text-[10px] text-emerald-400 border-emerald-500/30">Official N=100</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-2">
                <p className="text-xs text-slate-400 mb-4 font-mono">
                  Mean degradation in test MAE (in cycles) when feature column is shuffled at random.
                </p>
                <div className="h-60 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={rfPerm} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                      <XAxis type="number" unit=" cyc" stroke="#64748b" tick={{ fontSize: 11 }} />
                      <YAxis dataKey="name" type="category" stroke="#94a3b8" tick={{ fontSize: 12, fill: '#cbd5e1' }} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc' }}
                        formatter={(val: any) => [`+${val} cycles`, 'MAE Increase']}
                      />
                      <Bar dataKey="value" fill="#10b981" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </>
        ) : (
          <>
            {/* LSTM Sensor Saliency */}
            <Card className="bg-slate-900/60 border-slate-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-slate-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-emerald-400" />
                    <span>LSTM Sensor Gradient Saliency |dY/dX|</span>
                  </div>
                  <Badge variant="outline" className="text-[10px] text-emerald-400 border-emerald-500/30">Top Sensors</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-2">
                <p className="text-xs text-slate-400 mb-4 font-mono">
                  First-order Taylor sensitivity w.r.t input sensors across all 100 test engines and 30 cycles.
                </p>
                <div className="h-60 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={lstmSensors} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                      <XAxis type="number" stroke="#64748b" tick={{ fontSize: 11 }} />
                      <YAxis dataKey="name" type="category" stroke="#94a3b8" tick={{ fontSize: 12, fill: '#cbd5e1' }} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc' }}
                        formatter={(val: any, _: any, item: any) => [`${val} (${item.payload.pct}%)`, 'Saliency Score']}
                      />
                      <Bar dataKey="value" fill="#10b981" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* LSTM Temporal Saliency Curve */}
            <Card className="bg-slate-900/60 border-slate-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-slate-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-sky-400" />
                    <span>Temporal Sensitivity Across 30-Cycle Input Window</span>
                  </div>
                  <Badge variant="outline" className="text-[10px] text-sky-400 border-sky-500/30">T-29 to T</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-2">
                <p className="text-xs text-slate-400 mb-4 font-mono">
                  Recurrent sensitivity dynamics: gradient attribution concentrates sharply in final cycles.
                </p>
                <div className="h-60 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={lstmTemporal} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                      <XAxis dataKey="cycle" stroke="#64748b" tick={{ fontSize: 10, fill: '#94a3b8' }} angle={-35} textAnchor="end" interval={3} />
                      <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc' }}
                        formatter={(val: any) => [val, 'Gradient Attribution']}
                      />
                      <Line type="monotone" dataKey="saliency" stroke="#38bdf8" strokeWidth={2.5} dot={{ fill: '#0284c7', r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>

      {/* Main Section 2: Representative Engine Inspector (Local Perturbations) */}
      <Card className="bg-slate-900/60 border-slate-800">
        <CardHeader className="pb-3 border-b border-slate-800">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div>
              <CardTitle className="text-base text-white flex items-center gap-2">
                <Gauge className="w-5 h-5 text-accent" />
                <span>Local Engine Explainability & Perturbation Attribution</span>
              </CardTitle>
              <p className="text-xs text-slate-400 mt-1 font-mono">
                Inspect local feature attributions and degradation drivers for representative test-set engines.
              </p>
            </div>

            {/* Engine ID Selector */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-slate-400">Select Engine:</span>
              <div className="flex items-center gap-1.5">
                {[1, 34, 45, 77, 82].map((id) => (
                  <button
                    key={id}
                    onClick={() => setSelectedEngineId(id)}
                    className={`px-3 py-1 rounded-md text-xs font-mono font-bold transition ${
                      selectedEngineId === id
                        ? 'bg-accent text-slate-950 shadow'
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                    }`}
                  >
                    Unit #{id}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardHeader>

        <CardContent className="pt-4">
          {activeEngineExplanation ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Left Column: Diagnostics Summary */}
              <div className="bg-slate-950/50 border border-slate-800/80 rounded-xl p-4 font-mono text-xs space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                  <span className="text-slate-400">ENGINE ID</span>
                  <span className="text-white font-bold">Unit #{activeEngineExplanation.engine_id}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">ACTUAL CAPPED RUL</span>
                  <span className="text-emerald-400 font-bold">{activeEngineExplanation.actual_rul} cycles</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">MODEL PREDICTED RUL</span>
                  <span className="text-sky-400 font-bold">{activeEngineExplanation.predicted_rul} cycles</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">RESIDUAL ERROR</span>
                  <span className={activeEngineExplanation.residual >= 0 ? "text-amber-400 font-bold" : "text-rose-400 font-bold"}>
                    {activeEngineExplanation.residual > 0 ? `+${activeEngineExplanation.residual}` : activeEngineExplanation.residual} cycles
                  </span>
                </div>
                {activeEngineExplanation.final_observed_cycle && (
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">FINAL CYCLE OBSERVED</span>
                    <span className="text-slate-200">{activeEngineExplanation.final_observed_cycle}</span>
                  </div>
                )}
                <div className="pt-2 text-[11px] text-slate-400 border-t border-slate-800">
                  {activeEngineExplanation.actual_rul <= 15 ? (
                    <span className="text-rose-400 font-bold">⚠️ CRITICAL FAILURE DOMAIN (RUL &le; 15)</span>
                  ) : activeEngineExplanation.actual_rul <= 50 ? (
                    <span className="text-amber-400 font-bold">⚠️ MID-LIFE WEAR REGIME (RUL &le; 50)</span>
                  ) : (
                    <span className="text-emerald-400 font-bold">✓ HEALTHY / UPPER BOUND REGIME</span>
                  )}
                </div>
              </div>

              {/* Right Column: Top 5 Contributing Features */}
              <div className="md:col-span-2 space-y-3">
                <div className="text-xs font-mono font-bold text-slate-300 uppercase tracking-wider">
                  Top 5 Physical Drivers for Unit #{activeEngineExplanation.engine_id}
                </div>
                <div className="space-y-2">
                  {(selectedModel === 'rf' 
                    ? (activeEngineExplanation.top_5_contributing_features || [])
                    : (activeEngineExplanation.top_contributing_sensors || [])
                  ).map((item: any, idx: number) => {
                    const featName = item.feature || item.sensor;
                    const sensitivity = item.sensitivity_cycles_per_5pct || item.attribution;
                    return (
                      <div key={idx} className="bg-slate-950/40 border border-slate-800/80 rounded-lg p-2.5 flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-slate-400 w-4">#{idx + 1}</span>
                          <span className="font-semibold text-slate-100">{featName}</span>
                          {item.baseline_value !== undefined && (
                            <span className="text-[11px] font-mono text-slate-400 bg-slate-800/50 px-1.5 py-0.5 rounded">
                              val: {item.baseline_value}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-3 font-mono">
                          <span className="text-sky-400 font-bold">
                            {selectedModel === 'rf' ? `±${sensitivity} cyc response` : `Saliency: ${sensitivity}`}
                          </span>
                          {item.effect_direction && (
                            <Badge variant="outline" className={item.effect_direction === 'negative' ? 'border-rose-500/40 text-rose-400' : 'border-emerald-500/40 text-emerald-400'}>
                              {item.effect_direction}
                            </Badge>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400 font-mono">No explanation available for selected engine.</p>
          )}
        </CardContent>
      </Card>

      {/* Main Section 3: Sensor Noise Robustness & Fault Tolerance Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Noise Sensitivity Table */}
        <Card className="bg-slate-900/60 border-slate-800">
          <CardHeader className="pb-3 border-b border-slate-800">
            <CardTitle className="text-base text-slate-200 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-sky-400" />
              <span>Gaussian Sensor Noise Degradation Matrix</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4 overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="pb-2">Noise Level</th>
                  <th className="pb-2">RF Mean Δ</th>
                  <th className="pb-2">RF Stability</th>
                  <th className="pb-2">LSTM Mean Δ</th>
                  <th className="pb-2">LSTM Stability</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {(robustData?.gaussian_noise_sensitivity || []).map((row: any, i: number) => (
                  <tr key={i} className="hover:bg-slate-800/30">
                    <td className="py-2.5 font-bold text-slate-100">{row.noise_level}</td>
                    <td className="py-2.5 text-sky-400 font-bold">{row.rf_mean_delta_rul} cyc</td>
                    <td className="py-2.5 text-emerald-400 font-bold">{row.rf_stability_pct}%</td>
                    <td className="py-2.5 text-emerald-400 font-bold">{row.lstm_mean_delta_rul} cyc</td>
                    <td className="py-2.5 text-emerald-400 font-bold">{row.lstm_stability_pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        {/* Fault & Boundary Stress Audit */}
        <Card className="bg-slate-900/60 border-slate-800">
          <CardHeader className="pb-3 border-b border-slate-800">
            <CardTitle className="text-base text-slate-200 flex items-center gap-2">
              <Activity className="w-5 h-5 text-amber-400" />
              <span>Boundary Fault & Schema Stress Audit</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4 overflow-x-auto">
            <div className="space-y-2.5 font-mono text-xs">
              {(robustData?.stress_and_fault_scenarios || []).slice(0, 4).map((sc: any, i: number) => (
                <div key={i} className="bg-slate-950/40 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between">
                  <div>
                    <div className="font-bold text-slate-200">{sc.scenario}</div>
                    <div className="text-[11px] text-slate-400">{sc.handling_mechanism}</div>
                  </div>
                  <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 font-bold">
                    {sc.status}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Section 4: Performance Profiling & Deployment Sizing */}
      <Card className="bg-slate-900/60 border-slate-800">
        <CardHeader className="pb-3 border-b border-slate-800">
          <CardTitle className="text-base text-slate-200 flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-purple-400" />
            <span>Inference Latency & Deployment Footprint Benchmarks</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-4 overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="pb-2">Model</th>
                <th className="pb-2">Artifact Size</th>
                <th className="pb-2">Model Load (Cold)</th>
                <th className="pb-2">First Inference (Cold)</th>
                <th className="pb-2">Warm Mean</th>
                <th className="pb-2">P95 Latency</th>
                <th className="pb-2">Throughput</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {(perfData?.models || []).map((m: any, i: number) => (
                <tr key={i} className="hover:bg-slate-800/30">
                  <td className="py-2.5 font-bold text-slate-100">{m.model}</td>
                  <td className="py-2.5 text-purple-400">{m.artifact_size_mb} MB</td>
                  <td className="py-2.5 text-slate-300">{m.load_time_ms} ms</td>
                  <td className="py-2.5 text-slate-300">{m.cold_start_lat_ms} ms</td>
                  <td className="py-2.5 text-sky-400 font-bold">{m.warm_mean_lat_ms} ms</td>
                  <td className="py-2.5 text-amber-400 font-bold">{m.warm_p95_lat_ms} ms</td>
                  <td className="py-2.5 text-emerald-400 font-bold">{m.throughput_req_per_sec} req/s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}
