import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import { Card, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { formatDistanceToNow, format } from 'date-fns';
import { Search, AlertOctagon, Filter, ArrowRight, Sparkles, X, ShieldAlert } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { ModelExplanationView } from '../training/ModelExplanationView';
import type { RegisteredModel } from '../../types/training';

export function AnomaliesPage() {
  const navigate = useNavigate();
  const [anomalies, setAnomalies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [activeAnomalyModel, setActiveAnomalyModel] = useState<RegisteredModel | null>(null);
  const [selectedAnomalyForExplain, setSelectedAnomalyForExplain] = useState<any | null>(null);
  const [explainBenchmarkId, setExplainBenchmarkId] = useState<string | null>(null);
  const [explainModelId, setExplainModelId] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnomalies = async () => {
      setLoading(true);
      try {
        const [data, model] = await Promise.all([
          api.diagnostics.listFleetAnomalies(),
          api.models.getActive('anomaly_detection').catch(() => null),
        ]);
        setAnomalies(data);
        setActiveAnomalyModel(model || null);
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

      {/* ML Source Banner */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase font-bold">ML SOURCE</span>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-500/20 text-sky-400 border border-sky-500/30">
              BENCHMARK BASELINE
            </span>
          </div>
          <div className="font-bold text-slate-100">C-MAPSS Sensor Residual Baseline</div>
          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>Task: <span className="text-accent">Anomaly Detection</span> | Status: <span className="text-emerald-400 font-bold">ACTIVE</span></span>
            <button
              type="button"
              onClick={() => setExplainBenchmarkId('isolation_forest')}
              className="text-[10px] text-purple-400 hover:text-purple-300 font-semibold flex items-center gap-1 border border-purple-500/30 px-2 py-0.5 rounded bg-purple-500/10 transition-colors"
            >
              <Sparkles className="w-3 h-3" />
              <span>Explain Detector</span>
            </button>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase font-bold">ML SOURCE</span>
            {activeAnomalyModel ? (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                ACTIVE USER MODEL
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-500">
                NO ACTIVE MODEL
              </span>
            )}
          </div>
          <div className="font-bold text-slate-100">
            {activeAnomalyModel ? `${activeAnomalyModel.name} (v${activeAnomalyModel.version})` : 'User Isolation Forest Detector'}
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-slate-400">
              Algorithm: <span className="text-slate-200">{activeAnomalyModel?.model_type || 'Isolation Forest'}</span>
            </span>
            <div className="flex items-center gap-2">
              {activeAnomalyModel && (
                <button
                  type="button"
                  onClick={() => setExplainModelId(activeAnomalyModel.id)}
                  className="text-[10px] text-purple-400 hover:text-purple-300 font-semibold flex items-center gap-1 border border-purple-500/30 px-2 py-0.5 rounded bg-purple-500/10 transition-colors"
                >
                  <Sparkles className="w-3 h-3" />
                  <span>Explain Model</span>
                </button>
              )}
              <button
                type="button"
                onClick={() => navigate('/model-training')}
                className="text-accent hover:underline flex items-center gap-1"
              >
                <span>{activeAnomalyModel ? 'Test Detector' : 'Train Custom Detector'}</span>
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          </div>
        </div>
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
                            type="button"
                            onClick={() => setSelectedAnomalyForExplain(anom)}
                            className="inline-flex items-center justify-center px-2 py-1 rounded text-xs font-medium text-purple-400 bg-purple-500/10 border border-purple-500/30 hover:bg-purple-500/20 transition-colors"
                            title="Inspect decision semantics and score interpretation"
                          >
                            <Sparkles className="w-3 h-3 mr-1" /> Explain Semantics
                          </button>
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
      {/* ANOMALY DECISION SEMANTICS & SENSITIVITY MODAL */}
      {selectedAnomalyForExplain && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-xl w-full max-h-[85vh] flex flex-col font-mono text-xs shadow-2xl">
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <AlertOctagon className="w-4 h-4 text-amber-400" />
                <h3 className="font-bold text-slate-100">
                  Anomaly Decision Semantics: {selectedAnomalyForExplain.id}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedAnomalyForExplain(null)}
                className="text-slate-400 hover:text-slate-200 p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-4 overflow-y-auto flex-1 space-y-4">
              {/* Critical Methodology Warning */}
              <div className="bg-amber-500/10 border border-amber-500/30 rounded p-3 text-amber-300 space-y-1">
                <div className="flex items-center gap-2 font-bold text-amber-400 text-xs">
                  <ShieldAlert className="w-4 h-4" />
                  <span>Important ML Governance Notice</span>
                </div>
                <p className="text-[11px] leading-relaxed">
                  Anomaly scores represent an unsupervised statistical distance or divergence from normal multi-sensor baseline behavior. Anomaly scores are <strong className="text-amber-200 underline">NOT</strong> failure probabilities or confirmed hardware defects.
                </p>
              </div>

              {/* Observed Telemetry Metric */}
              <div className="bg-slate-950 p-3 rounded border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-slate-400">
                  <span>Target Machine:</span>
                  <span className="text-slate-100 font-bold">{selectedAnomalyForExplain.machineName} ({selectedAnomalyForExplain.machineId})</span>
                </div>
                <div className="flex justify-between items-center text-slate-400">
                  <span>Affected Telemetry Channel:</span>
                  <span className="text-slate-200 font-bold">{selectedAnomalyForExplain.sensor}</span>
                </div>
                <div className="flex justify-between items-center text-slate-400">
                  <span>Normalized Divergence Score:</span>
                  <span className="text-xl font-bold text-amber-400">{selectedAnomalyForExplain.score.toFixed(2)}</span>
                </div>
                <div className="flex justify-between items-center text-slate-400">
                  <span>Classification Severity:</span>
                  <span className="uppercase text-slate-200 font-bold">{selectedAnomalyForExplain.severity}</span>
                </div>
              </div>

              {/* Decision Function Semantics */}
              <div className="space-y-1.5">
                <span className="text-slate-400 font-bold block">MODEL DECISION FUNCTION:</span>
                <p className="text-slate-300 text-[11px] leading-relaxed bg-slate-950 p-3 rounded border border-slate-800">
                  The model utilizes an unsupervised tree isolation algorithm. Normal sensor snapshots require deep partitioning paths to isolate, whereas outlier spikes or anomalous sensor drift are isolated quickly in early splits. The normalized score reflects the inverse average path length across all isolation trees.
                </p>
              </div>

              {/* Model Attribution Status */}
              <div className="space-y-1.5">
                <span className="text-slate-400 font-bold block">ATTRIBUTION LIMITATIONS:</span>
                <p className="text-slate-400 text-[11px] leading-relaxed bg-slate-950 p-3 rounded border border-slate-800">
                  Tree-based Shapley (SHAP) or Mean Decrease in Impurity (MDI) attribution is mathematically invalid for unsupervised Isolation Forests because split criteria do not optimize target label variance or Gini impurity. Global multi-sensor sensitivity checks can be reviewed in the Model Explanation view.
                </p>
              </div>
            </div>

            <div className="p-3 border-t border-slate-800 flex justify-between items-center">
              <button
                type="button"
                onClick={() => {
                  setSelectedAnomalyForExplain(null);
                  setExplainBenchmarkId('isolation_forest');
                }}
                className="text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1 font-bold"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>View Isolation Forest Method Details</span>
              </button>
              <button
                type="button"
                onClick={() => setSelectedAnomalyForExplain(null)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-1.5 rounded text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* BENCHMARK EXPLANATION MODAL */}
      {explainBenchmarkId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <ModelExplanationView
              benchmarkId={explainBenchmarkId}
              modelName="NASA C-MAPSS Isolation Forest Detector"
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
    </div>
  );
}
