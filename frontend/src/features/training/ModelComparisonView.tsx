import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../../services/api';
import type { ModelComparisonResponse, ModelComparisonMetric } from '../../types/comparison';
import type { RegisteredModel } from '../../types/training';
import { GitCompare, CheckCircle2, AlertTriangle, XCircle, Info, Trophy, RefreshCw } from 'lucide-react';

interface ModelComparisonViewProps {
  initialModelIds?: string[];
}

const BENCHMARK_MODELS = [
  { id: 'benchmark_rf', name: 'NASA Random Forest Regressor', task: 'rul_regression' },
  { id: 'benchmark_lstm', name: 'NASA PyTorch LSTM Regressor', task: 'rul_regression' },
  { id: 'benchmark_isolation_forest', name: 'NASA Isolation Forest Anomaly', task: 'anomaly_detection' },
];

export const ModelComparisonView: React.FC<ModelComparisonViewProps> = ({ initialModelIds }) => {
  const [registeredModels, setRegisteredModels] = useState<RegisteredModel[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>(
    initialModelIds && initialModelIds.length >= 2 ? initialModelIds : ['benchmark_rf', 'benchmark_lstm']
  );
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ModelComparisonResponse | null>(null);

  useEffect(() => {
    async function loadModels() {
      try {
        const res = await api.models.list();
        setRegisteredModels(res.models || []);
      } catch (err: any) {
        console.error('Failed to list registered models for comparison:', err);
      }
    }
    loadModels();
  }, []);

  const runComparison = useCallback(async (idsToCompare: string[]) => {
    if (idsToCompare.length < 2) {
      setError('Select at least 2 models to perform a comparison.');
      setComparison(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await api.comparison.compare(idsToCompare);
      setComparison(res);
    } catch (err: any) {
      setError(err.message || 'Comparison failed.');
      setComparison(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    const targets = initialModelIds && initialModelIds.length >= 2 ? initialModelIds : ['benchmark_rf', 'benchmark_lstm'];
    const fetchInitial = async () => {
      try {
        const res = await api.comparison.compare(targets);
        if (!ignore) {
          setComparison(res);
        }
      } catch (err: any) {
        if (!ignore) {
          setError(err.message || 'Comparison failed.');
        }
      }
    };
    void fetchInitial();
    return () => {
      ignore = true;
    };
  }, [initialModelIds]);

  const handleToggleModel = (id: string) => {
    const next = selectedIds.includes(id)
      ? selectedIds.filter((item) => item !== id)
      : [...selectedIds, id];
    setSelectedIds(next);
    if (next.length >= 2) {
      runComparison(next);
    } else {
      setComparison(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header and Model Selector */}
      <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 text-slate-100 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center space-x-2.5">
              <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                <GitCompare className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-white">Comparative Model Evaluation</h3>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Select 2 or more registered or benchmark models to evaluate performance metrics, architectural differences, and evaluation protocols.
            </p>
          </div>
          <button
            onClick={() => runComparison(selectedIds)}
            disabled={loading || selectedIds.length < 2}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold rounded-xl text-white transition shadow-sm"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Recalculate Comparison</span>
          </button>
        </div>

        {/* Available Models Selection Chips */}
        <div>
          <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block mb-2">
            Select Models to Compare ({selectedIds.length} selected):
          </span>
          <div className="flex flex-wrap gap-2">
            {/* Benchmark Models */}
            {BENCHMARK_MODELS.map((bm) => {
              const isSelected = selectedIds.includes(bm.id);
              return (
                <button
                  key={bm.id}
                  onClick={() => handleToggleModel(bm.id)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-medium border transition flex items-center space-x-1.5 ${
                    isSelected
                      ? 'bg-blue-600/20 border-blue-500 text-blue-200'
                      : 'bg-slate-800/60 border-slate-700/60 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                  <span>{bm.name}</span>
                  <span className="text-[10px] text-blue-400/80 font-mono">[{bm.task}]</span>
                </button>
              );
            })}

            {/* User-Registered Models */}
            {registeredModels.map((rm) => {
              const isSelected = selectedIds.includes(rm.id);
              return (
                <button
                  key={rm.id}
                  onClick={() => handleToggleModel(rm.id)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-medium border transition flex items-center space-x-1.5 ${
                    isSelected
                      ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200'
                      : 'bg-slate-800/60 border-slate-700/60 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                  <span>
                    {rm.name} v{rm.version}
                  </span>
                  <span className="text-[10px] text-indigo-400/80 font-mono">[{rm.task}]</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="p-8 text-center bg-slate-900/50 rounded-2xl border border-slate-800 text-slate-400">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500 mb-3" />
          <p className="text-xs">Validating task schemas, target alignments, and held-out evaluation splits...</p>
        </div>
      )}

      {/* Error state */}
      {error && !loading && (
        <div className="p-4 rounded-xl bg-red-950/20 border border-red-800/40 text-red-300 text-xs flex items-start space-x-2">
          <XCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block mb-0.5">Comparison Error</span>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Incompatibility State */}
      {comparison && !comparison.is_compatible && (
        <div className="p-5 rounded-2xl bg-amber-950/20 border border-amber-800/50 text-slate-200">
          <div className="flex items-center space-x-2 text-amber-400 font-semibold text-sm mb-2">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            <h4>Incompatible Models for Direct Comparison</h4>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed mb-4">
            {comparison.incompatibility_reason}
          </p>
          <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-[11px] text-slate-400">
            <strong className="text-slate-300">Scientific Rule:</strong> Models must share the identical ML task (e.g. regression vs regression) to ensure metric alignment. Comparing regression error (MAE/RMSE) against anomaly detection scores or classification precision is mathematically invalid.
          </div>
        </div>
      )}

      {/* Valid Comparison Results */}
      {comparison && comparison.is_compatible && (
        <div className="space-y-6">
          {/* Validity & Protocol Banner */}
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 text-xs space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="font-semibold text-emerald-400 uppercase tracking-wider text-[11px]">
                  Task Compatibility Verified
                </span>
                <span className="px-2 py-0.5 rounded bg-slate-800 font-mono text-slate-300 text-[11px]">
                  {comparison.task}
                </span>
              </div>
              <span className="text-slate-400 text-[11px]">
                {comparison.identical_evaluation_dataset
                  ? '✅ Identical Evaluation Dataset'
                  : '⚠️ Heterogeneous Datasets'}
              </span>
            </div>
            <p className="text-slate-300 leading-relaxed">{comparison.narrative}</p>
            {comparison.dataset_warning && (
              <div className="p-2.5 rounded-lg bg-amber-950/20 border border-amber-900/30 text-amber-300 text-[11px] flex items-start space-x-1.5">
                <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-amber-400" />
                <span>{comparison.dataset_warning}</span>
              </div>
            )}
          </div>

          {/* Metric Comparison Matrix */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-semibold text-white flex items-center space-x-2">
                <Trophy className="w-4 h-4 text-amber-400" />
                <span>Evaluated Metric Matrix</span>
              </h4>
              <span className="text-[11px] text-slate-400">
                {comparison.identical_evaluation_dataset
                  ? 'Gold badges indicate top performance on shared held-out test split'
                  : 'Independent evaluation metrics across respective datasets'}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-200">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px] tracking-wider">
                    <th className="pb-3 pr-4 font-semibold">Evaluation Metric</th>
                    <th className="pb-3 px-3 font-semibold">Unit</th>
                    <th className="pb-3 px-3 font-semibold">Target Direction</th>
                    {comparison.models.map((m) => (
                      <th key={m.id} className="pb-3 px-3 font-semibold text-slate-200">
                        {m.name} {m.version ? `v${m.version}` : ''}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {comparison.metric_comparison.map((metric: ModelComparisonMetric) => (
                    <tr key={metric.metric_key} className="hover:bg-slate-800/30 transition">
                      <td className="py-3 pr-4 font-sans font-medium text-slate-200">
                        {metric.label}
                      </td>
                      <td className="py-3 px-3 text-slate-400 text-[11px] font-sans">
                        {metric.unit}
                      </td>
                      <td className="py-3 px-3 text-[11px] font-sans">
                        {metric.direction === 'lower_is_better' && (
                          <span className="text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                            Lower is better ↓
                          </span>
                        )}
                        {metric.direction === 'higher_is_better' && (
                          <span className="text-cyan-400 bg-cyan-500/10 px-1.5 py-0.5 rounded">
                            Higher is better ↑
                          </span>
                        )}
                        {metric.direction === 'neutral' && (
                          <span className="text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded">
                            Reference metric
                          </span>
                        )}
                      </td>
                      {comparison.models.map((m) => {
                        const val = metric.values[m.id];
                        const isBest = metric.best_model_id === m.id;
                        return (
                          <td key={m.id} className="py-3 px-3">
                            {val !== null && val !== undefined ? (
                              <span
                                className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded font-semibold ${
                                  isBest
                                    ? 'bg-amber-500/10 text-amber-300 border border-amber-500/30'
                                    : 'text-slate-300'
                                }`}
                              >
                                {isBest && <Trophy className="w-3 h-3 text-amber-400 inline" />}
                                <span>{typeof val === 'number' ? val.toFixed(4) : val}</span>
                              </span>
                            ) : (
                              <span className="text-slate-500 text-[11px] font-sans italic">
                                Not recorded
                              </span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model Specification Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {comparison.models.map((m) => (
              <div
                key={m.id}
                className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 text-xs space-y-3"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h5 className="font-semibold text-white text-sm">{m.name}</h5>
                    <span className="text-[10px] text-slate-400 font-mono">
                      {m.model_type} • v{m.version}
                    </span>
                  </div>
                  {m.is_benchmark ? (
                    <span className="px-2 py-0.5 text-[9px] uppercase font-bold tracking-wider rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                      Benchmark
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 text-[9px] uppercase font-bold tracking-wider rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                      User Trained
                    </span>
                  )}
                </div>

                <div className="space-y-1.5 text-slate-400 text-[11px] border-t border-slate-800/60 pt-2">
                  <p>
                    <strong className="text-slate-300">Dataset:</strong> {m.dataset_name}
                  </p>
                  <p>
                    <strong className="text-slate-300">Protocol:</strong> {m.split_protocol}
                  </p>
                  {m.sample_count && (
                    <p>
                      <strong className="text-slate-300">Held-Out Samples:</strong> {m.sample_count}
                    </p>
                  )}
                  {m.duration_seconds && (
                    <p>
                      <strong className="text-slate-300">Training Time:</strong> {m.duration_seconds}s
                    </p>
                  )}
                  {m.strengths && (
                    <p className="text-emerald-300">
                      <strong>Strengths:</strong> {m.strengths}
                    </p>
                  )}
                  {m.limitations && (
                    <p className="text-slate-400 italic">
                      <strong>Limitations:</strong> {m.limitations}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Operational Guidance */}
          {comparison.guidance && (
            <div className="p-4 rounded-2xl bg-slate-950/40 border border-slate-800/80 text-xs text-slate-400 flex items-start space-x-2">
              <Info className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-slate-200 block mb-0.5">Operational Guidance</span>
                <p className="text-[11px] leading-relaxed">{comparison.guidance}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
