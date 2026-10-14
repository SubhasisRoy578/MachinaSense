import React, { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { GlobalExplanationResponse } from '../../types/explainability';
import { Sparkles, AlertCircle, Info, ShieldAlert, BarChart3, HelpCircle } from 'lucide-react';

interface ModelExplanationViewProps {
  modelId?: string;
  benchmarkId?: string;
  modelName?: string;
  onClose?: () => void;
}

export const ModelExplanationView: React.FC<ModelExplanationViewProps> = ({ modelId, benchmarkId, modelName, onClose }) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<GlobalExplanationResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function loadExplanation() {
      if (!modelId && !benchmarkId) return;
      setLoading(true);
      setError(null);
      try {
        const res = benchmarkId
          ? await api.explainability.getBenchmark(benchmarkId)
          : await api.explainability.getModelGlobal(modelId!);
        if (isMounted) setData(res);
      } catch (err: any) {
        if (isMounted) setError(err.message || 'Failed to load model explainability.');
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadExplanation();
    return () => { isMounted = false; };
  }, [modelId, benchmarkId]);

  if (loading) {
    return (
      <div className="p-8 text-center bg-slate-900/50 rounded-xl border border-slate-800">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500 mb-3" />
        <p className="text-sm text-slate-400">Computing model feature attributions & sensitivity surface...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-6 bg-red-950/20 rounded-xl border border-red-800/50 text-slate-200">
        <div className="flex items-center space-x-2 text-red-400 mb-2">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <h4 className="font-semibold text-sm">Explainability Unavailable</h4>
        </div>
        <p className="text-xs text-slate-400 mb-4">{error || 'No explanation data returned.'}</p>
        {onClose && (
          <button
            onClick={onClose}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs font-medium rounded-lg transition"
          >
            Close
          </button>
        )}
      </div>
    );
  }

  if (!data.supported) {
    return (
      <div className="p-6 bg-amber-950/20 rounded-xl border border-amber-800/50 text-slate-200">
        <div className="flex items-center space-x-2 text-amber-400 mb-2">
          <HelpCircle className="w-5 h-5 flex-shrink-0" />
          <h4 className="font-semibold text-sm">Architecture Not Directly Supported</h4>
        </div>
        <p className="text-xs text-slate-400 mb-2">{data.method_description}</p>
        <p className="text-xs text-slate-500 italic mb-4">{data.limitations}</p>
        {onClose && (
          <button
            onClick={onClose}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs font-medium rounded-lg transition"
          >
            Dismiss
          </button>
        )}
      </div>
    );
  }

  const maxImp = Math.max(...(data.features?.map((f) => f.importance) || [1]), 0.001);

  return (
    <div className="space-y-6 text-slate-100">
      {/* Header Info */}
      <div className="flex flex-wrap items-start justify-between gap-4 p-4 rounded-xl bg-slate-900/80 border border-slate-800">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <Sparkles className="w-4 h-4" />
            </span>
            <h3 className="font-semibold text-base text-white">{data.model_name || modelName}</h3>
            {data.is_benchmark && (
              <span className="px-2 py-0.5 text-[10px] uppercase font-bold tracking-wider rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                NASA Benchmark
              </span>
            )}
            <span className="px-2 py-0.5 text-[10px] uppercase font-bold tracking-wider rounded bg-slate-800 text-slate-300">
              v{data.version}
            </span>
          </div>
          <p className="mt-1.5 text-xs text-slate-400">
            Method:{' '}
            <span className="font-mono text-indigo-300 font-medium">
              {data.explanation_method.replace(/_/g, ' ')}
            </span>
          </p>
          <p className="text-xs text-slate-400 mt-0.5">{data.method_description}</p>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white text-xs px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 transition"
          >
            Close
          </button>
        )}
      </div>

      {/* Feature Attribution Rankings */}
      <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4 text-indigo-400" />
            <h4 className="text-sm font-semibold text-white">Feature Importance & Sensor Sensitivity</h4>
          </div>
          <span className="text-[11px] text-slate-400">Normalized Contribution (Sum = 100%)</span>
        </div>

        <div className="space-y-3">
          {data.features.map((feat) => {
            const pct = Math.round(feat.importance * 100);
            const barWidth = Math.max(4, Math.round((feat.importance / maxImp) * 100));

            return (
              <div key={feat.feature} className="group">
                <div className="flex items-center justify-between text-xs mb-1">
                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] font-mono text-slate-500 w-4 text-right">
                      #{feat.rank}
                    </span>
                    <span className="font-medium text-slate-200">{feat.label || feat.feature}</span>
                    <span className="text-[11px] font-mono text-slate-400">({feat.feature})</span>
                    {feat.unit && (
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-400">
                        {feat.unit}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center space-x-2 font-mono">
                    <span className="text-xs font-semibold text-indigo-300">{pct}%</span>
                    <span className="text-[10px] text-slate-500">
                      (raw: {feat.raw_score.toFixed(4)})
                    </span>
                  </div>
                </div>
                <div className="h-2 w-full bg-slate-800/80 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 rounded-full transition-all duration-500"
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Evaluation Context & Limitations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {data.evaluation_context && (
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs">
            <div className="flex items-center space-x-2 text-slate-300 font-semibold mb-2">
              <Info className="w-3.5 h-3.5 text-cyan-400" />
              <span>Evaluation Context</span>
            </div>
            <div className="space-y-1.5 text-slate-400">
              <p>
                <strong className="text-slate-300">Dataset:</strong>{' '}
                {data.evaluation_context.dataset_name || data.evaluation_context.dataset || 'N/A'}
              </p>
              {data.evaluation_context.split && (
                <p>
                  <strong className="text-slate-300">Protocol:</strong> {data.evaluation_context.split}
                </p>
              )}
              {data.evaluation_context.sample_count && (
                <p>
                  <strong className="text-slate-300">Test Samples:</strong>{' '}
                  {data.evaluation_context.sample_count} units
                </p>
              )}
              {data.evaluation_context.metrics && (
                <div className="mt-2 pt-2 border-t border-slate-800/60">
                  <span className="text-slate-400 block mb-1 font-semibold">Persisted Benchmark Metrics:</span>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(data.evaluation_context.metrics).map(([k, v]) => (
                      <span key={k} className="px-2 py-0.5 rounded bg-slate-800 font-mono text-[11px] text-slate-300">
                        {k.toUpperCase()}: {typeof v === 'number' ? v.toFixed(3) : String(v)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="p-4 rounded-xl bg-slate-900/60 border border-amber-900/30 text-xs">
          <div className="flex items-center space-x-2 text-amber-400 font-semibold mb-2">
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>Scientific Scope & Limitations</span>
          </div>
          <p className="text-slate-300 leading-relaxed">{data.limitations}</p>
          <p className="mt-2 text-[11px] text-slate-500 italic">
            Statistical feature attribution measures variance contribution and model sensitivity. It does not establish proven mechanical fault causation.
          </p>
        </div>
      </div>
    </div>
  );
};
