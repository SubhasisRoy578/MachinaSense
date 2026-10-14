import React, { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { LocalExplanationResponse } from '../../types/explainability';
import { Sparkles, AlertCircle, X, ShieldAlert, TrendingUp, TrendingDown, Layers } from 'lucide-react';

interface PredictionExplanationModalProps {
  predictionId?: string;
  modelId?: string;
  features?: Record<string, any>;
  onClose: () => void;
}

export const PredictionExplanationModal: React.FC<PredictionExplanationModalProps> = ({
  predictionId,
  modelId,
  features,
  onClose,
}) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<LocalExplanationResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function loadLocalExplanation() {
      setLoading(true);
      setError(null);
      try {
        let res: LocalExplanationResponse;
        if (predictionId) {
          res = await api.explainability.getPredictionLocal(predictionId);
        } else if (modelId && features) {
          res = await api.explainability.explainLocal(modelId, features);
        } else {
          throw new Error('Either predictionId or modelId+features must be provided.');
        }
        if (isMounted) setData(res);
      } catch (err: any) {
        if (isMounted) setError(err.message || 'Failed to compute local prediction explanation.');
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadLocalExplanation();
    return () => { isMounted = false; };
  }, [predictionId, modelId, features]);

  const maxContribution = Math.max(
    ...(data?.contributions?.map((c) => Math.abs(c.contribution)) || [1]),
    0.01
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-6 text-slate-100">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="flex items-center space-x-3 mb-5">
          <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Local Prediction Attribution</h3>
            <p className="text-xs text-slate-400">
              Model:{' '}
              <span className="text-slate-200 font-medium">
                {data?.model_name || modelId || 'Machine Model'}
              </span>{' '}
              ({data?.task?.replace(/_/g, ' ') || 'ML Task'})
            </p>
          </div>
        </div>

        {loading ? (
          <div className="py-12 text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500 mb-3" />
            <p className="text-xs text-slate-400">Evaluating marginal sensor deviations against baseline...</p>
          </div>
        ) : error ? (
          <div className="p-4 bg-red-950/20 border border-red-800/40 rounded-xl text-xs text-red-300">
            <div className="flex items-center space-x-2 font-semibold mb-1">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <span>Explanation Unavailable</span>
            </div>
            <p className="text-slate-400">{error}</p>
          </div>
        ) : data ? (
          <div className="space-y-5">
            {/* Prediction Summary Card */}
            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center justify-between text-xs">
              <div>
                <span className="text-slate-400 block text-[11px] uppercase font-semibold">
                  Original Prediction Output
                </span>
                <div className="text-base font-bold text-white mt-0.5">
                  {data.prediction_output.rul_cycles !== undefined && (
                    <span>{data.prediction_output.rul_cycles} Cycles RUL</span>
                  )}
                  {data.prediction_output.condition !== undefined && (
                    <span>Condition: {data.prediction_output.condition}</span>
                  )}
                  {data.prediction_output.is_anomaly !== undefined && (
                    <span>
                      {data.prediction_output.is_anomaly ? '⚠️ Anomalous' : '✅ Normal'} (Score: {data.prediction_output.anomaly_score})
                    </span>
                  )}
                </div>
              </div>
              <div className="text-right">
                <span className="text-slate-500 block text-[10px]">Method</span>
                <span className="text-indigo-400 font-mono text-xs">
                  {data.explanation_method.replace(/_/g, ' ')}
                </span>
              </div>
            </div>

            {/* Contributing Features */}
            <div>
              <div className="flex items-center justify-between text-xs mb-3">
                <span className="font-semibold text-slate-200 flex items-center space-x-1.5">
                  <Layers className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Top Contributing Sensor Deviations</span>
                </span>
                <span className="text-[11px] text-slate-500">Relative to nominal baseline</span>
              </div>

              <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
                {data.contributions.map((c) => {
                  const isPositive = c.contribution >= 0;
                  const isNegativeImpact =
                    c.impact_direction.includes('decreases') ||
                    c.impact_direction.includes('elevates') ||
                    c.impact_direction.includes('opposes');
                  const barWidth = Math.max(5, Math.round((Math.abs(c.contribution) / maxContribution) * 100));

                  return (
                    <div
                      key={c.feature}
                      className="p-2.5 rounded-lg bg-slate-950/40 border border-slate-800/80 text-xs"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center space-x-2">
                          {isNegativeImpact ? (
                            <TrendingDown className="w-3.5 h-3.5 text-rose-400 flex-shrink-0" />
                          ) : (
                            <TrendingUp className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                          )}
                          <span className="font-medium text-slate-200">{c.label}</span>
                          <span className="text-[10px] font-mono text-slate-500">({c.feature})</span>
                          <span className="text-[10px] text-slate-400 bg-slate-800 px-1.5 py-0.2 rounded font-mono">
                            {c.value !== undefined ? c.value : 'N/A'} {c.unit}
                          </span>
                        </div>
                        <div className="font-mono text-xs font-semibold">
                          <span className={isNegativeImpact ? 'text-rose-400' : 'text-emerald-400'}>
                            {isPositive ? `+${c.contribution}` : c.contribution}
                          </span>
                        </div>
                      </div>

                      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden mb-1">
                        <div
                          className={`h-full rounded-full ${
                            isNegativeImpact ? 'bg-rose-500' : 'bg-emerald-500'
                          }`}
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>

                      <p className="text-[11px] text-slate-400">{c.description}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Scientific Scope Disclaimer */}
            <div className="p-3 rounded-xl bg-amber-950/20 border border-amber-900/30 text-xs text-slate-400 flex items-start space-x-2">
              <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-amber-300 block mb-0.5">Scientific Interpretation</span>
                <p className="text-[11px] leading-relaxed">{data.limitations}</p>
              </div>
            </div>
          </div>
        ) : null}

        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-xs font-medium text-white rounded-xl transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
