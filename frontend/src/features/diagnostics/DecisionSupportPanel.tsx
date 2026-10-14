import React, { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { DecisionSupportResponse } from '../../types/decisionSupport';
import {
  FileText,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  ShieldCheck,
  Stethoscope,
  Clock,
  BookOpen,
  Activity,
  Send,
} from 'lucide-react';

interface DecisionSupportPanelProps {
  machineId?: string;
  predictionId?: string;
}

export const DecisionSupportPanel: React.FC<DecisionSupportPanelProps> = ({
  machineId,
  predictionId,
}) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<DecisionSupportResponse | null>(null);
  const [symptomsInput, setSymptomsInput] = useState<string>('');

  const loadDecisionSupport = async (querySymptoms?: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.decisionSupport.generate({
        machine_id: machineId,
        prediction_id: predictionId,
        symptoms: querySymptoms || undefined,
      });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to generate decision support report.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDecisionSupport();
  }, [machineId, predictionId]);

  const handleSearchSymptoms = (e: React.FormEvent) => {
    e.preventDefault();
    if (symptomsInput.trim()) {
      loadDecisionSupport(symptomsInput.trim());
    }
  };

  return (
    <div className="space-y-6 text-slate-100">
      {/* Panel Header */}
      <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
              <Stethoscope className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-white">
                Evidence-Backed Maintenance Decision Support
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Synthesizes verified ML predictions with retrieved technical documentation for grounded maintenance guidance.
              </p>
            </div>
          </div>

          {/* Quick Symptoms Prompt */}
          <form onSubmit={handleSearchSymptoms} className="flex items-center space-x-2">
            <input
              type="text"
              placeholder="Filter by symptoms (e.g. vibration, seal leak)..."
              value={symptomsInput}
              onChange={(e) => setSymptomsInput(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 w-64"
            />
            <button
              type="submit"
              disabled={loading}
              className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-semibold flex items-center space-x-1 transition disabled:opacity-50"
            >
              <Send className="w-3.5 h-3.5" />
              <span>Search</span>
            </button>
          </form>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="p-8 text-center bg-slate-900/50 rounded-2xl border border-slate-800 text-slate-400">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500 mb-3" />
          <p className="text-xs">Correlating telemetry observations with technical library passages...</p>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="p-4 rounded-xl bg-red-950/20 border border-red-800/40 text-red-300 text-xs flex items-start space-x-2">
          <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block mb-0.5">Decision Support Error</span>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Content */}
      {data && !loading && (
        <div className="space-y-6">
          {/* Section 1: Observed Evidence */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <h4 className="text-sm font-semibold text-white">1. Observed Telemetry & ML Evidence</h4>
              </div>
              <span
                className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                  data.observed_evidence.risk_level === 'critical'
                    ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                    : data.observed_evidence.risk_level === 'warning'
                    ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                    : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                }`}
              >
                Risk: {data.observed_evidence.risk_level}
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/60">
                <span className="text-slate-400 block text-[10px] uppercase font-semibold">Predicted RUL</span>
                <div className="text-base font-bold text-white mt-0.5">
                  {data.observed_evidence.predicted_rul_cycles !== null && data.observed_evidence.predicted_rul_cycles !== undefined
                    ? `${data.observed_evidence.predicted_rul_cycles.toFixed(1)} Cycles`
                    : 'Not Available'}
                </div>
                <span className="text-[10px] text-slate-500">
                  Model: {data.observed_evidence.rul_model_used}
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/60">
                <span className="text-slate-400 block text-[10px] uppercase font-semibold">Anomaly Status</span>
                <div className="text-base font-bold text-white mt-0.5 capitalize">
                  {data.observed_evidence.anomaly_severity}
                </div>
                <span className="text-[10px] text-slate-500 font-mono">
                  Score: {data.observed_evidence.anomaly_score.toFixed(3)}
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/60">
                <span className="text-slate-400 block text-[10px] uppercase font-semibold">Confidence Interval</span>
                <div className="text-base font-bold text-white mt-0.5">
                  {data.observed_evidence.confidence_interval
                    ? `[${data.observed_evidence.confidence_interval.min_cycles} – ${data.observed_evidence.confidence_interval.max_cycles}]`
                    : 'FD001 Bounds'}
                </div>
                <span className="text-[10px] text-slate-500">Cycles span</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/60">
                <span className="text-slate-400 block text-[10px] uppercase font-semibold">Equipment Asset</span>
                <div className="text-base font-bold text-white mt-0.5 truncate">
                  {data.observed_evidence.machine_name}
                </div>
                <span className="text-[10px] text-slate-500">
                  {data.observed_evidence.machine_type}
                </span>
              </div>
            </div>

            {/* Sensor channels summary if available */}
            {data.observed_evidence.telemetry_observations && (
              <div className="pt-2">
                <span className="text-[11px] font-semibold text-slate-400 block mb-2">
                  Observed Sensor Transducer Values:
                </span>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(data.observed_evidence.telemetry_observations).map(([key, val]: [string, any]) => {
                    if (typeof val === 'object' && val !== null) {
                      return (
                        <div
                          key={key}
                          className="px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-300"
                        >
                          <span className="text-slate-400 font-sans">{val.label || key}:</span>{' '}
                          <span className="text-cyan-400 font-semibold">{val.value}</span> {val.unit}
                        </div>
                      );
                    }
                    return (
                      <div
                        key={key}
                        className="px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-300"
                      >
                        <span className="text-slate-400 font-sans">{key}:</span> {String(val)}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Section 2: Supporting Technical Documentation (RAG Provenance) */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center space-x-2">
                <BookOpen className="w-4 h-4 text-indigo-400" />
                <h4 className="text-sm font-semibold text-white">
                  2. Grounded Technical Documentation (RAG Provenance)
                </h4>
              </div>
              <span className="text-[11px] text-slate-400">
                {data.supporting_documentation.length} verified passages cited
              </span>
            </div>

            {data.supporting_documentation.length === 0 ? (
              <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-800 text-xs text-slate-400 flex items-center space-x-2">
                <HelpCircle className="w-4 h-4 text-slate-500 flex-shrink-0" />
                <span>
                  No matching technical manual passages were indexed in the Knowledge Base for this query. Upload OEM maintenance manuals to enrich citations.
                </span>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {data.supporting_documentation.map((doc) => (
                  <div
                    key={doc.chunk_id}
                    className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs space-y-2 flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <div className="flex items-center space-x-1.5 text-indigo-300 font-semibold">
                          <FileText className="w-3.5 h-3.5 flex-shrink-0" />
                          <span className="truncate">{doc.document_title}</span>
                        </div>
                        <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-1.5 py-0.5 rounded border border-cyan-500/20">
                          Relevance: {Math.round(doc.relevance_score * 100)}%
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-500 font-mono space-x-2 mb-2">
                        {doc.section && <span>Section: {doc.section}</span>}
                        {doc.page && <span>Page: {doc.page}</span>}
                      </div>
                      <p className="text-slate-300 text-[11px] leading-relaxed italic bg-slate-900/50 p-2.5 rounded-lg border border-slate-800/60">
                        "{doc.excerpt}"
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Section 3: Suggested Diagnostic Checks */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
            <div className="flex items-center space-x-2 border-b border-slate-800/80 pb-3">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              <h4 className="text-sm font-semibold text-white">3. Suggested Diagnostic Inspection Steps</h4>
            </div>

            <div className="space-y-2.5">
              {data.suggested_checks.map((chk) => (
                <div
                  key={chk.step}
                  className="p-3.5 rounded-xl bg-slate-950/40 border border-slate-800/80 text-xs flex items-start space-x-3"
                >
                  <span className="w-5 h-5 rounded-full bg-slate-800 text-cyan-400 flex items-center justify-center font-bold text-xs flex-shrink-0 mt-0.5">
                    {chk.step}
                  </span>
                  <div className="space-y-1 flex-1">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-200">{chk.category}</span>
                      <span
                        className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded ${
                          chk.priority === 'High'
                            ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {chk.priority} Priority
                      </span>
                    </div>
                    <p className="text-slate-300 font-medium">{chk.procedure}</p>
                    <p className="text-[11px] text-slate-500">{chk.rationale}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Section 4: Suggested Advisory Action */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
            <div className="flex items-center space-x-2 border-b border-slate-800/80 pb-3">
              <Clock className="w-4 h-4 text-amber-400" />
              <h4 className="text-sm font-semibold text-white">4. Suggested Advisory Action</h4>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 text-xs space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-sm text-amber-300">
                  {data.suggested_action.action_type}
                </span>
                <span className="px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono text-[11px]">
                  Timeframe: {data.suggested_action.recommended_timeframe}
                </span>
              </div>
              <p className="text-slate-200 leading-relaxed">{data.suggested_action.summary}</p>

              <div className="pt-2 text-[11px] text-amber-400/90 font-medium flex items-center space-x-1.5">
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
                <span>
                  {data.suggested_action.requires_human_approval
                    ? '⚠️ Requires Certified Technician Approval before work order issuance.'
                    : 'Standard operational guideline.'}
                </span>
              </div>
            </div>
          </div>

          {/* Section 5: Uncertainty & Limitations */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3 text-xs">
            <div className="flex items-center space-x-2 border-b border-slate-800/80 pb-3">
              <ShieldCheck className="w-4 h-4 text-slate-400" />
              <h4 className="text-sm font-semibold text-white">5. Uncertainty & Operational Boundaries</h4>
            </div>

            <ul className="list-disc list-inside space-y-1.5 text-slate-400 text-[11px]">
              {data.uncertainties_and_limitations.map((u, idx) => (
                <li key={idx} className="leading-relaxed">
                  {u}
                </li>
              ))}
            </ul>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-[11px] text-slate-500 italic mt-3">
              {data.compliance_notice}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
