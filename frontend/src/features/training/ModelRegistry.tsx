import { useState, useEffect, useCallback } from 'react';
import {
  Boxes,
  History,
  RotateCw,
  Search,
  Filter,
  Eye,
  Play,
  Cpu,
  AlertCircle,
  Check,
  X,
  Loader2,
  Sparkles,
} from 'lucide-react';
import { Badge } from '../../components/ui/Badge';
import { api } from '../../services/api';
import { ModelExplanationView } from './ModelExplanationView';
import type { RegisteredModel, ModelVersion } from '../../types/training';

interface ModelRegistryProps {
  onSelectModelForInference?: (model: RegisteredModel) => void;
  onNavigateToConfigure?: () => void;
}

export function ModelRegistry({
  onSelectModelForInference,
  onNavigateToConfigure,
}: ModelRegistryProps) {
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [taskFilter, setTaskFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Modals state
  const [selectedModelForDetails, setSelectedModelForDetails] = useState<RegisteredModel | null>(null);
  const [selectedModelForVersions, setSelectedModelForVersions] = useState<RegisteredModel | null>(null);
  const [selectedModelForExplain, setSelectedModelForExplain] = useState<RegisteredModel | null>(null);
  const [versionsList, setVersionsList] = useState<ModelVersion[]>([]);
  const [loadingVersions, setLoadingVersions] = useState(false);
  const [versionError, setVersionError] = useState<string | null>(null);

  // Action status
  const [actionLoadingId, setActionLoadingId] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params: { task?: string; status?: string } = {};
      if (taskFilter !== 'all') params.task = taskFilter;
      if (statusFilter !== 'all') params.status = statusFilter;
      const res = await api.models.list(params);
      setModels(res.models || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch registered models.');
    } finally {
      setLoading(false);
    }
  }, [taskFilter, statusFilter]);

  useEffect(() => {
    let active = true;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const params: { task?: string; status?: string } = {};
        if (taskFilter !== 'all') params.task = taskFilter;
        if (statusFilter !== 'all') params.status = statusFilter;
        const res = await api.models.list(params);
        if (active) setModels(res.models || []);
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : 'Failed to fetch registered models.');
      } finally {
        if (active) setLoading(false);
      }
    };
    void load();
    return () => {
      active = false;
    };
  }, [taskFilter, statusFilter]);

  // Activate model
  const handleActivate = async (modelId: string) => {
    setActionLoadingId(modelId);
    setActionMessage(null);
    try {
      await api.models.activate(modelId);
      setActionMessage('Model successfully activated as production inference target.');
      await fetchModels();
      if (selectedModelForVersions) {
        await loadVersions(selectedModelForVersions.id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Activation failed.');
    } finally {
      setActionLoadingId(null);
    }
  };

  // Deactivate model
  const handleDeactivate = async (modelId: string) => {
    setActionLoadingId(modelId);
    setActionMessage(null);
    try {
      await api.models.deactivate(modelId);
      setActionMessage('Model successfully deactivated.');
      await fetchModels();
      if (selectedModelForVersions) {
        await loadVersions(selectedModelForVersions.id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Deactivation failed.');
    } finally {
      setActionLoadingId(null);
    }
  };

  // Load versions
  const loadVersions = async (modelId: string) => {
    setLoadingVersions(true);
    setVersionError(null);
    try {
      const vers = await api.models.getVersions(modelId);
      setVersionsList(vers);
    } catch (err) {
      setVersionError(err instanceof Error ? err.message : 'Failed to load model versions.');
    } finally {
      setLoadingVersions(false);
    }
  };

  const handleOpenVersionsModal = (model: RegisteredModel) => {
    setSelectedModelForVersions(model);
    void loadVersions(model.id);
  };

  // Filter models by search
  const filteredModels = models.filter((m) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      m.name.toLowerCase().includes(q) ||
      m.task.toLowerCase().includes(q) ||
      m.model_type.toLowerCase().includes(q) ||
      m.id.toLowerCase().includes(q)
    );
  });

  return (
    <div className="space-y-6">
      {/* Top Controls Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900/80 border border-slate-800 p-4 rounded-lg">
        <div>
          <h2 className="text-base font-bold font-mono text-slate-100 flex items-center gap-2">
            <Boxes className="w-5 h-5 text-accent" />
            Machine Learning Model Registry
          </h2>
          <p className="text-xs font-mono text-slate-400 mt-0.5">
            Production-grade version tracking, atomic activation, and runtime lifecycle control
          </p>
        </div>

        <button
          type="button"
          onClick={fetchModels}
          disabled={loading}
          className="self-start sm:self-auto text-xs font-mono bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors disabled:opacity-50"
        >
          <RotateCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Action Notification */}
      {actionMessage && (
        <div className="bg-emerald-500/10 border border-emerald-500/30 p-3 rounded-lg text-xs font-mono text-emerald-400 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Check className="w-4 h-4" />
            <span>{actionMessage}</span>
          </div>
          <button
            type="button"
            onClick={() => setActionMessage(null)}
            className="text-slate-400 hover:text-slate-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Error Banner */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 p-3 rounded-lg text-xs font-mono text-red-400 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-slate-400 hover:text-slate-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 bg-slate-950/60 p-3 border border-slate-800 rounded-lg text-xs font-mono">
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5">
          <Search className="w-4 h-4 text-slate-500 shrink-0" />
          <input
            type="text"
            placeholder="Search by model name or algorithm..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-transparent text-slate-200 placeholder-slate-500 focus:outline-none w-full text-xs"
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-slate-500 shrink-0" />
          <span className="text-slate-400 shrink-0">Task:</span>
          <select
            value={taskFilter}
            onChange={(e) => setTaskFilter(e.target.value)}
            className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 w-full focus:outline-none focus:border-accent"
          >
            <option value="all">All ML Tasks</option>
            <option value="rul_regression">RUL Regression</option>
            <option value="condition_classification">Condition Classification</option>
            <option value="anomaly_detection">Anomaly Detection</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-slate-400 shrink-0">Status:</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 w-full focus:outline-none focus:border-accent"
          >
            <option value="all">All States</option>
            <option value="active">Active Only</option>
            <option value="inactive">Inactive Only</option>
          </select>
        </div>
      </div>

      {/* Models Table */}
      {loading ? (
        <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-12 text-center text-slate-400 font-mono text-xs flex flex-col items-center gap-2">
          <Loader2 className="w-6 h-6 animate-spin text-accent" />
          <span>Querying Model Registry from backend...</span>
        </div>
      ) : filteredModels.length === 0 ? (
        <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-12 text-center text-slate-500 font-mono text-xs space-y-3">
          <Boxes className="w-8 h-8 text-slate-600 mx-auto" />
          <p className="font-semibold text-slate-300">No Models Found in Registry</p>
          <p className="text-slate-500 max-w-md mx-auto">
            {models.length === 0
              ? 'Train a model using the Datasets & Training workflow, then register its completed artifact to activate it for live inference.'
              : 'No registered models match the selected filter criteria.'}
          </p>
          {onNavigateToConfigure && (
            <button
              type="button"
              onClick={onNavigateToConfigure}
              className="bg-accent text-slate-950 font-bold px-4 py-2 rounded text-xs transition-colors"
            >
              Train a New Model
            </button>
          )}
        </div>
      ) : (
        <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-950/60">
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono text-left">
              <thead className="bg-slate-900 text-slate-400 uppercase text-[11px] border-b border-slate-800">
                <tr>
                  <th className="px-4 py-3">Model Name & Version</th>
                  <th className="px-4 py-3">ML Task</th>
                  <th className="px-4 py-3">Algorithm</th>
                  <th className="px-4 py-3">Key Metrics</th>
                  <th className="px-4 py-3">Registered</th>
                  <th className="px-4 py-3 text-center">Status</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filteredModels.map((m) => {
                  const isActive = m.status === 'active';
                  const isActionBusy = actionLoadingId === m.id;

                  // Render compact metrics summary
                  let metricsSummary = 'N/A';
                  if (m.task === 'rul_regression') {
                    const r2 = m.metrics.r2 !== undefined ? `R²: ${m.metrics.r2}` : '';
                    const rmse = m.metrics.rmse !== undefined ? `RMSE: ${m.metrics.rmse}` : '';
                    metricsSummary = [r2, rmse].filter(Boolean).join(' | ') || 'N/A';
                  } else if (m.task === 'condition_classification') {
                    const acc =
                      m.metrics.accuracy !== undefined
                        ? `Acc: ${(m.metrics.accuracy * 100).toFixed(1)}%`
                        : '';
                    const f1 =
                      m.metrics.f1 !== undefined ? `F1: ${(m.metrics.f1 * 100).toFixed(1)}%` : '';
                    metricsSummary = [acc, f1].filter(Boolean).join(' | ') || 'N/A';
                  } else if (m.task === 'anomaly_detection') {
                    const rate =
                      m.metrics.anomaly_rate !== undefined
                        ? `Rate: ${(m.metrics.anomaly_rate * 100).toFixed(1)}%`
                        : '';
                    metricsSummary = rate || 'N/A';
                  }

                  return (
                    <tr key={m.id} className="hover:bg-slate-900/50 transition-colors">
                      <td className="px-4 py-3.5">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-slate-100">{m.name}</span>
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 border border-slate-700 text-accent font-semibold">
                            v{m.version}
                          </span>
                        </div>
                        <span className="text-[10px] text-slate-500 block truncate max-w-[180px]">
                          {m.id}
                        </span>
                      </td>

                      <td className="px-4 py-3.5">
                        <span className="text-slate-300 font-semibold">{m.task}</span>
                      </td>

                      <td className="px-4 py-3.5 text-slate-400">
                        {m.model_type}
                      </td>

                      <td className="px-4 py-3.5">
                        <span className="text-slate-300 font-bold bg-slate-900 px-2 py-1 rounded border border-slate-800 text-[11px]">
                          {metricsSummary}
                        </span>
                      </td>

                      <td className="px-4 py-3.5 text-slate-500 text-[11px]">
                        {m.created_at ? new Date(m.created_at).toLocaleDateString() : 'N/A'}
                      </td>

                      <td className="px-4 py-3.5 text-center">
                        {isActive ? (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 animate-pulse">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                            ACTIVE
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-slate-900 text-slate-500 border border-slate-800">
                            INACTIVE
                          </span>
                        )}
                      </td>

                      <td className="px-4 py-3.5 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          {isActive ? (
                            <button
                              type="button"
                              onClick={() => handleDeactivate(m.id)}
                              disabled={isActionBusy}
                              className="px-2.5 py-1 rounded text-[11px] font-mono text-amber-400 hover:text-amber-300 bg-amber-500/10 border border-amber-500/20 transition-colors disabled:opacity-50"
                              title="Deactivate this model"
                            >
                              {isActionBusy ? '...' : 'Deactivate'}
                            </button>
                          ) : (
                            <button
                              type="button"
                              onClick={() => handleActivate(m.id)}
                              disabled={isActionBusy}
                              className="px-2.5 py-1 rounded text-[11px] font-mono font-bold text-emerald-400 hover:text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 transition-colors disabled:opacity-50"
                              title="Activate as production inference target"
                            >
                              {isActionBusy ? '...' : 'Activate'}
                            </button>
                          )}

                          <button
                            type="button"
                            onClick={() => handleOpenVersionsModal(m)}
                            className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                            title="View version history"
                          >
                            <History className="w-3.5 h-3.5" />
                          </button>

                          <button
                            type="button"
                            onClick={() => setSelectedModelForDetails(m)}
                            className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                            title="View model details & metrics"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>

                          <button
                            type="button"
                            onClick={() => setSelectedModelForExplain(m)}
                            className="p-1 rounded text-purple-400 hover:text-purple-200 hover:bg-purple-950/40 border border-purple-500/20 transition-colors"
                            title="Explain feature importances & sensitivities"
                          >
                            <Sparkles className="w-3.5 h-3.5" />
                          </button>

                          {onSelectModelForInference && (
                            <button
                              type="button"
                              onClick={() => onSelectModelForInference(m)}
                              className="px-2 py-1 rounded text-[11px] font-mono text-accent hover:text-white bg-accent/10 hover:bg-accent/20 border border-accent/30 transition-colors flex items-center gap-1"
                              title="Test inference with this model"
                            >
                              <Play className="w-3 h-3" />
                              <span>Test</span>
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* VERSION HISTORY MODAL */}
      {selectedModelForVersions && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-2xl w-full max-h-[85vh] flex flex-col font-mono text-xs shadow-2xl">
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-accent" />
                <h3 className="font-bold text-slate-100">
                  Version History: {selectedModelForVersions.name}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedModelForVersions(null)}
                className="text-slate-400 hover:text-slate-200 p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-4 overflow-y-auto flex-1 space-y-4">
              <p className="text-slate-400 text-[11px]">
                Backend-managed lineage for model family{' '}
                <span className="text-slate-200 font-bold">{selectedModelForVersions.name}</span>.
                Activating any specific version sets it as the authoritative inference model for{' '}
                <span className="text-accent">{selectedModelForVersions.task}</span>.
              </p>

              {loadingVersions ? (
                <div className="p-8 text-center text-slate-500 flex items-center justify-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-accent" />
                  <span>Loading version tree...</span>
                </div>
              ) : versionError ? (
                <div className="p-3 bg-red-500/10 border border-red-500/20 text-red-400 rounded">
                  {versionError}
                </div>
              ) : versionsList.length === 0 ? (
                <div className="p-6 text-center text-slate-500">No versions found.</div>
              ) : (
                <div className="space-y-2">
                  {versionsList.map((v) => {
                    const isVActive = v.status === 'active';
                    return (
                      <div
                        key={v.id}
                        className={`p-3 rounded border flex items-center justify-between gap-3 ${
                          isVActive
                            ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300'
                            : 'bg-slate-950 border-slate-800 text-slate-300'
                        }`}
                      >
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-slate-100">v{v.version}</span>
                            <span className="text-slate-500 text-[10px]">ID: {v.id}</span>
                            {isVActive && (
                              <Badge variant="healthy" className="text-[9px]">
                                ACTIVE
                              </Badge>
                            )}
                          </div>
                          <div className="text-[11px] text-slate-400">
                            Algorithm: <span className="text-slate-200">{v.model_type}</span> | Job:{' '}
                            <span className="text-slate-400">{v.training_job_id}</span>
                          </div>
                          <div className="text-[10px] text-slate-500">
                            Created:{' '}
                            {v.created_at ? new Date(v.created_at).toLocaleString() : 'N/A'}
                          </div>
                        </div>

                        <div>
                          {isVActive ? (
                            <span className="text-[11px] text-emerald-400 font-bold px-2.5 py-1 bg-emerald-500/20 rounded border border-emerald-500/40">
                              Active Target
                            </span>
                          ) : (
                            <button
                              type="button"
                              onClick={() => handleActivate(v.id)}
                              className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded transition-colors font-bold"
                            >
                              Activate v{v.version}
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="p-3 border-t border-slate-800 flex justify-end">
              <button
                type="button"
                onClick={() => setSelectedModelForVersions(null)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-1.5 rounded text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* TECHNICAL DETAILS MODAL */}
      {selectedModelForDetails && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-2xl w-full max-h-[85vh] flex flex-col font-mono text-xs shadow-2xl">
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-accent" />
                <h3 className="font-bold text-slate-100">
                  Model Inspection: {selectedModelForDetails.name} (v{selectedModelForDetails.version})
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedModelForDetails(null)}
                className="text-slate-400 hover:text-slate-200 p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-4 overflow-y-auto flex-1 space-y-4">
              <div className="grid grid-cols-2 gap-3 bg-slate-950 p-3 rounded border border-slate-800">
                <div>
                  <span className="text-slate-500 block text-[10px]">MODEL ID</span>
                  <span className="text-slate-200">{selectedModelForDetails.id}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">STATUS</span>
                  <span
                    className={
                      selectedModelForDetails.status === 'active'
                        ? 'text-emerald-400 font-bold'
                        : 'text-slate-400'
                    }
                  >
                    {selectedModelForDetails.status.toUpperCase()}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">TASK</span>
                  <span className="text-slate-200">{selectedModelForDetails.task}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">ALGORITHM</span>
                  <span className="text-slate-200">{selectedModelForDetails.model_type}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">ORIGIN TRAINING JOB</span>
                  <span className="text-slate-200">{selectedModelForDetails.training_job_id}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">DATASET ID</span>
                  <span className="text-slate-200">{selectedModelForDetails.dataset_id}</span>
                </div>
              </div>

              {/* Evaluation Metrics JSON */}
              <div>
                <span className="text-slate-400 font-bold block mb-1">
                  EVALUATION METRICS
                </span>
                <pre className="bg-slate-950 p-3 rounded border border-slate-800 text-[11px] text-slate-300 overflow-x-auto">
                  {JSON.stringify(selectedModelForDetails.metrics, null, 2)}
                </pre>
              </div>

              {/* Training Metadata if available */}
              {selectedModelForDetails.training_metadata && (
                <div>
                  <span className="text-slate-400 font-bold block mb-1">
                    TRAINING METADATA & SCHEMA
                  </span>
                  <pre className="bg-slate-950 p-3 rounded border border-slate-800 text-[11px] text-slate-300 overflow-x-auto">
                    {JSON.stringify(selectedModelForDetails.training_metadata, null, 2)}
                  </pre>
                </div>
              )}
            </div>

            <div className="p-3 border-t border-slate-800 flex justify-between items-center">
              <button
                type="button"
                onClick={() => {
                  const target = selectedModelForDetails;
                  setSelectedModelForDetails(null);
                  setSelectedModelForExplain(target);
                }}
                className="bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 border border-purple-500/30 px-3 py-1.5 rounded text-xs flex items-center gap-1.5 font-bold transition-colors"
              >
                <Sparkles className="w-3 h-3 text-purple-400" />
                <span>Explain Model</span>
              </button>
              <button
                type="button"
                onClick={() => setSelectedModelForDetails(null)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-1.5 rounded text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODEL EXPLANATION MODAL */}
      {selectedModelForExplain && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <ModelExplanationView
              modelId={selectedModelForExplain.id}
              onClose={() => setSelectedModelForExplain(null)}
            />
          </div>
        </div>
      )}
    </div>
  );
}
