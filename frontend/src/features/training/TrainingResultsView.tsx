import { useState } from 'react';
import {
  CheckCircle2,
  AlertCircle,
  Database,
  Tag,
  ArrowRight,
  BookmarkCheck,
  Check,
  Loader2,
  Table,
  Cpu,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { api } from '../../services/api';
import type { TrainingJob, RegisteredModel } from '../../types/training';

interface TrainingResultsViewProps {
  job: TrainingJob;
  onNavigateToRegistry?: () => void;
  onNavigateToInference?: (model: RegisteredModel) => void;
}

function formatDuration(startedAt?: string, completedAt?: string | null, durationSeconds?: number | null): string {
  if (typeof durationSeconds === 'number' && durationSeconds >= 0) {
    if (durationSeconds < 60) return `${durationSeconds}s`;
    const minutes = Math.floor(durationSeconds / 60);
    const rem = Math.round(durationSeconds % 60);
    return `${minutes}m ${rem}s`;
  }
  if (!startedAt || !completedAt) return 'Not available from backend';
  try {
    const start = new Date(startedAt).getTime();
    const end = new Date(completedAt).getTime();
    if (isNaN(start) || isNaN(end) || end < start) return 'Not available from backend';
    const seconds = Math.round((end - start) / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const rem = seconds % 60;
    return `${minutes}m ${rem}s`;
  } catch {
    return 'Not available from backend';
  }
}

export function TrainingResultsView({
  job,
  onNavigateToRegistry,
  onNavigateToInference,
}: TrainingResultsViewProps) {
  const metrics = job.metrics || {};
  const [modelName, setModelName] = useState(
    `${job.task.replace('_', '-')}-${job.model_type.replace('_', '-')}`
  );
  const [isRegistering, setIsRegistering] = useState(false);
  const [registeredModel, setRegisteredModel] = useState<RegisteredModel | null>(null);
  const [registerError, setRegisterError] = useState<string | null>(null);
  const [isActivating, setIsActivating] = useState(false);

  const durationStr = formatDuration(job.started_at, job.completed_at, job.duration_seconds);
  const featureCount = job.config?.feature_columns?.length;
  const splitStr = job.config
    ? `${job.config.train_split}% / ${job.config.validation_split}% / ${job.config.test_split}%`
    : 'Not available from backend';
  const seedStr =
    job.config?.random_seed !== undefined && job.config?.random_seed !== null
      ? String(job.config.random_seed)
      : 'Not available from backend';

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelName.trim()) {
      setRegisterError('Please provide a valid model name.');
      return;
    }
    setIsRegistering(true);
    setRegisterError(null);
    try {
      const res = await api.models.register({
        training_job_id: job.id,
        name: modelName.trim(),
      });
      setRegisteredModel(res);
    } catch (err) {
      setRegisterError(
        err instanceof Error ? err.message : 'Failed to register model artifact.'
      );
    } finally {
      setIsRegistering(false);
    }
  };

  const handleActivate = async () => {
    if (!registeredModel) return;
    setIsActivating(true);
    try {
      const updated = await api.models.activate(registeredModel.id);
      setRegisteredModel(updated);
    } catch (err) {
      setRegisterError(
        err instanceof Error ? err.message : 'Failed to activate registered model.'
      );
    } finally {
      setIsActivating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
              <h3 className="text-base font-bold text-slate-100 font-mono">
                Model Evaluation Results
              </h3>
              <Badge variant="healthy" className="font-mono text-[10px]">
                REAL EVALUATION
              </Badge>
            </div>
            <p className="text-xs text-slate-400 font-mono mt-1">
              Truthful test split evaluation computed by the backend ML engine.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="text-slate-500">Duration:</span>
            <span className="text-slate-200 font-bold bg-slate-950 px-2 py-1 rounded border border-slate-800">
              {durationStr}
            </span>
          </div>
        </div>

        {/* Training Metadata Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-4 border-t border-slate-800/80 text-xs font-mono">
          <div>
            <span className="text-slate-500 block text-[10px]">ALGORITHM</span>
            <span className="text-slate-200 font-bold">{job.model_type}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">FEATURE COUNT</span>
            <span className="text-slate-200 font-bold">
              {featureCount !== undefined ? `${featureCount} features` : 'Not available from backend'}
            </span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">TRAIN/VAL/TEST SPLIT</span>
            <span className="text-slate-200 font-bold">{splitStr}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">RANDOM SEED</span>
            <span className="text-slate-200 font-bold">{seedStr}</span>
          </div>
        </div>
      </div>

      {/* RUL REGRESSION METRICS VIEW */}
      {job.task === 'rul_regression' && (
        <Card className="border-slate-800 bg-slate-950/60">
          <CardHeader className="pb-3 border-b border-slate-800/60">
            <CardTitle className="text-sm font-mono flex items-center gap-2 text-slate-200">
              <Cpu className="w-4 h-4 text-accent" />
              RUL Regression Performance Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4 space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="bg-slate-900 border border-slate-800 rounded p-4">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Mean Absolute Error (MAE)
                </span>
                <div className="text-2xl font-mono font-bold text-accent mt-1">
                  {metrics.mae !== undefined && metrics.mae !== null
                    ? `${metrics.mae} cycles`
                    : 'Not available from backend'}
                </div>
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Average deviation from actual remaining life
                </span>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded p-4">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Root Mean Squared Error (RMSE)
                </span>
                <div className="text-2xl font-mono font-bold text-sky-400 mt-1">
                  {metrics.rmse !== undefined && metrics.rmse !== null
                    ? `${metrics.rmse} cycles`
                    : 'Not available from backend'}
                </div>
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Quadratic penalty on large prediction misses
                </span>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded p-4">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Coefficient of Determination (R²)
                </span>
                <div className="text-2xl font-mono font-bold text-emerald-400 mt-1">
                  {metrics.r2 !== undefined && metrics.r2 !== null
                    ? metrics.r2
                    : 'Not available from backend'}
                </div>
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Proportion of variance explained by features
                </span>
              </div>
            </div>

            {metrics.test_samples && (
              <div className="text-xs font-mono text-slate-500">
                Evaluation computed over{' '}
                <span className="text-slate-300 font-bold">{metrics.test_samples}</span> test
                telemetry samples.
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* CONDITION CLASSIFICATION METRICS VIEW */}
      {job.task === 'condition_classification' && (
        <Card className="border-slate-800 bg-slate-950/60">
          <CardHeader className="pb-3 border-b border-slate-800/60">
            <CardTitle className="text-sm font-mono flex items-center gap-2 text-slate-200">
              <Cpu className="w-4 h-4 text-accent" />
              Health Condition Classification Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4 space-y-5">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-slate-900 border border-slate-800 rounded p-3">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Accuracy
                </span>
                <div className="text-xl font-mono font-bold text-emerald-400 mt-0.5">
                  {metrics.accuracy !== undefined && metrics.accuracy !== null
                    ? `${(metrics.accuracy * 100).toFixed(2)}%`
                    : 'Not available from backend'}
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded p-3">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Precision (Weighted)
                </span>
                <div className="text-xl font-mono font-bold text-sky-400 mt-0.5">
                  {metrics.precision !== undefined && metrics.precision !== null
                    ? `${(metrics.precision * 100).toFixed(2)}%`
                    : 'Not available from backend'}
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded p-3">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Recall (Weighted)
                </span>
                <div className="text-xl font-mono font-bold text-accent mt-0.5">
                  {metrics.recall !== undefined && metrics.recall !== null
                    ? `${(metrics.recall * 100).toFixed(2)}%`
                    : 'Not available from backend'}
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded p-3">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  F1 Score (Weighted)
                </span>
                <div className="text-xl font-mono font-bold text-purple-400 mt-0.5">
                  {metrics.f1 !== undefined && metrics.f1 !== null
                    ? (metrics.f1 * 100).toFixed(2) + '%'
                    : 'Not available from backend'}
                </div>
              </div>
            </div>

            {/* Confusion Matrix */}
            {metrics.confusion_matrix && Array.isArray(metrics.confusion_matrix) && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Table className="w-4 h-4 text-slate-400" />
                  <span className="text-xs font-mono font-semibold text-slate-300 uppercase">
                    Test Set Confusion Matrix
                  </span>
                </div>
                <div className="overflow-x-auto border border-slate-800 rounded bg-slate-900/60 p-3">
                  <table className="text-xs font-mono w-full text-center">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-500">
                        <th className="p-2 text-left">Actual \ Predicted</th>
                        {(metrics.classes || metrics.confusion_matrix[0].map((_: any, idx: number) => `Class ${idx}`)).map(
                          (c: string, idx: number) => (
                            <th key={idx} className="p-2 text-slate-300">
                              {c}
                            </th>
                          )
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.confusion_matrix.map((row: number[], rowIdx: number) => {
                        const rowLabel = metrics.classes?.[rowIdx] ?? `Class ${rowIdx}`;
                        return (
                          <tr key={rowIdx} className="border-b border-slate-800/40 hover:bg-slate-800/20">
                            <td className="p-2 text-left font-bold text-slate-300">{rowLabel}</td>
                            {row.map((val: number, colIdx: number) => {
                              const isDiagonal = rowIdx === colIdx;
                              return (
                                <td
                                  key={colIdx}
                                  className={`p-2 font-bold ${
                                    isDiagonal
                                      ? 'text-emerald-400 bg-emerald-500/10'
                                      : val > 0
                                      ? 'text-amber-400 bg-amber-500/10'
                                      : 'text-slate-600'
                                  }`}
                                >
                                  {val}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* ANOMALY DETECTION METRICS VIEW */}
      {job.task === 'anomaly_detection' && (
        <Card className="border-slate-800 bg-slate-950/60">
          <CardHeader className="pb-3 border-b border-slate-800/60">
            <CardTitle className="text-sm font-mono flex items-center gap-2 text-slate-200">
              <Cpu className="w-4 h-4 text-accent" />
              Isolation Forest Anomaly Metrics & Diagnostics
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4 space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-slate-900 border border-slate-800 rounded p-3">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Total Samples Evaluated
                </span>
                <div className="text-xl font-mono font-bold text-slate-100 mt-0.5">
                  {metrics.total_samples !== undefined && metrics.total_samples !== null
                    ? metrics.total_samples
                    : 'Not available from backend'}
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded p-3">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Detected Outliers
                </span>
                <div className="text-xl font-mono font-bold text-amber-400 mt-0.5">
                  {metrics.anomaly_count !== undefined && metrics.anomaly_count !== null
                    ? metrics.anomaly_count
                    : 'Not available from backend'}
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded p-3">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Anomaly Rate
                </span>
                <div className="text-xl font-mono font-bold text-red-400 mt-0.5">
                  {metrics.anomaly_rate !== undefined && metrics.anomaly_rate !== null
                    ? `${(metrics.anomaly_rate * 100).toFixed(2)}%`
                    : 'Not available from backend'}
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded p-3">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">
                  Contamination Setting
                </span>
                <div className="text-xl font-mono font-bold text-sky-400 mt-0.5">
                  {metrics.contamination !== undefined && metrics.contamination !== null
                    ? metrics.contamination
                    : 'Not available from backend'}
                </div>
              </div>
            </div>

            {/* Score Statistics */}
            <div className="bg-slate-900 border border-slate-800 rounded p-4 font-mono text-xs">
              <span className="text-slate-400 font-semibold block mb-2 uppercase text-[11px]">
                Decision Function Anomaly Score Statistics
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-slate-300">
                <div>
                  <span className="text-slate-500 block text-[10px]">MEAN SCORE</span>
                  <span>{metrics.score_mean !== undefined ? metrics.score_mean : 'Not available from backend'}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">MIN SCORE</span>
                  <span>{metrics.score_min !== undefined ? metrics.score_min : 'Not available from backend'}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">MAX SCORE</span>
                  <span>{metrics.score_max !== undefined ? metrics.score_max : 'Not available from backend'}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">STD DEVIATION</span>
                  <span>{metrics.score_std !== undefined ? metrics.score_std : 'Not available from backend'}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* MODEL REGISTRY ACTION CARD */}
      <Card className="border-accent/40 bg-slate-950/80">
        <CardHeader className="pb-3 border-b border-slate-800">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-mono flex items-center gap-2 text-slate-100">
              <BookmarkCheck className="w-4 h-4 text-accent" />
              Register Artifact in Model Registry
            </CardTitle>
            {registeredModel && (
              <Badge variant={registeredModel.status === 'active' ? 'healthy' : 'outline'}>
                {registeredModel.status.toUpperCase()}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="pt-4 space-y-4">
          {!registeredModel ? (
            <form onSubmit={handleRegister} className="space-y-4">
              <p className="text-xs font-mono text-slate-300">
                Register this verified training run into the MachinaSense Model Registry. A new model
                family or version will be created authoritatively on the backend.
              </p>

              <div className="flex flex-col sm:flex-row gap-3">
                <div className="flex-1">
                  <label className="text-[11px] font-mono text-slate-400 block mb-1">
                    MODEL IDENTIFIER / NAME
                  </label>
                  <input
                    type="text"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    placeholder="e.g. turbofan-lstm-v1"
                    className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-xs font-mono text-slate-100 focus:outline-none focus:border-accent"
                  />
                </div>
                <div className="flex items-end">
                  <button
                    type="submit"
                    disabled={isRegistering || !modelName.trim()}
                    className="w-full sm:w-auto bg-accent hover:bg-accent/90 text-slate-950 font-mono font-bold text-xs px-5 py-2 rounded flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                  >
                    {isRegistering ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Registering...</span>
                      </>
                    ) : (
                      <>
                        <Tag className="w-3.5 h-3.5" />
                        <span>Register Model</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {registerError && (
                <div className="p-3 rounded bg-red-500/10 border border-red-500/20 text-xs font-mono text-red-400 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{registerError}</span>
                </div>
              )}
            </form>
          ) : (
            <div className="space-y-4">
              <div className="p-4 rounded bg-emerald-500/10 border border-emerald-500/30 flex items-start gap-3">
                <Check className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                <div className="text-xs font-mono space-y-1">
                  <p className="font-bold text-emerald-400">
                    Model Successfully Registered in Registry!
                  </p>
                  <p className="text-slate-300">
                    Model: <span className="text-slate-100 font-bold">{registeredModel.name}</span> |
                    Version: <span className="text-accent font-bold">v{registeredModel.version}</span> |
                    ID: <span className="text-slate-400">{registeredModel.id}</span>
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3 pt-2">
                {registeredModel.status !== 'active' ? (
                  <button
                    type="button"
                    onClick={handleActivate}
                    disabled={isActivating}
                    className="bg-emerald-600 hover:bg-emerald-500 text-white font-mono font-bold text-xs px-4 py-2 rounded flex items-center gap-1.5 transition-colors disabled:opacity-50"
                  >
                    {isActivating ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Activating...</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Activate This Model</span>
                      </>
                    )}
                  </button>
                ) : (
                  <span className="text-xs font-mono text-emerald-400 font-bold flex items-center gap-1.5 bg-emerald-500/20 border border-emerald-500/40 px-3 py-1.5 rounded">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Model is Currently ACTIVE for {registeredModel.task}
                  </span>
                )}

                {onNavigateToRegistry && (
                  <button
                    type="button"
                    onClick={onNavigateToRegistry}
                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono text-xs px-4 py-2 rounded flex items-center gap-1.5 transition-colors"
                  >
                    <Database className="w-3.5 h-3.5" />
                    <span>View in Model Registry</span>
                    <ArrowRight className="w-3.5 h-3.5 ml-1" />
                  </button>
                )}

                {onNavigateToInference && (
                  <button
                    type="button"
                    onClick={() => onNavigateToInference(registeredModel)}
                    className="bg-accent hover:bg-accent/90 text-slate-950 font-mono font-bold text-xs px-4 py-2 rounded flex items-center gap-1.5 transition-colors"
                  >
                    <Cpu className="w-3.5 h-3.5" />
                    <span>Run Inference with this Model</span>
                  </button>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
