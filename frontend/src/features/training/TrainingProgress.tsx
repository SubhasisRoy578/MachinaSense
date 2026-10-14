import { useEffect, useCallback } from 'react';
import {
  CheckCircle2,
  XCircle,
  RotateCcw,
  ArrowLeft,
  Loader2,
  Clock,
  Cpu,
  Layers,
  StopCircle,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { api } from '../../services/api';
import type { TrainingJob, TrainingJobStatus, RegisteredModel } from '../../types/training';
import { TrainingResultsView } from './TrainingResultsView';

interface TrainingProgressProps {
  job: TrainingJob;
  onUpdateJob: (job: TrainingJob) => void;
  onReturnToConfig: () => void;
  onRetry: () => void;
  isCancelling: boolean;
  onCancelJob: () => void;
  onNavigateToRegistry?: () => void;
  onNavigateToInference?: (model: RegisteredModel) => void;
}

const STAGES: { key: TrainingJobStatus; label: string }[] = [
  { key: 'queued', label: 'Queued' },
  { key: 'validating', label: 'Validation' },
  { key: 'preprocessing', label: 'Preprocessing' },
  { key: 'training', label: 'Model Training' },
  { key: 'evaluating', label: 'Evaluation' },
  { key: 'saving', label: 'Artifact Persistence' },
];

function safeFormatTime(isoString?: string | null): string {
  if (!isoString) return 'Pending';
  try {
    const d = new Date(isoString);
    return isNaN(d.getTime()) ? isoString : d.toLocaleTimeString();
  } catch {
    return isoString;
  }
}

function safeFormatDate(isoString?: string | null): string {
  if (!isoString) return 'Just now';
  try {
    const d = new Date(isoString);
    return isNaN(d.getTime()) ? isoString : d.toLocaleString();
  } catch {
    return isoString;
  }
}

export function TrainingProgress({
  job,
  onUpdateJob,
  onReturnToConfig,
  onRetry,
  isCancelling,
  onCancelJob,
  onNavigateToRegistry,
  onNavigateToInference,
}: TrainingProgressProps) {
  const isTerminal = job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled';

  // Status Polling Effect (only when job is in active execution states)
  const pollStatus = useCallback(async () => {
    if (isTerminal) return;
    try {
      const updated = await api.training.getStatus(job.id);
      if (updated) {
        onUpdateJob(updated);
      }
    } catch (err) {
      console.warn('Training status poll update:', err);
    }
  }, [job.id, isTerminal, onUpdateJob]);

  useEffect(() => {
    if (isTerminal) return;
    const interval = setInterval(() => {
      void pollStatus();
    }, 2500);
    return () => clearInterval(interval);
  }, [pollStatus, isTerminal]);

  const getStageState = (stageKey: TrainingJobStatus) => {
    if (job.status === 'failed') return 'failed';
    if (job.status === 'completed') return 'completed';
    const stageIndex = STAGES.findIndex((s) => s.key === stageKey);
    const currentIndex = STAGES.findIndex((s) => s.key === job.status);
    if (stageIndex < currentIndex) return 'completed';
    if (stageIndex === currentIndex) return 'active';
    return 'pending';
  };

  return (
    <Card className="border-slate-800 bg-slate-900/60 max-w-4xl mx-auto">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-accent" />
          <CardTitle>ML Training Execution Monitor</CardTitle>
        </div>
        <div className="flex items-center gap-2">
          <Badge
            variant={
              job.status === 'completed'
                ? 'healthy'
                : job.status === 'failed'
                ? 'critical'
                : 'warning'
            }
          >
            STATUS: {job.status.toUpperCase()}
          </Badge>
          {!isTerminal && (
            <button
              type="button"
              onClick={onCancelJob}
              disabled={isCancelling}
              className="text-xs font-mono text-red-400 hover:text-red-300 px-2.5 py-1 bg-red-500/10 border border-red-500/20 rounded flex items-center gap-1 transition-colors disabled:opacity-50"
            >
              <StopCircle className="w-3.5 h-3.5" />
              <span>{isCancelling ? 'Cancelling...' : 'Cancel Job'}</span>
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Job Identification Header */}
        <div className="bg-slate-950/60 border border-slate-800 rounded p-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div>
            <span className="text-slate-500 block text-[10px]">JOB ID</span>
            <span className="text-slate-200 font-bold truncate block">{job.id}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">MODEL / TASK</span>
            <span className="text-accent font-semibold">{job.model_type}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">DATASET</span>
            <span className="text-slate-200 truncate block">{job.dataset_name || job.dataset_id}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">STARTED AT</span>
            <span className="text-slate-300">
              {safeFormatTime(job.started_at)}
            </span>
          </div>
        </div>

        {/* Execution Pipeline Stepper */}
        <div>
          <h4 className="text-xs font-mono font-semibold text-slate-300 uppercase tracking-wider mb-3">
            Execution Pipeline
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
            {STAGES.map((stage) => {
              const state = getStageState(stage.key);
              return (
                <div
                  key={stage.key}
                  className={`p-3 rounded border font-mono text-xs flex flex-col justify-between transition-colors ${
                    state === 'completed'
                      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400'
                      : state === 'active'
                      ? 'border-accent bg-accent/10 text-accent animate-pulse'
                      : state === 'failed'
                      ? 'border-red-500/30 bg-red-500/10 text-slate-400'
                      : 'border-slate-800 bg-slate-950/40 text-slate-500'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] text-slate-500">STAGE</span>
                    {state === 'completed' ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    ) : state === 'active' ? (
                      <Loader2 className="w-3.5 h-3.5 text-accent animate-spin" />
                    ) : (
                      <Clock className="w-3.5 h-3.5 text-slate-600" />
                    )}
                  </div>
                  <span className="font-semibold text-[11px]">{stage.label}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Active execution feedback */}
        {!isTerminal && (
          <div className="bg-slate-950/80 border border-slate-800 rounded p-4 flex items-center gap-3">
            <Loader2 className="w-5 h-5 text-accent animate-spin shrink-0" />
            <div className="text-xs font-mono">
              <p className="text-slate-200 font-semibold">
                Executing training job on backend worker...
              </p>
              <p className="text-slate-400 mt-0.5">
                Current step:{' '}
                <span className="text-accent">{job.current_step || job.status.toUpperCase()}</span>
                {job.progress !== null && job.progress !== undefined && ` (${job.progress}%)`}
              </p>
            </div>
          </div>
        )}

        {/* Failure state */}
        {job.status === 'failed' && (
          <div className="bg-red-500/10 border border-red-500/20 rounded p-4 space-y-3">
            <div className="flex items-center gap-2 text-red-400 font-mono text-sm font-semibold">
              <XCircle className="w-5 h-5 shrink-0" />
              <span>Training Job Failed</span>
            </div>
            <p className="text-xs font-mono text-slate-300 leading-relaxed">
              {job.error_message ||
                'Training process halted due to a backend error or missing training endpoint.'}
            </p>
            <div className="flex items-center gap-2 pt-2">
              <button
                type="button"
                onClick={onRetry}
                className="bg-red-600 hover:bg-red-500 text-white font-mono text-xs px-4 py-2 rounded flex items-center gap-1.5 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Retry Training</span>
              </button>
              <button
                type="button"
                onClick={onReturnToConfig}
                className="bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono text-xs px-4 py-2 rounded flex items-center gap-1.5 transition-colors"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                <span>Return to Configuration</span>
              </button>
            </div>
          </div>
        )}

        {/* Success state */}
        {job.status === 'completed' && (
          <div className="space-y-6">
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded p-5 space-y-4">
              <div className="flex items-center gap-2 text-emerald-400 font-mono text-sm font-bold">
                <CheckCircle2 className="w-5 h-5 shrink-0" />
                <span>Training Pipeline Execution Succeeded</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
                <div className="bg-slate-950/60 p-2.5 rounded border border-slate-800">
                  <span className="text-slate-500 block">MODEL ARTIFACT</span>
                  <span className="text-slate-100 font-bold">{job.model_type}</span>
                </div>
                <div className="bg-slate-950/60 p-2.5 rounded border border-slate-800">
                  <span className="text-slate-500 block">PREDICTION TASK</span>
                  <span className="text-slate-100 font-bold">{job.task}</span>
                </div>
                <div className="bg-slate-950/60 p-2.5 rounded border border-slate-800">
                  <span className="text-slate-500 block">COMPLETED AT</span>
                  <span className="text-slate-100 font-bold">
                    {safeFormatDate(job.completed_at)}
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between pt-2">
                <p className="text-[11px] font-mono text-slate-400">
                  Artifact binary and preprocessor serialized. Review authentic evaluation metrics below and register model.
                </p>
                <button
                  type="button"
                  onClick={onReturnToConfig}
                  className="bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono text-xs px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors shrink-0"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  <span>Configure Another Job</span>
                </button>
              </div>
            </div>

            {/* REAL EVALUATION & MODEL REGISTRY SECTION */}
            <TrainingResultsView
              job={job}
              onNavigateToRegistry={onNavigateToRegistry}
              onNavigateToInference={onNavigateToInference}
            />
          </div>
        )}

        {/* Execution Logs (if provided by backend) */}
        {job.logs && job.logs.length > 0 && (
          <div>
            <h4 className="text-xs font-mono font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-accent" />
              <span>Backend Training Logs</span>
            </h4>
            <div className="bg-slate-950 border border-slate-800 rounded p-3 text-xs font-mono text-slate-300 max-h-48 overflow-y-auto custom-scrollbar space-y-1">
              {job.logs.map((log, idx) => (
                <p key={idx} className="leading-relaxed">
                  <span className="text-slate-600 mr-2">[{idx + 1}]</span>
                  {log}
                </p>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
