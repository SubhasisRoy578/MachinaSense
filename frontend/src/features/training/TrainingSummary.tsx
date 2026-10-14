import { ClipboardCheck, Play, AlertOctagon, Loader2 } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { TASK_DEFINITIONS, TASK_MODELS } from '../../store/trainingStore';
import type { Dataset } from '../../types/dataset';
import type { MLTask } from '../../types/training';

interface TrainingSummaryProps {
  dataset: Dataset;
  task: MLTask;
  targetColumn: string | null;
  selectedFeatures: string[];
  modelType: string;
  trainSplit: number;
  validationSplit: number;
  testSplit: number;
  randomSeed: number;
  validationErrors: Record<string, string>;
  onStartTraining: () => void;
  isStarting: boolean;
}

export function TrainingSummary({
  dataset,
  task,
  targetColumn,
  selectedFeatures,
  modelType,
  trainSplit,
  validationSplit,
  testSplit,
  randomSeed,
  validationErrors,
  onStartTraining,
  isStarting,
}: TrainingSummaryProps) {
  const taskDef = TASK_DEFINITIONS[task];
  const modelDef = TASK_MODELS[task]?.find((m) => m.id === modelType);
  const hasErrors = Object.keys(validationErrors).length > 0;

  return (
    <Card className={`border-slate-800 bg-slate-900/60 ${hasErrors ? 'border-amber-500/40' : 'border-accent/40'}`}>
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <ClipboardCheck className="w-5 h-5 text-accent" />
          <CardTitle>6. Training Job Configuration Summary</CardTitle>
        </div>
        <Badge variant={hasErrors ? 'warning' : 'healthy'}>
          {hasErrors ? 'CONFIG VALIDATION FAILED' : 'READY TO TRAIN'}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Summary Spec Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 text-xs font-mono">
          <div className="bg-slate-950/60 border border-slate-800 rounded p-2.5">
            <span className="text-slate-500 block text-[10px]">DATASET</span>
            <span className="text-slate-200 font-semibold truncate block" title={dataset.name}>
              {dataset.name}
            </span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-2.5">
            <span className="text-slate-500 block text-[10px]">PREDICTION TASK</span>
            <span className="text-accent font-semibold truncate block">{taskDef?.label}</span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-2.5">
            <span className="text-slate-500 block text-[10px]">TARGET VARIABLE</span>
            <span className={targetColumn ? 'text-emerald-400 font-semibold' : 'text-slate-500 italic'}>
              {targetColumn || (task === 'anomaly_detection' ? 'None (Unsupervised)' : 'Not Selected')}
            </span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-2.5">
            <span className="text-slate-500 block text-[10px]">FEATURE COUNT</span>
            <span className="text-slate-200 font-semibold">{selectedFeatures.length} columns</span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-2.5">
            <span className="text-slate-500 block text-[10px]">ALGORITHM</span>
            <span className="text-slate-200 font-semibold truncate block" title={modelDef?.name}>
              {modelDef?.name || modelType}
            </span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-2.5">
            <span className="text-slate-500 block text-[10px]">SPLIT RATIO</span>
            <span className="text-slate-200 font-semibold">
              {trainSplit} / {validationSplit} / {testSplit}
            </span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-2.5">
            <span className="text-slate-500 block text-[10px]">RANDOM SEED</span>
            <span className="text-slate-200 font-semibold">{randomSeed}</span>
          </div>
        </div>

        {/* Validation Errors Notice */}
        {hasErrors && (
          <div className="bg-amber-500/10 border border-amber-500/20 rounded p-3 text-xs font-mono text-amber-400 space-y-1">
            <div className="flex items-center gap-1.5 font-bold">
              <AlertOctagon className="w-4 h-4 shrink-0" />
              <span>Resolve Configuration Errors to Proceed:</span>
            </div>
            <ul className="list-disc list-inside space-y-0.5 pl-2 text-[11px] text-amber-300">
              {Object.entries(validationErrors).map(([key, msg]) => (
                <li key={key}>{msg}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Trigger Button */}
        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={onStartTraining}
            disabled={isStarting || hasErrors}
            className="bg-accent hover:bg-accent/90 text-slate-950 font-bold px-6 py-2.5 rounded text-sm transition-all flex items-center gap-2 shadow-lg shadow-accent/10 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isStarting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Initializing Training Job...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Start Training Job</span>
              </>
            )}
          </button>
        </div>
      </CardContent>
    </Card>
  );
}
