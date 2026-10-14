import { Target, Info, AlertCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import type { DatasetColumn } from '../../types/dataset';
import type { MLTask } from '../../types/training';

interface TargetSelectorProps {
  task: MLTask;
  columns: DatasetColumn[];
  targetColumn: string | null;
  onSelectTarget: (col: string | null) => void;
  validationError?: string;
}

export function TargetSelector({
  task,
  columns,
  targetColumn,
  onSelectTarget,
  validationError,
}: TargetSelectorProps) {
  const isSupervised = task !== 'anomaly_detection';

  if (!isSupervised) {
    return (
      <Card className="border-slate-800 bg-slate-900/40 opacity-80">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Target className="w-5 h-5 text-slate-500" />
            <CardTitle className="text-slate-400">2. Target Variable Selection</CardTitle>
          </div>
          <Badge variant="outline">UNSUPERVISED TASK</Badge>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-950/40 p-3 rounded border border-slate-800">
            <Info className="w-4 h-4 text-accent shrink-0" />
            <span>
              Anomaly Detection operates without a target label. The model learns baseline density directly from multi-sensor telemetry features.
            </span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const selectedColMeta = columns.find((c) => c.name === targetColumn);

  return (
    <Card className={`border-slate-800 bg-slate-900/60 ${validationError ? 'border-amber-500/50' : ''}`}>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Target className="w-5 h-5 text-accent" />
          <CardTitle>2. Select Target Column (Dependent Variable)</CardTitle>
        </div>
        <Badge variant={targetColumn ? 'healthy' : 'warning'}>
          {targetColumn ? `TARGET: ${targetColumn}` : 'SELECTION REQUIRED'}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <label htmlFor="target-column-select" className="block text-xs font-mono text-slate-400 mb-1.5">
            Choose ground-truth label for{' '}
            <span className="text-slate-200">
              {task === 'rul_regression' ? 'RUL Regression (e.g. cycle, rul, hours)' : 'Condition Classification (e.g. health, state, fault)'}
            </span>
          </label>
          <select
            id="target-column-select"
            value={targetColumn || ''}
            onChange={(e) => onSelectTarget(e.target.value || null)}
            className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 font-mono focus:outline-none focus:border-accent"
          >
            <option value="">-- Choose Target Column --</option>
            {columns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.dtype || (c.isNumeric ? 'numeric' : 'categorical')}, {c.uniqueCount} unique)
              </option>
            ))}
          </select>
        </div>

        {/* Selected target metadata breakdown */}
        {selectedColMeta && (
          <div className="bg-slate-950/60 border border-slate-800 rounded p-3 text-xs font-mono grid grid-cols-2 sm:grid-cols-4 gap-2">
            <div>
              <span className="text-slate-500 block">TYPE</span>
              <span className="text-slate-200 font-semibold">{selectedColMeta.dtype}</span>
            </div>
            <div>
              <span className="text-slate-500 block">MISSING VALUES</span>
              <span className={selectedColMeta.missingCount > 0 ? 'text-amber-400' : 'text-slate-200'}>
                {selectedColMeta.missingCount}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block">CARDINALITY</span>
              <span className="text-slate-200">{selectedColMeta.uniqueCount} unique</span>
            </div>
            <div>
              <span className="text-slate-500 block">FEATURE EXCLUSION</span>
              <span className="text-emerald-400">Auto-excluded from input features</span>
            </div>
          </div>
        )}

        {validationError && (
          <div className="flex items-center gap-2 text-amber-400 text-xs font-mono bg-amber-500/10 p-2.5 rounded border border-amber-500/20">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{validationError}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
