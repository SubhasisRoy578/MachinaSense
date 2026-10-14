import { Sliders, CheckSquare, Square, AlertCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import type { DatasetColumn } from '../../types/dataset';

interface FeatureSelectorProps {
  columns: DatasetColumn[];
  selectedFeatures: string[];
  targetColumn: string | null;
  onToggleFeature: (col: string) => void;
  onSelectAll: (cols: string[]) => void;
  onClearAll: () => void;
  validationError?: string;
}

export function FeatureSelector({
  columns,
  selectedFeatures,
  targetColumn,
  onToggleFeature,
  onSelectAll,
  onClearAll,
  validationError,
}: FeatureSelectorProps) {
  // Available features exclude target column
  const availableColumns = columns.filter((c) => c.name !== targetColumn);
  const availableColNames = availableColumns.map((c) => c.name);

  return (
    <Card className={`border-slate-800 bg-slate-900/60 ${validationError ? 'border-amber-500/50' : ''}`}>
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <Sliders className="w-5 h-5 text-accent" />
          <CardTitle>3. Select Feature Columns (Inputs)</CardTitle>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={selectedFeatures.length > 0 ? 'healthy' : 'warning'}>
            {selectedFeatures.length} OF {availableColumns.length} SELECTED
          </Badge>
          <button
            type="button"
            onClick={() => onSelectAll(availableColNames)}
            className="text-xs font-mono text-slate-300 hover:text-accent px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
          >
            Select All
          </button>
          <button
            type="button"
            onClick={onClearAll}
            className="text-xs font-mono text-slate-400 hover:text-red-400 px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
          >
            Clear All
          </button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {targetColumn && (
          <div className="text-xs font-mono text-slate-400 bg-slate-950/40 px-3 py-2 rounded border border-slate-800 flex items-center justify-between">
            <span>
              Target column <span className="text-accent font-semibold">"{targetColumn}"</span> is reserved and excluded from input features.
            </span>
          </div>
        )}

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 max-h-[300px] overflow-y-auto p-1 custom-scrollbar">
          {availableColumns.map((col) => {
            const isChecked = selectedFeatures.includes(col.name);
            const isNum = col.isNumeric;

            return (
              <div
                key={col.name}
                onClick={() => onToggleFeature(col.name)}
                className={`p-2.5 rounded border cursor-pointer transition-colors flex items-center justify-between ${
                  isChecked
                    ? 'border-accent/40 bg-accent/10'
                    : 'border-slate-800 bg-slate-950/40 hover:border-slate-700 hover:bg-slate-800/30'
                }`}
              >
                <div className="flex items-center gap-2 overflow-hidden">
                  {isChecked ? (
                    <CheckSquare className="w-4 h-4 text-accent shrink-0" />
                  ) : (
                    <Square className="w-4 h-4 text-slate-600 shrink-0" />
                  )}
                  <div className="overflow-hidden">
                    <p className="text-xs font-mono text-slate-200 truncate" title={col.name}>
                      {col.name}
                    </p>
                    <span className="text-[10px] font-mono text-slate-500">
                      {col.dtype || (isNum ? 'num' : 'cat')}
                    </span>
                  </div>
                </div>

                {!isNum && (
                  <Badge variant="outline" className="text-[9px] px-1 py-0 text-amber-400 border-amber-500/30">
                    CAT
                  </Badge>
                )}
              </div>
            );
          })}
        </div>

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
