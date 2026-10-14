import { Database, Table, Hash, AlertTriangle, Copy, HardDrive, Calendar, ArrowRight } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import type { Dataset } from '../../types/dataset';

interface DatasetInspectionProps {
  dataset: Dataset;
  onProceedToTraining?: () => void;
}

export function DatasetInspection({ dataset, onProceedToTraining }: DatasetInspectionProps) {
  const stats = dataset.statistics;

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const formatDate = (isoString: string) => {
    try {
      return new Date(isoString).toLocaleString();
    } catch {
      return isoString;
    }
  };

  return (
    <Card className="border-slate-800 bg-slate-900/60">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-accent" />
          <CardTitle>Dataset Profile: {dataset.name}</CardTitle>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={dataset.status === 'ready' ? 'healthy' : 'warning'}>
            STATUS: {dataset.status.toUpperCase()}
          </Badge>
          {onProceedToTraining && (
            <button
              type="button"
              onClick={onProceedToTraining}
              className="bg-accent hover:bg-accent/90 text-slate-950 font-semibold text-xs px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors"
            >
              <span>Configure Training</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono mb-1">
              <Table className="w-3.5 h-3.5 text-accent" />
              <span>ROWS</span>
            </div>
            <p className="text-lg font-bold font-mono text-slate-100">
              {stats?.rowCount !== undefined ? stats.rowCount.toLocaleString() : dataset.preview?.totalRows ?? '—'}
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono mb-1">
              <Hash className="w-3.5 h-3.5 text-accent" />
              <span>COLUMNS</span>
            </div>
            <p className="text-lg font-bold font-mono text-slate-100">
              {stats?.columnCount !== undefined ? stats.columnCount : dataset.columns.length}
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono mb-1">
              <Hash className="w-3.5 h-3.5 text-emerald-400" />
              <span>NUMERIC</span>
            </div>
            <p className="text-lg font-bold font-mono text-emerald-400">
              {stats?.numericColumnCount !== undefined
                ? stats.numericColumnCount
                : dataset.columns.filter((c) => c.isNumeric).length}
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono mb-1">
              <Hash className="w-3.5 h-3.5 text-amber-400" />
              <span>CATEGORICAL</span>
            </div>
            <p className="text-lg font-bold font-mono text-amber-400">
              {stats?.categoricalColumnCount !== undefined
                ? stats.categoricalColumnCount
                : dataset.columns.filter((c) => !c.isNumeric).length}
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono mb-1">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              <span>MISSING</span>
            </div>
            <p className="text-lg font-bold font-mono text-slate-100">
              {stats?.missingValueCount !== undefined ? stats.missingValueCount.toLocaleString() : '0'}
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono mb-1">
              <Copy className="w-3.5 h-3.5 text-slate-400" />
              <span>DUPLICATES</span>
            </div>
            <p className="text-lg font-bold font-mono text-slate-100">
              {stats?.duplicateRowCount !== undefined ? stats.duplicateRowCount.toLocaleString() : '0'}
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono mb-1">
              <HardDrive className="w-3.5 h-3.5 text-blue-400" />
              <span>SIZE</span>
            </div>
            <p className="text-sm font-bold font-mono text-slate-100 mt-1">
              {formatFileSize(dataset.fileSize)}
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono mb-1">
              <Calendar className="w-3.5 h-3.5 text-slate-400" />
              <span>UPLOADED</span>
            </div>
            <p className="text-xs font-mono text-slate-300 truncate mt-1" title={formatDate(dataset.uploadedAt)}>
              {formatDate(dataset.uploadedAt)}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
