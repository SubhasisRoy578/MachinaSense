import { useState } from 'react';
import { Database, Trash2, Eye, Play, AlertCircle, Loader2, RefreshCw } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { api } from '../../services/api';
import type { DatasetListItem } from '../../types/dataset';

interface DatasetListProps {
  datasets: DatasetListItem[];
  selectedDatasetId?: string | null;
  onSelectDataset: (id: string) => void;
  onConfigureTraining: (id: string) => void;
  onRefresh: () => void;
  isLoading: boolean;
  onDeleteSuccess?: (deletedId: string) => void;
}

export function DatasetList({
  datasets,
  selectedDatasetId,
  onSelectDataset,
  onConfigureTraining,
  onRefresh,
  isLoading,
  onDeleteSuccess,
}: DatasetListProps) {
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const formatDate = (isoString: string) => {
    try {
      return new Date(isoString).toLocaleDateString();
    } catch {
      return isoString;
    }
  };

  const confirmDelete = async () => {
    if (!deleteId) return;
    setIsDeleting(true);
    setDeleteError(null);

    try {
      const targetId = deleteId;
      await api.datasets.delete(targetId);
      setDeleteId(null);
      if (onDeleteSuccess) {
        onDeleteSuccess(targetId);
      }
      onRefresh();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete dataset.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <Card className="border-slate-800 bg-slate-900/60">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-accent" />
          <CardTitle>Uploaded Telemetry Datasets ({datasets.length})</CardTitle>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          disabled={isLoading}
          className="text-slate-400 hover:text-slate-200 p-1.5 rounded transition-colors disabled:opacity-50"
          title="Refresh dataset inventory"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </CardHeader>
      <CardContent className="p-0">
        {deleteError && (
          <div className="m-4 flex items-center gap-2 bg-red-500/10 border border-red-500/20 text-red-400 rounded p-3 text-xs">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{deleteError}</span>
          </div>
        )}

        {datasets.length === 0 ? (
          <div className="p-8 text-center text-slate-500 font-mono text-xs">
            {isLoading ? (
              <div className="flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin text-accent" />
                <span>Loading datasets from repository...</span>
              </div>
            ) : (
              <span>No user telemetry datasets uploaded yet. Upload a CSV file above to begin.</span>
            )}
          </div>
        ) : (
          <div className="divide-y divide-slate-800/60 font-mono text-xs">
            {datasets.map((ds) => {
              const isSelected = ds.id === selectedDatasetId;
              return (
                <div
                  key={ds.id}
                  className={`p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 transition-colors ${
                    isSelected ? 'bg-accent/10 border-l-2 border-accent' : 'hover:bg-slate-800/30'
                  }`}
                >
                  <div className="overflow-hidden">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-slate-100 text-sm">{ds.name || ds.fileName}</span>
                      <Badge variant={ds.status === 'ready' ? 'healthy' : 'warning'}>
                        {ds.status.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-4 text-slate-400 text-[11px] mt-1">
                      <span>File: {ds.fileName}</span>
                      <span>Size: {formatFileSize(ds.fileSize)}</span>
                      <span>Rows: {ds.rowCount !== null ? ds.rowCount.toLocaleString() : '—'}</span>
                      <span>Cols: {ds.columnCount !== null ? ds.columnCount : '—'}</span>
                      <span>Uploaded: {formatDate(ds.uploadedAt)}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={() => onSelectDataset(ds.id)}
                      className={`px-2.5 py-1.5 rounded flex items-center gap-1.5 transition-colors ${
                        isSelected
                          ? 'bg-slate-700 text-slate-200 cursor-default'
                          : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                      }`}
                    >
                      <Eye className="w-3.5 h-3.5" />
                      <span>{isSelected ? 'Inspecting' : 'Inspect'}</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => onConfigureTraining(ds.id)}
                      className="bg-accent/20 hover:bg-accent/30 text-accent border border-accent/40 px-2.5 py-1.5 rounded flex items-center gap-1.5 transition-colors font-semibold"
                    >
                      <Play className="w-3.5 h-3.5" />
                      <span>Train</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => setDeleteId(ds.id)}
                      className="text-slate-500 hover:text-red-400 p-1.5 rounded transition-colors"
                      title="Delete dataset"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Delete Confirmation Modal */}
        {deleteId && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
            <div className="bg-slate-900 border border-slate-700 rounded-lg max-w-sm w-full p-5 space-y-4 shadow-xl">
              <div className="flex items-center gap-2 text-red-400">
                <AlertCircle className="w-5 h-5" />
                <h4 className="font-semibold text-sm">Delete Dataset</h4>
              </div>
              <p className="text-xs text-slate-300 font-mono">
                Are you sure you want to delete this dataset? This will remove all associated columns, previews, and statistics.
              </p>
              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setDeleteId(null)}
                  disabled={isDeleting}
                  className="px-3 py-1.5 rounded text-xs bg-slate-800 text-slate-300 hover:bg-slate-700 font-mono"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={confirmDelete}
                  disabled={isDeleting}
                  className="px-3 py-1.5 rounded text-xs bg-red-600 hover:bg-red-500 text-white font-mono flex items-center gap-1.5"
                >
                  {isDeleting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                  <span>{isDeleting ? 'Deleting...' : 'Confirm Delete'}</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
