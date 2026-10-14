import { useState, useRef, type DragEvent, type ChangeEvent } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, X, Loader2 } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { api } from '../../services/api';
import type { DatasetUploadResponse } from '../../types/dataset';

interface DatasetUploadProps {
  onUploadSuccess: (dataset: DatasetUploadResponse) => void;
}

export function DatasetUpload({ onUploadSuccess }: DatasetUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const validateAndSetFile = (file: File) => {
    setUploadError(null);
    setUploadSuccess(null);

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUploadError('Invalid file type. Only standard CSV (.csv) files are supported for ML training.');
      setSelectedFile(null);
      return;
    }

    if (file.size === 0) {
      setUploadError('The selected CSV file is empty (0 bytes).');
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    if (!datasetName) {
      setDatasetName(file.name.replace(/\.[^/.]+$/, ''));
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setDatasetName('');
    setUploadError(null);
    setUploadSuccess(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const response = await api.datasets.upload(selectedFile, datasetName.trim() || undefined);
      setUploadSuccess(`Dataset "${response.name || selectedFile.name}" successfully uploaded.`);
      setSelectedFile(null);
      setDatasetName('');
      if (fileInputRef.current) fileInputRef.current.value = '';
      onUploadSuccess(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Dataset upload failed. Backend service may be unavailable.';
      setUploadError(message);
    } finally {
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <Card className="border-slate-800 bg-slate-900/60">
      <CardHeader>
        <div className="flex items-center gap-2">
          <UploadCloud className="w-5 h-5 text-accent" />
          <CardTitle>CSV Dataset Ingestion</CardTitle>
        </div>
        <Badge variant="outline">RFC 4180 CSV</Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Dropzone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => !selectedFile && fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer flex flex-col items-center justify-center ${
            dragActive
              ? 'border-accent bg-accent/5'
              : selectedFile
              ? 'border-emerald-500/40 bg-slate-950/40'
              : 'border-slate-700 hover:border-slate-500 bg-slate-950/20'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            onChange={handleChange}
            className="hidden"
            id="csv-file-input"
          />

          {!selectedFile ? (
            <div className="flex flex-col items-center gap-2">
              <div className="w-12 h-12 rounded-full bg-slate-800/80 flex items-center justify-center text-slate-400">
                <UploadCloud className="w-6 h-6 text-accent" />
              </div>
              <p className="text-sm font-medium text-slate-200">
                Drag and drop your telemetry CSV file here, or{' '}
                <span className="text-accent underline">browse</span>
              </p>
              <p className="text-xs text-slate-400">
                Supports machine telemetry, vibration recordings, C-MAPSS logs, or sensor datasets
              </p>
            </div>
          ) : (
            <div className="flex items-center justify-between w-full max-w-md bg-slate-800/80 border border-slate-700 rounded-md p-3">
              <div className="flex items-center gap-3 overflow-hidden text-left">
                <FileText className="w-6 h-6 text-accent shrink-0" />
                <div className="overflow-hidden">
                  <p className="text-sm font-mono text-slate-200 truncate">{selectedFile.name}</p>
                  <p className="text-xs text-slate-400">{formatFileSize(selectedFile.size)}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleClear();
                }}
                disabled={isUploading}
                className="text-slate-400 hover:text-red-400 p-1 rounded transition-colors disabled:opacity-50"
                title="Remove file"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>

        {/* Dataset Label and Submit */}
        {selectedFile && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
            <div className="sm:col-span-2">
              <label htmlFor="dataset-name-input" className="block text-xs font-mono text-slate-400 mb-1">
                Dataset Name (Optional)
              </label>
              <input
                id="dataset-name-input"
                type="text"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                placeholder="e.g. Turbofan_Bearing_Run_01"
                disabled={isUploading}
                className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-accent font-mono"
              />
            </div>
            <div className="flex items-end">
              <button
                type="button"
                onClick={handleUpload}
                disabled={isUploading}
                className="w-full bg-accent hover:bg-accent/90 text-slate-950 font-semibold px-4 py-2 rounded text-sm transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Uploading...</span>
                  </>
                ) : (
                  <>
                    <UploadCloud className="w-4 h-4" />
                    <span>Upload CSV</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Feedback states */}
        {uploadError && (
          <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 text-red-400 rounded-md p-3 text-xs">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-semibold">Upload Error</p>
              <p>{uploadError}</p>
            </div>
          </div>
        )}

        {uploadSuccess && (
          <div className="flex items-start gap-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-md p-3 text-xs">
            <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-semibold">Upload Complete</p>
              <p>{uploadSuccess}</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
