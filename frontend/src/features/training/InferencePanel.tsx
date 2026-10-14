import { useState, useEffect, useCallback } from 'react';
import {
  Cpu,
  Play,
  Upload,
  FileText,
  Clock,
  History,
  AlertCircle,
  CheckCircle2,
  Download,
  RotateCw,
  Eye,
  Sliders,
  Sparkles,
  X,
  Loader2,
  Check,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { api } from '../../services/api';
import { PredictionExplanationModal } from '../predictions/PredictionExplanationModal';
import type { RegisteredModel, MLTask } from '../../types/training';
import type {
  SingleInferenceResponse,
  BatchInferenceResponse,
  PredictionHistoryRecord,
} from '../../types/inference';

interface InferencePanelProps {
  initialTask?: MLTask;
  task?: MLTask;
  preselectedModel?: RegisteredModel | null;
  onNavigateToRegistry?: () => void;
}

export function InferencePanel({
  initialTask = 'rul_regression',
  task: propTask,
  preselectedModel,
  onNavigateToRegistry,
}: InferencePanelProps) {
  const [subTab, setSubTab] = useState<'single' | 'batch' | 'history'>('single');
  const [task, setTask] = useState<MLTask>(propTask || initialTask);

  // Active Model state
  const [activeModel, setActiveModel] = useState<RegisteredModel | null>(preselectedModel || null);
  const [loadingActiveModel, setLoadingActiveModel] = useState(false);
  const [activeModelError, setActiveModelError] = useState<string | null>(null);

  // Single inference state
  const [featureInputs, setFeatureInputs] = useState<Record<string, string>>({});
  const [isPredictingSingle, setIsPredictingSingle] = useState(false);
  const [singleResult, setSingleResult] = useState<SingleInferenceResponse | null>(null);
  const [singleError, setSingleError] = useState<string | null>(null);

  // Batch inference state
  const [batchRawInput, setBatchRawInput] = useState('');
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [isPredictingBatch, setIsPredictingBatch] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchInferenceResponse | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  // History state
  const [historyList, setHistoryList] = useState<PredictionHistoryRecord[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selectedHistoryRecord, setSelectedHistoryRecord] = useState<PredictionHistoryRecord | null>(null);
  const [explainPredictionId, setExplainPredictionId] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const loadModel = async () => {
      if (!preselectedModel) {
        setLoadingActiveModel(true);
        setActiveModelError(null);
        try {
          const model = await api.models.getActive(task);
          if (!active) return;
          setActiveModel(model);
          if (model.training_metadata?.feature_columns) {
            const initialMap: Record<string, string> = {};
            model.training_metadata.feature_columns.forEach((col: string) => {
              initialMap[col] = '';
            });
            setFeatureInputs(initialMap);
          }
        } catch (err) {
          if (!active) return;
          setActiveModel(null);
          setActiveModelError(
            err instanceof Error
              ? err.message
              : `No active user-trained model found for ${task}.`
          );
        } finally {
          if (active) setLoadingActiveModel(false);
        }
      } else {
        setActiveModel(preselectedModel);
        setTask(preselectedModel.task);
        if (preselectedModel.training_metadata?.feature_columns) {
          const initialMap: Record<string, string> = {};
          preselectedModel.training_metadata.feature_columns.forEach((col: string) => {
            initialMap[col] = '';
          });
          setFeatureInputs(initialMap);
        }
      }
    };
    void loadModel();
    return () => {
      active = false;
    };
  }, [task, preselectedModel]);

  // Load History
  const fetchHistory = useCallback(async () => {
    setLoadingHistory(true);
    setHistoryError(null);
    try {
      const res = await api.inference.getHistory({ task, limit: 50 });
      setHistoryList(res.predictions || []);
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Failed to retrieve prediction history.');
    } finally {
      setLoadingHistory(false);
    }
  }, [task]);

  useEffect(() => {
    let active = true;
    const loadHistory = async () => {
      if (subTab === 'history') {
        setLoadingHistory(true);
        setHistoryError(null);
        try {
          const res = await api.inference.getHistory({ task, limit: 50 });
          if (active) setHistoryList(res.predictions || []);
        } catch (err) {
          if (active) setHistoryError(err instanceof Error ? err.message : 'Failed to retrieve prediction history.');
        } finally {
          if (active) setLoadingHistory(false);
        }
      }
    };
    void loadHistory();
    return () => {
      active = false;
    };
  }, [subTab, task]);

  // Populate sample values for single inference
  const handlePopulateSampleValues = () => {
    const featureCols: string[] = activeModel?.training_metadata?.feature_columns || Object.keys(featureInputs);
    const populated: Record<string, string> = {};
    featureCols.forEach((col: string, idx: number) => {
      // Deterministic realistic sensor telemetry samples
      const val = (25.0 + ((idx * 7) % 50) + Math.sin(idx) * 5).toFixed(2);
      populated[col] = val;
    });
    setFeatureInputs(populated);
  };

  // Run Single Inference
  const handleRunSingleInference = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeModel) return;

    // Convert string inputs to numeric floats
    const numericFeatures: Record<string, number> = {};
    const missing: string[] = [];
    for (const [key, val] of Object.entries(featureInputs)) {
      if (val === '' || val === null || val === undefined) {
        missing.push(key);
      } else {
        const num = parseFloat(val);
        if (isNaN(num)) {
          setSingleError(`Invalid numeric value for feature '${key}': "${val}"`);
          return;
        }
        numericFeatures[key] = num;
      }
    }

    if (missing.length > 0) {
      setSingleError(`Missing required feature values for: ${missing.slice(0, 3).join(', ')}${missing.length > 3 ? '...' : ''}`);
      return;
    }

    setIsPredictingSingle(true);
    setSingleError(null);
    setSingleResult(null);

    try {
      let res: SingleInferenceResponse;
      if (task === 'rul_regression') {
        res = await api.inference.predictRul(numericFeatures);
      } else if (task === 'condition_classification') {
        res = await api.inference.predictClassification(numericFeatures);
      } else {
        res = await api.inference.predictAnomaly(numericFeatures);
      }
      setSingleResult(res);
    } catch (err) {
      setSingleError(err instanceof Error ? err.message : 'Inference request failed.');
    } finally {
      setIsPredictingSingle(false);
    }
  };

  // Run Batch Inference
  const handleRunBatchInference = async () => {
    if (!activeModel) return;
    setIsPredictingBatch(true);
    setBatchError(null);
    setBatchResult(null);

    try {
      let rows: Record<string, any>[] = [];

      // Parse from CSV file or Raw input
      if (batchFile) {
        const text = await batchFile.text();
        rows = parseCsvToRows(text);
      } else if (batchRawInput.trim()) {
        if (batchRawInput.trim().startsWith('[')) {
          // JSON array
          rows = JSON.parse(batchRawInput.trim());
        } else {
          // CSV text
          rows = parseCsvToRows(batchRawInput);
        }
      } else {
        throw new Error('Please select a CSV file or enter CSV data for batch prediction.');
      }

      if (!Array.isArray(rows) || rows.length === 0) {
        throw new Error('Parsed 0 valid rows from batch input.');
      }

      if (rows.length > 500) {
        throw new Error(`Batch contains ${rows.length} rows, which exceeds the maximum backend limit (500).`);
      }

      const res = await api.inference.batch(task, rows);
      setBatchResult(res);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : 'Batch inference failed.');
    } finally {
      setIsPredictingBatch(false);
    }
  };

  // Helper CSV parser
  const parseCsvToRows = (csvText: string): Record<string, any>[] => {
    const lines = csvText.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    if (lines.length < 2) return [];
    const headers = lines[0].split(',').map((h) => h.trim().replace(/^"|"$/g, ''));
    const rows: Record<string, any>[] = [];
    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(',').map((p) => p.trim().replace(/^"|"$/g, ''));
      if (parts.length === headers.length) {
        const rowObj: Record<string, any> = {};
        headers.forEach((h, idx) => {
          const val = parts[idx];
          const num = parseFloat(val);
          rowObj[h] = isNaN(num) ? val : num;
        });
        rows.push(rowObj);
      }
    }
    return rows;
  };

  // Export batch results to CSV
  const handleExportBatchResults = () => {
    if (!batchResult || !batchResult.predictions) return;
    const preds = batchResult.predictions;
    if (preds.length === 0) return;

    const keys = Object.keys(preds[0]);
    const headerRow = ['row_index', ...keys].join(',');
    const dataRows = preds.map((p, idx) => {
      const vals = keys.map((k) => (typeof p[k] === 'object' ? JSON.stringify(p[k]) : p[k]));
      return [idx + 1, ...vals].join(',');
    });

    const csvContent = 'data:text/csv;charset=utf-8,' + [headerRow, ...dataRows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `machinasense_${task}_batch_predictions.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Top Header & Task Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900/80 border border-slate-800 p-4 rounded-lg font-mono">
        <div>
          <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-accent" />
            Live Machine Learning Inference Operations
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Execute validated single telemetry and high-throughput batch inference against active user-trained models
          </p>
        </div>

        {/* Task Selector */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400">Task:</span>
          <select
            value={task}
            onChange={(e) => {
              const newTask = e.target.value as MLTask;
              setTask(newTask);
              setSingleResult(null);
              setBatchResult(null);
            }}
            className="bg-slate-950 border border-slate-700 rounded px-2.5 py-1 text-slate-200 text-xs focus:outline-none focus:border-accent"
          >
            <option value="rul_regression">RUL Regression</option>
            <option value="condition_classification">Condition Classification</option>
            <option value="anomaly_detection">Anomaly Detection</option>
          </select>
        </div>
      </div>

      {/* ACTIVE MODEL STATUS CARD */}
      <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-4 font-mono text-xs">
        {loadingActiveModel ? (
          <div className="flex items-center gap-2 text-slate-400">
            <Loader2 className="w-4 h-4 animate-spin text-accent" />
            <span>Resolving active model for {task}...</span>
          </div>
        ) : activeModel ? (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-slate-500 uppercase text-[10px]">ACTIVE PRODUCTION MODEL:</span>
                <span className="text-slate-100 font-bold">{activeModel.name}</span>
                <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 text-accent font-semibold">
                  v{activeModel.version}
                </span>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  ACTIVE
                </span>
              </div>
              <div className="text-[11px] text-slate-400">
                Algorithm: <span className="text-slate-200">{activeModel.model_type}</span> | ID:{' '}
                <span className="text-slate-500">{activeModel.id}</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {onNavigateToRegistry && (
                <button
                  type="button"
                  onClick={onNavigateToRegistry}
                  className="bg-slate-900 hover:bg-slate-800 text-slate-300 px-3 py-1.5 rounded border border-slate-800 text-xs transition-colors"
                >
                  Manage in Registry
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-amber-400">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>
                {activeModelError || (
                  <>
                    No active user-trained model for <strong className="text-slate-100">{task}</strong>.
                    Please activate a registered model in the Model Registry.
                  </>
                )}
              </span>
            </div>
            {onNavigateToRegistry && (
              <button
                type="button"
                onClick={onNavigateToRegistry}
                className="bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 font-bold px-3 py-1.5 rounded border border-amber-500/40 text-xs transition-colors self-start sm:self-auto"
              >
                Go to Model Registry
              </button>
            )}
          </div>
        )}
      </div>

      {/* Subtabs Bar */}
      <div className="flex items-center gap-2 border-b border-slate-800 font-mono text-xs pb-2">
        <button
          type="button"
          onClick={() => setSubTab('single')}
          className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors ${
            subTab === 'single'
              ? 'bg-accent/20 text-accent border border-accent/40 font-bold'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Play className="w-3.5 h-3.5" />
          <span>Single Telemetry Inference</span>
        </button>

        <button
          type="button"
          onClick={() => setSubTab('batch')}
          className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors ${
            subTab === 'batch'
              ? 'bg-accent/20 text-accent border border-accent/40 font-bold'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Upload className="w-3.5 h-3.5" />
          <span>Batch Dataset Inference</span>
        </button>

        <button
          type="button"
          onClick={() => setSubTab('history')}
          className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors ${
            subTab === 'history'
              ? 'bg-accent/20 text-accent border border-accent/40 font-bold'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <History className="w-3.5 h-3.5" />
          <span>Prediction Audit History</span>
        </button>
      </div>

      {/* SUBTAB 1: SINGLE PREDICTION */}
      {subTab === 'single' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Feature Inputs Form */}
          <Card className="border-slate-800 bg-slate-950/60">
            <CardHeader className="pb-3 border-b border-slate-800">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-mono flex items-center gap-2 text-slate-100">
                  <Sliders className="w-4 h-4 text-accent" />
                  Required Input Features
                </CardTitle>
                <button
                  type="button"
                  onClick={handlePopulateSampleValues}
                  className="text-xs font-mono text-accent hover:text-white bg-accent/10 hover:bg-accent/20 border border-accent/30 px-2.5 py-1 rounded flex items-center gap-1 transition-colors"
                >
                  <Sparkles className="w-3 h-3" />
                  <span>Populate Sample Values</span>
                </button>
              </div>
            </CardHeader>
            <CardContent className="pt-4 space-y-4 font-mono text-xs">
              <p className="text-slate-400 text-[11px]">
                Enter numerical telemetry values corresponding to the active model's required input schema.
              </p>

              {singleError && (
                <div className="p-3 rounded bg-red-500/10 border border-red-500/20 text-red-400 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{singleError}</span>
                </div>
              )}

              <form onSubmit={handleRunSingleInference} className="space-y-4">
                <div className="max-h-[380px] overflow-y-auto pr-1 space-y-2.5 custom-scrollbar">
                  {Object.keys(featureInputs).length === 0 ? (
                    <div className="p-4 text-center text-slate-500 bg-slate-900 rounded border border-slate-800">
                      No feature columns found in model schema. Use "Manage in Registry" or verify model training metadata.
                    </div>
                  ) : (
                    Object.keys(featureInputs).map((col) => (
                      <div key={col} className="flex items-center justify-between gap-3 bg-slate-900/60 p-2 rounded border border-slate-800">
                        <label className="text-slate-300 font-bold truncate max-w-[200px]" title={col}>
                          {col}
                        </label>
                        <input
                          type="number"
                          step="any"
                          required
                          value={featureInputs[col]}
                          onChange={(e) =>
                            setFeatureInputs({ ...featureInputs, [col]: e.target.value })
                          }
                          placeholder="e.g. 100.5"
                          className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 text-right w-32 focus:outline-none focus:border-accent text-xs"
                        />
                      </div>
                    ))
                  )}
                </div>

                <div className="pt-2">
                  <button
                    type="submit"
                    disabled={isPredictingSingle || !activeModel || Object.keys(featureInputs).length === 0}
                    className="w-full bg-accent hover:bg-accent/90 text-slate-950 font-bold py-2.5 rounded text-xs flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                  >
                    {isPredictingSingle ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Running Inference...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        <span>Execute Real Model Prediction</span>
                      </>
                    )}
                  </button>
                </div>
              </form>
            </CardContent>
          </Card>

          {/* Single Prediction Output */}
          <Card className="border-slate-800 bg-slate-950/60 flex flex-col">
            <CardHeader className="pb-3 border-b border-slate-800">
              <CardTitle className="text-sm font-mono flex items-center gap-2 text-slate-100">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Prediction Output Authority
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4 flex-1 flex flex-col justify-center font-mono text-xs">
              {singleResult ? (
                <div className="space-y-5 animate-in fade-in duration-300">
                  {/* Result display tailored to task */}
                  {task === 'rul_regression' && (
                    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 text-center space-y-2">
                      <span className="text-slate-400 text-xs font-semibold uppercase">
                        PREDICTED REMAINING USEFUL LIFE (RUL)
                      </span>
                      <div className="text-4xl font-bold text-accent">
                        {singleResult.prediction.rul_cycles}
                        <span className="text-base text-slate-400 font-normal ml-2">cycles</span>
                      </div>
                      <p className="text-[11px] text-slate-500 mt-2">
                        Computed by active model {singleResult.model.name} (v{singleResult.model.version})
                      </p>
                    </div>
                  )}

                  {task === 'condition_classification' && (
                    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 text-center space-y-3">
                      <span className="text-slate-400 text-xs font-semibold uppercase">
                        PREDICTED HEALTH CONDITION
                      </span>
                      <div className="text-3xl font-bold text-emerald-400">
                        {singleResult.prediction.condition}
                      </div>

                      {/* Class Probabilities Bar */}
                      {singleResult.prediction.probabilities && (
                        <div className="mt-4 pt-4 border-t border-slate-800 text-left space-y-2">
                          <span className="text-[10px] text-slate-500 uppercase block">
                            CLASS PROBABILITY DISTRIBUTION:
                          </span>
                          {Object.entries(singleResult.prediction.probabilities).map(
                            ([cls, prob]: [string, any]) => (
                              <div key={cls} className="space-y-1">
                                <div className="flex justify-between text-[11px]">
                                  <span className="text-slate-300">{cls}</span>
                                  <span className="text-accent font-bold">
                                    {(prob * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden">
                                  <div
                                    className="bg-accent h-full rounded-full"
                                    style={{ width: `${Math.min(100, prob * 100)}%` }}
                                  />
                                </div>
                              </div>
                            )
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {task === 'anomaly_detection' && (
                    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 text-center space-y-3">
                      <span className="text-slate-400 text-xs font-semibold uppercase">
                        ANOMALY ASSESSMENT
                      </span>
                      <div
                        className={`text-3xl font-bold ${
                          singleResult.prediction.is_anomaly ? 'text-red-400' : 'text-emerald-400'
                        }`}
                      >
                        {singleResult.prediction.is_anomaly ? 'ANOMALY DETECTED' : 'NORMAL TELEMETRY'}
                      </div>
                      <div className="text-slate-300 text-xs">
                        Decision Function Score:{' '}
                        <span className="text-accent font-bold">
                          {singleResult.prediction.anomaly_score}
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Provenance Audit Tag */}
                  <div className="bg-slate-950 p-3 rounded border border-slate-800 text-[11px] text-slate-400 space-y-1">
                    <span className="font-bold text-slate-300 block">AUDIT PROVENANCE:</span>
                    <div>Model: {singleResult.model.name} (v{singleResult.model.version})</div>
                    <div>Model ID: {singleResult.model.id}</div>
                    <div>Architecture: {singleResult.model.model_type}</div>
                    <div className="text-emerald-400 flex items-center gap-1 pt-1">
                      <Check className="w-3.5 h-3.5" />
                      <span>Prediction persisted to PostgreSQL audit history</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-12 text-center text-slate-500 space-y-2">
                  <Cpu className="w-8 h-8 text-slate-600 mx-auto" />
                  <p>Enter telemetry features and execute prediction to inspect truthful backend inference.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* SUBTAB 2: BATCH INFERENCE */}
      {subTab === 'batch' && (
        <div className="space-y-6">
          <Card className="border-slate-800 bg-slate-950/60 font-mono text-xs">
            <CardHeader className="pb-3 border-b border-slate-800">
              <CardTitle className="text-sm font-mono flex items-center gap-2 text-slate-100">
                <Upload className="w-4 h-4 text-accent" />
                Batch Dataset Ingestion & Inference
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4 space-y-4">
              <p className="text-slate-400 text-[11px]">
                Upload a CSV file containing multiple telemetry rows or paste tabular comma-separated
                values. Maximum batch size permitted by backend:{' '}
                <strong className="text-slate-200">500 records</strong>.
              </p>

              {batchError && (
                <div className="p-3 rounded bg-red-500/10 border border-red-500/20 text-red-400 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{batchError}</span>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* File Upload Option */}
                <div className="border border-dashed border-slate-700 bg-slate-900/60 rounded-lg p-5 flex flex-col items-center justify-center text-center space-y-2">
                  <FileText className="w-8 h-8 text-slate-500" />
                  <span className="font-bold text-slate-200">Select Telemetry CSV File</span>
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        setBatchFile(e.target.files[0]);
                      }
                    }}
                    className="text-xs text-slate-400 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700"
                  />
                  {batchFile && (
                    <span className="text-[11px] text-accent">Selected: {batchFile.name}</span>
                  )}
                </div>

                {/* Direct CSV / JSON Paste Option */}
                <div className="space-y-1">
                  <span className="text-slate-400 text-[11px] block">
                    Or Paste CSV Data (with header line):
                  </span>
                  <textarea
                    rows={5}
                    value={batchRawInput}
                    onChange={(e) => setBatchRawInput(e.target.value)}
                    placeholder="sensor_1,sensor_2,sensor_3&#10;12.4,55.1,100.2&#10;14.1,56.8,102.0"
                    className="w-full bg-slate-900 border border-slate-700 rounded p-2.5 text-xs text-slate-100 font-mono focus:outline-none focus:border-accent"
                  />
                </div>
              </div>

              <div>
                <button
                  type="button"
                  onClick={handleRunBatchInference}
                  disabled={isPredictingBatch || !activeModel}
                  className="bg-accent hover:bg-accent/90 text-slate-950 font-bold px-6 py-2.5 rounded text-xs flex items-center gap-2 transition-colors disabled:opacity-50"
                >
                  {isPredictingBatch ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Executing Batch Predictions...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>Run Batch Inference</span>
                    </>
                  )}
                </button>
              </div>
            </CardContent>
          </Card>

          {/* Batch Results View */}
          {batchResult && (
            <Card className="border-slate-800 bg-slate-950/60 font-mono text-xs">
              <CardHeader className="pb-3 border-b border-slate-800 flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-mono flex items-center gap-2 text-slate-100">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  Batch Inference Results ({batchResult.total_rows} Records)
                </CardTitle>
                <button
                  type="button"
                  onClick={handleExportBatchResults}
                  className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors text-xs"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export to CSV</span>
                </button>
              </CardHeader>
              <CardContent className="pt-4 space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-900 p-3 rounded border border-slate-800">
                  <div>
                    <span className="text-slate-500 block text-[10px]">TOTAL ROWS PROCESSED</span>
                    <span className="text-slate-100 font-bold">{batchResult.total_rows}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">SUCCESSFUL PREDICTIONS</span>
                    <span className="text-emerald-400 font-bold">
                      {batchResult.predictions.length}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">ACTIVE INFERENCE MODEL</span>
                    <span className="text-slate-100 font-bold">
                      {batchResult.model.name} (v{batchResult.model.version})
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">PERSISTENCE STATUS</span>
                    <span className="text-emerald-400 font-bold">Batch Summary Audited</span>
                  </div>
                </div>

                {/* Table of predictions */}
                <div className="border border-slate-800 rounded overflow-x-auto max-h-72">
                  <table className="w-full text-xs font-mono text-left">
                    <thead className="bg-slate-900 text-slate-400 uppercase text-[10px] border-b border-slate-800">
                      <tr>
                        <th className="px-3 py-2">Row #</th>
                        {batchResult.predictions.length > 0 &&
                          Object.keys(batchResult.predictions[0]).map((key) => (
                            <th key={key} className="px-3 py-2">
                              {key}
                            </th>
                          ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/40">
                      {batchResult.predictions.map((p, idx) => (
                        <tr key={idx} className="hover:bg-slate-900/30">
                          <td className="px-3 py-2 text-slate-500 font-bold">{idx + 1}</td>
                          {Object.keys(p).map((key) => (
                            <td key={key} className="px-3 py-2 text-slate-200">
                              {typeof p[key] === 'object'
                                ? JSON.stringify(p[key])
                                : String(p[key])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* SUBTAB 3: PREDICTION AUDIT HISTORY */}
      {subTab === 'history' && (
        <Card className="border-slate-800 bg-slate-950/60 font-mono text-xs">
          <CardHeader className="pb-3 border-b border-slate-800 flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-sm font-mono flex items-center gap-2 text-slate-100">
                <History className="w-4 h-4 text-accent" />
                Authenticated Inference Audit Records
              </CardTitle>
              <p className="text-slate-400 text-[11px] mt-0.5">
                Every user-trained model prediction is persisted to PostgreSQL for traceability
              </p>
            </div>
            <button
              type="button"
              onClick={fetchHistory}
              disabled={loadingHistory}
              className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors"
            >
              <RotateCw className={`w-3.5 h-3.5 ${loadingHistory ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </CardHeader>
          <CardContent className="pt-4 space-y-4">
            {loadingHistory ? (
              <div className="p-12 text-center text-slate-400 flex items-center justify-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin text-accent" />
                <span>Loading prediction audit logs...</span>
              </div>
            ) : historyError ? (
              <div className="p-3 bg-red-500/10 border border-red-500/20 text-red-400 rounded">
                {historyError}
              </div>
            ) : historyList.length === 0 ? (
              <div className="p-12 text-center text-slate-500 space-y-2">
                <Clock className="w-8 h-8 text-slate-600 mx-auto" />
                <p>No prediction records found for {task}. Execute a single or batch inference to generate audit history.</p>
              </div>
            ) : (
              <div className="border border-slate-800 rounded overflow-x-auto">
                <table className="w-full text-xs font-mono text-left">
                  <thead className="bg-slate-900 text-slate-400 uppercase text-[11px] border-b border-slate-800">
                    <tr>
                      <th className="px-3 py-2.5">Prediction ID</th>
                      <th className="px-3 py-2.5">Task</th>
                      <th className="px-3 py-2.5">Model ID</th>
                      <th className="px-3 py-2.5">Type / Inputs</th>
                      <th className="px-3 py-2.5">Result</th>
                      <th className="px-3 py-2.5">Timestamp</th>
                      <th className="px-3 py-2.5 text-right">Detail</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/40">
                    {historyList.map((rec) => (
                      <tr key={rec.id} className="hover:bg-slate-900/40">
                        <td className="px-3 py-2.5 font-bold text-slate-200">{rec.id}</td>
                        <td className="px-3 py-2.5 text-slate-400">{rec.task}</td>
                        <td className="px-3 py-2.5 text-slate-300 truncate max-w-[140px]" title={rec.model_id}>
                          {rec.model_name ? (
                            <span>{rec.model_name} <span className="text-slate-500 font-mono">v{rec.model_version}</span></span>
                          ) : (
                            rec.model_id
                          )}
                        </td>
                        <td className="px-3 py-2.5 text-slate-400">
                          {rec.input_metadata?.type === 'batch'
                            ? `Batch (${rec.input_metadata.batch_size} rows)`
                            : 'Single'}
                        </td>
                        <td className="px-3 py-2.5 font-bold text-accent">
                          {rec.prediction.rul_cycles !== undefined
                            ? `${rec.prediction.rul_cycles} cycles`
                            : rec.prediction.condition
                            ? rec.prediction.condition
                            : rec.prediction.is_anomaly !== undefined
                            ? rec.prediction.is_anomaly
                              ? 'Anomaly'
                              : 'Normal'
                            : 'Batch Summary'}
                        </td>
                        <td className="px-3 py-2.5 text-slate-500 text-[11px]">
                          {rec.created_at ? new Date(rec.created_at).toLocaleString() : 'N/A'}
                        </td>
                        <td className="px-3 py-2.5 text-right">
                          <div className="flex items-center justify-end gap-1">
                            <button
                              type="button"
                              onClick={() => setSelectedHistoryRecord(rec)}
                              className="p-1 rounded text-slate-400 hover:text-slate-100 hover:bg-slate-800"
                              title="Inspect complete audit record"
                            >
                              <Eye className="w-3.5 h-3.5" />
                            </button>
                            <button
                              type="button"
                              onClick={() => setExplainPredictionId(rec.id)}
                              className="p-1 rounded text-purple-400 hover:text-purple-200 hover:bg-purple-950/40 border border-purple-500/20 transition-colors"
                              title="Explain feature attribution"
                            >
                              <Sparkles className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* DETAIL MODAL FOR PREDICTION AUDIT RECORD */}
      {selectedHistoryRecord && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-xl w-full max-h-[85vh] flex flex-col font-mono text-xs shadow-2xl">
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-accent" />
                <h3 className="font-bold text-slate-100">
                  Prediction Audit Record: {selectedHistoryRecord.id}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedHistoryRecord(null)}
                className="text-slate-400 hover:text-slate-200 p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-4 overflow-y-auto flex-1 space-y-4">
              <div className="grid grid-cols-2 gap-3 bg-slate-950 p-3 rounded border border-slate-800 text-[11px]">
                <div>
                  <span className="text-slate-500 block text-[10px]">RECORD ID</span>
                  <span className="text-slate-200 font-bold">{selectedHistoryRecord.id}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">TASK</span>
                  <span className="text-slate-200">{selectedHistoryRecord.task}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">MODEL</span>
                  <span className="text-slate-200">
                    {selectedHistoryRecord.model_name
                      ? `${selectedHistoryRecord.model_name} (v${selectedHistoryRecord.model_version})`
                      : selectedHistoryRecord.model_id}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">TIMESTAMP</span>
                  <span className="text-slate-200">
                    {selectedHistoryRecord.created_at
                      ? new Date(selectedHistoryRecord.created_at).toLocaleString()
                      : 'N/A'}
                  </span>
                </div>
              </div>

              <div>
                <span className="text-slate-400 font-bold block mb-1">INPUT METADATA:</span>
                <pre className="bg-slate-950 p-3 rounded border border-slate-800 text-[11px] text-slate-300 overflow-x-auto">
                  {JSON.stringify(selectedHistoryRecord.input_metadata, null, 2)}
                </pre>
              </div>

              <div>
                <span className="text-slate-400 font-bold block mb-1">PREDICTION OUTPUT:</span>
                <pre className="bg-slate-950 p-3 rounded border border-slate-800 text-[11px] text-accent overflow-x-auto">
                  {JSON.stringify(selectedHistoryRecord.prediction, null, 2)}
                </pre>
              </div>
            </div>

            <div className="p-3 border-t border-slate-800 flex justify-between items-center">
              <button
                type="button"
                onClick={() => {
                  const pid = selectedHistoryRecord.id;
                  setSelectedHistoryRecord(null);
                  setExplainPredictionId(pid);
                }}
                className="bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 border border-purple-500/30 px-3 py-1.5 rounded text-xs flex items-center gap-1.5 font-bold transition-colors"
              >
                <Sparkles className="w-3 h-3 text-purple-400" />
                <span>Explain Feature Attribution</span>
              </button>
              <button
                type="button"
                onClick={() => setSelectedHistoryRecord(null)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-1.5 rounded text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* PREDICTION ATTRIBUTION EXPLANATION MODAL */}
      {explainPredictionId && (
        <PredictionExplanationModal
          predictionId={explainPredictionId}
          onClose={() => setExplainPredictionId(null)}
        />
      )}
    </div>
  );
}
