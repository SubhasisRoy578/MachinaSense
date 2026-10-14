import { useEffect, useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  BrainCircuit,
  Database,
  Sliders,
  Activity,
  AlertCircle,
  RotateCw,
  FolderOpen,
  Boxes,
  Play,
  Scale,
} from 'lucide-react';
import { api } from '../../services/api';
import { useTrainingStore } from '../../store/trainingStore';
import { DatasetUpload } from './DatasetUpload';
import { DatasetInspection } from './DatasetInspection';
import { ColumnExplorer } from './ColumnExplorer';
import { DatasetPreview } from './DatasetPreview';
import { DatasetList } from './DatasetList';
import { TaskSelector } from './TaskSelector';
import { TargetSelector } from './TargetSelector';
import { FeatureSelector } from './FeatureSelector';
import { ModelSelector } from './ModelSelector';
import { SplitConfiguration } from './SplitConfiguration';
import { TrainingSummary } from './TrainingSummary';
import { TrainingProgress } from './TrainingProgress';
import { ModelRegistry } from './ModelRegistry';
import { InferencePanel } from './InferencePanel';
import { ModelComparisonView } from './ModelComparisonView';
import type { DatasetUploadResponse } from '../../types/dataset';
import type { TrainingJob, RegisteredModel } from '../../types/training';

export function ModelTrainingPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const datasetParam = searchParams.get('dataset');

  const {
    activeTab,
    setActiveTab,
    selectedDataset,
    setSelectedDataset,
    datasets,
    setDatasets,
    loadingDatasets,
    setLoadingDatasets,
    datasetError,
    setDatasetError,

    task,
    setTask,
    targetColumn,
    setTargetColumn,
    selectedFeatures,
    toggleFeature,
    selectAllFeatures,
    clearAllFeatures,
    modelType,
    setModelType,
    trainSplit,
    validationSplit,
    testSplit,
    setSplits,
    randomSeed,
    setRandomSeed,

    activeJob,
    setActiveJob,
    isStartingJob,
    setIsStartingJob,
    setTrainingError,
    isCancellingJob,
    setIsCancellingJob,

    selectedModelForInference,
    setSelectedModelForInference,

    getTrainingConfig,
  } = useTrainingStore();

  // Load datasets list from backend
  const fetchDatasets = useCallback(async () => {
    setLoadingDatasets(true);
    setDatasetError(null);
    try {
      const items = await api.datasets.list();
      setDatasets(items);

      // If URL param is set or no dataset selected, select first
      if (items.length > 0) {
        const targetId = datasetParam || items[0].id;
        const found = items.find((d) => d.id === targetId) || items[0];
        try {
          const full = await api.datasets.get(found.id);
          setSelectedDataset(full);
        } catch {
          // If detailed get fails, create minimal preview from list item
          setSelectedDataset({
            id: found.id,
            name: found.name,
            fileName: found.fileName,
            fileSize: found.fileSize,
            status: found.status,
            statistics: null,
            columns: [],
            preview: null,
            uploadedAt: found.uploadedAt,
          });
        }
      }
    } catch (err) {
      console.warn('Backend datasets API unreachable or not yet deployed:', err);
      setDatasetError(
        err instanceof Error
          ? err.message
          : 'Backend dataset service is currently unavailable. Datasets will load once backend API is active.'
      );
    } finally {
      setLoadingDatasets(false);
    }
  }, [datasetParam, setDatasets, setSelectedDataset, setLoadingDatasets, setDatasetError]);

  useEffect(() => {
    void fetchDatasets();
  }, [fetchDatasets]);

  // Handle successful dataset upload
  const handleUploadSuccess = async (response: DatasetUploadResponse) => {
    void fetchDatasets();
    try {
      const full = await api.datasets.get(response.id);
      setSelectedDataset(full);
    } catch {
      setSelectedDataset({
        id: response.id,
        name: response.name,
        fileName: response.fileName,
        fileSize: response.fileSize,
        status: response.status,
        statistics: response.statistics,
        columns: response.columns,
        preview: response.preview,
        uploadedAt: response.uploadedAt,
      });
    }
    setActiveTab('datasets');
  };

  // Switch selected dataset
  const handleSelectDataset = async (id: string) => {
    setSearchParams({ dataset: id });
    try {
      const full = await api.datasets.get(id);
      setSelectedDataset(full);
    } catch (err) {
      console.error('Failed to load dataset details:', err);
    }
  };

  // Configure training with selected dataset
  const handleConfigureTraining = async (id: string) => {
    await handleSelectDataset(id);
    setActiveTab('configure');
  };

  // Frontend Validation for Phase 2 Training Configuration
  const validationErrors = useMemo(() => {
    const errors: Record<string, string> = {};

    if (!selectedDataset) {
      errors.dataset = 'A telemetry dataset must be selected before training.';
    }

    if (task !== 'anomaly_detection') {
      if (!targetColumn) {
        errors.target = 'Supervised tasks (RUL Regression / Classification) require selecting a valid target column.';
      } else if (selectedFeatures.includes(targetColumn)) {
        errors.targetConflict = 'The target column cannot simultaneously be included in input features.';
      }
    }

    if (selectedFeatures.length === 0) {
      errors.features = 'At least one input feature column must be selected.';
    }

    const splitSum = trainSplit + validationSplit + testSplit;
    if (splitSum !== 100) {
      errors.split = `Train (${trainSplit}%), Validation (${validationSplit}%), and Test (${testSplit}%) splits must equal exactly 100% (currently ${splitSum}%).`;
    }

    if (isNaN(randomSeed) || !Number.isInteger(randomSeed)) {
      errors.seed = 'Random seed must be a valid integer.';
    }

    return errors;
  }, [selectedDataset, task, targetColumn, selectedFeatures, trainSplit, validationSplit, testSplit, randomSeed]);

  // Start Training Job execution
  const handleStartTraining = async () => {
    const config = getTrainingConfig();
    if (!config || Object.keys(validationErrors).length > 0) return;

    setIsStartingJob(true);
    setTrainingError(null);

    try {
      const job = await api.training.start(config);
      setActiveJob(job);
      setActiveTab('progress');
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Backend training endpoint error. The FastAPI training service may not be deployed yet.';

      // Construct truthful job failure object to display backend error
      const placeholderJob: TrainingJob = {
        id: `job-${Date.now().toString(36)}`,
        dataset_id: config.dataset_id,
        dataset_name: selectedDataset?.name,
        task: config.task,
        model_type: config.model_type,
        status: 'failed',
        error_message: errorMessage,
        started_at: new Date().toISOString(),
        completed_at: new Date().toISOString(),
        created_at: new Date().toISOString(),
        config,
      };

      setActiveJob(placeholderJob);
      setActiveTab('progress');
      setTrainingError(errorMessage);
    } finally {
      setIsStartingJob(false);
    }
  };

  // Cancel Job
  const handleCancelJob = async () => {
    if (!activeJob) return;
    setIsCancellingJob(true);
    try {
      await api.training.cancel(activeJob.id);
      setActiveJob({ ...activeJob, status: 'cancelled' });
    } catch (err) {
      console.warn('Failed to cancel training job:', err);
    } finally {
      setIsCancellingJob(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Page Title & Industrial Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded bg-accent/20 border border-accent/50 flex items-center justify-center">
              <BrainCircuit className="w-5 h-5 text-accent" />
            </div>
            <h1 className="text-xl font-bold tracking-wide text-primary uppercase font-mono">
              Machine Learning Training Operations
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1 font-mono">
            Telemetry Dataset Ingestion, Feature Engineering Configuration, and Model Training Pipeline
          </p>
        </div>

        {/* Workflow Phase Tabs */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 border border-slate-800 rounded-lg font-mono text-xs">
          <button
            type="button"
            onClick={() => setActiveTab('datasets')}
            className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors ${
              activeTab === 'datasets'
                ? 'bg-slate-800 text-accent font-semibold shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>1. Datasets</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('configure')}
            className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors ${
              activeTab === 'configure'
                ? 'bg-slate-800 text-accent font-semibold shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>2. Configure Training</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('progress')}
            className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors ${
              activeTab === 'progress'
                ? 'bg-slate-800 text-accent font-semibold shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>3. Monitor Job</span>
            {activeJob && (
              <span className="w-2 h-2 rounded-full bg-accent animate-pulse ml-0.5" />
            )}
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('registry')}
            className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors ${
              activeTab === 'registry'
                ? 'bg-slate-800 text-accent font-semibold shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Boxes className="w-3.5 h-3.5" />
            <span>4. Model Registry</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('inference')}
            className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors ${
              activeTab === 'inference'
                ? 'bg-slate-800 text-accent font-semibold shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Play className="w-3.5 h-3.5" />
            <span>5. Run Inference</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('comparison')}
            className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors ${
              activeTab === 'comparison'
                ? 'bg-slate-800 text-accent font-semibold shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Scale className="w-3.5 h-3.5" />
            <span>6. Model Comparison</span>
          </button>
        </div>
      </div>

      {/* Backend API Notice (if dataset endpoint unreachable) */}
      {datasetError && (
        <div className="bg-slate-900 border border-slate-700/80 rounded-lg p-3 text-xs font-mono text-slate-300 flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-bold text-amber-400">Backend Notice: </span>
            <span>{datasetError}</span>
          </div>
          <button
            type="button"
            onClick={fetchDatasets}
            className="text-slate-400 hover:text-slate-200 p-1"
            title="Retry connecting"
          >
            <RotateCw className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* TAB 1: DATASET INGESTION & INSPECTION (PHASE 1 FOUNDATION) */}
      {activeTab === 'datasets' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DatasetUpload onUploadSuccess={handleUploadSuccess} />
            <DatasetList
              datasets={datasets}
              selectedDatasetId={selectedDataset?.id}
              onSelectDataset={handleSelectDataset}
              onConfigureTraining={handleConfigureTraining}
              onRefresh={fetchDatasets}
              onDeleteSuccess={(deletedId) => {
                if (selectedDataset?.id === deletedId) {
                  setSelectedDataset(null);
                }
              }}
              isLoading={loadingDatasets}
            />
          </div>

          {selectedDataset ? (
            <div className="space-y-6">
              <DatasetInspection
                dataset={selectedDataset}
                onProceedToTraining={() => setActiveTab('configure')}
              />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ColumnExplorer columns={selectedDataset.columns} />
                <DatasetPreview preview={selectedDataset.preview} />
              </div>
            </div>
          ) : (
            <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-12 text-center text-slate-500 font-mono text-xs">
              <FolderOpen className="w-8 h-8 text-slate-600 mx-auto mb-2" />
              <p>Upload a telemetry CSV or select an existing dataset above to view schema inspection and preview.</p>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: TRAINING CONFIGURATION WORKFLOW (PHASE 2) */}
      {activeTab === 'configure' && (
        <div className="space-y-6">
          {/* Active Dataset Selector Bar */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 font-mono text-xs">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-accent" />
              <span className="text-slate-400">Target Dataset:</span>
              <span className="text-slate-100 font-bold">
                {selectedDataset ? selectedDataset.name : 'No dataset selected'}
              </span>
            </div>
            {datasets.length > 1 && (
              <div className="flex items-center gap-2">
                <span className="text-slate-500">Switch:</span>
                <select
                  value={selectedDataset?.id || ''}
                  onChange={(e) => handleSelectDataset(e.target.value)}
                  className="bg-slate-950 border border-slate-700 rounded px-2.5 py-1 text-slate-200 text-xs focus:outline-none focus:border-accent"
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name || d.fileName} ({d.rowCount ? `${d.rowCount} rows` : 'CSV'})
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {selectedDataset ? (
            <div className="space-y-6">
              {/* Step 1: Task Selection */}
              <TaskSelector selectedTask={task} onSelectTask={setTask} />

              {/* Step 2: Target Selection */}
              <TargetSelector
                task={task}
                columns={selectedDataset.columns}
                targetColumn={targetColumn}
                onSelectTarget={setTargetColumn}
                validationError={validationErrors.target || validationErrors.targetConflict}
              />

              {/* Step 3: Feature Selection */}
              <FeatureSelector
                columns={selectedDataset.columns}
                selectedFeatures={selectedFeatures}
                targetColumn={targetColumn}
                onToggleFeature={toggleFeature}
                onSelectAll={selectAllFeatures}
                onClearAll={clearAllFeatures}
                validationError={validationErrors.features}
              />

              {/* Step 4: Model Selection */}
              <ModelSelector
                task={task}
                selectedModel={modelType}
                onSelectModel={setModelType}
              />

              {/* Step 5: Split & Seed */}
              <SplitConfiguration
                trainSplit={trainSplit}
                validationSplit={validationSplit}
                testSplit={testSplit}
                randomSeed={randomSeed}
                onSetSplits={setSplits}
                onSetSeed={setRandomSeed}
                validationError={validationErrors.split || validationErrors.seed}
              />

              {/* Step 6: Dynamic Training Summary & Trigger */}
              <TrainingSummary
                dataset={selectedDataset}
                task={task}
                targetColumn={targetColumn}
                selectedFeatures={selectedFeatures}
                modelType={modelType}
                trainSplit={trainSplit}
                validationSplit={validationSplit}
                testSplit={testSplit}
                randomSeed={randomSeed}
                validationErrors={validationErrors}
                onStartTraining={handleStartTraining}
                isStarting={isStartingJob}
              />
            </div>
          ) : (
            <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-12 text-center text-slate-400 font-mono text-xs space-y-3">
              <Database className="w-8 h-8 text-accent mx-auto" />
              <p className="font-semibold text-slate-200">No Dataset Currently Selected</p>
              <p>Please upload or select a CSV dataset in the Datasets tab before configuring model training.</p>
              <button
                type="button"
                onClick={() => setActiveTab('datasets')}
                className="bg-accent text-slate-950 font-bold px-4 py-2 rounded text-xs transition-colors"
              >
                Go to Datasets Tab
              </button>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: LIVE TRAINING JOB MONITOR & REAL EVALUATION */}
      {activeTab === 'progress' && (
        <div>
          {activeJob ? (
            <TrainingProgress
              job={activeJob}
              onUpdateJob={setActiveJob}
              onReturnToConfig={() => setActiveTab('configure')}
              onRetry={handleStartTraining}
              isCancelling={isCancellingJob}
              onCancelJob={handleCancelJob}
              onNavigateToRegistry={() => setActiveTab('registry')}
              onNavigateToInference={(model: RegisteredModel) => {
                setSelectedModelForInference(model);
                setActiveTab('inference');
              }}
            />
          ) : (
            <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-12 text-center text-slate-400 font-mono text-xs space-y-3">
              <Activity className="w-8 h-8 text-slate-600 mx-auto" />
              <p className="font-semibold text-slate-200">No Active Training Job</p>
              <p>Configure model training parameters and click "Start Training Job" to initiate execution.</p>
              <button
                type="button"
                onClick={() => setActiveTab('configure')}
                className="bg-accent text-slate-950 font-bold px-4 py-2 rounded text-xs transition-colors"
              >
                Configure Training Job
              </button>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: MODEL REGISTRY & VERSION MANAGEMENT */}
      {activeTab === 'registry' && (
        <ModelRegistry
          onSelectModelForInference={(model: RegisteredModel) => {
            setSelectedModelForInference(model);
            setActiveTab('inference');
          }}
          onNavigateToConfigure={() => setActiveTab('configure')}
        />
      )}

      {/* TAB 5: REAL INFERENCE & PREDICTION AUDIT HISTORY */}
      {activeTab === 'inference' && (
        <InferencePanel
          task={task}
          preselectedModel={selectedModelForInference}
          onNavigateToRegistry={() => setActiveTab('registry')}
        />
      )}

      {/* TAB 6: COMPARATIVE MODEL EVALUATION & SCIENTIFIC BENCHMARKING */}
      {activeTab === 'comparison' && (
        <ModelComparisonView />
      )}
    </div>
  );
}
