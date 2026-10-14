import { create } from 'zustand';
import type { Dataset, DatasetListItem } from '../types/dataset';
import type { MLTask, TrainingConfig, TrainingJob, ModelAlgorithm, RegisteredModel } from '../types/training';

export const TASK_DEFINITIONS: Record<MLTask, { label: string; isSupervised: boolean; description: string }> = {
  rul_regression: {
    label: 'RUL Regression',
    isSupervised: true,
    description: 'Predict remaining useful life (in cycles or operating hours) from sensor telemetry.',
  },
  condition_classification: {
    label: 'Condition Classification',
    isSupervised: true,
    description: 'Classify machine health into operating states: Healthy, Degrading, Critical, or Failed.',
  },
  anomaly_detection: {
    label: 'Anomaly Detection',
    isSupervised: false,
    description: 'Identify abnormal telemetry drift and anomalous sensor behavior without ground-truth labels.',
  },
};

export const TASK_MODELS: Record<MLTask, ModelAlgorithm[]> = {
  rul_regression: [
    {
      id: 'rf_regressor',
      name: 'Random Forest Regressor',
      task: 'rul_regression',
      description: 'Robust ensemble of decision trees for non-linear telemetry degradation modeling.',
      isAvailable: true,
    },
    {
      id: 'gradient_boosting_regressor',
      name: 'Gradient Boosting Regressor',
      task: 'rul_regression',
      description: 'Sequentially boosted trees optimized for minimal mean squared RUL error.',
      isAvailable: true,
    },
    {
      id: 'lstm_regressor',
      name: 'PyTorch LSTM Temporal Regressor',
      task: 'rul_regression',
      description: 'Deep recurrent network capturing temporal patterns across consecutive operating cycles.',
      isAvailable: true,
    },
  ],
  condition_classification: [
    {
      id: 'rf_classifier',
      name: 'Random Forest Classifier',
      task: 'condition_classification',
      description: 'Multi-class tree ensemble predicting health states with calibrated probabilities.',
      isAvailable: true,
    },
    {
      id: 'gradient_boosting_classifier',
      name: 'Gradient Boosting Classifier',
      task: 'condition_classification',
      description: 'High-precision gradient boosted trees for multi-class condition diagnosis.',
      isAvailable: true,
    },
    {
      id: 'logistic_regression',
      name: 'Logistic Regression Classifier',
      task: 'condition_classification',
      description: 'Fast linear multi-class baseline for calibrated machine health condition probabilities.',
      isAvailable: true,
    },
  ],
  anomaly_detection: [
    {
      id: 'isolation_forest',
      name: 'Isolation Forest Anomaly Detector',
      task: 'anomaly_detection',
      description: 'Unsupervised partitioning algorithm isolating multi-sensor outliers and abnormal drift.',
      isAvailable: true,
    },
  ],
};

export type TrainingTab = 'datasets' | 'configure' | 'progress' | 'registry' | 'inference' | 'comparison';

interface TrainingState {
  // Navigation / Phase view
  activeTab: TrainingTab;
  setActiveTab: (tab: TrainingTab) => void;

  // Selected dataset
  selectedDataset: Dataset | null;
  setSelectedDataset: (dataset: Dataset | null) => void;

  // Datasets list
  datasets: DatasetListItem[];
  setDatasets: (datasets: DatasetListItem[]) => void;
  loadingDatasets: boolean;
  setLoadingDatasets: (loading: boolean) => void;
  datasetError: string | null;
  setDatasetError: (error: string | null) => void;

  // Training Configuration (Phase 2)
  task: MLTask;
  setTask: (task: MLTask) => void;
  targetColumn: string | null;
  setTargetColumn: (col: string | null) => void;
  selectedFeatures: string[];
  toggleFeature: (col: string) => void;
  setSelectedFeatures: (cols: string[]) => void;
  selectAllFeatures: (availableCols: string[]) => void;
  clearAllFeatures: () => void;
  modelType: string;
  setModelType: (model: string) => void;
  trainSplit: number;
  validationSplit: number;
  testSplit: number;
  setSplits: (train: number, val: number, test: number) => void;
  randomSeed: number;
  setRandomSeed: (seed: number) => void;

  // Training Job Execution State (Phase 2)
  activeJob: TrainingJob | null;
  setActiveJob: (job: TrainingJob | null) => void;
  isStartingJob: boolean;
  setIsStartingJob: (starting: boolean) => void;
  trainingError: string | null;
  setTrainingError: (error: string | null) => void;
  isCancellingJob: boolean;
  setIsCancellingJob: (cancelling: boolean) => void;

  // Model Registry State (Phase 3 & 4)
  registeredModels: RegisteredModel[];
  setRegisteredModels: (models: RegisteredModel[]) => void;
  loadingModels: boolean;
  setLoadingModels: (loading: boolean) => void;
  selectedModelForInference: RegisteredModel | null;
  setSelectedModelForInference: (model: RegisteredModel | null) => void;

  // Reset
  resetTrainingConfig: () => void;
  getTrainingConfig: () => TrainingConfig | null;
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  activeTab: 'datasets',
  setActiveTab: (activeTab) => set({ activeTab }),

  registeredModels: [],
  setRegisteredModels: (registeredModels) => set({ registeredModels }),
  loadingModels: false,
  setLoadingModels: (loadingModels) => set({ loadingModels }),
  selectedModelForInference: null,
  setSelectedModelForInference: (selectedModelForInference) => set({ selectedModelForInference }),

  selectedDataset: null,
  setSelectedDataset: (selectedDataset) => {
    // When a dataset is selected, reset target and auto-select all non-target columns
    const columns = selectedDataset?.columns.map((c) => c.name) ?? [];
    set({
      selectedDataset,
      targetColumn: null,
      selectedFeatures: columns,
      trainingError: null,
    });
  },

  datasets: [],
  setDatasets: (datasets) => set({ datasets }),
  loadingDatasets: false,
  setLoadingDatasets: (loadingDatasets) => set({ loadingDatasets }),
  datasetError: null,
  setDatasetError: (datasetError) => set({ datasetError }),

  task: 'rul_regression',
  setTask: (task) => {
    const defaultModel = TASK_MODELS[task][0]?.id ?? 'rf_regressor';
    set({
      task,
      modelType: defaultModel,
      targetColumn: task === 'anomaly_detection' ? null : get().targetColumn,
    });
  },

  targetColumn: null,
  setTargetColumn: (targetColumn) => {
    // If target is currently in features, remove it from features
    const currentFeatures = get().selectedFeatures;
    const filteredFeatures = targetColumn ? currentFeatures.filter((f) => f !== targetColumn) : currentFeatures;
    set({ targetColumn, selectedFeatures: filteredFeatures });
  },

  selectedFeatures: [],
  toggleFeature: (col) => {
    const current = get().selectedFeatures;
    if (current.includes(col)) {
      set({ selectedFeatures: current.filter((c) => c !== col) });
    } else {
      set({ selectedFeatures: [...current, col] });
    }
  },
  setSelectedFeatures: (selectedFeatures) => set({ selectedFeatures }),
  selectAllFeatures: (availableCols) => {
    const target = get().targetColumn;
    const features = target ? availableCols.filter((c) => c !== target) : availableCols;
    set({ selectedFeatures: features });
  },
  clearAllFeatures: () => set({ selectedFeatures: [] }),

  modelType: 'rf_regressor',
  setModelType: (modelType) => set({ modelType }),

  trainSplit: 70,
  validationSplit: 15,
  testSplit: 15,
  setSplits: (trainSplit, validationSplit, testSplit) => set({ trainSplit, validationSplit, testSplit }),

  randomSeed: 42,
  setRandomSeed: (randomSeed) => set({ randomSeed }),

  activeJob: null,
  setActiveJob: (activeJob) => set({ activeJob }),
  isStartingJob: false,
  setIsStartingJob: (isStartingJob) => set({ isStartingJob }),
  trainingError: null,
  setTrainingError: (trainingError) => set({ trainingError }),
  isCancellingJob: false,
  setIsCancellingJob: (isCancellingJob) => set({ isCancellingJob }),

  resetTrainingConfig: () => {
    const ds = get().selectedDataset;
    const cols = ds ? ds.columns.map((c) => c.name) : [];
    set({
      task: 'rul_regression',
      targetColumn: null,
      selectedFeatures: cols,
      modelType: 'rf_regressor',
      trainSplit: 70,
      validationSplit: 15,
      testSplit: 15,
      randomSeed: 42,
      trainingError: null,
    });
  },

  getTrainingConfig: () => {
    const { selectedDataset, task, targetColumn, selectedFeatures, modelType, trainSplit, validationSplit, testSplit, randomSeed } = get();
    if (!selectedDataset) return null;
    return {
      dataset_id: selectedDataset.id,
      task,
      target_column: task === 'anomaly_detection' ? null : targetColumn,
      feature_columns: selectedFeatures,
      model_type: modelType,
      train_split: trainSplit,
      validation_split: validationSplit,
      test_split: testSplit,
      random_seed: randomSeed,
    };
  },
}));
