// ---------------------------------------------------------------------------
// Training types for the ML Training workflow (Phase 2: Training Configuration & Job Execution)
// ---------------------------------------------------------------------------

export type MLTask = 'rul_regression' | 'condition_classification' | 'anomaly_detection';

export interface MLTaskDefinition {
  id: MLTask;
  label: string;
  isSupervised: boolean;
  description: string;
  targetDescription?: string;
}

export interface ModelAlgorithm {
  id: string;
  name: string;
  task: MLTask;
  description: string;
  isAvailable: boolean;
}

export interface TrainingConfig {
  dataset_id: string;
  task: MLTask;
  target_column: string | null;
  feature_columns: string[];
  model_type: string;
  train_split: number;
  validation_split: number;
  test_split: number;
  random_seed: number;
}

export type TrainingJobStatus =
  | 'queued'
  | 'validating'
  | 'preprocessing'
  | 'training'
  | 'evaluating'
  | 'saving'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface TrainingJob {
  id: string;
  dataset_id: string;
  dataset_name?: string;
  task: MLTask;
  model_type: string;
  status: TrainingJobStatus;
  progress?: number | null;
  current_step?: string | null;
  logs?: string[];
  error_message?: string | null;
  started_at?: string;
  completed_at?: string | null;
  duration_seconds?: number | null;
  created_at: string;
  config?: TrainingConfig;
  metrics?: Record<string, any> | null;
  artifactPath?: string | null;
}

export interface TrainingValidationResult {
  isValid: boolean;
  errors: Record<string, string>;
}

export interface RegisteredModel {
  id: string;
  name: string;
  task: MLTask;
  model_type: string;
  version: number;
  status: 'active' | 'inactive';
  training_job_id: string;
  dataset_id: string;
  metrics: Record<string, any>;
  training_metadata?: Record<string, any>;
  created_at: string;
  updated_at?: string | null;
  activated_at?: string | null;
  deactivated_at?: string | null;
}

export interface ModelVersion {
  id: string;
  name: string;
  task: MLTask;
  model_type: string;
  version: number;
  status: 'active' | 'inactive';
  training_job_id: string;
  dataset_id: string;
  metrics: Record<string, any>;
  created_at: string;
  activated_at?: string | null;
  deactivated_at?: string | null;
}

export interface RegisterModelPayload {
  training_job_id: string;
  name: string;
}

