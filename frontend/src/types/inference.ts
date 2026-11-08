import type { MLTask } from './training';

export interface ModelSummaryMeta {
  id: string;
  name: string;
  version: number;
  model_type: string;
}

export interface SingleInferenceResponse<T = any> {
  task: MLTask;
  prediction: T;
  model: ModelSummaryMeta;
}

export interface BatchInferenceResponse<T = any> {
  task: MLTask;
  predictions: T[];
  model: ModelSummaryMeta;
  total_rows: number;
}

export interface PredictionHistoryRecord {
  id: string;
  model_id: string;
  model_name?: string | null;
  model_version?: number | null;
  model_type?: string | null;
  task: MLTask;
  input_metadata: Record<string, any>;
  prediction: Record<string, any>;
  created_at: string;
}

export interface FD001Features {
  cycle: number;
  op_setting_1: number;
  op_setting_2: number;
  op_setting_3: number;
  sensor_2: number;
  sensor_3: number;
  sensor_4: number;
  sensor_6: number;
  sensor_7: number;
  sensor_8: number;
  sensor_9: number;
  sensor_11: number;
  sensor_12: number;
  sensor_13: number;
  sensor_14: number;
  sensor_15: number;
  sensor_17: number;
  sensor_20: number;
  sensor_21: number;
}

export interface FD001PredictionRequest {
  features: FD001Features;
}

export interface FD001EvaluationMetrics {
  MAE: number;
  RMSE: number;
  R2: number;
  validation_engines: number;
  training_engines: number;
}

export interface FD001PredictionResponse {
  predicted_rul: number;
  model_name: string;
  algorithm: string;
  dataset: string;
  target: string;
  rul_cap: number;
  evaluation: FD001EvaluationMetrics;
  features_used: string[];
}

