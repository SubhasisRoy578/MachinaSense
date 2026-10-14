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
