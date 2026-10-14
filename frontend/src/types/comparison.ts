export interface ModelComparisonMetric {
  metric_key: string;
  label: string;
  unit: string;
  direction: 'lower_is_better' | 'higher_is_better' | 'neutral';
  values: Record<string, number | null>;
  best_model_id?: string | null;
}

export interface ModelComparisonModelInfo {
  id: string;
  name: string;
  task: string;
  model_type: string;
  version: number;
  status: string;
  is_benchmark: boolean;
  dataset_name: string;
  dataset_id?: string;
  split_protocol: string;
  sample_count?: number;
  metrics: Record<string, any>;
  target_column?: string | null;
  feature_count?: number;
  duration_seconds?: number | null;
  created_at?: string | null;
  strengths?: string;
  limitations?: string;
}

export interface ModelComparisonResponse {
  is_compatible: boolean;
  incompatibility_reason?: string;
  task?: string;
  identical_evaluation_dataset?: boolean;
  dataset_warning?: string | null;
  target_warning?: string | null;
  narrative?: string;
  models: ModelComparisonModelInfo[];
  metric_comparison: ModelComparisonMetric[];
  guidance?: string;
}

export interface ModelEvaluationHistoryVersion {
  model_id: string;
  name: string;
  version: number;
  task: string;
  model_type: string;
  status: string;
  dataset_id?: string;
  dataset_name: string;
  metrics: Record<string, any>;
  duration_seconds?: number | null;
  created_at?: string | null;
  activated_at?: string | null;
}

export interface ModelEvaluationHistoryResponse {
  model_name: string;
  task: string;
  total_versions: number;
  versions: ModelEvaluationHistoryVersion[];
}
