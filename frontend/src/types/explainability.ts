export interface FeatureImportanceItem {
  feature: string;
  importance: number;
  raw_score: number;
  label: string;
  unit: string;
  rank?: number;
}

export interface GlobalExplanationResponse {
  model_id: string;
  model_name: string;
  model_type: string;
  task: string;
  version: number;
  status: string;
  is_benchmark: boolean;
  supported: boolean;
  explanation_method: string;
  method_description: string;
  evaluation_context: {
    dataset?: string;
    dataset_name?: string;
    target?: string;
    sample_count?: number;
    split?: string;
    metrics?: Record<string, any>;
  };
  features: FeatureImportanceItem[];
  limitations: string;
}

export interface FeatureContributionItem {
  feature: string;
  label: string;
  value: any;
  unit: string;
  contribution: number;
  impact_direction: string;
  description: string;
}

export interface LocalExplanationResponse {
  prediction_id?: string;
  model_id: string;
  model_name: string;
  model_version?: number;
  model_type?: string;
  task: string;
  prediction_output: Record<string, any>;
  explanation_method: string;
  contributions: FeatureContributionItem[];
  limitations: string;
}
