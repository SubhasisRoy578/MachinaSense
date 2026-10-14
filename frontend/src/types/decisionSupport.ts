export interface SupportingDocCitation {
  chunk_id: string;
  document_id: string;
  document_title: string;
  section?: string;
  page?: number;
  relevance_score: number;
  excerpt: string;
}

export interface SuggestedCheck {
  step: number;
  category: string;
  procedure: string;
  rationale: string;
  priority: 'High' | 'Medium' | 'Routine' | string;
}

export interface SuggestedAction {
  action_type: string;
  recommended_timeframe: string;
  summary: string;
  requires_human_approval: boolean;
  advisory_disclaimer: string;
}

export interface ObservedEvidence {
  machine_id?: string | null;
  machine_name: string;
  machine_type: string;
  predicted_rul_cycles?: number | null;
  rul_model_used: string;
  confidence_interval?: { min_cycles: number; max_cycles: number } | null;
  risk_level: 'healthy' | 'warning' | 'critical' | 'unknown' | string;
  anomaly_severity: string;
  anomaly_score: number;
  telemetry_observations?: Record<string, any>;
  open_maintenance_count?: number;
}

export interface DecisionSupportResponse {
  generated_at: string;
  observed_evidence: ObservedEvidence;
  supporting_documentation: SupportingDocCitation[];
  suggested_checks: SuggestedCheck[];
  suggested_action: SuggestedAction;
  uncertainties_and_limitations: string[];
  compliance_notice: string;
}
