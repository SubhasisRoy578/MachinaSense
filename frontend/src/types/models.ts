export type MachineStatus = 'healthy' | 'warning' | 'critical' | 'offline';

export interface Machine {
  id: string;
  name: string;
  type: string;
  location: string;
  status: MachineStatus;
  rulDays: number; // Remaining Useful Life in days
  rulConfidence: number; // Percentage 0-100
  failureRisk: number; // Percentage 0-100
  lastUpdated: string; // ISO date string
}

export interface SensorDataPoint {
  timestamp: string; // ISO date string
  vibration: number;
  temperature: number;
  pressure: number;
  isAnomaly: boolean;
}

export interface Alert {
  id: string;
  machineId: string;
  machineName: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: 'anomaly' | 'rul_warning' | 'maintenance_due';
  description: string;
  timestamp: string;
  acknowledged: boolean;
}

export interface MaintenanceTask {
  id: string;
  machineId: string;
  machineName: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  dueDate: string;
  status: 'pending' | 'in_progress' | 'completed';
}

export interface RulForecastPoint {
  timestamp: string; // ISO date string
  predictedRul: number;
  lowerBound: number;
  upperBound: number;
  isHistorical: boolean;
}

export interface AnomalyEvent {
  id: string;
  timestamp: string;
  sensor: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  score: number; // 0-1
  description: string;
}

export interface DiagnosticPreview {
  finding: string;
  explanation: string;
  evidence: string[];
  recommendedAction?: string;
}

export interface KnowledgeDocument {
  id: string;
  title: string;
  type: 'PDF' | 'DOCX' | 'TXT';
  size: number; // in bytes
  uploadDate: string;
  processingStatus: 'processing' | 'extracting' | 'indexing' | 'ready' | 'failed';
  indexedChunks?: number;
  sourceCategory: 'Manual' | 'Specification' | 'Maintenance Log' | 'Safety Guideline';
  metadata?: Record<string, string>;
}

export interface RetrievalResult {
  documentId: string;
  documentTitle: string;
  snippet: string;
  relevance: number; // 0.0 to 1.0
  section?: string;
  page?: number;
}

export interface DiagnosticCase {
  id: string;
  machineId: string;
  machineName: string;
  trigger: string;
  finding: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  createdTime: string;
  status: 'open' | 'investigating' | 'resolved';
  evidence: RetrievalResult[];
  explanation: string;
  recommendedAction: string;
  confidence?: number;
  generatorUsed?: string;
  isGroundedFallback?: boolean;
}
