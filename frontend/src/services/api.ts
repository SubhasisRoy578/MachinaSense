import type { Machine } from '../types/models';
import type { Dataset, DatasetListItem, DatasetUploadResponse } from '../types/dataset';
import type { TrainingConfig, TrainingJob, RegisteredModel, ModelVersion, RegisterModelPayload } from '../types/training';
import type { SingleInferenceResponse, BatchInferenceResponse, PredictionHistoryRecord } from '../types/inference';
import type { GlobalExplanationResponse, LocalExplanationResponse } from '../types/explainability';
import type { ModelComparisonResponse, ModelEvaluationHistoryResponse } from '../types/comparison';
import type { DecisionSupportResponse } from '../types/decisionSupport';

const BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.PROD ? 'https://machinasense-backend.onrender.com' : 'http://localhost:8000')
).replace(/\/$/, '');

let tokenGetter: (() => Promise<string | null>) | null = null;
export type SystemStatus = {
  mode: 'live_rag' | 'live_fallback' | 'live_no_rag' | 'error';
  label: string;
  message?: string;
  modelsLoaded?: boolean;
  database?: string;
  baseUrl?: string;
  llmStatus?: string;
};

export function configureApiAuth(getToken: () => Promise<string | null>) {
  tokenGetter = getToken;
}

async function request<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const token = tokenGetter ? await tokenGetter() : null;
  if (!token) throw new Error('Authenticated session required. Please sign in.');
  const headers = new Headers(init.headers);
  headers.set('Authorization', `Bearer ${token}`);
  if (!(init.body instanceof FormData)) headers.set('Content-Type', 'application/json');

  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch (err: any) {
    throw new Error(`Network error connecting to backend service (${err.message || 'connection failed'}). Please check your connection.`);
  }

  if (response.status === 204) return null as unknown as T;
  const body = await response.json().catch(() => ({}));
  const reqId = response.headers.get('x-request-id');
  const reqIdSuffix = reqId ? ` (Trace: ${reqId})` : '';

  if (response.status === 401) throw new Error((body.detail || 'Session expired or unauthenticated. Please sign in again.') + reqIdSuffix);
  if (response.status === 403) throw new Error((body.detail || 'Access denied. You do not have permission to access this resource.') + reqIdSuffix);
  if (response.status === 503) throw new Error((body.detail || 'Service is temporarily degraded or undergoing initialization. Please retry shortly.') + reqIdSuffix);
  if (!response.ok) throw new Error((body.detail || `Request failed (${response.status})`) + reqIdSuffix);
  return body;
}

function mapMachine(m: any): Machine {
  return {
    id: m.id,
    name: m.name,
    type: m.type || m.model,
    location: m.location,
    status: m.status === 'offline' ? 'offline' : m.status,
    rulDays: m.rulDays ?? 0,
    rulConfidence: m.rulConfidence ?? 0,
    failureRisk: m.failureRisk ?? 0,
    lastUpdated: m.lastUpdated,
  };
}

export function subscribeSystemStatus(listener: (value: SystemStatus) => void) {
  fetch(`${BASE}/health`)
    .then((r) => (r.ok ? r.json() : null))
    .then((data) => {
      if (!data) {
        listener({ mode: 'error', label: 'BACKEND ERROR', baseUrl: BASE });
      } else if (data.status === 'healthy') {
        listener({
          mode: 'live_rag',
          label: 'LIVE ML + RAG',
          modelsLoaded: data.models_loaded,
          database: data.database,
          baseUrl: BASE,
        });
      } else {
        listener({
          mode: 'live_fallback',
          label: 'DEGRADED SERVICE',
          message: 'Backend running in degraded mode',
          modelsLoaded: data.models_loaded,
          database: data.database,
          baseUrl: BASE,
        });
      }
    })
    .catch(() => listener({ mode: 'error', label: 'BACKEND UNREACHABLE', baseUrl: BASE }));
  return () => {};
}

export async function checkBackendHealth() {
  const r = await fetch(`${BASE}/health`).catch((err) => {
    throw new Error(`Backend service unreachable: ${err.message}`);
  });
  if (!r.ok) throw new Error(`Backend reported degraded status (${r.status})`);
  return r.json();
}
export const api = {
  machines: {
    list: async (): Promise<Machine[]> => (await request('/api/machines')).map(mapMachine),
    get: async (id: string): Promise<any> => mapMachine(await request(`/api/machines/${encodeURIComponent(id)}`)),
    create: async (input: { machine_id:string; name:string; machine_type:string; location:string; description?:string }) => mapMachine(await request('/api/machines', {method:'POST',body:JSON.stringify(input)})),
    delete: async (id: string) => request(`/api/machines/${encodeURIComponent(id)}`, { method: 'DELETE' }),
    uploadTelemetry: async (id:string,file:File) => { const form=new FormData();form.append('file',file);return request(`/api/machines/${encodeURIComponent(id)}/telemetry`,{method:'POST',body:form}); },
  },
  sensors: {
    getMachineData: async (id:string, _hours?:number): Promise<any[]> => request(`/api/machines/${encodeURIComponent(id)}/sensors`),
    listFleetData: async () => { const ms:any[] = await request('/api/machines'); return {total:ms.length,active:ms.filter(m=>m.currentCycle>0).length,warning:ms.filter(m=>m.anomalySeverity !== 'none').length,recentActivity:[]}; },
  },
  diagnostics: {
    getMachineAnomalies: async (id:string): Promise<any[]> => {
      const data = await request(`/api/machines/${encodeURIComponent(id)}/anomalies`);
      return (data.anomalyEvents || []).map((e: any) => ({
        ...e,
        machineId: id,
        status: e.status || 'active',
        sensor: e.sensor || (e.affectedSensors?.length ? e.affectedSensors.join(', ') : 'Telemetry Observation'),
      }));
    },
    listFleetAnomalies: async () => {
      const ms:any[] = await request('/api/machines');
      const parts = await Promise.all(ms.map(async m => {
        const res = await request(`/api/machines/${encodeURIComponent(m.id)}/anomalies`);
        return (res.anomalyEvents || []).map((e: any) => ({
          ...e,
          machineId: m.id,
          machineName: m.name,
          status: e.status || 'active',
          sensor: e.sensor || (e.affectedSensors?.length ? e.affectedSensors.join(', ') : 'Telemetry Observation'),
        }));
      }));
      return parts.flat();
    },
    getRulForecast: async (id:string): Promise<any[]> => { try { return (await request(`/api/machines/${encodeURIComponent(id)}/prediction`)).degradationCurve || []; } catch (e) { if ((e as Error).message.includes('No predictions')) return []; throw e; } },
    listFleetPredictions: async () => { const ms:any[]=await request('/api/machines'); return ms.filter(m=>m.predictedRul != null).map(m=>({machineId:m.id,machineName:m.name,predictedRul:m.predictedRul,rfPredictedRul:m.rfPredictedRul,confidence:m.rulConfidence,failureRisk:m.failureRisk,degradationState:m.status,nextMilestone:'Review uploaded telemetry',lastPredictionTime:m.lastUpdated})); },
    listCases: async (): Promise<any[]> => request('/api/diagnostics'),
    getCase: async (id:string): Promise<any> => request(`/api/diagnostics/${encodeURIComponent(id)}`),
    createCase: async (machine_id:string): Promise<any> => request('/api/diagnostics',{method:'POST',body:JSON.stringify({machine_id})}),
    explain: async (id:string): Promise<any> => request(`/api/diagnostics/${encodeURIComponent(id)}/explain`,{method:'POST'}),
    getMachineDiagnostics: async (id:string): Promise<any> => {
      const cases = await request('/api/diagnostics'); const latest = cases.find((item:any) => item.machineId === id);
      return latest ? {...latest, evidence: (latest.evidence || []).map((item:any) => item.snippet || item.documentTitle)} : null;
    },
  },
  maintenance: {
    listQueue: async (): Promise<any[]> => request('/api/maintenance'),
    create: async (body:any) => request('/api/maintenance',{method:'POST',body:JSON.stringify(body)}),
    update: async (id:string,body:any) => request(`/api/maintenance/${encodeURIComponent(id)}`,{method:'PATCH',body:JSON.stringify(body)}),
    remove: async (id:string) => request(`/api/maintenance/${encodeURIComponent(id)}`,{method:'DELETE'}),
  },
  knowledge: {
    listDocuments: async () => request('/api/knowledge/documents'),
    upload: async (file:File,category='Manual') => { const form=new FormData();form.append('file',file);form.append('category',category);return request('/api/knowledge/upload',{method:'POST',body:form}); },
    search: async (query:string) => (await request('/api/knowledge/search',{method:'POST',body:JSON.stringify({query,top_k:5})})).map((r:any)=>({documentId:r.document_id,documentTitle:r.document_name,snippet:r.excerpt,relevance:r.relevance_score,section:r.section,page:r.page})),
    getDocument: async (id: string) => request(`/api/knowledge/documents/${encodeURIComponent(id)}`),
    deleteDocument: async (id: string) => request(`/api/knowledge/documents/${encodeURIComponent(id)}`, { method: 'DELETE' }),
    getChunks: async (id: string) => request(`/api/knowledge/documents/${encodeURIComponent(id)}/chunks`),
  },
  analytics: {
    getModels: async () => request('/api/analytics/models'),
    getFleet: async () => request('/api/analytics/fleet'),
  },
  copilot: {
    conversations: () => request('/api/copilot/conversations'),
    create: (machine_id?: string, title?: string) => request('/api/copilot/conversations', { method: 'POST', body: JSON.stringify({ machine_id: machine_id || null, title }) }),
    get: (id: string) => request(`/api/copilot/conversations/${encodeURIComponent(id)}`),
    update: (id: string, payload: { machine_id?: string | null; title?: string }) => request(`/api/copilot/conversations/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(payload) }),
    send: (id: string, content: string, machine_id?: string | null) => request(`/api/copilot/conversations/${encodeURIComponent(id)}/messages`, { method: 'POST', body: JSON.stringify({ content, machine_id }) }),
    delete: (id: string) => request(`/api/copilot/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' }),
    status: () => request('/api/copilot/status'),
  },
  datasets: {
    list: async (): Promise<DatasetListItem[]> => {
      const res = await request('/api/datasets');
      return (res.datasets ?? res) as DatasetListItem[];
    },
    get: async (id: string): Promise<Dataset> => request(`/api/datasets/${encodeURIComponent(id)}`),
    upload: async (file: File, name?: string): Promise<DatasetUploadResponse> => {
      const form = new FormData();
      form.append('file', file);
      if (name) form.append('name', name);
      return request('/api/datasets', { method: 'POST', body: form });
    },
    delete: async (id: string): Promise<null> => request(`/api/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
  training: {
    start: async (config: TrainingConfig): Promise<TrainingJob> =>
      request('/api/training/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      }),
    getStatus: async (jobId: string): Promise<TrainingJob> =>
      request(`/api/training/jobs/${encodeURIComponent(jobId)}`),
    cancel: async (jobId: string): Promise<{ success: boolean; message?: string }> =>
      request(`/api/training/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }),
    listJobs: async (datasetId?: string): Promise<TrainingJob[]> => {
      const query = datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : '';
      const res = await request(`/api/training/jobs${query}`);
      return (res.jobs ?? res) as TrainingJob[];
    },
  },
  models: {
    register: async (payload: RegisterModelPayload): Promise<RegisteredModel> =>
      request('/api/models/register', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
    list: async (params?: { task?: string; status?: string; name?: string }): Promise<{ models: RegisteredModel[]; total: number }> => {
      const search = new URLSearchParams();
      if (params?.task) search.set('task', params.task);
      if (params?.status) search.set('status', params.status);
      if (params?.name) search.set('name', params.name);
      const qs = search.toString();
      const res = await request(`/api/models${qs ? `?${qs}` : ''}`);
      return res;
    },
    get: async (id: string): Promise<RegisteredModel> =>
      request(`/api/models/${encodeURIComponent(id)}`),
    getVersions: async (id: string): Promise<ModelVersion[]> =>
      request(`/api/models/${encodeURIComponent(id)}/versions`),
    activate: async (id: string): Promise<RegisteredModel> =>
      request(`/api/models/${encodeURIComponent(id)}/activate`, { method: 'POST' }),
    deactivate: async (id: string): Promise<RegisteredModel> =>
      request(`/api/models/${encodeURIComponent(id)}/deactivate`, { method: 'POST' }),
    getActive: async (task: string): Promise<RegisteredModel> =>
      request(`/api/models/active/${encodeURIComponent(task)}`),
  },
  inference: {
    predictRul: async (features: Record<string, any>): Promise<SingleInferenceResponse<{ rul_cycles: number }>> =>
      request('/api/inference/rul', {
        method: 'POST',
        body: JSON.stringify({ features }),
      }),
    predictClassification: async (features: Record<string, any>): Promise<SingleInferenceResponse<{ condition: string; probabilities?: Record<string, number> }>> =>
      request('/api/inference/classification', {
        method: 'POST',
        body: JSON.stringify({ features }),
      }),
    predictAnomaly: async (features: Record<string, any>): Promise<SingleInferenceResponse<{ is_anomaly: boolean; anomaly_score: number }>> =>
      request('/api/inference/anomaly', {
        method: 'POST',
        body: JSON.stringify({ features }),
      }),
    batch: async (task: string, rows: Record<string, any>[]): Promise<BatchInferenceResponse<any>> =>
      request(`/api/inference/${encodeURIComponent(task)}/batch`, {
        method: 'POST',
        body: JSON.stringify({ rows }),
      }),
    getHistory: async (params?: { task?: string; model_id?: string; limit?: number }): Promise<{ predictions: PredictionHistoryRecord[]; total: number }> => {
      const search = new URLSearchParams();
      if (params?.task) search.set('task', params.task);
      if (params?.model_id) search.set('model_id', params.model_id);
      if (params?.limit) search.set('limit', String(params.limit));
      const qs = search.toString();
      return request(`/api/inference/history${qs ? `?${qs}` : ''}`);
    },
    history: async (params?: { task?: string; model_id?: string; limit?: number }): Promise<{ predictions: PredictionHistoryRecord[]; total: number; records: PredictionHistoryRecord[] }> => {
      const search = new URLSearchParams();
      if (params?.task) search.set('task', params.task);
      if (params?.model_id) search.set('model_id', params.model_id);
      if (params?.limit) search.set('limit', String(params.limit));
      const qs = search.toString();
      const res = await request<{ predictions: PredictionHistoryRecord[]; total: number }>(`/api/inference/history${qs ? `?${qs}` : ''}`);
      return { ...res, records: res.predictions };
    },
    getHistoryDetail: async (id: string): Promise<PredictionHistoryRecord> =>
      request(`/api/inference/history/${encodeURIComponent(id)}`),
  },
  explainability: {
    getModelGlobal: async (id: string): Promise<GlobalExplanationResponse> =>
      request(`/api/models/${encodeURIComponent(id)}/explain`),
    getBenchmarkGlobal: async (id: string): Promise<GlobalExplanationResponse> =>
      request(`/api/models/benchmark/${encodeURIComponent(id)}/explain`),
    getBenchmark: async (id: string): Promise<GlobalExplanationResponse> =>
      request(`/api/models/benchmark/${encodeURIComponent(id)}/explain`),
    getPredictionLocal: async (id: string): Promise<LocalExplanationResponse> =>
      request(`/api/inference/history/${encodeURIComponent(id)}/explain`),
    explainLocal: async (modelId: string, features: Record<string, any>): Promise<LocalExplanationResponse> =>
      request(`/api/models/${encodeURIComponent(modelId)}/explain/local`, {
        method: 'POST',
        body: JSON.stringify({ features }),
      }),
  },
  comparison: {
    compare: async (modelIds: string[], task?: string): Promise<ModelComparisonResponse> =>
      request('/api/models/compare', {
        method: 'POST',
        body: JSON.stringify({ model_ids: modelIds, task }),
      }),
    getEvaluationHistory: async (modelName: string): Promise<ModelEvaluationHistoryResponse> =>
      request(`/api/models/${encodeURIComponent(modelName)}/evaluation-history`),
  },
  decisionSupport: {
    generate: async (payload: { machine_id?: string; prediction_id?: string; symptoms?: string }): Promise<DecisionSupportResponse> =>
      request('/api/decision-support', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
  },
};

