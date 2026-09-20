import type { Machine } from '../types/models';

const BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
let tokenGetter: (() => Promise<string | null>) | null = null;
export type SystemStatus = { mode: 'live_rag' | 'live_fallback' | 'live_no_rag' | 'error'; label: string; message?: string; modelsLoaded?: boolean; baseUrl?: string; llmStatus?: string };
export function configureApiAuth(getToken: () => Promise<string | null>) { tokenGetter = getToken; }
async function request(path: string, init: RequestInit = {}) {
  if (!BASE) throw new Error('VITE_API_BASE_URL is not configured.');
  const token = tokenGetter ? await tokenGetter() : null;
  if (!token) throw new Error('Authenticated session required.');
  const headers = new Headers(init.headers); headers.set('Authorization', `Bearer ${token}`);
  if (!(init.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  const response = await fetch(`${BASE}${path}`, { ...init, headers });
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}
function mapMachine(m: any): Machine { return { id:m.id, name:m.name, type:m.type || m.model, location:m.location, status:m.status === 'offline' ? 'offline' : m.status, rulDays:m.rulDays ?? 0, rulConfidence:m.rulConfidence ?? 0, failureRisk:m.failureRisk ?? 0, lastUpdated:m.lastUpdated }; }
export function subscribeSystemStatus(listener: (value: SystemStatus) => void) {
  if (!BASE) { listener({ mode: 'error', label: 'BACKEND NOT CONFIGURED', message: 'Set VITE_API_BASE_URL.' }); return () => {}; }
  fetch(`${BASE}/health`).then(r => r.ok ? listener({mode:'live_rag',label:'LIVE ML + RAG'}) : listener({mode:'error',label:'BACKEND ERROR'})).catch(() => listener({mode:'error',label:'BACKEND UNREACHABLE'}));
  return () => {};
}
export async function checkBackendHealth() { if (!BASE) throw new Error('VITE_API_BASE_URL is not configured.'); const r = await fetch(`${BASE}/health`); if (!r.ok) throw new Error('Backend unavailable'); return r.json(); }
export const api = {
  machines: {
    list: async (): Promise<Machine[]> => (await request('/api/machines')).map(mapMachine),
    get: async (id: string): Promise<any> => mapMachine(await request(`/api/machines/${encodeURIComponent(id)}`)),
    create: async (input: { machine_id:string; name:string; machine_type:string; location:string; description?:string }) => mapMachine(await request('/api/machines', {method:'POST',body:JSON.stringify(input)})),
    uploadTelemetry: async (id:string,file:File) => { const form=new FormData();form.append('file',file);return request(`/api/machines/${encodeURIComponent(id)}/telemetry`,{method:'POST',body:form}); },
  },
  sensors: {
    getMachineData: async (id:string, _hours?:number): Promise<any[]> => request(`/api/machines/${encodeURIComponent(id)}/sensors`),
    listFleetData: async () => { const ms:any[] = await request('/api/machines'); return {total:ms.length,active:ms.filter(m=>m.currentCycle>0).length,warning:ms.filter(m=>m.anomalySeverity !== 'none').length,recentActivity:[]}; },
  },
  diagnostics: {
    getMachineAnomalies: async (id:string): Promise<any[]> => (await request(`/api/machines/${encodeURIComponent(id)}/anomalies`)).anomalyEvents || [],
    listFleetAnomalies: async () => { const ms:any[] = await request('/api/machines'); const parts=await Promise.all(ms.map(m=>request(`/api/machines/${encodeURIComponent(m.id)}/anomalies`))); return parts.flatMap(p=>p.anomalyEvents || []); },
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
    getDocument: async () => null,
  },
  analytics: { getModels: async () => request('/api/analytics/models') },
  copilot: {
    conversations: () => request('/api/copilot/conversations'), create: (machine_id?:string) => request('/api/copilot/conversations',{method:'POST',body:JSON.stringify({machine_id})}),
    get: (id:string) => request(`/api/copilot/conversations/${id}`), send: (id:string,content:string) => request(`/api/copilot/conversations/${id}/messages`,{method:'POST',body:JSON.stringify({content})}),
  }
};
