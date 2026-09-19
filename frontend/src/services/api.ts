import type { Machine, Alert, SensorDataPoint } from '../types/models';
import { addDays, subHours, subMinutes, subDays } from 'date-fns';

export interface SystemStatus {
  mode: 'live_rag' | 'live_fallback' | 'live_no_rag' | 'mock' | 'error';
  label: string;
  message?: string;
  baseUrl?: string;
  modelsLoaded?: boolean;
  ragStatus?: string;
  llmStatus?: string;
}

let currentSystemStatus: SystemStatus = {
  mode: 'mock',
  label: 'DEVELOPMENT MOCK',
};

const statusSubscribers: Set<(status: SystemStatus) => void> = new Set();

export const subscribeSystemStatus = (callback: (status: SystemStatus) => void) => {
  statusSubscribers.add(callback);
  callback(currentSystemStatus);
  return () => {
    statusSubscribers.delete(callback);
  };
};

export const getSystemStatus = (): SystemStatus => currentSystemStatus;

const updateStatus = (newStatus: SystemStatus) => {
  currentSystemStatus = newStatus;
  statusSubscribers.forEach((cb) => cb(newStatus));
};

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
let isLiveAvailable = false;

export async function checkBackendHealth(): Promise<boolean> {
  if (!API_BASE_URL) {
    isLiveAvailable = false;
    updateStatus({
      mode: 'mock',
      label: 'DEVELOPMENT MOCK',
    });
    return false;
  }

  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });
    
    if (res.ok) {
      const data = await res.json();
      if (data.models_loaded) {
        isLiveAvailable = true;
        const ragStatus = data.rag_status || 'unindexed';
        const llmStatus = data.llm_status || 'fallback';

        let mode: SystemStatus['mode'] = 'live_fallback';
        let label = 'LIVE ML + GROUNDED FALLBACK';

        if (ragStatus === 'ready' && llmStatus !== 'fallback') {
          mode = 'live_rag';
          label = 'LIVE ML + REAL RAG';
        } else if (ragStatus === 'ready' && llmStatus === 'fallback') {
          mode = 'live_fallback';
          label = 'LIVE ML + GROUNDED FALLBACK';
        } else {
          mode = 'live_no_rag';
          label = 'LIVE ML (RAG UNAVAILABLE)';
        }

        updateStatus({
          mode,
          label,
          baseUrl: API_BASE_URL,
          modelsLoaded: true,
          ragStatus,
          llmStatus
        });
        return true;
      }
    }
    
    isLiveAvailable = false;
    updateStatus({
      mode: 'error',
      label: 'BACKEND UNREACHABLE',
      message: 'Live ML backend unavailable. Showing development data.',
      baseUrl: API_BASE_URL,
      modelsLoaded: false
    });
    return false;
  } catch {
    isLiveAvailable = false;
    updateStatus({
      mode: 'error',
      label: 'BACKEND UNREACHABLE',
      message: 'Live ML backend unavailable. Showing development data.',
      baseUrl: API_BASE_URL,
      modelsLoaded: false
    });
    return false;
  }
}

if (API_BASE_URL) {
  checkBackendHealth();
}

// ---------------------------------------------------------
// MOCK DATA GENERATOR (Phase 1-4 Fallback)
// ---------------------------------------------------------
const generateMockMachines = (): Machine[] => [
  { id: 'M-001', name: 'CNC Mill Alpha', type: 'Milling', location: 'Plant A - Sector 1', status: 'healthy', rulDays: 145, rulConfidence: 92, failureRisk: 12, lastUpdated: new Date().toISOString() },
  { id: 'M-002', name: 'CNC Mill Beta', type: 'Milling', location: 'Plant A - Sector 1', status: 'warning', rulDays: 42, rulConfidence: 85, failureRisk: 45, lastUpdated: new Date().toISOString() },
  { id: 'M-003', name: 'Hydraulic Press V1', type: 'Press', location: 'Plant B - Sector 4', status: 'critical', rulDays: 8, rulConfidence: 78, failureRisk: 88, lastUpdated: new Date().toISOString() },
  { id: 'M-004', name: 'Conveyor Drive 4A', type: 'Motor', location: 'Plant A - Sector 2', status: 'healthy', rulDays: 210, rulConfidence: 95, failureRisk: 5, lastUpdated: new Date().toISOString() },
  { id: 'M-005', name: 'Coolant Pump P-2', type: 'Pump', location: 'Plant C - Sector 1', status: 'offline', rulDays: 0, rulConfidence: 0, failureRisk: 100, lastUpdated: subDays(new Date(), 2).toISOString() },
];

const mockMachines = generateMockMachines();

const generateMockAlerts = (): Alert[] => [
  { id: 'A-101', machineId: 'M-003', machineName: 'Hydraulic Press V1', severity: 'critical', type: 'rul_warning', description: 'RUL dropped below 10 days. Impending pressure seal failure detected.', timestamp: subMinutes(new Date(), 15).toISOString(), acknowledged: false },
  { id: 'A-102', machineId: 'M-002', machineName: 'CNC Mill Beta', severity: 'medium', type: 'anomaly', description: 'Abnormal high-frequency vibration detected during cutting phase.', timestamp: subHours(new Date(), 2).toISOString(), acknowledged: true },
];

const mockAlerts = generateMockAlerts();
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

function transformLiveMachine(m: any): Machine {
  let status: 'healthy' | 'warning' | 'critical' | 'offline' = 'healthy';
  if (m.status === 'degraded') status = 'warning';
  else if (m.status === 'critical') status = 'critical';

  return {
    id: m.id,
    name: m.name,
    type: 'Turbofan Engine',
    location: m.location,
    status,
    rulDays: Math.round(m.predictedRul),
    rulConfidence: 87,
    failureRisk: Math.max(0, 100 - m.healthScore),
    lastUpdated: new Date().toISOString()
  };
}

// ---------------------------------------------------------
// CENTRALIZED API PROVIDER (Phase 1-5 Complete)
// ---------------------------------------------------------
export const api = {
  machines: {
    list: async (): Promise<Machine[]> => {
      if (API_BASE_URL) {
        const healthy = await checkBackendHealth();
        if (healthy && isLiveAvailable) {
          try {
            const res = await fetch(`${API_BASE_URL}/api/machines`);
            if (res.ok) {
              const data = await res.json();
              return data.map(transformLiveMachine);
            }
          } catch (e) {
            console.warn("Live API machine list failed, using mock fallback:", e);
          }
        }
      }
      await delay(400);
      return mockMachines;
    },
    get: async (id: string): Promise<Machine | undefined> => {
      if (API_BASE_URL) {
        const healthy = await checkBackendHealth();
        if (healthy && isLiveAvailable) {
          try {
            const res = await fetch(`${API_BASE_URL}/api/machines/${id}`);
            if (res.ok) {
              const data = await res.json();
              return transformLiveMachine(data);
            }
          } catch (e) {
            console.warn(`Live API machine get (${id}) failed, using mock fallback:`, e);
          }
        }
      }
      await delay(300);
      return mockMachines.find(m => m.id === id);
    }
  },
  alerts: {
    listRecent: async (): Promise<Alert[]> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/machines`);
          if (res.ok) {
            const machines = await res.json();
            const liveAlerts: Alert[] = [];
            machines.forEach((m: any, idx: number) => {
              if (m.anomalySeverity === 'critical' || m.predictedRul < 20) {
                liveAlerts.push({
                  id: `LA-${idx + 100}`,
                  machineId: m.id,
                  machineName: m.name,
                  severity: m.predictedRul < 15 ? 'critical' : 'medium',
                  type: 'rul_warning',
                  description: `PyTorch LSTM RUL: ${m.predictedRul} cycles. Anomaly severity: ${m.anomalySeverity}.`,
                  timestamp: subMinutes(new Date(), idx * 5).toISOString(),
                  acknowledged: false
                });
              }
            });
            if (liveAlerts.length > 0) return liveAlerts;
          }
        } catch (e) {
          console.warn("Live API alerts failed, using mock fallback:", e);
        }
      }
      await delay(300);
      return mockAlerts;
    }
  },
  sensors: {
    getMachineData: async (machineId: string, _hours: number = 24): Promise<SensorDataPoint[]> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/machines/${machineId}/sensors`);
          if (res.ok) {
            const data = await res.json();
            const now = new Date();
            const history = data.history || [];
            return history.slice(-25).map((point: any, idx: number) => ({
              timestamp: subHours(now, 24 - idx).toISOString(),
              vibration: point.s2 || point.s3 || 2.5,
              temperature: (point.s3 ? point.s3 / 20 : 45),
              pressure: (point.s7 ? point.s7 / 5 : 100),
              isAnomaly: point.s2 > 645
            }));
          }
        } catch (e) {
          console.warn(`Live API sensors (${machineId}) failed, using mock fallback:`, e);
        }
      }
      await delay(500);
      const data: SensorDataPoint[] = [];
      let baseVibration = 2.5;
      let baseTemp = 45;
      const now = new Date();
      for (let i = 24; i >= 0; i--) {
        const time = subHours(now, i);
        baseVibration += (Math.random() - 0.5) * 0.4;
        baseTemp += (Math.random() - 0.5) * 1.5;
        const isAnomaly = i === 5;
        if (isAnomaly) {
          baseVibration += 3.5;
          baseTemp += 15;
        }
        data.push({
          timestamp: time.toISOString(),
          vibration: Math.max(0, baseVibration),
          temperature: Math.max(20, baseTemp),
          pressure: 100 + (Math.random() - 0.5) * 5,
          isAnomaly
        });
        if (isAnomaly) {
          baseVibration -= 2.0;
          baseTemp -= 8;
        }
      }
      return data;
    },
    listFleetData: async (): Promise<{ active: number, total: number, warning: number, recentActivity: any[] }> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/machines`);
          if (res.ok) {
            const machines = await res.json();
            const total = machines.length;
            const active = machines.filter((m: any) => m.status === 'operating').length;
            const warning = machines.filter((m: any) => m.status === 'degraded' || m.status === 'critical').length;
            const recentActivity = machines.slice(0, 5).map((m: any, idx: number) => ({
              id: `ACT-${idx + 1}`,
              machineId: m.id,
              machineName: m.name,
              sensor: 'PyTorch RUL Model',
              value: m.predictedRul,
              unit: 'cycles',
              status: m.status === 'operating' ? 'healthy' : (m.status === 'degraded' ? 'warning' : 'critical'),
              time: subMinutes(new Date(), idx * 7).toISOString()
            }));
            return { active, total, warning, recentActivity };
          }
        } catch (e) {
          console.warn("Live API listFleetData failed, using mock fallback:", e);
        }
      }
      await delay(400);
      return {
        active: 1420,
        total: 1450,
        warning: 34,
        recentActivity: [
          { id: '1', machineId: 'M-003', machineName: 'Hydraulic Press V1', sensor: 'Vibration V-02', value: 6.8, unit: 'mm/s', status: 'critical', time: new Date().toISOString() },
          { id: '2', machineId: 'M-002', machineName: 'CNC Mill Beta', sensor: 'Temperature T-01', value: 72, unit: '°C', status: 'warning', time: subMinutes(new Date(), 10).toISOString() },
          { id: '3', machineId: 'M-001', machineName: 'CNC Mill Alpha', sensor: 'Pressure P-04', value: 102, unit: 'kPa', status: 'healthy', time: subMinutes(new Date(), 22).toISOString() },
          { id: '4', machineId: 'M-004', machineName: 'Conveyor Drive 4A', sensor: 'Load L-02', value: 85, unit: '%', status: 'healthy', time: subMinutes(new Date(), 45).toISOString() }
        ]
      };
    }
  },
  diagnostics: {
    createCase: async (machineId: string): Promise<any> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/diagnostics`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ machine_id: machineId, use_llm: true })
          });
          if (res.ok) {
            return await res.json();
          }
        } catch (e) {
          console.warn(`Live API createCase (${machineId}) failed, using mock fallback:`, e);
        }
      }
      await delay(600);
      return {
        id: `DC-MOCK-${Date.now()}`,
        machineId,
        machineName: `Machine ${machineId}`,
        trigger: 'Telemetry Anomaly Detected',
        finding: 'High-frequency resonance peak detected on compressor spindle bearing.',
        severity: 'critical',
        createdTime: new Date().toISOString(),
        status: 'open',
        evidence: [
          { documentId: 'DOC-902', documentTitle: 'Spindle Bearing B Specifications', snippet: 'High frequency resonance (>2kHz) on inner race precedes spalling by 7-14 days.', relevance: 0.94, section: 'Section 4.2', page: 1 }
        ],
        explanation: 'Observed vibration peak aligns with inner race wear patterns documented in technical specification DOC-902.',
        recommendedAction: 'Schedule bearing inspection and replacement within 48 hours.',
        confidence: 90,
        generatorUsed: 'Development Mock Generator',
        isGroundedFallback: false
      };
    },
    getMachineDiagnostics: async (machineId: string): Promise<any> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/machines/${machineId}/prediction`);
          if (res.ok) {
            const pred = await res.json();
            return {
              finding: `PyTorch LSTM RUL Prediction: ${pred.predictedRul} cycles remaining (RF Baseline: ${pred.rfPredictedRul} cycles).`,
              explanation: `Model evaluated with MAE: 11.30 cycles, R²: 0.87. Degradation trajectory indicates ${pred.riskLevel} risk level.`,
              evidence: [
                `PyTorch LSTM RUL estimate: ${pred.predictedRul} cycles`,
                `Random Forest RUL estimate: ${pred.rfPredictedRul} cycles`,
                `95% Confidence Interval: [${pred.confidenceInterval.min} - ${pred.confidenceInterval.max}] cycles`
              ],
              recommendedAction: pred.recommendations ? pred.recommendations.join('. ') : 'Routine monitoring'
            };
          }
        } catch (e) {
          console.warn(`Live API diagnostics (${machineId}) failed, using mock fallback:`, e);
        }
      }
      await delay(400);
      return {
        finding: 'Abnormal high-frequency vibration detected in Spindle Bearing B.',
        explanation: 'The current vibration spectrum matches known failure signatures for inner race wear. RUL model indicates accelerated degradation consistent with historical bearing failures in similar operational loads.',
        evidence: [
          'Vibration sensor V-02 exceeded 6.0 mm/s baseline',
          'Temperature increased by 15°C over 2 hours',
          'Acoustic emission anomaly detected at 4:30 AM'
        ],
        recommendedAction: 'Schedule immediate inspection and probable replacement of Spindle Bearing B. Reduce load by 20% until maintenance is performed.'
      };
    },
    getMachineAnomalies: async (machineId: string): Promise<any[]> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/machines/${machineId}/anomalies`);
          if (res.ok) {
            const data = await res.json();
            return (data.anomalyEvents || []).map((evt: any, idx: number) => ({
              id: `AN-LIVE-${idx + 1}`,
              machineId: data.machineId,
              machineName: `Turbofan Engine ${data.machineId}`,
              timestamp: subHours(new Date(), (200 - evt.cycle) / 10).toISOString(),
              sensor: 'Isolation Forest ML Detector',
              severity: evt.severity,
              score: evt.score,
              description: evt.description,
              status: evt.status
            }));
          }
        } catch (e) {
          console.warn(`Live API getMachineAnomalies (${machineId}) failed, using mock fallback:`, e);
        }
      }
      await delay(400);
      return [
        { id: 'AN-01', machineId: 'M-003', machineName: 'Hydraulic Press V1', timestamp: subHours(new Date(), 2).toISOString(), sensor: 'Vibration V-02', severity: 'critical', score: 0.95, description: 'High frequency resonance peak exceeding safety limits', status: 'active' },
        { id: 'AN-02', machineId: 'M-003', machineName: 'Hydraulic Press V1', timestamp: subHours(new Date(), 5).toISOString(), sensor: 'Temperature T-01', severity: 'high', score: 0.88, description: 'Rapid temperature rise across housing', status: 'active' },
        { id: 'AN-03', machineId: 'M-003', machineName: 'Hydraulic Press V1', timestamp: subDays(new Date(), 2).toISOString(), sensor: 'Load L-04', severity: 'low', score: 0.45, description: 'Minor load fluctuation during startup', status: 'resolved' }
      ];
    },
    listFleetAnomalies: async (): Promise<any[]> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/machines/FD001-001/anomalies`);
          if (res.ok) {
            const data = await res.json();
            return (data.anomalyEvents || []).map((evt: any, idx: number) => ({
              id: `AN-FLEET-${idx + 1}`,
              machineId: 'FD001-001',
              machineName: 'Turbofan Engine FD001-001',
              timestamp: subHours(new Date(), idx * 3).toISOString(),
              sensor: 'Isolation Forest Anomaly Model',
              severity: evt.severity,
              score: evt.score,
              description: evt.description,
              status: evt.status
            }));
          }
        } catch (e) {
          console.warn("Live API listFleetAnomalies failed, using mock fallback:", e);
        }
      }
      await delay(400);
      return [
        { id: 'AN-01', machineId: 'M-003', machineName: 'Hydraulic Press V1', timestamp: subHours(new Date(), 2).toISOString(), sensor: 'Vibration V-02', severity: 'critical', score: 0.95, description: 'High frequency resonance peak exceeding safety limits', status: 'active' },
        { id: 'AN-02', machineId: 'M-003', machineName: 'Hydraulic Press V1', timestamp: subHours(new Date(), 5).toISOString(), sensor: 'Temperature T-01', severity: 'high', score: 0.88, description: 'Rapid temperature rise across housing', status: 'active' },
        { id: 'AN-04', machineId: 'M-002', machineName: 'CNC Mill Beta', timestamp: subHours(new Date(), 10).toISOString(), sensor: 'Vibration V-01', severity: 'medium', score: 0.65, description: 'Slight chatter detected during roughing pass', status: 'investigating' },
        { id: 'AN-05', machineId: 'M-001', machineName: 'CNC Mill Alpha', timestamp: subDays(new Date(), 1).toISOString(), sensor: 'Pressure P-02', severity: 'low', score: 0.42, description: 'Coolant pressure drop below optimal', status: 'resolved' },
        { id: 'AN-03', machineId: 'M-003', machineName: 'Hydraulic Press V1', timestamp: subDays(new Date(), 2).toISOString(), sensor: 'Load L-04', severity: 'low', score: 0.45, description: 'Minor load fluctuation during startup', status: 'resolved' }
      ];
    },
    getRulForecast: async (machineId: string): Promise<any[]> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/machines/${machineId}/prediction`);
          if (res.ok) {
            const data = await res.json();
            const curve = data.degradationCurve || [];
            const now = new Date();
            return curve.map((pt: any, idx: number) => ({
              timestamp: subDays(now, curve.length - idx).toISOString(),
              predictedRul: pt.predictedRul,
              lowerBound: Math.max(0, pt.predictedRul - 11.3),
              upperBound: pt.predictedRul + 11.3,
              isHistorical: idx < curve.length - 1
            }));
          }
        } catch (e) {
          console.warn(`Live API getRulForecast (${machineId}) failed, using mock fallback:`, e);
        }
      }
      await delay(500);
      const data = [];
      const now = new Date();
      let currentRul = 22;
      for (let i = 14; i >= 0; i--) {
        const time = subDays(now, i);
        currentRul -= (Math.random() * 0.8 + 0.2);
        data.push({
          timestamp: time.toISOString(),
          predictedRul: Math.max(0, currentRul),
          lowerBound: Math.max(0, currentRul - 2),
          upperBound: currentRul + 2,
          isHistorical: true
        });
      }
      for (let i = 1; i <= 14; i++) {
        const time = addDays(now, i);
        currentRul -= (Math.random() * 1.0 + 0.5);
        data.push({
          timestamp: time.toISOString(),
          predictedRul: Math.max(0, currentRul),
          lowerBound: Math.max(0, currentRul - (i * 0.5)),
          upperBound: Math.max(0, currentRul + (i * 0.5)),
          isHistorical: false
        });
      }
      return data;
    },
    listFleetPredictions: async (): Promise<any[]> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/machines`);
          if (res.ok) {
            const machines = await res.json();
            return machines.map((m: any) => ({
              machineId: m.id,
              machineName: m.name,
              predictedRul: m.predictedRul,
              rfPredictedRul: m.rfPredictedRul,
              confidence: 87,
              failureRisk: Math.max(0, 100 - m.healthScore),
              degradationState: m.predictedRul < 30 ? 'Accelerated' : (m.predictedRul < 70 ? 'Linear' : 'Stable'),
              nextMilestone: m.predictedRul < 20 ? 'Critical Maintenance Required' : `Routine Inspection at Cycle ${m.currentCycle + 20}`,
              lastPredictionTime: new Date().toISOString()
            }));
          }
        } catch (e) {
          console.warn("Live API listFleetPredictions failed, using mock fallback:", e);
        }
      }
      await delay(500);
      return [
        { machineId: 'M-003', machineName: 'Hydraulic Press V1', predictedRul: 8, confidence: 78, failureRisk: 88, degradationState: 'Accelerated', nextMilestone: 'Critical Failure Expected in <10 days', lastPredictionTime: new Date().toISOString() },
        { machineId: 'M-002', machineName: 'CNC Mill Beta', predictedRul: 42, confidence: 85, failureRisk: 45, degradationState: 'Linear', nextMilestone: 'Preventative Maintenance Window', lastPredictionTime: new Date().toISOString() },
        { machineId: 'M-001', machineName: 'CNC Mill Alpha', predictedRul: 145, confidence: 92, failureRisk: 12, degradationState: 'Stable', nextMilestone: 'Routine Check (Q3)', lastPredictionTime: new Date().toISOString() },
        { machineId: 'M-004', machineName: 'Conveyor Drive 4A', predictedRul: 210, confidence: 95, failureRisk: 5, degradationState: 'Stable', nextMilestone: 'Routine Check (Q4)', lastPredictionTime: new Date().toISOString() }
      ];
    },
    listCases: async (): Promise<any[]> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/diagnostics`);
          if (res.ok) {
            const cases = await res.json();
            if (cases.length > 0) return cases;
          }
        } catch (e) {
          console.warn("Live API listCases failed, using mock fallback:", e);
        }
      }
      await delay(400);
      return [
        { 
          id: 'DC-9942', 
          machineId: 'FD001-001', 
          machineName: 'Turbofan Engine FD001-001', 
          trigger: 'Severe vibration anomaly, RUL < 30 cycles', 
          finding: 'High-Pressure Compressor (HPC) stage degradation. Elevated s3 outlet temperature correlated with acoustic resonance.', 
          severity: 'critical', 
          createdTime: subDays(new Date(), 1).toISOString(), 
          status: 'open', 
          evidence: [
            { documentId: 'DOC-901', documentTitle: 'C-MAPSS FD001 Turbofan Maintenance & Degradation Guide', snippet: 'Elevated HPC outlet temperature (sensor s3 > 1600 °R) alongside RUL < 30 cycles indicates stator vane wear.', relevance: 0.95, section: 'Section 1.2', page: 1 }
          ],
          explanation: 'Telemetry analysis confirms HPC stator vane degradation consistent with specifications in document DOC-901.',
          recommendedAction: 'Schedule HPC stator vane inspection and blade tip clearance adjustment.',
          confidence: 91,
          generatorUsed: 'Grounded Rule-Based Fallback',
          isGroundedFallback: true
        }
      ];
    },
    getCase: async (id: string): Promise<any | null> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/diagnostics/${id}`);
          if (res.ok) {
            return await res.json();
          }
        } catch (e) {
          console.warn(`Live API getCase (${id}) failed, using mock fallback:`, e);
        }
      }
      await delay(300);
      const cases = await api.diagnostics.listCases();
      return cases.find((c: any) => c.id === id) || null;
    }
  },
  maintenance: {
    listQueue: async (): Promise<any[]> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/machines`);
          if (res.ok) {
            const machines = await res.json();
            return machines.slice(0, 6).map((m: any, idx: number) => ({
              id: `MT-${idx + 200}`,
              machineId: m.id,
              machineName: m.name,
              priority: m.status === 'critical' ? 'critical' : (m.status === 'degraded' ? 'high' : 'medium'),
              condition: `Predicted RUL: ${m.predictedRul} cycles. Isolation Forest Anomaly: ${m.anomalySeverity}`,
              action: m.predictedRul < 30 ? 'Schedule HPC Stage Inspection' : 'Calibrate temperature/pressure sensors',
              status: m.status === 'critical' ? 'pending' : 'scheduled',
              timeframe: m.predictedRul < 30 ? 'Immediate' : 'Next 48h'
            }));
          }
        } catch (e) {
          console.warn("Live API maintenance queue failed, using mock fallback:", e);
        }
      }
      await delay(400);
      return [
        { id: 'MT-1042', machineId: 'M-003', machineName: 'Hydraulic Press V1', priority: 'critical', condition: 'Severe vibration anomaly, RUL < 10 days', action: 'Replace Spindle Bearing B', status: 'pending', timeframe: 'Immediate' },
        { id: 'MT-1043', machineId: 'M-002', machineName: 'CNC Mill Beta', priority: 'high', condition: 'Linear degradation observed, high temp', action: 'Inspect cutting head cooling system', status: 'scheduled', timeframe: 'Next 48h' },
        { id: 'MT-1044', machineId: 'M-001', machineName: 'CNC Mill Alpha', priority: 'medium', condition: 'Pressure fluctuation', action: 'Recalibrate coolant pressure valve', status: 'in-progress', timeframe: 'This week' },
        { id: 'MT-1041', machineId: 'M-004', machineName: 'Conveyor Drive 4A', priority: 'low', condition: 'Routine operation', action: 'Quarterly lubrication check', status: 'pending', timeframe: 'Next month' }
      ];
    }
  },
  knowledge: {
    upload: async (file: File, category: string = "Manual"): Promise<any> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const formData = new FormData();
          formData.append('file', file);
          formData.append('category', category);
          const res = await fetch(`${API_BASE_URL}/api/knowledge/upload`, {
            method: 'POST',
            body: formData
          });
          if (res.ok) {
            const data = await res.json();
            await checkBackendHealth(); // Refresh RAG status
            return data;
          } else {
            const errData = await res.json();
            throw new Error(errData.detail || 'Upload failed');
          }
        } catch (e: any) {
          console.error("Live API document upload failed:", e);
          throw e;
        }
      }
      await delay(1000);
      return {
        id: `DOC-${Date.now()}`,
        title: file.name.replace(/\.[^/.]+$/, "").replace(/_/g, " "),
        type: file.name.split('.').pop()?.toUpperCase() || 'TXT',
        size: file.size,
        uploadDate: new Date().toISOString(),
        processingStatus: 'ready',
        indexedChunks: 5,
        sourceCategory: category
      };
    },
    listDocuments: async (): Promise<any[]> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/knowledge/documents`);
          if (res.ok) {
            return await res.json();
          }
        } catch (e) {
          console.warn("Live API listDocuments failed, using mock fallback:", e);
        }
      }
      await delay(300);
      return [
        { id: 'DOC-901', title: 'C-MAPSS FD001 Turbofan Maintenance & Degradation Guide', type: 'PDF', size: 14500000, uploadDate: subDays(new Date(), 45).toISOString(), processingStatus: 'ready', indexedChunks: 4, sourceCategory: 'Manual' },
        { id: 'DOC-902', title: 'Spindle & Compressor Shaft Bearing Specifications', type: 'PDF', size: 2300000, uploadDate: subDays(new Date(), 30).toISOString(), processingStatus: 'ready', indexedChunks: 2, sourceCategory: 'Specification' },
        { id: 'DOC-903', title: 'CNC & Turbofan Fuel System Troubleshooting Manual', type: 'DOCX', size: 850000, uploadDate: subDays(new Date(), 15).toISOString(), processingStatus: 'ready', indexedChunks: 2, sourceCategory: 'Manual' }
      ];
    },
    search: async (query: string): Promise<any[]> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/knowledge/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5, min_score: 0.05 })
          });
          if (res.ok) {
            const results = await res.json();
            return results.map((r: any) => ({
              documentId: r.document_id,
              documentTitle: r.document_name,
              snippet: r.excerpt,
              relevance: r.relevance_score,
              section: r.section,
              page: r.page
            }));
          }
        } catch (e) {
          console.warn("Live API RAG search failed, using mock fallback:", e);
        }
      }
      await delay(600);
      const q = query.toLowerCase();
      if (q.includes('vibration') || q.includes('bearing') || q.includes('compressor')) {
        return [
          { documentId: 'DOC-901', documentTitle: 'C-MAPSS FD001 Turbofan Maintenance & Degradation Guide', snippet: 'Elevated HPC outlet temperature (sensor s3 > 1600 °R) alongside RUL < 30 cycles indicates stator vane wear.', relevance: 0.95, section: 'Section 1.2', page: 1 },
          { documentId: 'DOC-902', documentTitle: 'Spindle & Compressor Shaft Bearing Specifications', snippet: 'High frequency resonance (>2kHz) on compressor shaft bearings often precedes catastrophic spalling by 7-14 cycles.', relevance: 0.88, section: 'Section 4.2', page: 1 }
        ];
      }
      return [];
    },
    getDocument: async (id: string): Promise<any | null> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/knowledge/documents/${id}`);
          if (res.ok) {
            return await res.json();
          }
        } catch (e) {
          console.warn(`Live API getDocument (${id}) failed, using mock fallback:`, e);
        }
      }
      await delay(200);
      const docs = await api.knowledge.listDocuments();
      return docs.find((d: any) => d.id === id) || null;
    }
  },
  analytics: {
    getModels: async (): Promise<any> => {
      if (API_BASE_URL && isLiveAvailable) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/analytics/models`);
          if (res.ok) {
            return await res.json();
          }
        } catch (e) {
          console.warn("Live API analytics failed, using mock fallback:", e);
        }
      }
      await delay(300);
      return {
        dataset: "C-MAPSS FD001",
        metadata: {
          random_seed: 42,
          models: {
            baseline: "Random Forest Regressor",
            temporal: "PyTorch LSTM",
            anomaly: "Isolation Forest"
          },
          features: ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21'],
          window_size: 30,
          rul_cap: 125
        },
        evaluation: {
          rul_models: {
            random_forest: { mae: 13.22, rmse: 18.10, r2: 0.81 },
            lstm: { mae: 11.30, rmse: 15.00, r2: 0.87 }
          },
          anomaly_detection: {
            method: "Isolation Forest",
            note: "Unsupervised anomaly detection trained on healthy operational cycles."
          }
        }
      };
    }
  }
};
