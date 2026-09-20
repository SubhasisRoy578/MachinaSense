import { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { api } from '../../services/api';
import type { DiagnosticCase } from '../../types/models';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Stethoscope, AlertOctagon, FileCheck, Target, ArrowRight, BrainCircuit, ShieldAlert, Plus } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

export function DiagnosticsPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const machineFilter = searchParams.get('machine');
  
  const [cases, setCases] = useState<DiagnosticCase[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [createMachineId, setCreateMachineId] = useState('');
  const [error, setError] = useState('');

  const fetchCases = async () => {
    setLoading(true);
    try {
      setError('');
      const data = await api.diagnostics.listCases();
      if (machineFilter) {
        setCases(data.filter(c => c.machineId === machineFilter));
      } else {
        setCases(data);
      }
    } catch (error) {
      console.error("Failed to fetch diagnostics", error);
      setError(error instanceof Error ? error.message : 'Diagnostics are unavailable.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCases();
  }, [machineFilter]);

  const handleCreateDiagnostic = async () => {
    if (!createMachineId.trim()) return;
    setCreating(true);
    try {
      await api.diagnostics.createCase(createMachineId.trim());
      await fetchCases();
      setCreateMachineId('');
    } catch (error) {
      console.error("Failed to create diagnostic case", error);
      setError(error instanceof Error ? error.message : 'Unable to create a diagnostic.');
    } finally {
      setCreating(false);
    }
  };

  const openCount = cases.filter(c => c.status === 'open' || c.status === 'investigating').length;
  const criticalCount = cases.filter(c => c.severity === 'critical' && (c.status === 'open' || c.status === 'investigating')).length;
  const evidenceBacked = cases.filter(c => c.evidence && c.evidence.length > 0).length;
  const uniqueMachines = new Set(cases.map(c => c.machineId)).size;

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500 pb-12">
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-primary flex items-center gap-2">
            <Stethoscope className="w-6 h-6 text-accent" /> AI Diagnostics
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Engineering investigations generated from telemetry, anomaly detection, RUL prediction, and failure-risk signals.
          </p>
        </div>
        {/* Quick create diagnostic */}
        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Your machine ID"
            value={createMachineId}
            onChange={(e) => setCreateMachineId(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-md px-3 py-2 text-sm text-primary focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent w-48"
          />
          <button
            onClick={handleCreateDiagnostic}
            disabled={creating || !createMachineId.trim()}
            className="inline-flex items-center justify-center px-4 py-2 bg-accent/10 border border-accent/30 text-accent rounded-md hover:bg-accent/20 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {creating ? (
              <><BrainCircuit className="w-4 h-4 mr-2 animate-spin" /> Running...</>
            ) : (
              <><Plus className="w-4 h-4 mr-1" /> Run Diagnostic</>
            )}
          </button>
        </div>
      </div>

      {machineFilter && (
        <div className="bg-accent/10 border border-accent/20 px-4 py-2 rounded text-sm text-accent flex items-center justify-between">
          <span>Filtering investigations for machine: <strong>{machineFilter}</strong></span>
          <button onClick={() => navigate('/diagnostics')} className="text-xs underline hover:text-accent/80">Clear Filter</button>
        </div>
      )}
      {error && <div className="rounded border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">{error}</div>}

      {/* KPI Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <Card className="border-l-4 border-l-amber-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2">Open Investigations</div>
            <div className="text-3xl font-mono text-amber-500">{openCount}</div>
          </CardContent>
        </Card>
        <Card className={criticalCount > 0 ? "border-l-4 border-l-red-500 bg-red-950/10" : "border-l-4 border-l-slate-500"}>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <AlertOctagon className={criticalCount > 0 ? "text-red-500 w-4 h-4" : "w-4 h-4"} /> Critical Cases
            </div>
            <div className={`text-3xl font-mono ${criticalCount > 0 ? "text-red-500" : "text-primary"}`}>{criticalCount}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-emerald-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <FileCheck className="w-4 h-4 text-emerald-500" /> Evidence-Backed
            </div>
            <div className="text-3xl font-mono text-emerald-400">{evidenceBacked}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-accent">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <Target className="w-4 h-4 text-accent" /> Machines Under Review
            </div>
            <div className="text-3xl font-mono text-accent">{uniqueMachines}</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Diagnostic History</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-slate-400 uppercase bg-slate-950/50 border-b border-slate-800">
                <tr>
                  <th className="px-6 py-4 font-medium tracking-wider">Severity</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Machine</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Investigation Trigger & ML Finding</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Generator</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Status & Evidence</th>
                  <th className="px-6 py-4 font-medium tracking-wider text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {loading ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-slate-500 font-mono">
                      [LOADING_DIAGNOSTICS...]
                    </td>
                  </tr>
                ) : cases.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-slate-500 italic">
                      No diagnostic investigations yet. Select one of your machines and run an evidence-bound diagnostic.
                    </td>
                  </tr>
                ) : (
                  cases.map((diag) => (
                    <tr key={diag.id} className="hover:bg-slate-800/20 transition-colors group">
                      <td className="px-6 py-4">
                        <Badge variant={
                          diag.severity === 'critical' ? 'critical' :
                          diag.severity === 'high' ? 'warning' :
                          diag.severity === 'medium' ? 'healthy' : 'outline'
                        }>
                          {diag.severity.toUpperCase()}
                        </Badge>
                      </td>
                      <td className="px-6 py-4">
                        <div 
                          className="font-medium text-primary hover:text-accent cursor-pointer transition-colors"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/machines/${diag.machineId}`);
                          }}
                        >
                          {diag.machineName}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5 font-mono">{diag.machineId}</div>
                      </td>
                      <td className="px-6 py-4 max-w-sm">
                        <div className="font-mono text-xs text-accent mb-1 border-b border-slate-800 pb-1">
                          Trigger: {diag.trigger}
                        </div>
                        <div className="text-slate-300 line-clamp-2" title={diag.finding}>
                          {diag.finding}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        {diag.isGroundedFallback ? (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-amber-950/60 text-amber-400 border border-amber-800/60">
                            <ShieldAlert className="w-3 h-3" />
                            Grounded Fallback
                          </span>
                        ) : diag.generatorUsed ? (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-emerald-950/60 text-emerald-400 border border-emerald-800/60">
                            <BrainCircuit className="w-3 h-3" />
                            {diag.generatorUsed.includes('Gemini') ? 'Gemini RAG' : 
                             diag.generatorUsed.includes('Grok') ? 'Grok' : 'Grounded analysis'}
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-slate-800/60 text-slate-400 border border-slate-700">
                            <BrainCircuit className="w-3 h-3" />
                            No provider metadata
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4">
                        <div className="capitalize font-medium text-slate-200 mb-1">{diag.status}</div>
                        <div className="text-xs text-slate-500 flex flex-col gap-1">
                          <span>{diag.evidence.length} Retrieved Snippets</span>
                          {diag.confidence && <span className="font-mono">Conf: {diag.confidence}%</span>}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="text-slate-500 text-xs mb-2 font-mono">{formatDistanceToNow(new Date(diag.createdTime), { addSuffix: true })}</div>
                        <button 
                          onClick={() => navigate(`/diagnostics/${diag.id}`)}
                          className="inline-flex items-center justify-center px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider text-accent border border-accent/30 hover:bg-accent/10 transition-colors"
                        >
                          Inspect Case <ArrowRight className="w-3 h-3 ml-1" />
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
