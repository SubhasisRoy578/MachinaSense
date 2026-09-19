import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../../services/api';
import type { DiagnosticCase } from '../../types/models';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Stethoscope, AlertTriangle, Cpu, BrainCircuit, FileText, ArrowRight, CheckCircle2, ShieldAlert, Sparkles, ShieldCheck } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

export function DiagnosticCaseView() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [caseData, setCaseData] = useState<DiagnosticCase | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCase = async () => {
      setLoading(true);
      try {
        if (id) {
          const data = await api.diagnostics.getCase(id);
          setCaseData(data);
        }
      } catch (error) {
        console.error("Failed to fetch case details", error);
      } finally {
        setLoading(false);
      }
    };
    fetchCase();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500 font-mono animate-pulse">
        [LOADING_DIAGNOSTIC_CASE_DATA...]
      </div>
    );
  }

  if (!caseData) {
    return (
      <div className="flex flex-col items-center justify-center h-64 bg-slate-900/50 border border-slate-800 rounded-lg">
        <AlertTriangle className="w-8 h-8 text-amber-500 mb-4" />
        <h2 className="text-lg font-bold text-slate-200">Diagnostic case not found.</h2>
        <p className="text-sm text-slate-400 mt-2 text-center max-w-md">
          The requested investigation ID does not exist or has been archived.
        </p>
        <button 
          onClick={() => navigate('/diagnostics')}
          className="mt-6 px-4 py-2 bg-slate-800 text-slate-200 rounded hover:bg-slate-700 transition-colors text-sm"
        >
          Return to Diagnostics
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex flex-wrap items-center gap-3 mb-2">
            <h1 className="text-2xl font-bold tracking-tight text-primary flex items-center gap-2">
              <Stethoscope className="w-6 h-6 text-accent" /> Investigation: {caseData.id}
            </h1>
            <Badge variant={
              caseData.severity === 'critical' ? 'critical' :
              caseData.severity === 'high' ? 'warning' :
              caseData.severity === 'medium' ? 'healthy' : 'outline'
            }>
              {caseData.severity.toUpperCase()}
            </Badge>
            <Badge variant="outline" className="capitalize text-slate-300">
              {caseData.status}
            </Badge>
            
            {/* Generator Badge */}
            {caseData.isGroundedFallback ? (
              <Badge variant="outline" className="bg-amber-950/80 text-amber-400 border-amber-800 flex items-center gap-1.5 font-mono text-xs">
                <ShieldAlert className="w-3.5 h-3.5" /> Grounded Fallback
              </Badge>
            ) : caseData.generatorUsed ? (
              <Badge variant="outline" className="bg-emerald-950/80 text-emerald-400 border-emerald-800 flex items-center gap-1.5 font-mono text-xs">
                <Sparkles className="w-3.5 h-3.5" /> {caseData.generatorUsed}
              </Badge>
            ) : (
              <Badge variant="outline" className="bg-slate-800 text-slate-300 border-slate-700 flex items-center gap-1.5 font-mono text-xs">
                Development Mock
              </Badge>
            )}
          </div>
          <p className="text-sm text-slate-400">
            Created {formatDistanceToNow(new Date(caseData.createdTime), { addSuffix: true })}
          </p>
        </div>
      </div>

      {/* Grounded Fallback Banner Alert */}
      {caseData.isGroundedFallback && (
        <div className="p-4 bg-amber-950/40 border border-amber-800/60 rounded-lg flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-bold text-amber-300">Grounded Fallback Diagnostic (No LLM Synthesis)</h4>
            <p className="text-xs text-amber-200/80 mt-1 leading-relaxed">
              This diagnostic was generated deterministically by matching machine telemetry findings with technical manual chunks stored in the vector index. An external LLM API key was not available or was disabled.
            </p>
          </div>
        </div>
      )}

      {/* Real RAG Banner */}
      {!caseData.isGroundedFallback && caseData.generatorUsed && (
        <div className="p-3 bg-emerald-950/30 border border-emerald-800/40 rounded-lg flex items-center justify-between gap-3 text-xs text-emerald-300">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Diagnostic generated with <strong>{caseData.generatorUsed}</strong> grounded in technical manuals.</span>
          </div>
          <span className="font-mono text-slate-400">TF-IDF Vector Retrieval</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Context & Findings */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          
          {/* Machine Context */}
          <Card>
            <CardHeader className="pb-3 border-b border-slate-800/50">
              <CardTitle className="flex items-center gap-2 text-sm uppercase tracking-wider text-slate-400">
                <Cpu className="w-4 h-4" /> Machine Context
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              <div className="mb-4">
                <div className="text-lg font-bold text-primary">{caseData.machineName}</div>
                <div className="text-sm text-slate-500 font-mono">{caseData.machineId}</div>
              </div>
              <button 
                onClick={() => navigate(`/machines/${caseData.machineId}`)}
                className="w-full py-2 bg-slate-900 border border-slate-700 hover:border-accent/50 text-slate-300 hover:text-accent rounded text-sm font-medium transition-colors flex items-center justify-center gap-2"
              >
                View Machine Telemetry <ArrowRight className="w-4 h-4" />
              </button>
            </CardContent>
          </Card>

          {/* Trigger */}
          <Card className="border-l-4 border-l-amber-500">
            <CardHeader className="pb-3 border-b border-slate-800/50">
              <CardTitle className="flex items-center gap-2 text-sm uppercase tracking-wider text-slate-400">
                <AlertTriangle className="w-4 h-4 text-amber-500" /> Investigation Trigger
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              <p className="text-sm text-slate-300 font-medium">
                {caseData.trigger}
              </p>
            </CardContent>
          </Card>

          {/* ML Finding */}
          <Card className="border-l-4 border-l-primary bg-slate-900/30">
            <CardHeader className="pb-3 border-b border-slate-800/50">
              <CardTitle className="flex items-center gap-2 text-sm uppercase tracking-wider text-slate-400">
                <BrainCircuit className="w-4 h-4 text-primary" /> ML System Finding
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              <p className="text-sm text-primary leading-relaxed">
                {caseData.finding}
              </p>
              <div className="mt-4 pt-3 border-t border-slate-800/50 text-xs text-slate-500 flex justify-between">
                <span>Source: Anomaly Detection Model</span>
                {caseData.confidence !== undefined && <span className="font-mono">Conf: {caseData.confidence}%</span>}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: RAG Explanation & Evidence */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          
          {/* Grounded Explanation & Recommendation */}
          <Card className="border-accent/30 shadow-[0_0_15px_rgba(168,85,247,0.05)]">
            <CardHeader className="pb-4 border-b border-slate-800 flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-accent">
                <Stethoscope className="w-5 h-5" /> AI Diagnostic Explanation
              </CardTitle>
              {caseData.generatorUsed && (
                <span className="text-xs font-mono text-slate-500">{caseData.generatorUsed}</span>
              )}
            </CardHeader>
            <CardContent className="pt-6 space-y-8">
              
              {/* Explanation Text */}
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 border-l-2 border-slate-600 pl-2">
                  Engineering Assessment
                </h3>
                <p className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap">
                  {caseData.explanation}
                </p>
              </div>

              {/* Recommended Action */}
              <div className="bg-slate-900/80 border border-slate-700 rounded-lg p-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500" /> Recommended Action
                </h3>
                <p className="text-primary font-medium text-sm">
                  {caseData.recommendedAction}
                </p>
              </div>

            </CardContent>
          </Card>

          {/* Retrieved Evidence */}
          <div>
            <h3 className="text-lg font-bold tracking-tight text-primary flex items-center gap-2 mb-4">
              <FileText className="w-5 h-5 text-slate-400" /> Supporting Evidence ({caseData.evidence?.length || 0})
            </h3>
            
            {(!caseData.evidence || caseData.evidence.length === 0) ? (
              <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 flex items-start gap-3">
                 <ShieldAlert className="w-5 h-5 text-amber-500 shrink-0" />
                 <div>
                   <h4 className="text-sm font-bold text-slate-200">No supporting documentation found.</h4>
                   <p className="text-xs text-slate-400 mt-1">
                     The diagnostic explanation is derived solely from telemetry baselines. No specific technical manuals or maintenance logs were retrieved from the Knowledge Base for this anomaly signature.
                   </p>
                 </div>
              </div>
            ) : (
              <div className="space-y-4">
                {caseData.evidence.map((ev, idx) => (
                  <Card key={idx} className="bg-slate-950 border-slate-800 hover:border-slate-700 transition-colors">
                    <CardContent className="p-5">
                      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-3 pb-3 border-b border-slate-800">
                        <div>
                          <div className="text-sm font-bold text-primary flex items-center gap-2">
                            <FileText className="w-4 h-4 text-slate-400" /> {ev.documentTitle}
                          </div>
                          <div className="text-xs text-slate-500 mt-1 font-mono flex items-center gap-2 flex-wrap">
                            <span>Doc ID: {ev.documentId}</span>
                            {ev.section && <span>• Section: {ev.section}</span>}
                            {ev.page !== undefined && <span className="text-amber-400 font-semibold">• Page {ev.page}</span>}
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <Badge variant="outline" className="font-mono text-xs text-slate-400 border-slate-700">
                            Relevance: {(ev.relevance * 100).toFixed(0)}%
                          </Badge>
                          <button 
                            onClick={() => navigate('/knowledge-base')}
                            className="text-xs text-accent hover:text-accent/80 font-medium whitespace-nowrap"
                          >
                            View Source
                          </button>
                        </div>
                      </div>
                      <p className="text-sm text-slate-300 leading-relaxed italic border-l-2 border-accent/40 pl-4 bg-slate-900/40 py-2 rounded-r">
                        "{ev.snippet}"
                      </p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
