import { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { api } from '../../services/api';
import type { Machine, SensorDataPoint, RulForecastPoint, AnomalyEvent, DiagnosticPreview } from '../../types/models';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceLine
} from 'recharts';
import { format, formatDistanceToNow } from 'date-fns';
import { ArrowLeft, Activity, Wrench, AlertTriangle, BrainCircuit, Sparkles, Loader2 } from 'lucide-react';

export function MachineDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [machine, setMachine] = useState<Machine | null>(null);
  const [sensorData, setSensorData] = useState<SensorDataPoint[]>([]);
  const [rulData, setRulData] = useState<RulForecastPoint[]>([]);
  const [anomalies, setAnomalies] = useState<AnomalyEvent[]>([]);
  const [diagnostics, setDiagnostics] = useState<DiagnosticPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    if (!id) return;
    const fetchMachineData = async () => {
      setLoading(true);
      try {
        const [m, sensors, rul, anom, diag] = await Promise.all([
          api.machines.get(id),
          api.sensors.getMachineData(id, 24),
          api.diagnostics.getRulForecast(id),
          api.diagnostics.getMachineAnomalies(id),
          api.diagnostics.getMachineDiagnostics(id)
        ]);
        setMachine(m || null);
        setSensorData(sensors);
        setRulData(rul);
        setAnomalies(anom);
        setDiagnostics(diag);
      } catch (error) {
        console.error("Failed to fetch machine details", error);
      } finally {
        setLoading(false);
      }
    };
    fetchMachineData();
  }, [id]);

  const handleRunDiagnostic = async () => {
    if (!id || !machine) return;
    setIsGenerating(true);
    try {
      const result = await api.diagnostics.createCase(machine.id);
      if (result && result.id) {
        navigate(`/diagnostics/${result.id}`);
      } else {
        navigate(`/diagnostics?machine=${machine.id}`);
      }
    } catch (err) {
      console.error("Failed to trigger diagnostic", err);
      navigate(`/diagnostics?machine=${machine.id}`);
    } finally {
      setIsGenerating(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 font-mono text-sm">
        [LOADING_MACHINE_TELEMETRY...]
      </div>
    );
  }

  if (!machine) {
    return <div className="text-red-500">Machine not found.</div>;
  }

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500 pb-12">
      {/* 1. Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-start justify-between border-b border-slate-800 pb-6">
        <div>
          <Link to="/machines" className="inline-flex items-center text-sm text-slate-400 hover:text-accent mb-4 transition-colors">
            <ArrowLeft className="w-4 h-4 mr-1" /> Back to Machines
          </Link>
          <h1 className="text-3xl font-bold tracking-tight text-primary flex items-center gap-3">
            {machine.name}
            <Badge variant={machine.status as any} className="text-sm px-3 py-1">{machine.status.toUpperCase()}</Badge>
          </h1>
          <p className="text-sm text-slate-400 mt-2 font-mono">
            ID: {machine.id} <span className="mx-2">•</span> TYPE: {machine.type.toUpperCase()} <span className="mx-2">•</span> LOC: {machine.location.toUpperCase()}
          </p>
        </div>
        <div className="text-right flex flex-col items-end gap-2">
          <div className="text-xs text-slate-500 uppercase font-bold tracking-wider">Last Updated</div>
          <div className="text-sm text-slate-300 font-mono">
            {format(new Date(machine.lastUpdated), 'yyyy-MM-dd HH:mm:ss')}
          </div>
          <button 
            onClick={handleRunDiagnostic}
            disabled={isGenerating}
            className="inline-flex items-center justify-center gap-2 mt-1 px-4 py-2 bg-accent hover:bg-accent/90 text-slate-950 font-semibold rounded-md transition-colors text-xs uppercase tracking-wider disabled:opacity-50"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Generating...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" /> Run Grounded AI Diagnostic
              </>
            )}
          </button>
        </div>
      </div>

      {/* 2. Health & Risk Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-500" /> Overall Health
            </div>
            <div className="text-3xl font-mono text-primary">{100 - machine.failureRisk}<span className="text-lg text-slate-500">%</span></div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-500" /> Failure Risk
            </div>
            <div className="text-3xl font-mono text-amber-500">{machine.failureRisk}<span className="text-lg text-slate-500">%</span></div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <Wrench className="w-4 h-4 text-blue-500" /> Est. RUL
            </div>
            <div className="text-3xl font-mono text-primary">{machine.rulDays || 0} <span className="text-xs text-slate-500">DAYS</span></div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <BrainCircuit className="w-4 h-4 text-accent" /> Active Anomalies
            </div>
            <div className="text-3xl font-mono text-accent">{anomalies.length}</div>
          </CardContent>
        </Card>
      </div>

      {/* Main Grid Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left 2 Columns: Charts */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          
          {/* 3. Sensor Telemetry Stream */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-base font-medium text-slate-300">Live Telemetry (Vibration & Temperature)</CardTitle>
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-accent inline-block"></span> Vib (mm/s)</span>
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block"></span> Temp (°C)</span>
              </div>
            </CardHeader>
            <CardContent className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={sensorData}>
                  <defs>
                    <linearGradient id="vibColor" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#a855f7" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="tempColor" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="timestamp" stroke="#64748b" tickFormatter={(t) => format(new Date(t), 'HH:mm')} />
                  <YAxis stroke="#64748b" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.375rem', color: '#f8fafc' }} 
                    labelFormatter={(t) => t ? format(new Date(String(t)), 'yyyy-MM-dd HH:mm:ss') : ''}
                  />
                  <Area type="monotone" dataKey="vibration" stroke="#a855f7" fillOpacity={1} fill="url(#vibColor)" strokeWidth={2} />
                  <Area type="monotone" dataKey="temperature" stroke="#f59e0b" fillOpacity={1} fill="url(#tempColor)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* 4. RUL Forecast Curve */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-base font-medium text-slate-300">Remaining Useful Life (RUL) Trajectory</CardTitle>
              <div className="text-xs text-slate-500 font-mono">LSTM PyTorch Temporal Degradation Model</div>
            </CardHeader>
            <CardContent className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rulData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="timestamp" stroke="#64748b" tickFormatter={(t) => format(new Date(t), 'MM-dd')} />
                  <YAxis stroke="#64748b" label={{ value: 'Hours', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.375rem', color: '#f8fafc' }}
                  />
                  <ReferenceLine y={20} label="Critical Maintenance Threshold" stroke="#ef4444" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="predictedRul" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="lowerBound" stroke="#1d4ed8" strokeDasharray="2 2" dot={false} />
                  <Line type="monotone" dataKey="upperBound" stroke="#1d4ed8" strokeDasharray="2 2" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

        </div>

        {/* Right Column: Context & Diagnostics */}
        <div className="flex flex-col gap-6">
          
          {/* 8. AI Diagnostic Preview */}
          <Card className="border-accent/30 bg-accent/5 overflow-visible relative">
            <div className="absolute -top-3 -right-3 w-6 h-6 rounded-full bg-accent text-slate-950 flex items-center justify-center shadow-[0_0_12px_rgba(59,130,246,0.5)]">
              <BrainCircuit className="w-4 h-4" />
            </div>
            <CardHeader className="border-b-accent/20 flex flex-row items-center justify-between">
              <CardTitle className="text-accent flex items-center gap-2">AI Diagnostic Summary</CardTitle>
              <button
                onClick={handleRunDiagnostic}
                disabled={isGenerating}
                className="text-xs text-accent hover:underline font-medium flex items-center gap-1"
              >
                <Sparkles className="w-3 h-3" /> New Investigation
              </button>
            </CardHeader>
            <CardContent className="pt-4 text-sm">
              <div className="mb-4">
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">ML Finding</div>
                <div className="text-primary font-medium">{diagnostics?.finding}</div>
              </div>
              <div className="mb-4">
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Explanation</div>
                <div className="text-slate-300 leading-relaxed">{diagnostics?.explanation}</div>
              </div>
              <div>
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Supporting Evidence</div>
                <ul className="list-disc pl-4 space-y-1 text-slate-400 font-mono text-xs">
                  {diagnostics?.evidence.map((ev, i) => (
                    <li key={i}>{ev}</li>
                  ))}
                </ul>
              </div>
            </CardContent>
          </Card>

          {/* 5. Anomaly Timeline */}
          <Card className="flex-1 min-h-[300px]">
            <CardHeader>
              <CardTitle>Recent Anomalies</CardTitle>
            </CardHeader>
            <CardContent className="pt-4 overflow-y-auto">
              <div className="relative border-l border-slate-700 ml-3 space-y-6">
                {anomalies.map((anom) => (
                  <div key={anom.id} className="relative pl-6">
                    <div className={`absolute -left-1.5 top-1.5 w-3 h-3 rounded-full border-2 border-slate-900 ${
                      anom.severity === 'critical' ? 'bg-red-500' :
                      anom.severity === 'high' ? 'bg-amber-500' :
                      anom.severity === 'medium' ? 'bg-amber-400' : 'bg-slate-400'
                    }`}></div>
                    <div className="text-xs font-mono text-slate-400 mb-1">
                      {formatDistanceToNow(new Date(anom.timestamp), { addSuffix: true })}
                    </div>
                    <div className="text-sm font-medium text-primary">{anom.description}</div>
                    <div className="text-xs text-slate-500 mt-1 flex items-center justify-between">
                      <span>Sensor: {anom.sensor}</span>
                      <span className="font-mono">Score: {anom.score.toFixed(2)}</span>
                    </div>
                  </div>
                ))}
                {anomalies.length === 0 && (
                  <div className="text-sm text-slate-500 pl-4 py-4 italic">No recent anomalies detected.</div>
                )}
              </div>
            </CardContent>
          </Card>
          
        </div>
      </div>
    </div>
  );
}
