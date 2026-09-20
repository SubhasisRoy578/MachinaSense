import { useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import { api } from '../../services/api';
import { Card, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Wrench, Calendar, ArrowRight, Activity, AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export function MaintenancePage() {
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({ machine_id: '', title: '', description: '', recommended_action: '', priority: 'medium' });

  useEffect(() => {
    const fetchMaintenance = async () => {
      setLoading(true);
      try {
        setError('');
        const data = await api.maintenance.listQueue();
        setTasks(data);
      } catch (error) {
        console.error("Failed to fetch maintenance queue", error);
        setError(error instanceof Error ? error.message : 'Maintenance records are unavailable.');
      } finally {
        setLoading(false);
      }
    };
    fetchMaintenance();
  }, []);
  async function createTask(event: FormEvent) {
    event.preventDefault();
    try { const task = await api.maintenance.create(form); setTasks(current => [task, ...current]); setShowCreate(false); setForm({ machine_id: '', title: '', description: '', recommended_action: '', priority: 'medium' }); }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Unable to create maintenance record.'); }
  }
  async function removeTask(id: string) {
    try { await api.maintenance.remove(id); setTasks(current => current.filter(task => task.id !== id)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Unable to delete maintenance record.'); }
  }

  const criticalCount = tasks.filter(t => t.priority === 'critical').length;
  const pendingCount = tasks.filter(t => t.status === 'pending').length;

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500 pb-12">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-primary flex items-center gap-2">
          <Wrench className="w-6 h-6 text-accent" /> Maintenance Priority
        </h1>
        <p className="text-sm text-slate-400 mt-1">Actionable maintenance queue driven by AI predictive health scores and real-time anomalies.</p>
      </div>
      <button onClick={() => setShowCreate(value => !value)} className="self-start rounded bg-accent px-4 py-2 text-sm font-semibold text-slate-950">Create maintenance action</button>
      {showCreate && <form onSubmit={createTask} className="grid grid-cols-1 gap-3 rounded border border-slate-800 bg-slate-900/50 p-4 md:grid-cols-2">{[['machine_id','Machine ID'],['title','Action title'],['description','Condition / context'],['recommended_action','Recommended action']].map(([key,label]) => <input key={key} required value={(form as any)[key]} onChange={event => setForm({...form,[key]:event.target.value})} placeholder={label} className="rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"/>)}<select value={form.priority} onChange={event=>setForm({...form,priority:event.target.value})} className="rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option><option value="critical">Critical</option></select><button className="rounded bg-accent px-3 py-2 text-sm font-semibold text-slate-950">Save action</button></form>}
      {error && <div className="rounded border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">{error}</div>}

      {/* KPI Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className={criticalCount > 0 ? "border-l-4 border-l-red-500 bg-red-950/10" : "border-l-4 border-l-slate-500"}>
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <AlertTriangle className={criticalCount > 0 ? "text-red-500 w-4 h-4" : "w-4 h-4"} /> Critical Priority
            </div>
            <div className={`text-3xl font-mono ${criticalCount > 0 ? "text-red-500" : "text-primary"}`}>{criticalCount}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-amber-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <Calendar className="w-4 h-4 text-amber-500" /> Pending Actions
            </div>
            <div className="text-3xl font-mono text-amber-400">{pendingCount}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-emerald-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-500" /> Total Active Queue
            </div>
            <div className="text-3xl font-mono text-emerald-400">{tasks.length}</div>
          </CardContent>
        </Card>
      </div>

      {/* Maintenance Queue */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-slate-400 uppercase bg-slate-950/50 border-b border-slate-800">
                <tr>
                  <th className="px-6 py-4 font-medium tracking-wider">Priority</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Machine</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Machine Condition Context</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Recommended Action</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Timeframe</th>
                  <th className="px-6 py-4 font-medium tracking-wider text-right">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {loading ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-slate-500 font-mono">
                      [LOADING_MAINTENANCE_QUEUE...]
                    </td>
                  </tr>
                ) : tasks.length === 0 ? <tr><td colSpan={6} className="px-6 py-12 text-center text-slate-500">No maintenance actions available.</td></tr> : (
                  tasks.map((task) => (
                    <tr key={task.id} className="hover:bg-slate-800/20 transition-colors group">
                      <td className="px-6 py-4">
                        <Badge variant={
                          task.priority === 'critical' ? 'critical' :
                          task.priority === 'high' ? 'warning' :
                          task.priority === 'medium' ? 'healthy' : 'outline'
                        }>
                          {task.priority.toUpperCase()}
                        </Badge>
                      </td>
                      <td className="px-6 py-4">
                        <div 
                          className="font-medium text-primary hover:text-accent cursor-pointer transition-colors"
                          onClick={() => navigate(`/machines/${task.machineId}`)}
                        >
                          {task.machineName}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5 font-mono">{task.machineId}</div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="text-slate-300">{task.condition}</div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="font-medium text-primary bg-slate-900 border border-slate-700 px-3 py-1.5 rounded inline-block">
                          {task.action}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className={`font-mono ${task.timeframe === 'Immediate' ? 'text-red-400 font-bold' : 'text-slate-400'}`}>
                          {task.timeframe}
                        </div>
                        <div className="text-xs text-slate-500 mt-1 capitalize">{task.status}</div>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex flex-col gap-2 items-end">
                          <button 
                            onClick={() => navigate(`/diagnostics?machine=${task.machineId}`)}
                            className="inline-flex items-center justify-center px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider text-primary border border-slate-700 bg-slate-900 hover:bg-slate-800 transition-colors"
                          >
                            View Diagnostic
                          </button>
                          <button onClick={() => removeTask(task.id)} className="text-xs text-red-400 hover:text-red-300">Delete</button>
                          <button 
                            onClick={() => navigate(`/machines/${task.machineId}`)}
                            className="inline-flex items-center justify-center px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider text-accent border border-accent/30 hover:bg-accent/10 transition-colors"
                          >
                            Inspect Machine <ArrowRight className="w-3 h-3 ml-1" />
                          </button>
                        </div>
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
