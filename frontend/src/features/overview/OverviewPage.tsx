
import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { Machine, SensorDataPoint } from '../../types/models';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import { format } from 'date-fns';

export function OverviewPage() {
  const [machines, setMachines] = useState<Machine[]>([]);
  const [sensorData, setSensorData] = useState<SensorDataPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const machinesData = await api.machines.list();
        const sensorSeries = machinesData.length ? await api.sensors.getMachineData(machinesData[0].id, 24) : [];
        setMachines(machinesData);
        setSensorData(sensorSeries);
      } catch (error) {
        console.error("Error fetching overview data", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const healthyCount = machines.filter(m => m.status === 'healthy').length;
  const warningCount = machines.filter(m => m.status === 'warning').length;
  const criticalCount = machines.filter(m => m.status === 'critical').length;

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-primary">Operations Overview</h1>
        <p className="text-sm text-slate-400 mt-1">Real-time fleet health and predictive analytics.</p>
      </div>
      
      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-l-4 border-l-slate-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-1">Total Monitored</div>
            <div className="text-3xl font-mono font-light text-primary">{loading ? '-' : machines.length}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-emerald-500 bg-emerald-950/10">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-1 text-emerald-500/80">Healthy</div>
            <div className="text-3xl font-mono font-light text-emerald-400">{loading ? '-' : healthyCount}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-amber-500 bg-amber-950/10">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-1 text-amber-500/80">Warning</div>
            <div className="text-3xl font-mono font-light text-amber-400">{loading ? '-' : warningCount}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-red-500 bg-red-950/10 shadow-[0_0_15px_rgba(239,68,68,0.1)]">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-1 text-red-500/80">Critical</div>
            <div className="text-3xl font-mono font-light text-red-400">{loading ? '-' : criticalCount}</div>
          </CardContent>
        </Card>
      </div>

      {/* Main Chart Area */}
      <Card className="h-96">
        <CardHeader>
          <CardTitle>Fleet Vibration Average vs Anomaly Baseline (Last 24h)</CardTitle>
          <Badge variant="outline">Model: Isolation Forest v2.1</Badge>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="h-full w-full flex items-center justify-center text-slate-500 font-mono text-sm">
              [LOADING_SENSOR_DATA...]
            </div>
          ) : sensorData.length === 0 ? <div className="h-full w-full flex items-center justify-center text-slate-500 text-sm">No telemetry data available.</div> : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={sensorData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorVib" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2b3038" vertical={false} />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(val) => format(new Date(val), 'HH:mm')}
                  stroke="#475569" 
                  tick={{ fill: '#64748b', fontSize: 12 }} 
                  dy={10}
                />
                <YAxis 
                  stroke="#475569" 
                  tick={{ fill: '#64748b', fontSize: 12, fontFamily: 'monospace' }} 
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f1115', borderColor: '#2b3038', borderRadius: '6px' }}
                  labelStyle={{ color: '#94a3b8' }}
                  itemStyle={{ color: '#e2e8f0', fontFamily: 'monospace' }}
                  labelFormatter={(val) => format(new Date(val as string | number), 'MMM dd, HH:mm')}
                />
                <Area 
                  type="monotone" 
                  dataKey="vibration" 
                  name="Vibration (mm/s)"
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  fillOpacity={1} 
                  fill="url(#colorVib)" 
                  activeDot={{ r: 4, strokeWidth: 0, fill: '#60a5fa' }}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
