import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { SensorDataPoint } from '../../types/models';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import { format, formatDistanceToNow } from 'date-fns';
import { Activity, AlertTriangle, TrendingDown, CheckCircle2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export function SensorsPage() {
  const navigate = useNavigate();
  const [fleetData, setFleetData] = useState<any>(null);
  const [sensorData, setSensorData] = useState<SensorDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedMachine, setSelectedMachine] = useState('M-003');
  const [availableMachines, setAvailableMachines] = useState<{ id: string; name: string }[]>([]);

  useEffect(() => {
    api.machines.list().then((machines) => {
      if (machines && machines.length > 0) {
        setAvailableMachines(machines.map(m => ({ id: m.id, name: m.name })));
        setSelectedMachine(prev => machines.some(m => m.id === prev) ? prev : machines[0].id);
      }
    }).catch(err => console.warn("Could not load machines for sensors page:", err));
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [fleet, telemetry] = await Promise.all([
          api.sensors.listFleetData(),
          api.sensors.getMachineData(selectedMachine, 24)
        ]);
        setFleetData(fleet);
        setSensorData(telemetry);
      } catch (error) {
        console.error("Error fetching sensor data", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [selectedMachine]);

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500 pb-12">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-primary flex items-center gap-2">
          <TrendingDown className="w-6 h-6 text-accent" /> Sensor Intelligence
        </h1>
        <p className="text-sm text-slate-400 mt-1">Fleet telemetry and individual machine sensor monitoring.</p>
      </div>

      {/* Fleet Sensor KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-l-4 border-l-slate-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <Activity className="w-4 h-4" /> Total Sensors
            </div>
            <div className="text-3xl font-mono text-primary">{fleetData?.total || '-'}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-emerald-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2 text-emerald-500">
              <CheckCircle2 className="w-4 h-4" /> Active Telemetry
            </div>
            <div className="text-3xl font-mono text-emerald-400">{fleetData?.active || '-'}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-amber-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2 text-amber-500">
              <AlertTriangle className="w-4 h-4" /> Abnormal Values
            </div>
            <div className="text-3xl font-mono text-amber-400">{fleetData?.warning || '-'}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-red-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2 text-red-500">
              <AlertTriangle className="w-4 h-4" /> Sensor Failures
            </div>
            <div className="text-3xl font-mono text-red-400">0</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Sensor Monitoring Chart */}
        <Card className="xl:col-span-2 h-[500px] flex flex-col">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Vibration Telemetry Analysis</CardTitle>
              <div className="text-sm text-slate-400 mt-1 font-mono">Last 24 Hours</div>
            </div>
            <div className="flex gap-2">
              <select 
                className="bg-slate-950 border border-slate-700 rounded text-sm px-3 py-1.5 text-primary focus:outline-none focus:border-accent"
                value={selectedMachine}
                onChange={(e) => setSelectedMachine(e.target.value)}
              >
                {availableMachines.length > 0 ? (
                  availableMachines.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.id} — {m.name}
                    </option>
                  ))
                ) : (
                  <>
                    <option value="M-001">M-001 - CNC Mill Alpha</option>
                    <option value="M-002">M-002 - CNC Mill Beta</option>
                    <option value="M-003">M-003 - Hydraulic Press V1 (Critical)</option>
                    <option value="M-004">M-004 - Conveyor Drive 4A</option>
                  </>
                )}
              </select>
            </div>
          </CardHeader>
          <CardContent className="flex-1 min-h-0">
            {loading ? (
              <div className="flex items-center justify-center h-full text-slate-500 font-mono text-sm">
                [LOADING_SENSOR_DATA...]
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={sensorData} margin={{ top: 20, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorVibSensors" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorTempSensors" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2b3038" vertical={false} />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(val) => format(new Date(val), 'HH:mm')}
                    stroke="#475569" 
                    tick={{ fill: '#64748b', fontSize: 12 }} 
                  />
                  <YAxis yAxisId="left" stroke="#475569" tick={{ fill: '#3b82f6', fontSize: 12, fontFamily: 'monospace' }} />
                  <YAxis yAxisId="right" orientation="right" stroke="#475569" tick={{ fill: '#f59e0b', fontSize: 12, fontFamily: 'monospace' }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f1115', borderColor: '#2b3038', borderRadius: '6px' }}
                    labelStyle={{ color: '#94a3b8' }}
                    itemStyle={{ fontFamily: 'monospace' }}
                    labelFormatter={(val) => format(new Date(val as string | number), 'MMM dd, yyyy HH:mm')}
                  />
                  <Area yAxisId="left" type="monotone" dataKey="vibration" name="Vibration (mm/s)" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorVibSensors)" activeDot={{ r: 4 }} />
                  <Area yAxisId="right" type="monotone" dataKey="temperature" name="Temperature (°C)" stroke="#f59e0b" strokeWidth={2} fillOpacity={1} fill="url(#colorTempSensors)" activeDot={{ r: 4 }} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Recent Sensor Activity */}
        <Card className="h-[500px] flex flex-col">
          <CardHeader>
            <CardTitle>Recent Telemetry Activity</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto pt-0">
            {loading ? (
              <div className="flex justify-center mt-10 text-slate-500 font-mono text-sm">[LOADING...]</div>
            ) : (
              <div className="space-y-4">
                {fleetData?.recentActivity?.map((activity: any) => (
                  <div 
                    key={activity.id} 
                    className="p-4 rounded-lg bg-slate-950/50 border border-slate-800 hover:border-slate-700 cursor-pointer transition-colors"
                    onClick={() => navigate(`/machines/${activity.machineId}`)}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <div className="text-sm font-medium text-primary hover:text-accent transition-colors">
                          {activity.machineName}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">{activity.machineId}</div>
                      </div>
                      <Badge variant={activity.status}>{activity.status}</Badge>
                    </div>
                    <div className="flex justify-between items-end mt-4">
                      <div>
                        <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">{activity.sensor}</div>
                        <div className="text-lg font-mono text-primary">
                          {activity.value} <span className="text-xs text-slate-500 font-sans">{activity.unit}</span>
                        </div>
                      </div>
                      <div className="text-xs text-slate-500 font-mono">
                        {formatDistanceToNow(new Date(activity.time), { addSuffix: true })}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
