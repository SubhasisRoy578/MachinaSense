import { Activity, ShieldAlert, Cpu } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { TASK_DEFINITIONS } from '../../store/trainingStore';
import type { MLTask } from '../../types/training';

interface TaskSelectorProps {
  selectedTask: MLTask;
  onSelectTask: (task: MLTask) => void;
}

export function TaskSelector({ selectedTask, onSelectTask }: TaskSelectorProps) {
  const tasks: { id: MLTask; icon: typeof Activity }[] = [
    { id: 'rul_regression', icon: Activity },
    { id: 'condition_classification', icon: Cpu },
    { id: 'anomaly_detection', icon: ShieldAlert },
  ];

  return (
    <Card className="border-slate-800 bg-slate-900/60">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-accent" />
          <CardTitle>1. Select Machine Learning Task</CardTitle>
        </div>
        <Badge variant="outline">MODEL OBJECTIVE</Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {tasks.map(({ id, icon: Icon }) => {
            const def = TASK_DEFINITIONS[id];
            const isSelected = selectedTask === id;

            return (
              <div
                key={id}
                onClick={() => onSelectTask(id)}
                className={`p-4 rounded-lg border cursor-pointer transition-all flex flex-col justify-between ${
                  isSelected
                    ? 'border-accent bg-accent/10 shadow-lg shadow-accent/5'
                    : 'border-slate-800 bg-slate-950/40 hover:border-slate-700 hover:bg-slate-800/30'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-7 h-7 rounded flex items-center justify-center ${
                          isSelected ? 'bg-accent/20 text-accent' : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                      </div>
                      <h4 className="text-sm font-semibold text-slate-100">{def.label}</h4>
                    </div>
                    <Badge variant={def.isSupervised ? 'healthy' : 'warning'}>
                      {def.isSupervised ? 'SUPERVISED' : 'UNSUPERVISED'}
                    </Badge>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed font-mono">
                    {def.description}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-[11px] font-mono text-slate-500">
                  <span>
                    {id === 'rul_regression'
                      ? 'Target: Remaining Cycles / Hours'
                      : id === 'condition_classification'
                      ? 'Target: Multi-class State'
                      : 'Target: None (Self-Supervised)'}
                  </span>
                  <div
                    className={`w-3.5 h-3.5 rounded-full border flex items-center justify-center ${
                      isSelected ? 'border-accent' : 'border-slate-700'
                    }`}
                  >
                    {isSelected && <div className="w-2 h-2 rounded-full bg-accent" />}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
