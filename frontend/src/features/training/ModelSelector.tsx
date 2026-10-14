import { Cpu, CheckCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { TASK_MODELS } from '../../store/trainingStore';
import type { MLTask } from '../../types/training';

interface ModelSelectorProps {
  task: MLTask;
  selectedModel: string;
  onSelectModel: (model: string) => void;
}

export function ModelSelector({ task, selectedModel, onSelectModel }: ModelSelectorProps) {
  const models = TASK_MODELS[task] || [];

  return (
    <Card className="border-slate-800 bg-slate-900/60">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-accent" />
          <CardTitle>4. Select ML Algorithm</CardTitle>
        </div>
        <Badge variant="outline">{models.length} ALGORITHMS SUPPORTED</Badge>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {models.map((model) => {
            const isSelected = selectedModel === model.id;

            return (
              <div
                key={model.id}
                onClick={() => onSelectModel(model.id)}
                className={`p-3.5 rounded-lg border cursor-pointer transition-all flex flex-col justify-between ${
                  isSelected
                    ? 'border-accent bg-accent/10 shadow-md shadow-accent/5'
                    : 'border-slate-800 bg-slate-950/40 hover:border-slate-700 hover:bg-slate-800/30'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <h4 className="text-xs font-bold text-slate-100 font-mono">{model.name}</h4>
                    {isSelected && <CheckCircle className="w-4 h-4 text-accent shrink-0" />}
                  </div>
                  <p className="text-[11px] text-slate-400 font-mono leading-relaxed">
                    {model.description}
                  </p>
                </div>

                <div className="mt-3 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono text-slate-500">
                  <span>ID: {model.id}</span>
                  <Badge variant="healthy" className="text-[9px] px-1 py-0">
                    AVAILABLE
                  </Badge>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
