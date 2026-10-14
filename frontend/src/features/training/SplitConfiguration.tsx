import { PieChart, Shuffle, AlertCircle, Info } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';

interface SplitConfigurationProps {
  trainSplit: number;
  validationSplit: number;
  testSplit: number;
  randomSeed: number;
  onSetSplits: (train: number, val: number, test: number) => void;
  onSetSeed: (seed: number) => void;
  validationError?: string;
}

export function SplitConfiguration({
  trainSplit,
  validationSplit,
  testSplit,
  randomSeed,
  onSetSplits,
  onSetSeed,
  validationError,
}: SplitConfigurationProps) {
  const sum = trainSplit + validationSplit + testSplit;
  const isSumValid = sum === 100;

  const handleTrainChange = (val: number) => {
    onSetSplits(val, validationSplit, testSplit);
  };

  const handleValChange = (val: number) => {
    onSetSplits(trainSplit, val, testSplit);
  };

  const handleTestChange = (val: number) => {
    onSetSplits(trainSplit, validationSplit, val);
  };

  const handleResetDefault = () => {
    onSetSplits(70, 15, 15);
    onSetSeed(42);
  };

  return (
    <Card className={`border-slate-800 bg-slate-900/60 ${!isSumValid ? 'border-red-500/50' : ''}`}>
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <PieChart className="w-5 h-5 text-accent" />
          <CardTitle>5. Dataset Split & Reproducibility</CardTitle>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={isSumValid ? 'healthy' : 'critical'}>
            SPLIT SUM: {sum}% {isSumValid ? '(VALID)' : '(MUST EQUAL 100%)'}
          </Badge>
          <button
            type="button"
            onClick={handleResetDefault}
            className="text-xs font-mono text-slate-400 hover:text-slate-200 px-2 py-1 bg-slate-800 rounded transition-colors"
          >
            Reset (70/15/15)
          </button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Split Visual Bar */}
        <div className="w-full h-3 bg-slate-950 rounded-full overflow-hidden flex border border-slate-800">
          <div
            style={{ width: `${Math.max(0, Math.min(100, trainSplit))}%` }}
            className="bg-emerald-500/80 transition-all duration-300"
            title={`Train: ${trainSplit}%`}
          />
          <div
            style={{ width: `${Math.max(0, Math.min(100, validationSplit))}%` }}
            className="bg-amber-500/80 transition-all duration-300"
            title={`Validation: ${validationSplit}%`}
          />
          <div
            style={{ width: `${Math.max(0, Math.min(100, testSplit))}%` }}
            className="bg-blue-500/80 transition-all duration-300"
            title={`Test: ${testSplit}%`}
          />
        </div>

        {/* Sliders / Number inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center justify-between text-xs font-mono mb-1">
              <span className="text-emerald-400 font-semibold">TRAIN SPLIT</span>
              <span className="text-slate-200">{trainSplit}%</span>
            </div>
            <input
              type="range"
              min="10"
              max="90"
              value={trainSplit}
              onChange={(e) => handleTrainChange(parseInt(e.target.value, 10))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center justify-between text-xs font-mono mb-1">
              <span className="text-amber-400 font-semibold">VALIDATION SPLIT</span>
              <span className="text-slate-200">{validationSplit}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="50"
              value={validationSplit}
              onChange={(e) => handleValChange(parseInt(e.target.value, 10))}
              className="w-full accent-amber-500 cursor-pointer"
            />
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded p-3">
            <div className="flex items-center justify-between text-xs font-mono mb-1">
              <span className="text-blue-400 font-semibold">TEST SPLIT</span>
              <span className="text-slate-200">{testSplit}%</span>
            </div>
            <input
              type="range"
              min="5"
              max="50"
              value={testSplit}
              onChange={(e) => handleTestChange(parseInt(e.target.value, 10))}
              className="w-full accent-blue-500 cursor-pointer"
            />
          </div>
        </div>

        {/* Random Seed and Time-Series Note */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
          <div>
            <label htmlFor="random-seed-input" className="block text-xs font-mono text-slate-400 mb-1 flex items-center gap-1.5">
              <Shuffle className="w-3.5 h-3.5 text-accent" />
              <span>Random Seed (Reproducibility)</span>
            </label>
            <input
              id="random-seed-input"
              type="number"
              step="1"
              value={randomSeed}
              onChange={(e) => onSetSeed(parseInt(e.target.value, 10) || 0)}
              className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 font-mono focus:outline-none focus:border-accent"
            />
          </div>

          <div className="bg-slate-950/40 border border-slate-800 rounded p-3 flex items-start gap-2 text-xs font-mono text-slate-400">
            <Info className="w-4 h-4 text-accent shrink-0 mt-0.5" />
            <p className="leading-relaxed text-[11px]">
              <strong className="text-slate-200">Temporal / Run-to-Failure Data:</strong> For sequential telemetry datasets, time-ordered splits are preserved to prevent future telemetry leakage into training partitions.
            </p>
          </div>
        </div>

        {validationError && (
          <div className="flex items-center gap-2 text-red-400 text-xs font-mono bg-red-500/10 p-2.5 rounded border border-red-500/20">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{validationError}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
