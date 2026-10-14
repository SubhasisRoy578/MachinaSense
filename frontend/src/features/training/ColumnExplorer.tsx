import { Columns, Hash } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import type { DatasetColumn } from '../../types/dataset';

interface ColumnExplorerProps {
  columns: DatasetColumn[];
}

export function ColumnExplorer({ columns }: ColumnExplorerProps) {
  if (!columns || columns.length === 0) {
    return (
      <Card className="border-slate-800 bg-slate-900/60">
        <CardContent className="p-6 text-center text-slate-500 font-mono text-xs">
          No column definitions detected for this dataset.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-slate-800 bg-slate-900/60">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Columns className="w-5 h-5 text-accent" />
          <CardTitle>Column Schema & Distribution ({columns.length} Features)</CardTitle>
        </div>
        <Badge variant="outline">DETECTION COMPLETE</Badge>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto max-h-[380px] custom-scrollbar">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead className="bg-slate-950/80 sticky top-0 border-b border-slate-800 text-slate-400 z-10">
              <tr>
                <th className="py-2.5 px-4 font-semibold">COLUMN NAME</th>
                <th className="py-2.5 px-4 font-semibold">TYPE</th>
                <th className="py-2.5 px-4 font-semibold text-right">MISSING</th>
                <th className="py-2.5 px-4 font-semibold text-right">UNIQUE VALUES</th>
                <th className="py-2.5 px-4 font-semibold">STATISTICAL SUMMARY</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {columns.map((col) => {
                const isNum = col.isNumeric;
                const stats = col.numericStats;
                const catStats = col.categoricalStats;

                return (
                  <tr key={col.name} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-2.5 px-4 font-medium text-slate-100 flex items-center gap-1.5">
                      <Hash className="w-3.5 h-3.5 text-slate-500" />
                      <span>{col.name}</span>
                    </td>
                    <td className="py-2.5 px-4">
                      <Badge variant={isNum ? 'healthy' : 'warning'}>
                        {col.dtype || (isNum ? 'float64' : 'object')}
                      </Badge>
                    </td>
                    <td className="py-2.5 px-4 text-right">
                      {col.missingCount > 0 ? (
                        <span className="text-amber-400 font-semibold">{col.missingCount}</span>
                      ) : (
                        <span className="text-slate-500">0</span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 text-right text-slate-300">
                      {col.uniqueCount !== undefined ? col.uniqueCount.toLocaleString() : '—'}
                    </td>
                    <td className="py-2.5 px-4 text-slate-400">
                      {stats ? (
                        <div className="flex items-center gap-3 text-[11px]">
                          <span>
                            μ: <span className="text-slate-200">{stats.mean.toFixed(2)}</span>
                          </span>
                          <span>
                            σ: <span className="text-slate-200">{stats.std.toFixed(2)}</span>
                          </span>
                          <span>
                            range: [{stats.min.toFixed(2)} .. {stats.max.toFixed(2)}]
                          </span>
                        </div>
                      ) : catStats ? (
                        <div className="text-[11px] text-slate-400">
                          top: <span className="text-slate-200">"{catStats.topValue}"</span> ({catStats.topFrequency} occurrences)
                        </div>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
