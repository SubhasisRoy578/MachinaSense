import { Table, Eye } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import type { DatasetPreview as DatasetPreviewType } from '../../types/dataset';

interface DatasetPreviewProps {
  preview: DatasetPreviewType | null;
}

export function DatasetPreview({ preview }: DatasetPreviewProps) {
  if (!preview || !preview.rows || preview.rows.length === 0) {
    return (
      <Card className="border-slate-800 bg-slate-900/60">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Eye className="w-5 h-5 text-accent" />
            <CardTitle>Raw Data Preview</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="p-6 text-center text-slate-500 font-mono text-xs">
          No preview records returned by the backend for this dataset.
        </CardContent>
      </Card>
    );
  }

  const columns = preview.columns.length > 0 ? preview.columns : Object.keys(preview.rows[0] || {});

  return (
    <Card className="border-slate-800 bg-slate-900/60">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <Table className="w-5 h-5 text-accent" />
          <CardTitle>Telemetry Data Preview</CardTitle>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline">
            SHOWING {preview.rows.length} OF {preview.totalRows.toLocaleString()} ROWS
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto max-h-[380px] custom-scrollbar">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead className="bg-slate-950/80 sticky top-0 border-b border-slate-800 text-slate-400 z-10">
              <tr>
                <th className="py-2.5 px-3 font-semibold text-slate-500 w-12 text-center">#</th>
                {columns.map((col) => (
                  <th key={col} className="py-2.5 px-4 font-semibold text-slate-300 whitespace-nowrap">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {preview.rows.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-2 px-3 text-center text-slate-600 select-none">{idx + 1}</td>
                  {columns.map((col) => {
                    const val = row[col];
                    const isNull = val === null || val === undefined;
                    return (
                      <td key={col} className="py-2 px-4 whitespace-nowrap text-slate-200">
                        {isNull ? (
                          <span className="text-slate-600 italic">null</span>
                        ) : typeof val === 'number' ? (
                          <span>{Number.isInteger(val) ? val : val.toFixed(4)}</span>
                        ) : (
                          <span>{String(val)}</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
