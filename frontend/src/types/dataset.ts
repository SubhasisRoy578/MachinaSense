// ---------------------------------------------------------------------------
// Dataset types for the ML Training workflow (Phase 1: Dataset Foundation)
// These types represent the planned backend contract for dataset management.
// ---------------------------------------------------------------------------

export type DatasetStatus = 'uploading' | 'processing' | 'ready' | 'failed';
export type ColumnDtype = 'int64' | 'float64' | 'object' | 'bool' | 'datetime64' | 'category' | string;

/** Summary statistics for a single numeric column. */
export interface NumericColumnStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  q25: number;
  q75: number;
}

/** Summary statistics for a single categorical column. */
export interface CategoricalColumnStats {
  uniqueCount: number;
  topValue: string;
  topFrequency: number;
}

/** Metadata and stats for a single column within a dataset. */
export interface DatasetColumn {
  name: string;
  dtype: ColumnDtype;
  missingCount: number;
  uniqueCount: number;
  isNumeric: boolean;
  numericStats: NumericColumnStats | null;
  categoricalStats: CategoricalColumnStats | null;
}

/** Aggregate statistics for the entire dataset. */
export interface DatasetStatistics {
  rowCount: number;
  columnCount: number;
  numericColumnCount: number;
  categoricalColumnCount: number;
  missingValueCount: number;
  duplicateRowCount: number;
  memorySizeBytes: number;
}

/** A preview table: first N rows of the dataset. */
export interface DatasetPreview {
  columns: string[];
  rows: Record<string, string | number | boolean | null>[];
  totalRows: number;
  previewRows: number;
}

/** Full dataset resource returned by the backend. */
export interface Dataset {
  id: string;
  name: string;
  fileName: string;
  fileSize: number;
  status: DatasetStatus;
  statistics: DatasetStatistics | null;
  columns: DatasetColumn[];
  preview: DatasetPreview | null;
  uploadedAt: string; // ISO 8601
}

/** Response when listing the current user's datasets. */
export interface DatasetListResponse {
  datasets: DatasetListItem[];
  total: number;
}

/** Lightweight dataset item used in list views. */
export interface DatasetListItem {
  id: string;
  name: string;
  fileName: string;
  fileSize: number;
  status: DatasetStatus;
  rowCount: number | null;
  columnCount: number | null;
  uploadedAt: string;
}

/** Response returned immediately after a successful upload. */
export interface DatasetUploadResponse {
  id: string;
  name: string;
  fileName: string;
  fileSize: number;
  status: DatasetStatus;
  statistics: DatasetStatistics | null;
  columns: DatasetColumn[];
  preview: DatasetPreview | null;
  uploadedAt: string;
}
