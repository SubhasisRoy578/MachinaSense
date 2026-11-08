import React, { useState } from 'react';
import { api } from '../../services/api';
import type { FD001Features, FD001PredictionResponse } from '../../types/inference';
import {
  Cpu,
  X,
  Play,
  RotateCcw,
  FileCheck2,
  AlertCircle,
  Loader2,
  BarChart3,
  CheckCircle2,
  Info,
  Gauge,
  Sliders,
  Thermometer,
  GaugeCircle,
  Wind,
} from 'lucide-react';

interface FD001PredictionModalProps {
  onClose: () => void;
}

// Verified real dataset row from NASA C-MAPSS FD001 (train_FD001.txt, Engine #1, Cycle 1)
const VERIFIED_FD001_SAMPLE: Record<keyof FD001Features, string> = {
  cycle: '1',
  op_setting_1: '-0.0007',
  op_setting_2: '-0.0004',
  op_setting_3: '100.0',
  sensor_2: '641.82',
  sensor_3: '1589.70',
  sensor_4: '1400.60',
  sensor_6: '21.61',
  sensor_7: '554.36',
  sensor_8: '2388.06',
  sensor_9: '9046.19',
  sensor_11: '47.47',
  sensor_12: '521.66',
  sensor_13: '2388.02',
  sensor_14: '8138.62',
  sensor_15: '8.4195',
  sensor_17: '392.0',
  sensor_20: '39.06',
  sensor_21: '23.4190',
};

const FEATURE_DEFINITIONS: Array<{
  key: keyof FD001Features;
  label: string;
  unit: string;
  group: 'operating' | 'temperature' | 'pressure' | 'speed' | 'bleed';
}> = [
  { key: 'cycle', label: 'Engine Cycle', unit: 'cycles', group: 'operating' },
  { key: 'op_setting_1', label: 'Op Setting 1 (Altitude)', unit: '--', group: 'operating' },
  { key: 'op_setting_2', label: 'Op Setting 2 (TRA)', unit: '--', group: 'operating' },
  { key: 'op_setting_3', label: 'Op Setting 3 (Mach)', unit: '--', group: 'operating' },

  { key: 'sensor_2', label: 'LPC Outlet Temp (T24)', unit: '°R', group: 'temperature' },
  { key: 'sensor_3', label: 'HPC Outlet Temp (T30)', unit: '°R', group: 'temperature' },
  { key: 'sensor_4', label: 'LPT Outlet Temp (T50)', unit: '°R', group: 'temperature' },

  { key: 'sensor_6', label: 'Bypass Duct Press (P15)', unit: 'psia', group: 'pressure' },
  { key: 'sensor_7', label: 'HPC Outlet Press (P30)', unit: 'psia', group: 'pressure' },
  { key: 'sensor_11', label: 'HPC Static Press (Ps30)', unit: 'psia', group: 'pressure' },
  { key: 'sensor_12', label: 'Fuel Flow Ratio (phi)', unit: 'pps/psi', group: 'pressure' },

  { key: 'sensor_8', label: 'Physical Fan Speed (Nf)', unit: 'rpm', group: 'speed' },
  { key: 'sensor_9', label: 'Physical Core Speed (Nc)', unit: 'rpm', group: 'speed' },
  { key: 'sensor_13', label: 'Corrected Fan Speed (NRf)', unit: 'rpm', group: 'speed' },
  { key: 'sensor_14', label: 'Corrected Core Speed (NRc)', unit: 'rpm', group: 'speed' },

  { key: 'sensor_15', label: 'Bypass Ratio (BPR)', unit: '--', group: 'bleed' },
  { key: 'sensor_17', label: 'Bleed Enthalpy (ht)', unit: 'BTU/lb', group: 'bleed' },
  { key: 'sensor_20', label: 'HPT Cool Bleed (W31)', unit: 'lbm/s', group: 'bleed' },
  { key: 'sensor_21', label: 'LPT Cool Bleed (W32)', unit: 'lbm/s', group: 'bleed' },
];

export const FD001PredictionModal: React.FC<FD001PredictionModalProps> = ({ onClose }) => {
  const [formValues, setFormValues] = useState<Record<string, string>>({
    cycle: '',
    op_setting_1: '',
    op_setting_2: '',
    op_setting_3: '',
    sensor_2: '',
    sensor_3: '',
    sensor_4: '',
    sensor_6: '',
    sensor_7: '',
    sensor_8: '',
    sensor_9: '',
    sensor_11: '',
    sensor_12: '',
    sensor_13: '',
    sensor_14: '',
    sensor_15: '',
    sensor_17: '',
    sensor_20: '',
    sensor_21: '',
  });

  const [validationError, setValidationError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [result, setResult] = useState<FD001PredictionResponse | null>(null);

  const handleInputChange = (key: string, value: string) => {
    setFormValues((prev) => ({ ...prev, [key]: value }));
    if (validationError) setValidationError(null);
  };

  const handleLoadSample = () => {
    setFormValues({ ...VERIFIED_FD001_SAMPLE });
    setValidationError(null);
    setApiError(null);
  };

  const handleReset = () => {
    const empty: Record<string, string> = {};
    FEATURE_DEFINITIONS.forEach((f) => {
      empty[f.key] = '';
    });
    setFormValues(empty);
    setValidationError(null);
    setApiError(null);
    setResult(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);
    setApiError(null);

    // Validate all 19 inputs are present and finite numbers
    const missing: string[] = [];
    const invalid: string[] = [];
    const payloadFeatures: Partial<FD001Features> = {};

    FEATURE_DEFINITIONS.forEach((f) => {
      const rawVal = formValues[f.key]?.trim();
      if (!rawVal && rawVal !== '0') {
        missing.push(f.label);
      } else {
        const num = Number(rawVal);
        if (isNaN(num) || !isFinite(num)) {
          invalid.push(f.label);
        } else {
          payloadFeatures[f.key] = num;
        }
      }
    });

    if (missing.length > 0) {
      setValidationError(`Missing required feature${missing.length > 1 ? 's' : ''}: ${missing.join(', ')}.`);
      return;
    }

    if (invalid.length > 0) {
      setValidationError(`Invalid non-numeric or infinite value${invalid.length > 1 ? 's' : ''} in: ${invalid.join(', ')}.`);
      return;
    }

    setIsPredicting(true);
    try {
      const response = await api.inference.predictFd001(payloadFeatures as FD001Features);
      setResult(response);
    } catch (err: any) {
      setApiError(err.message || 'Inference request failed. Please verify backend connection.');
    } finally {
      setIsPredicting(false);
    }
  };

  // Group definitions for structured layout
  const operatingFeatures = FEATURE_DEFINITIONS.filter((f) => f.group === 'operating');
  const tempFeatures = FEATURE_DEFINITIONS.filter((f) => f.group === 'temperature');
  const pressureFeatures = FEATURE_DEFINITIONS.filter((f) => f.group === 'pressure');
  const speedFeatures = FEATURE_DEFINITIONS.filter((f) => f.group === 'speed');
  const bleedFeatures = FEATURE_DEFINITIONS.filter((f) => f.group === 'bleed');

  const rulPercentage = result
    ? Math.min(100, Math.max(0, (result.predicted_rul / result.rul_cap) * 100))
    : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-4xl max-h-[92vh] flex flex-col bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl text-slate-100 overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800/80 bg-slate-950/40">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-sky-500/10 border border-sky-500/20 text-sky-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-white tracking-tight">NASA C-MAPSS FD001 RUL Predictor</h3>
                <span className="px-2 py-0.5 text-[10px] font-mono font-bold rounded bg-sky-500/20 text-sky-400 border border-sky-500/30 uppercase">
                  Random Forest
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Execute single-point RUL inference on NASA C-MAPSS FD001 turbofan run-to-failure telemetry (capped at 125 cycles).
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
            aria-label="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body: Scrollable */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Quick Preset Action Toolbar */}
          <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-950/60 border border-slate-800/80 p-3.5 rounded-xl text-xs">
            <div className="flex items-center gap-2 text-slate-400">
              <Info className="w-4 h-4 text-sky-400 shrink-0" />
              <span>Provide 19 unscaled operational and sensor parameters, or load an authenticated dataset record.</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleLoadSample}
                className="px-3 py-1.5 rounded-lg font-mono text-xs font-semibold bg-sky-500/10 border border-sky-500/30 text-sky-300 hover:bg-sky-500/20 transition-colors flex items-center gap-1.5"
                title="Populate row from NASA C-MAPSS FD001 Engine #1, Cycle 1"
              >
                <FileCheck2 className="w-3.5 h-3.5" />
                <span>Load Verified Unit #1 (Cycle 1)</span>
              </button>
              <button
                type="button"
                onClick={handleReset}
                className="px-3 py-1.5 rounded-lg font-mono text-xs font-semibold bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 transition-colors flex items-center gap-1.5"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset</span>
              </button>
            </div>
          </div>

          {/* Validation & API Error Banners */}
          {validationError && (
            <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-2.5 animate-in fade-in">
              <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold block">Input Validation Advisory:</span>
                <p className="mt-0.5 text-amber-200/90 leading-relaxed">{validationError}</p>
              </div>
            </div>
          )}

          {apiError && (
            <div className="p-3.5 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-xs flex items-start gap-2.5 animate-in fade-in">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold block">Inference Execution Error:</span>
                <p className="mt-0.5 text-red-200/90 leading-relaxed">{apiError}</p>
              </div>
            </div>
          )}

          {/* Form */}
          <form id="fd001-form" onSubmit={handleSubmit} className="space-y-6">
            {/* 1. Operating Profile */}
            <div className="space-y-2.5">
              <div className="flex items-center gap-2 text-xs font-bold font-mono uppercase text-slate-400 tracking-wider">
                <Sliders className="w-3.5 h-3.5 text-accent" />
                <span>1. Operational Profile & Operating Regime</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 bg-slate-950/30 p-3.5 rounded-xl border border-slate-800/60">
                {operatingFeatures.map((f) => (
                  <div key={f.key} className="space-y-1">
                    <label className="text-[11px] font-mono font-medium text-slate-300 flex items-center justify-between">
                      <span>{f.key}</span>
                      <span className="text-[10px] text-slate-500">{f.unit}</span>
                    </label>
                    <input
                      type="text"
                      value={formValues[f.key] || ''}
                      onChange={(e) => handleInputChange(f.key, e.target.value)}
                      placeholder="e.g. 1"
                      className="w-full bg-slate-900 border border-slate-700 focus:border-accent focus:ring-1 focus:ring-accent rounded-lg px-3 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-600 transition"
                      disabled={isPredicting}
                    />
                    <div className="text-[10px] text-slate-500 truncate" title={f.label}>
                      {f.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 2. Compressor & Turbine Temperatures */}
            <div className="space-y-2.5">
              <div className="flex items-center gap-2 text-xs font-bold font-mono uppercase text-slate-400 tracking-wider">
                <Thermometer className="w-3.5 h-3.5 text-rose-400" />
                <span>2. Core Temperatures</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 bg-slate-950/30 p-3.5 rounded-xl border border-slate-800/60">
                {tempFeatures.map((f) => (
                  <div key={f.key} className="space-y-1">
                    <label className="text-[11px] font-mono font-medium text-slate-300 flex items-center justify-between">
                      <span>{f.key}</span>
                      <span className="text-[10px] text-slate-500">{f.unit}</span>
                    </label>
                    <input
                      type="text"
                      value={formValues[f.key] || ''}
                      onChange={(e) => handleInputChange(f.key, e.target.value)}
                      placeholder="e.g. 641.82"
                      className="w-full bg-slate-900 border border-slate-700 focus:border-accent focus:ring-1 focus:ring-accent rounded-lg px-3 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-600 transition"
                      disabled={isPredicting}
                    />
                    <div className="text-[10px] text-slate-500 truncate" title={f.label}>
                      {f.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 3. Pressures & Fuel Flow */}
            <div className="space-y-2.5">
              <div className="flex items-center gap-2 text-xs font-bold font-mono uppercase text-slate-400 tracking-wider">
                <GaugeCircle className="w-3.5 h-3.5 text-amber-400" />
                <span>3. Pressures & Flow Ratios</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 bg-slate-950/30 p-3.5 rounded-xl border border-slate-800/60">
                {pressureFeatures.map((f) => (
                  <div key={f.key} className="space-y-1">
                    <label className="text-[11px] font-mono font-medium text-slate-300 flex items-center justify-between">
                      <span>{f.key}</span>
                      <span className="text-[10px] text-slate-500">{f.unit}</span>
                    </label>
                    <input
                      type="text"
                      value={formValues[f.key] || ''}
                      onChange={(e) => handleInputChange(f.key, e.target.value)}
                      placeholder="e.g. 554.36"
                      className="w-full bg-slate-900 border border-slate-700 focus:border-accent focus:ring-1 focus:ring-accent rounded-lg px-3 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-600 transition"
                      disabled={isPredicting}
                    />
                    <div className="text-[10px] text-slate-500 truncate" title={f.label}>
                      {f.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 4. Rotational Speeds */}
            <div className="space-y-2.5">
              <div className="flex items-center gap-2 text-xs font-bold font-mono uppercase text-slate-400 tracking-wider">
                <Gauge className="w-3.5 h-3.5 text-emerald-400" />
                <span>4. Rotational Speeds (Physical & Corrected)</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 bg-slate-950/30 p-3.5 rounded-xl border border-slate-800/60">
                {speedFeatures.map((f) => (
                  <div key={f.key} className="space-y-1">
                    <label className="text-[11px] font-mono font-medium text-slate-300 flex items-center justify-between">
                      <span>{f.key}</span>
                      <span className="text-[10px] text-slate-500">{f.unit}</span>
                    </label>
                    <input
                      type="text"
                      value={formValues[f.key] || ''}
                      onChange={(e) => handleInputChange(f.key, e.target.value)}
                      placeholder="e.g. 2388.06"
                      className="w-full bg-slate-900 border border-slate-700 focus:border-accent focus:ring-1 focus:ring-accent rounded-lg px-3 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-600 transition"
                      disabled={isPredicting}
                    />
                    <div className="text-[10px] text-slate-500 truncate" title={f.label}>
                      {f.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 5. Bypass & Bleed Enthalpy */}
            <div className="space-y-2.5">
              <div className="flex items-center gap-2 text-xs font-bold font-mono uppercase text-slate-400 tracking-wider">
                <Wind className="w-3.5 h-3.5 text-cyan-400" />
                <span>5. Bypass Ratio & Cooling Bleeds</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 bg-slate-950/30 p-3.5 rounded-xl border border-slate-800/60">
                {bleedFeatures.map((f) => (
                  <div key={f.key} className="space-y-1">
                    <label className="text-[11px] font-mono font-medium text-slate-300 flex items-center justify-between">
                      <span>{f.key}</span>
                      <span className="text-[10px] text-slate-500">{f.unit}</span>
                    </label>
                    <input
                      type="text"
                      value={formValues[f.key] || ''}
                      onChange={(e) => handleInputChange(f.key, e.target.value)}
                      placeholder="e.g. 8.4195"
                      className="w-full bg-slate-900 border border-slate-700 focus:border-accent focus:ring-1 focus:ring-accent rounded-lg px-3 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-600 transition"
                      disabled={isPredicting}
                    />
                    <div className="text-[10px] text-slate-500 truncate" title={f.label}>
                      {f.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </form>

          {/* Results Display */}
          {result && (
            <div className="p-5 rounded-xl bg-slate-950/80 border border-sky-500/30 space-y-4 animate-in fade-in duration-300">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-200">
                    FD001 Model Inference Result
                  </span>
                </div>
                <div className="flex items-center gap-2 font-mono text-[11px]">
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                    Dataset: {result.dataset}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-sky-400 font-bold">
                    {result.algorithm}
                  </span>
                </div>
              </div>

              {/* RUL Main Metric and Visual Gauge */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-xl">
                  <div className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider">
                    Predicted Remaining Useful Life
                  </div>
                  <div className="flex items-baseline gap-2 mt-1.5 font-mono">
                    <span className="text-3xl font-extrabold text-white">
                      {result.predicted_rul.toFixed(1)}
                    </span>
                    <span className="text-sm text-slate-400 font-medium">cycles</span>
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">
                    Target: {result.target} (Capped at {result.rul_cap} cycles)
                  </div>
                </div>

                <div className="md:col-span-2 bg-slate-900/90 border border-slate-800 p-4 rounded-xl space-y-2">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-400 font-medium">Lifecycle Remaining:</span>
                    <span
                      className={`font-bold ${
                        result.predicted_rul > 75
                          ? 'text-emerald-400'
                          : result.predicted_rul > 35
                          ? 'text-amber-400'
                          : 'text-rose-400'
                      }`}
                    >
                      {rulPercentage.toFixed(1)}% of maximum operational threshold
                    </span>
                  </div>
                  <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-slate-700/60">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        result.predicted_rul > 75
                          ? 'bg-gradient-to-r from-emerald-500 to-emerald-400'
                          : result.predicted_rul > 35
                          ? 'bg-gradient-to-r from-amber-500 to-amber-400'
                          : 'bg-gradient-to-r from-rose-500 to-rose-400'
                      }`}
                      style={{ width: `${rulPercentage}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] font-mono text-slate-500">
                    <span>0 cycles (Failure boundary)</span>
                    <span>125 cycles (Upper cap)</span>
                  </div>
                </div>
              </div>

              {/* Model Validation Performance (from API response) */}
              <div className="space-y-2 pt-2 border-t border-slate-800/80">
                <div className="flex items-center justify-between">
                  <div className="text-[11px] font-bold font-mono uppercase text-slate-400 flex items-center gap-1.5">
                    <BarChart3 className="w-3.5 h-3.5 text-accent" />
                    <span>Model Validation Performance Metrics</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {result.evaluation.validation_engines} Val Engines / {result.evaluation.training_engines} Train Engines
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-3 font-mono">
                  <div className="bg-slate-900/70 border border-slate-800/80 p-2.5 rounded-lg text-center">
                    <div className="text-[10px] text-slate-500">Validation MAE</div>
                    <div className="text-sm font-bold text-slate-200 mt-0.5">
                      {result.evaluation.MAE.toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-slate-900/70 border border-slate-800/80 p-2.5 rounded-lg text-center">
                    <div className="text-[10px] text-slate-500">Validation RMSE</div>
                    <div className="text-sm font-bold text-slate-200 mt-0.5">
                      {result.evaluation.RMSE.toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-slate-900/70 border border-slate-800/80 p-2.5 rounded-lg text-center">
                    <div className="text-[10px] text-slate-500">Validation R² Score</div>
                    <div className="text-sm font-bold text-emerald-400 mt-0.5">
                      {result.evaluation.R2.toFixed(4)}
                    </div>
                  </div>
                </div>
                <p className="text-[10px] text-slate-500 leading-normal italic pt-1">
                  * Metrics represent holdout validation performance across 20 test engines in NASA C-MAPSS FD001. They reflect general model validation accuracy, not an individual cycle's confidence interval.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between p-5 border-t border-slate-800 bg-slate-950/60">
          <div className="text-[11px] text-slate-500 font-mono">
            {FEATURE_DEFINITIONS.length} features required • Raw unscaled inference
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-xs font-semibold text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-700 transition"
              disabled={isPredicting}
            >
              Close
            </button>
            <button
              type="submit"
              form="fd001-form"
              disabled={isPredicting}
              className="px-5 py-2 rounded-lg text-xs font-semibold font-mono bg-sky-500 hover:bg-sky-400 text-slate-950 flex items-center gap-2 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-sky-500/20"
            >
              {isPredicting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Computing RUL...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Execute Prediction</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

