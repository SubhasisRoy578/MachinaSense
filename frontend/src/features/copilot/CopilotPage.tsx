import { useEffect, useState, useRef } from 'react';
import type { FormEvent } from 'react';
import { 
  Send, BrainCircuit, Cpu, Sparkles, AlertTriangle, 
  Layers, ShieldAlert, Plus, Trash2, Bot, User as UserIcon, BookOpen, Clock
} from 'lucide-react';
import { api } from '../../services/api';
import type { Machine } from '../../types/models';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';

export function CopilotPage() {
  const [conversations, setConversations] = useState<any[]>([]);
  const [active, setActive] = useState<any>(null);
  const [machines, setMachines] = useState<Machine[]>([]);
  const [selectedMachineId, setSelectedMachineId] = useState<string>('');
  const [text, setText] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // 1. Fetch user's real machines
    api.machines.list()
      .then(mList => setMachines(mList))
      .catch(err => console.error("Failed to load user machines for copilot:", err));

    // 2. Fetch conversations
    api.copilot.conversations()
      .then(convos => {
        setConversations(convos);
        if (convos.length > 0) {
          api.copilot.get(convos[0].id)
            .then(fullConvo => {
              setActive(fullConvo);
              setSelectedMachineId(fullConvo.machineId || '');
            })
            .catch(err => setError(err.message));
        }
      })
      .catch(err => setError(err.message));
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [active?.messages, busy]);

  const activeMachine = machines.find(m => m.id === (active?.machineId || selectedMachineId));

  async function handleStartNewConversation() {
    setError('');
    try {
      const mid = selectedMachineId || undefined;
      const title = mid ? `Investigation: ${mid}` : "General Engineering Session";
      const newConvo = await api.copilot.create(mid, title);
      setConversations(prev => [newConvo, ...prev]);
      setActive({ ...newConvo, messages: [] });
    } catch (e: any) {
      setError(e.message || "Failed to create new conversation.");
    }
  }

  async function handleSelectConversation(id: string) {
    setError('');
    try {
      const full = await api.copilot.get(id);
      setActive(full);
      setSelectedMachineId(full.machineId || '');
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function handleDeleteConversation(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    try {
      await api.copilot.delete(id);
      const remaining = conversations.filter(c => c.id !== id);
      setConversations(remaining);
      if (active?.id === id) {
        if (remaining.length > 0) {
          const next = await api.copilot.get(remaining[0].id);
          setActive(next);
          setSelectedMachineId(next.machineId || '');
        } else {
          setActive(null);
        }
      }
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function handleMachineChange(newMid: string) {
    setSelectedMachineId(newMid);
    if (!active) return;
    try {
      const updated = await api.copilot.update(active.id, { machine_id: newMid || null });
      setActive((prev: any) => ({ ...prev, machineId: updated.machineId }));
      setConversations(items => items.map(c => c.id === active.id ? { ...c, machineId: updated.machineId } : c));
    } catch (e: any) {
      setError(e.message || "Failed to update conversation asset.");
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const query = text.trim();
    if (!query) return;

    if (!active) {
      // Auto-start a conversation if none exists
      try {
        const mid = selectedMachineId || undefined;
        const title = mid ? `Investigation: ${mid}` : "General Engineering Session";
        const newConvo = await api.copilot.create(mid, title);
        setConversations(prev => [newConvo, ...prev]);
        const fullNew = { ...newConvo, messages: [{ sender: 'user', content: query }] };
        setActive(fullNew);
        setText('');
        setBusy(true);
        setError('');
        const result = await api.copilot.send(newConvo.id, query, mid || null);
        setActive((prev: any) => ({
          ...prev,
          messages: [
            ...prev.messages,
            { sender: 'copilot', content: result.content, provider_used: result.provider_used, fallback_level: result.fallback_level, evidence: result.evidence || [] }
          ]
        }));
      } catch (err: any) {
        setError(err.message);
      } finally {
        setBusy(false);
      }
      return;
    }

    setBusy(true);
    setError('');
    const userMsg = { sender: 'user', content: query };
    setActive((prev: any) => ({
      ...prev,
      messages: [...(prev.messages || []), userMsg]
    }));
    setText('');

    try {
      const result = await api.copilot.send(active.id, query, selectedMachineId || null);
      setActive((prev: any) => ({
        ...prev,
        messages: [
          ...prev.messages,
          { 
            sender: 'copilot', 
            content: result.content, 
            provider_used: result.provider_used, 
            fallback_level: result.fallback_level,
            evidence: result.evidence || []
          }
        ]
      }));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const promptSuggestions = activeMachine ? [
    "Why is this machine degrading?",
    "What is this machine's RUL?",
    "What anomalies were detected?",
    "What maintenance should be investigated?"
  ] : [
    "What is predictive maintenance?",
    "Explain RUL",
    "Explain anomaly detection",
    "What can you do?"
  ];

  return (
    <div className="flex flex-col gap-5 h-[calc(100vh-7.5rem)] animate-in fade-in duration-500 pb-2">
      {/* Top Header & Machine Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-primary flex items-center gap-2">
            <BrainCircuit className="w-6 h-6 text-accent" />
            Engineering Copilot
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Grounded diagnostic assistance integrating real-time telemetry, model forecasts, and technical documentation.
          </p>
        </div>

        {/* Machine Selector */}
        <div className="flex items-center gap-3 bg-slate-900/90 border border-slate-700/80 p-2 rounded-lg">
          <Cpu className="w-4 h-4 text-accent shrink-0" />
          <span className="text-xs font-mono text-slate-400 whitespace-nowrap">Target Asset:</span>
          <select
            value={selectedMachineId}
            onChange={(e) => handleMachineChange(e.target.value)}
            className="bg-slate-950 border border-slate-700 rounded px-2.5 py-1 text-xs text-primary font-medium focus:outline-none focus:border-accent"
          >
            <option value="">🌐 General Assistant (No Machine Selected)</option>
            {machines.map(m => (
              <option key={m.id} value={m.id}>
                ⚙️ {m.name} ({m.id})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Grid: Sidebar + Chat Stream */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 flex-1 min-h-0">
        
        {/* Left Column: Sessions List */}
        <Card className="md:col-span-1 flex flex-col min-h-0 bg-slate-900/40 border-slate-800">
          <CardHeader className="py-3 px-4 border-b border-slate-800 flex flex-row items-center justify-between">
            <CardTitle className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-slate-500" /> Sessions
            </CardTitle>
            <button
              onClick={handleStartNewConversation}
              className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-1 bg-accent text-slate-950 rounded hover:bg-accent/90 transition-colors"
            >
              <Plus className="w-3.5 h-3.5" /> New
            </button>
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto p-2 space-y-1.5 custom-scrollbar">
            {conversations.length === 0 ? (
              <div className="p-4 text-center text-xs text-slate-500 font-mono">
                No conversations yet. Click "New" to start.
              </div>
            ) : (
              conversations.map(c => {
                const isSelected = active?.id === c.id;
                return (
                  <div
                    key={c.id}
                    onClick={() => handleSelectConversation(c.id)}
                    className={`group relative flex items-center justify-between p-2.5 rounded-md cursor-pointer transition-all text-xs ${
                      isSelected
                        ? 'bg-slate-800 border border-slate-700 text-primary shadow-sm'
                        : 'hover:bg-slate-800/40 text-slate-400 border border-transparent'
                    }`}
                  >
                    <div className="flex-1 min-w-0 pr-2">
                      <div className="font-medium truncate text-primary">{c.title || 'Engineering Session'}</div>
                      <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                        {c.machineId ? `Asset: ${c.machineId}` : 'General Inquiry'}
                      </div>
                    </div>
                    <button
                      onClick={(e) => handleDeleteConversation(e, c.id)}
                      className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-red-400 transition-opacity"
                      title="Delete conversation"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })
            )}
          </CardContent>
        </Card>

        {/* Right Column: Chat Stream Area */}
        <Card className="md:col-span-3 flex flex-col min-h-0 bg-slate-900/30 border-slate-800">
          
          {/* Active Context Banner */}
          <div className="px-4 py-2.5 border-b border-slate-800 bg-slate-950/60 flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2">
              <Bot className="w-4 h-4 text-accent" />
              {activeMachine ? (
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-primary">{activeMachine.name}</span>
                  <span className="text-slate-500 font-mono">({activeMachine.id})</span>
                  <Badge variant={activeMachine.status as any} className="text-[10px] py-0 px-2">
                    {activeMachine.status.toUpperCase()}
                  </Badge>
                  <span className="text-slate-400 font-mono text-[11px] ml-1">
                    RUL: {activeMachine.rulDays || 0}d
                  </span>
                </div>
              ) : (
                <span className="text-slate-300 font-medium">
                  General Domain Assistant — Concept Explanations & Knowledge Base Search
                </span>
              )}
            </div>

            <div className="text-[10px] font-mono text-slate-500 flex items-center gap-2">
              <span>Primary: Gemini 2.5</span>
              <span>•</span>
              <span>Secondary: Grok 3</span>
              <span>•</span>
              <span>Fallback: Deterministic RAG</span>
            </div>
          </div>

          {/* Messages Stream */}
          <CardContent className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
            {!active && (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-400">
                <BrainCircuit className="w-12 h-12 text-slate-600 mb-3" />
                <h3 className="text-base font-semibold text-slate-200">Start a Grounded Engineering Session</h3>
                <p className="text-xs text-slate-400 max-w-md mt-1 mb-4">
                  Select a machine to examine real-time sensor trends and degradation signals, or ask general questions about predictive maintenance principles.
                </p>
                <button
                  onClick={handleStartNewConversation}
                  className="px-4 py-2 bg-accent text-slate-950 rounded text-xs font-semibold hover:bg-accent/90 transition-colors"
                >
                  Start New Session
                </button>
              </div>
            )}

            {active?.messages?.map((m: any, idx: number) => {
              const isUser = m.sender === 'user';
              return (
                <div
                  key={idx}
                  className={`flex gap-3 max-w-3xl ${isUser ? 'ml-auto flex-row-reverse' : 'mr-auto'}`}
                >
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 text-xs font-bold ${
                    isUser ? 'bg-accent/20 border border-accent/40 text-accent' : 'bg-slate-800 border border-slate-700 text-slate-300'
                  }`}>
                    {isUser ? <UserIcon className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5 text-accent" />}
                  </div>

                  <div className={`rounded-lg p-3.5 text-xs leading-relaxed ${
                    isUser
                      ? 'bg-accent/15 border border-accent/30 text-slate-100 rounded-tr-none'
                      : 'bg-slate-950 border border-slate-800 text-slate-200 rounded-tl-none shadow-sm'
                  }`}>
                    {/* Metadata Header for Copilot messages */}
                    {!isUser && (
                      <div className="flex items-center gap-2 mb-2 pb-1.5 border-b border-slate-800/80">
                        {m.provider_used === 'gemini' && (
                          <Badge variant="outline" className="bg-emerald-950/80 text-emerald-400 border-emerald-800 text-[10px] font-mono flex items-center gap-1">
                            <Sparkles className="w-3 h-3" /> GEMINI (PRIMARY)
                          </Badge>
                        )}
                        {m.provider_used === 'grok' && (
                          <Badge variant="outline" className="bg-sky-950/80 text-sky-400 border-sky-800 text-[10px] font-mono flex items-center gap-1">
                            <Layers className="w-3 h-3" /> GROK (SECONDARY)
                          </Badge>
                        )}
                        {m.provider_used === 'grounded_fallback' && (
                          <Badge variant="outline" className="bg-amber-950/80 text-amber-400 border-amber-800 text-[10px] font-mono flex items-center gap-1">
                            <ShieldAlert className="w-3 h-3" /> GROUNDED DETERMINISTIC
                          </Badge>
                        )}
                        {m.provider_used === 'ml_only' && (
                          <Badge variant="outline" className="bg-purple-950/80 text-purple-400 border-purple-800 text-[10px] font-mono flex items-center gap-1">
                            <Cpu className="w-3 h-3" /> ML TELEMETRY ONLY
                          </Badge>
                        )}
                        {m.provider_used === 'unavailable' && (
                          <Badge variant="outline" className="bg-slate-900 text-slate-400 border-slate-700 text-[10px] font-mono flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" /> DATA UNAVAILABLE
                          </Badge>
                        )}

                        {m.fallback_level != null && (
                          <span className="text-[10px] font-mono text-slate-500">
                            Tier {m.fallback_level}
                          </span>
                        )}
                      </div>
                    )}

                    <div className="whitespace-pre-wrap">{m.content}</div>

                    {/* Evidence chunk previews if attached */}
                    {!isUser && m.evidence && Array.isArray(m.evidence) && m.evidence.length > 0 && (
                      <div className="mt-3 pt-2 border-t border-slate-800/80">
                        <div className="text-[10px] font-mono text-slate-400 font-semibold mb-1 flex items-center gap-1">
                          <BookOpen className="w-3 h-3 text-accent" /> Retrieved Technical Citations:
                        </div>
                        <ul className="space-y-1">
                          {m.evidence.map((ev: any, evIdx: number) => (
                            <li key={evIdx} className="text-[11px] font-mono text-slate-400 bg-slate-900/80 p-1.5 rounded border border-slate-800">
                              • {ev.documentTitle} (Section: {ev.section || 'General'}, p.{ev.page})
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {busy && (
              <div className="flex gap-3 max-w-xl mr-auto">
                <div className="w-7 h-7 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0">
                  <Bot className="w-3.5 h-3.5 text-accent animate-pulse" />
                </div>
                <div className="rounded-lg p-3 bg-slate-950 border border-slate-800 text-slate-400 text-xs flex items-center gap-2 font-mono">
                  <div className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                  Synthesizing grounded engineering findings...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </CardContent>

          {/* Quick Suggestions & Input Footer */}
          <div className="p-3 border-t border-slate-800 bg-slate-950/80 space-y-2">
            
            {/* Quick Suggestions Pills */}
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-[11px]">
              <span className="text-slate-500 font-mono shrink-0">Prompts:</span>
              {promptSuggestions.map((promptText, pIdx) => (
                <button
                  key={pIdx}
                  type="button"
                  onClick={() => setText(promptText)}
                  className="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 transition-colors whitespace-nowrap"
                >
                  {promptText}
                </button>
              ))}
            </div>

            {/* Input Form */}
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                value={text}
                onChange={e => setText(e.target.value)}
                disabled={busy}
                placeholder={
                  activeMachine
                    ? `Ask about ${activeMachine.name} (RUL, anomalies, sensor trends, maintenance)...`
                    : "Ask a general predictive maintenance question, or select a machine above..."
                }
                className="flex-1 bg-slate-900 border border-slate-700 focus:border-accent rounded px-3 py-2 text-xs text-primary placeholder:text-slate-500 focus:outline-none"
              />
              <button
                type="submit"
                disabled={!text.trim() || busy}
                className="px-4 py-2 bg-accent text-slate-950 font-semibold rounded text-xs hover:bg-accent/90 disabled:opacity-50 transition-colors flex items-center gap-1.5"
              >
                <Send className="w-3.5 h-3.5" />
                <span>Send</span>
              </button>
            </form>

            {error && (
              <div className="text-[11px] text-red-400 font-mono px-1 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3 shrink-0" />
                {error}
              </div>
            )}
          </div>

        </Card>

      </div>
    </div>
  );
}
