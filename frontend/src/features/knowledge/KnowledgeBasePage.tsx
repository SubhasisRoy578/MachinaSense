import { useCallback, useEffect, useState, useRef } from 'react';
import { api } from '../../services/api';
import type { KnowledgeDocument, RetrievalResult } from '../../types/models';
import { Card, CardHeader, CardTitle, CardContent } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { BookOpen, Search, UploadCloud, FileText, File, FileType, CheckCircle2, Clock, AlertCircle, X } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt'];
const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25 MB

export function KnowledgeBasePage() {
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Search State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<RetrievalResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // Upload State
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    try {
      const docs = await api.knowledge.listDocuments();
      setDocuments(docs);
    } catch (error) {
      console.error("Failed to fetch documents", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const requestId = window.setTimeout(() => {
      void fetchDocs();
    }, 0);
    return () => window.clearTimeout(requestId);
  }, [fetchDocs]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    setHasSearched(true);
    try {
      const results = await api.knowledge.search(searchQuery);
      setSearchResults(results);
    } catch (error) {
      console.error("Search failed", error);
    } finally {
      setIsSearching(false);
    }
  };

  const validateFile = (file: File): string | null => {
    const ext = '.' + (file.name.split('.').pop()?.toLowerCase() || '');
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      return `Unsupported file type "${ext}". Only PDF, DOCX, and TXT files are allowed.`;
    }
    if (file.size === 0) {
      return 'File is empty (0 bytes).';
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File size (${(file.size / 1024 / 1024).toFixed(1)} MB) exceeds the 25 MB limit.`;
    }
    return null;
  };

  const handleFileUpload = async (file: File) => {
    setUploadError(null);
    setUploadSuccess(null);

    const validationError = validateFile(file);
    if (validationError) {
      setUploadError(validationError);
      return;
    }

    setIsUploading(true);
    try {
      const result = await api.knowledge.upload(file, 'Manual');
      setUploadSuccess(`Successfully uploaded "${result.title}" — ${result.indexedChunks} chunks indexed.`);
      await fetchDocs(); // Refresh document list
    } catch (error: any) {
      setUploadError(error.message || 'Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
    // Reset input so the same file can be re-selected
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const readyCount = documents.filter(d => d.processingStatus === 'ready').length;
  const totalChunks = documents.reduce((acc, doc) => acc + (doc.indexedChunks || 0), 0);

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500 pb-12">
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-primary flex items-center gap-2">
            <BookOpen className="w-6 h-6 text-accent" /> Technical Knowledge Base
          </h1>
          <p className="text-sm text-slate-400 mt-1">Indexed machine manuals, specifications, and maintenance logs for AI RAG retrieval.</p>
        </div>
        <button 
          onClick={handleUploadClick}
          disabled={isUploading}
          className="inline-flex items-center justify-center px-4 py-2 bg-accent/10 border border-accent/30 text-accent rounded-md hover:bg-accent/20 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isUploading ? (
            <><Clock className="w-4 h-4 mr-2 animate-spin" /> Processing...</>
          ) : (
            <><UploadCloud className="w-4 h-4 mr-2" /> Upload Document</>
          )}
        </button>
        <input 
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleFileSelect}
          className="hidden"
        />
      </div>

      {/* Upload Feedback */}
      {uploadError && (
        <div className="bg-red-950/30 border border-red-800/60 rounded-lg px-4 py-3 flex items-start gap-3 animate-in fade-in duration-300">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="text-sm font-medium text-red-300">Upload Failed</div>
            <div className="text-xs text-red-400 mt-0.5">{uploadError}</div>
          </div>
          <button onClick={() => setUploadError(null)} className="text-red-500 hover:text-red-300">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
      {uploadSuccess && (
        <div className="bg-emerald-950/30 border border-emerald-800/60 rounded-lg px-4 py-3 flex items-start gap-3 animate-in fade-in duration-300">
          <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="text-sm font-medium text-emerald-300">Upload Successful</div>
            <div className="text-xs text-emerald-400 mt-0.5">{uploadSuccess}</div>
          </div>
          <button onClick={() => setUploadSuccess(null)} className="text-emerald-500 hover:text-emerald-300">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Drag and Drop Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`border-2 border-dashed rounded-lg p-6 text-center transition-all duration-200 cursor-pointer ${
          isDragOver 
            ? 'border-accent bg-accent/10 shadow-[0_0_20px_rgba(168,85,247,0.1)]' 
            : 'border-slate-700 bg-slate-900/30 hover:border-slate-600 hover:bg-slate-900/50'
        }`}
        onClick={handleUploadClick}
      >
        <UploadCloud className={`w-8 h-8 mx-auto mb-2 ${isDragOver ? 'text-accent' : 'text-slate-500'}`} />
        <p className={`text-sm font-medium ${isDragOver ? 'text-accent' : 'text-slate-400'}`}>
          {isDragOver ? 'Drop file to upload' : 'Drag & drop a document here, or click to browse'}
        </p>
        <p className="text-xs text-slate-500 mt-1">
          Supported: PDF, DOCX, TXT — Max 25 MB
        </p>
      </div>

      {/* KPI Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-l-4 border-l-slate-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2">Total Documents</div>
            <div className="text-3xl font-mono text-primary">{documents.length}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-emerald-500">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500" /> Processed & Ready
            </div>
            <div className="text-3xl font-mono text-emerald-400">{readyCount}</div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-accent">
          <CardContent className="py-5">
            <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-2">Total Vector Chunks</div>
            <div className="text-3xl font-mono text-accent">{totalChunks.toLocaleString()}</div>
          </CardContent>
        </Card>
      </div>

      {/* RAG Search Interface */}
      <Card className="border-accent/30 bg-accent/5">
        <CardHeader>
          <CardTitle className="text-accent flex items-center gap-2">
            <Search className="w-4 h-4" /> TF-IDF + Cosine Similarity Vector Retrieval
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSearch} className="flex gap-3 mb-6">
            <input 
              type="text" 
              placeholder="e.g., What could cause increasing vibration in a spindle bearing?" 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 bg-slate-950 border border-slate-700 rounded-md px-4 py-2 text-sm focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent text-primary transition-shadow"
            />
            <button 
              type="submit"
              disabled={isSearching || !searchQuery.trim()}
              className="px-6 py-2 bg-accent text-slate-950 rounded-md font-medium text-sm hover:bg-accent/90 transition-colors disabled:opacity-50"
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
          </form>

          {/* Search Results */}
          {hasSearched && (
            <div className="space-y-4 mt-4">
              <div className="text-xs font-bold text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-2">
                Retrieval Results
              </div>
              
              {isSearching ? (
                <div className="text-sm text-slate-500 font-mono py-8 text-center animate-pulse">
                  [QUERYING_TFIDF_VECTOR_INDEX...]
                </div>
              ) : searchResults.length === 0 ? (
                <div className="text-sm text-slate-500 italic py-8 text-center bg-slate-900/50 rounded border border-slate-800">
                  No relevant technical documents found for this query.
                </div>
              ) : (
                <div className="space-y-4">
                  {searchResults.map((result, idx) => (
                    <div key={idx} className="bg-slate-900 border border-slate-700 rounded p-4">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-accent" />
                          <span className="text-sm font-medium text-primary">{result.documentTitle}</span>
                        </div>
                        <Badge variant="outline" className="text-xs font-mono">Rel: {result.relevance.toFixed(2)}</Badge>
                      </div>
                      <div className="text-xs text-slate-500 mb-3 font-mono border-b border-slate-800 pb-2">
                        Section: {result.section || 'General'} 
                        {result.page != null && <><span className="mx-2">•</span> Page: {result.page}</>}
                        <span className="mx-2">•</span> ID: {result.documentId}
                      </div>
                      <p className="text-sm text-slate-300 leading-relaxed italic border-l-2 border-accent/50 pl-3">
                        "{result.snippet}"
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Document Library */}
      <Card>
        <CardHeader>
          <CardTitle>Indexed Document Library</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-slate-400 uppercase bg-slate-950/50 border-b border-slate-800">
                <tr>
                  <th className="px-6 py-4 font-medium tracking-wider">Document</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Category</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Size</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Status</th>
                  <th className="px-6 py-4 font-medium tracking-wider text-right">Added</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {loading ? (
                  <tr>
                    <td colSpan={5} className="px-6 py-12 text-center text-slate-500 font-mono">
                      [LOADING_DOCUMENT_INDEX...]
                    </td>
                  </tr>
                ) : documents.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-6 py-12 text-center text-slate-500 italic">
                      No documents in the knowledge base.
                    </td>
                  </tr>
                ) : (
                  documents.map((doc) => (
                    <tr key={doc.id} className="hover:bg-slate-800/20 transition-colors group">
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="flex items-center justify-center w-8 h-8 rounded bg-slate-900 border border-slate-700">
                            {doc.type === 'PDF' ? <File className="w-4 h-4 text-red-400" /> : 
                             doc.type === 'DOCX' ? <FileText className="w-4 h-4 text-blue-400" /> : 
                             <FileType className="w-4 h-4 text-slate-400" />}
                          </div>
                          <div>
                            <div className="font-medium text-primary mb-1">{doc.title}</div>
                            <div className="text-xs text-slate-500 font-mono">ID: {doc.id}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-slate-300">{doc.sourceCategory}</td>
                      <td className="px-6 py-4 text-slate-400 font-mono text-xs">{(doc.size / 1024 / 1024).toFixed(1)} MB</td>
                      <td className="px-6 py-4">
                        <Badge variant={
                          doc.processingStatus === 'ready' ? 'healthy' : 
                          doc.processingStatus === 'failed' ? 'critical' : 'warning'
                        }>
                          {doc.processingStatus.toUpperCase()}
                        </Badge>
                        {doc.indexedChunks && (
                          <div className="text-xs text-slate-500 mt-1 font-mono">
                            {doc.indexedChunks} chunks
                          </div>
                        )}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="text-slate-400">{formatDistanceToNow(new Date(doc.uploadDate), { addSuffix: true })}</div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
