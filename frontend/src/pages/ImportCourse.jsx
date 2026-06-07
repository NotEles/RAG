import React, { useState, useEffect, useRef } from 'react';
import { apiBaseUrl as API_BASE_URL } from '../config/config';

const EMBEDDING_MODELS = {
  huggingface: ['BAAI/bge-small-zh-v1.5', 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
  openai: ['text-embedding-3-small', 'text-embedding-3-large'],
};

const CHUNKING_METHODS = [
  { value: 'by_pages', label: '按页分块' },
  { value: 'fixed_size', label: '固定大小' },
  { value: 'by_paragraphs', label: '按段落' },
  { value: 'by_sentences', label: '按句子' },
  { value: 'semantic', label: '语义分块' },
];

const LOADING_METHODS = [
  { value: 'pymupdf', label: 'PyMuPDF' },
  { value: 'pypdf', label: 'PyPDF' },
  { value: 'pdfplumber', label: 'PDFPlumber' },
  { value: 'unstructured', label: 'Unstructured' },
];

const SUPPORTED_EXTENSIONS = ['.pdf', '.json', '.jsonl', '.docx', '.md', '.markdown', '.txt', '.csv', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'];
const ACCEPTED_FILE_TYPES = SUPPORTED_EXTENSIONS.join(',');

export default function ImportCourse() {
  const [file, setFile] = useState(null);
  const [loadingMethod, setLoadingMethod] = useState('pymupdf');
  const [chunkingMethod, setChunkingMethod] = useState('by_pages');
  const [chunkSize, setChunkSize] = useState(1000);
  const [minQuality, setMinQuality] = useState(0);
  const [embeddingProvider, setEmbeddingProvider] = useState('huggingface');
  const [embeddingModel, setEmbeddingModel] = useState('BAAI/bge-small-zh-v1.5');
  const [apiKey, setApiKey] = useState('');
  const [importing, setImporting] = useState(false);
  const [logs, setLogs] = useState([]);
  const [collectionInfo, setCollectionInfo] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  const isPdf = file && file.name.toLowerCase().endsWith('.pdf');
  const isJson = file && file.name.toLowerCase().endsWith('.json');
  const isSupportedFile = file && SUPPORTED_EXTENSIONS.some(ext => file.name.toLowerCase().endsWith(ext));

  useEffect(() => {
    setEmbeddingModel(EMBEDDING_MODELS[embeddingProvider][0]);
  }, [embeddingProvider]);

  useEffect(() => {
    fetchCollectionInfo();
  }, []);

  const fetchCollectionInfo = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/course/info`);
      const data = await res.json();
      setCollectionInfo(data);
    } catch {
      setCollectionInfo(null);
    }
  };

  const addLog = (msg, type = 'info') => {
    setLogs(prev => [...prev, { msg, type, time: new Date().toLocaleTimeString() }]);
  };

  const handleFileDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  };

  const handleImport = async () => {
    if (!file) return;
    if (!isSupportedFile) {
      addLog('请上传支持的文档格式', 'error');
      return;
    }
    setImporting(true);
    addLog(`开始导入: ${file.name}`, 'info');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('loading_method', loadingMethod);
      formData.append('chunking_method', chunkingMethod);
      formData.append('chunk_size', chunkSize);
      formData.append('embedding_provider', embeddingProvider);
      formData.append('embedding_model', embeddingModel);
      if (apiKey) formData.append('api_key', apiKey);
      formData.append('min_quality', minQuality);

      const res = await fetch(`${API_BASE_URL}/import-to-course`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || '导入失败');
      }

      const data = await res.json();
      addLog(`✓ ${data.message}`, 'success');
      if (data.collection_info) {
        setCollectionInfo(data.collection_info);
      }
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      await fetchCollectionInfo();
    } catch (e) {
      addLog(`✗ 错误: ${e.message}`, 'error');
    } finally {
      setImporting(false);
    }
  };

  const handleClearCollection = async () => {
    if (!window.confirm('确定要清空 course 数据库吗？此操作不可撤销。')) return;
    try {
      const res = await fetch(`${API_BASE_URL}/collections/chroma/course`, { method: 'DELETE' });
      if (res.ok) {
        addLog('course 数据库已清空', 'info');
        await fetchCollectionInfo();
      }
    } catch (e) {
      addLog(`清空失败: ${e.message}`, 'error');
    }
  };

  return (
    <div className="p-6 h-screen flex flex-col">
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-gray-800">文件导入</h1>
        <p className="text-gray-500 text-sm mt-1">将文件导入到 <span className="font-mono bg-gray-100 px-1 rounded">course</span> 知识库</p>
      </div>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Left: Controls */}
        <div className="w-80 shrink-0 space-y-4 overflow-y-auto">
          {/* File upload */}
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold text-gray-700 mb-3">上传文件</h2>
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleFileDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
                dragging ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-blue-300 hover:bg-gray-50'
              }`}
            >
              <div className="text-3xl mb-1">{file ? '📄' : '📂'}</div>
              {file ? (
                <p className="text-sm text-gray-700 font-medium break-all">{file.name}</p>
              ) : (
                <p className="text-sm text-gray-400">拖拽或点击上传不同格式的文件</p>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPTED_FILE_TYPES}
                className="hidden"
                onChange={(e) => setFile(e.target.files[0])}
              />
            </div>
          </div>

          {/* Loading method (PDF only) */}
          {isPdf && (
            <div className="bg-white rounded-lg shadow p-4">
              <h2 className="font-semibold text-gray-700 mb-3">加载方式</h2>
              <div className="grid grid-cols-2 gap-2">
                {LOADING_METHODS.map(m => (
                  <button
                    key={m.value}
                    onClick={() => setLoadingMethod(m.value)}
                    className={`text-xs py-2 px-3 rounded border transition-colors ${
                      loadingMethod === m.value
                        ? 'bg-blue-500 text-white border-blue-500'
                        : 'bg-white text-gray-600 border-gray-300 hover:border-blue-300'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Chunking method */}
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold text-gray-700 mb-3">分块方式</h2>
            <div className="space-y-1">
              {CHUNKING_METHODS.map(m => (
                <label key={m.value} className="flex items-center gap-2 cursor-pointer py-1 px-2 rounded hover:bg-gray-50">
                  <input
                    type="radio"
                    name="chunking"
                    value={m.value}
                    checked={chunkingMethod === m.value}
                    onChange={() => setChunkingMethod(m.value)}
                    className="accent-blue-500"
                  />
                  <span className="text-sm text-gray-700">{m.label}</span>
                </label>
              ))}
            </div>
            {chunkingMethod === 'fixed_size' && (
              <div className="mt-3">
                <label className="text-xs text-gray-500">块大小（字符数）</label>
                <input
                  type="number"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  className="mt-1 w-full border rounded px-2 py-1 text-sm"
                  min={100}
                  max={10000}
                />
              </div>
            )}
          </div>

          {/* QA 质量过滤（仅 JSON 问答格式） */}
          {isJson && (
            <div className="bg-white rounded-lg shadow p-4">
              <h2 className="font-semibold text-gray-700 mb-3">问答质量过滤</h2>
              <p className="text-xs text-gray-400 mb-3">
                仅对 QA 格式 JSON 生效。每个问题的每条回答都会生成独立向量，此处过滤掉低于设定分数的答案（0 = 保留全部）。
              </p>
              <div className="flex items-center gap-3">
                <input
                  type="range" min={0} max={9} step={1} value={minQuality}
                  onChange={(e) => setMinQuality(Number(e.target.value))}
                  className="flex-1 accent-blue-500"
                />
                <span className="text-sm font-medium text-blue-500 w-6 text-right">{minQuality}</span>
              </div>
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>0（全部）</span>
                <span>9（最高）</span>
              </div>
            </div>
          )}

          {/* Embedding */}
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold text-gray-700 mb-3">编码方式</h2>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-500">提供商</label>
                <select
                  value={embeddingProvider}
                  onChange={(e) => setEmbeddingProvider(e.target.value)}
                  className="mt-1 w-full border rounded px-2 py-1.5 text-sm bg-white"
                >
                  <option value="huggingface">HuggingFace（本地）</option>
                  <option value="openai">OpenAI</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-500">模型</label>
                <select
                  value={embeddingModel}
                  onChange={(e) => setEmbeddingModel(e.target.value)}
                  className="mt-1 w-full border rounded px-2 py-1.5 text-sm bg-white"
                >
                  {EMBEDDING_MODELS[embeddingProvider].map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
              {embeddingProvider === 'openai' && (
                <div>
                  <label className="text-xs text-gray-500">API Key</label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-..."
                    className="mt-1 w-full border rounded px-2 py-1.5 text-sm font-mono"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Import button */}
          <button
            onClick={handleImport}
            disabled={!file || importing}
            className={`w-full py-3 rounded-lg font-semibold text-white transition-colors ${
              !file || importing
                ? 'bg-gray-300 cursor-not-allowed'
                : 'bg-blue-500 hover:bg-blue-600 active:bg-blue-700'
            }`}
          >
            {importing ? '导入中...' : '开始导入'}
          </button>
        </div>

        {/* Right: Info + Logs */}
        <div className="flex-1 flex flex-col gap-4 min-h-0 min-w-0">
          {/* Collection info */}
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-semibold text-gray-700">
                course 知识库状态
              </h2>
              <div className="flex gap-2">
                <button
                  onClick={fetchCollectionInfo}
                  className="text-xs px-3 py-1 border rounded text-gray-600 hover:bg-gray-50"
                >
                  刷新
                </button>
                <button
                  onClick={handleClearCollection}
                  className="text-xs px-3 py-1 border border-red-200 rounded text-red-500 hover:bg-red-50"
                >
                  清空数据库
                </button>
              </div>
            </div>
            {collectionInfo ? (
              <div className="flex gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-500">{collectionInfo.num_entities ?? 0}</div>
                  <div className="text-xs text-gray-500 mt-1">已存储向量</div>
                </div>
                {collectionInfo.schema && Object.keys(collectionInfo.schema).length > 0 && (
                  <div className="text-sm text-gray-600">
                    <div className="text-xs text-gray-400 mb-1">最近文件信息</div>
                    {collectionInfo.schema.document_name && (
                      <div>文档: <span className="font-mono">{collectionInfo.schema.document_name}</span></div>
                    )}
                    {collectionInfo.schema.embedding_model && (
                      <div>模型: {collectionInfo.schema.embedding_model}</div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-gray-400">数据库为空或尚未创建</p>
            )}
          </div>

          {/* Logs */}
          <div className="flex-1 bg-white rounded-lg shadow p-4 flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-semibold text-gray-700">导入日志</h2>
              {logs.length > 0 && (
                <button onClick={() => setLogs([])} className="text-xs text-gray-400 hover:text-gray-600">
                  清除
                </button>
              )}
            </div>
            <div className="flex-1 overflow-y-auto space-y-1 min-h-0">
              {logs.length === 0 ? (
                <p className="text-sm text-gray-400">导入后日志将显示在这里</p>
              ) : (
                logs.map((log, i) => (
                  <div
                    key={i}
                    className={`text-sm px-3 py-1.5 rounded font-mono ${
                      log.type === 'success'
                        ? 'bg-green-50 text-green-700'
                        : log.type === 'error'
                        ? 'bg-red-50 text-red-600'
                        : 'bg-gray-50 text-gray-600'
                    }`}
                  >
                    <span className="text-gray-400 text-xs mr-2">{log.time}</span>
                    {log.msg}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
