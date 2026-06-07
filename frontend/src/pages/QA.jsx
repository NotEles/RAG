import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiBaseUrl as API_BASE_URL } from '../config/config';

const PROVIDERS = [
  { value: 'deepseek', label: 'DeepSeek' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'aliyun', label: '阿里云百炼' },
  { value: 'huggingface', label: '本地模型' },
];

const MODELS = {
  deepseek: [
    { value: 'deepseek-v3', label: 'DeepSeek V3' },
    { value: 'deepseek-r1', label: 'DeepSeek R1（含推理）' },
  ],
  openai: [
    { value: 'gpt-4o', label: 'GPT-4o' },
    { value: 'gpt-4o-mini', label: 'GPT-4o mini' },
    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' },
  ],
  aliyun: [
    { value: 'qwen-turbo', label: 'Qwen Turbo' },
    { value: 'qwen-plus', label: 'Qwen Plus' },
  ],
  huggingface: [
    { value: 'Qwen3-0.6B', label: 'Qwen3 0.6B（本地）' },
  ],
};

export default function QA() {
  const [provider, setProvider] = useState('deepseek');
  const [model, setModel] = useState('deepseek-v3');
  const [apiKey, setApiKey] = useState('');
  const [showReasoning, setShowReasoning] = useState(false);
  const [collection, setCollection] = useState('course');
  const [collections, setCollections] = useState([]);
  const [topK, setTopK] = useState(3);
  const [threshold, setThreshold] = useState(0.5);
  const [wordCountThreshold, setWordCountThreshold] = useState(20);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('response');
  const [searchResults, setSearchResults] = useState([]);
  const [response, setResponse] = useState('');
  const [error, setError] = useState('');
  const [taskType, setTaskType] = useState('auto');
  const [taskTypes, setTaskTypes] = useState([]);

  // ── 查询优化 ──
  const [queryStrategies, setQueryStrategies] = useState([]);       // 勾选的策略
  const [strategyMeta, setStrategyMeta] = useState([]);             // 可用策略元数据
  const [rewriteProvider, setRewriteProvider] = useState('deepseek');
  const [rewriteModel, setRewriteModel] = useState('deepseek-v3');
  const [showQueryOpt, setShowQueryOpt] = useState(false);          // 折叠/展开

  useEffect(() => {
    setModel(MODELS[provider][0].value);
    setShowReasoning(false);
  }, [provider]);

  useEffect(() => {
    fetchCollections();
    fetchTaskTypes();
    fetchQueryStrategies();
  }, []);

  const fetchCollections = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/collections?provider=chroma`);
      const data = await res.json();
      if (data.collections) {
        setCollections(data.collections.map(c => (typeof c === 'string' ? c : c.name || c.id)));
      }
    } catch {
      setCollections(['course']);
    }
  };

  const fetchTaskTypes = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/task-types`);
      const data = await res.json();
      if (data.task_types) {
        setTaskTypes(data.task_types);
      }
    } catch {
      setTaskTypes([{ value: 'auto', label: '自动识别' }]);
    }
  };

  const fetchQueryStrategies = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/query-strategies`);
      const data = await res.json();
      if (data.strategies) {
        setStrategyMeta(data.strategies);
      }
    } catch {
      setStrategyMeta([]);
    }
  };

  const toggleStrategy = (value) => {
    setQueryStrategies((prev) =>
      prev.includes(value) ? prev.filter((v) => v !== value) : [...prev, value]
    );
  };

  const handleAsk = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    setSearchResults([]);
    setResponse('');
    setActiveTab('response');

    try {
      const res = await fetch(`${API_BASE_URL}/qa`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          api_key: apiKey || null,
          provider,
          model_name: model,
          collection,
          top_k: topK,
          threshold,
          word_count_threshold: wordCountThreshold,
          show_reasoning: showReasoning,
          task_type: taskType === 'auto' ? null : taskType,
          query_strategies: queryStrategies,
          rewrite_model_provider: rewriteProvider,
          rewrite_model_name: rewriteModel,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || '请求失败');
      }

      const data = await res.json();
      setSearchResults(data.search_results || []);
      setResponse(data.response || '');
      if ((data.search_results || []).length === 0) {
        setActiveTab('response');
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleAsk();
    }
  };

  const isDeepseekR1 = provider === 'deepseek' && model === 'deepseek-r1';

  return (
    <div className="p-6 h-screen flex flex-col">
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-gray-800">知识库问答</h1>
        <p className="text-gray-500 text-sm mt-1">基于 RAG 检索增强的智能问答</p>
      </div>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Left: Settings */}
        <div className="w-72 shrink-0 space-y-4 overflow-y-auto">
          {/* Model */}
          <div className="bg-white rounded-lg shadow p-4 space-y-3">
            <h2 className="font-semibold text-gray-700">模型设置</h2>
            <div>
              <label className="text-xs text-gray-500">提供商</label>
              <div className="mt-1 flex gap-1 flex-wrap">
                {PROVIDERS.map(p => (
                  <button
                    key={p.value}
                    onClick={() => setProvider(p.value)}
                    className={`text-xs px-3 py-1.5 rounded border transition-colors ${
                      provider === p.value
                        ? 'bg-blue-500 text-white border-blue-500'
                        : 'bg-white text-gray-600 border-gray-300 hover:border-blue-300'
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="text-xs text-gray-500">模型</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="mt-1 w-full border rounded px-2 py-1.5 text-sm bg-white"
              >
                {MODELS[provider].map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500">API Key</label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="请输入 API Key..."
                className="mt-1 w-full border rounded px-2 py-1.5 text-sm font-mono"
              />
            </div>
            {isDeepseekR1 && (
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showReasoning}
                  onChange={(e) => setShowReasoning(e.target.checked)}
                  className="accent-blue-500"
                />
                <span className="text-sm text-gray-600">显示推理过程</span>
              </label>
            )}
          </div>

          {/* Search options */}
          <div className="bg-white rounded-lg shadow p-4 space-y-3">
            <h2 className="font-semibold text-gray-700">检索设置</h2>
            <div>
              <label className="text-xs text-gray-500">数据库</label>
              <select
                value={collection}
                onChange={(e) => setCollection(e.target.value)}
                className="mt-1 w-full border rounded px-2 py-1.5 text-sm bg-white"
              >
                <option value="course">course（默认）</option>
                {collections.filter(c => c !== 'course').map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
            <div>
              <div className="flex justify-between">
                <label className="text-xs text-gray-500">返回数量 (Top K)</label>
                <span className="text-xs font-medium text-blue-500">{topK}</span>
              </div>
              <input
                type="range" min={1} max={10} value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full mt-1 accent-blue-500"
              />
            </div>
            <div>
              <div className="flex justify-between">
                <label className="text-xs text-gray-500">相似度阈值</label>
                <span className="text-xs font-medium text-blue-500">{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range" min={0} max={1} step={0.05} value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full mt-1 accent-blue-500"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500">最少字数过滤</label>
              <input
                type="number" value={wordCountThreshold} min={0} max={500}
                onChange={(e) => setWordCountThreshold(Number(e.target.value))}
                className="mt-1 w-full border rounded px-2 py-1 text-sm"
              />
            </div>
          </div>

          {/* Task type */}
          <div className="bg-white rounded-lg shadow p-4 space-y-3">
            <h2 className="font-semibold text-gray-700">任务类型</h2>
            <div>
              <label className="text-xs text-gray-500">Prompt 模板</label>
              <select
                value={taskType}
                onChange={(e) => setTaskType(e.target.value)}
                className="mt-1 w-full border rounded px-2 py-1.5 text-sm bg-white"
              >
                {taskTypes.map(t => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
              {taskType !== 'auto' && taskTypes.find(t => t.value === taskType) && (
                <p className="text-xs text-gray-400 mt-1">
                  {taskTypes.find(t => t.value === taskType).description}
                </p>
              )}
              {taskType === 'auto' && (
                <p className="text-xs text-gray-400 mt-1">
                  系统将自动识别问题类型并选择最优 prompt 和参数
                </p>
              )}
            </div>
          </div>

          {/* Query Optimization */}
          <div className="bg-white rounded-lg shadow p-4 space-y-3">
            <button
              onClick={() => setShowQueryOpt(!showQueryOpt)}
              className="w-full flex items-center justify-between font-semibold text-gray-700"
            >
              <span>🔍 查询优化</span>
              <span className={`text-xs text-gray-400 transition-transform ${showQueryOpt ? 'rotate-90' : ''}`}>
                {showQueryOpt ? '▼' : '▶'} {queryStrategies.length > 0 && (
                  <span className="ml-1 text-blue-500">({queryStrategies.length})</span>
                )}
              </span>
            </button>
            {!showQueryOpt && queryStrategies.length > 0 && (
              <p className="text-xs text-gray-400">
                已开启：{queryStrategies.map(s => strategyMeta.find(m => m.value === s)?.label || s).join('、')}
              </p>
            )}
            {showQueryOpt && (
              <div className="space-y-2 pt-1">
                <p className="text-xs text-gray-400">
                  选择查询优化策略（默认关闭，不增加额外耗时）
                </p>
                {strategyMeta.map((s) => (
                  <label
                    key={s.value}
                    className={`flex items-start gap-2 p-2 rounded border cursor-pointer transition-colors ${
                      queryStrategies.includes(s.value)
                        ? 'border-blue-300 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={queryStrategies.includes(s.value)}
                      onChange={() => toggleStrategy(s.value)}
                      className="mt-0.5 accent-blue-500"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm font-medium text-gray-700">{s.label}</span>
                        {s.requires_llm && (
                          <span className="text-[10px] px-1 py-0.5 rounded bg-amber-100 text-amber-600">LLM</span>
                        )}
                        {!s.requires_llm && (
                          <span className="text-[10px] px-1 py-0.5 rounded bg-green-100 text-green-600">无开销</span>
                        )}
                      </div>
                      <p className="text-xs text-gray-400 mt-0.5">{s.description}</p>
                    </div>
                  </label>
                ))}
                {/* LLM 配置 — 仅当选择了需要 LLM 的策略时显示 */}
                {queryStrategies.some(s => strategyMeta.find(m => m.value === s)?.requires_llm) && (
                  <div className="mt-3 pt-3 border-t border-gray-200 space-y-2">
                    <p className="text-xs text-gray-500 font-medium">查询优化模型</p>
                    <select
                      value={rewriteProvider}
                      onChange={(e) => { setRewriteProvider(e.target.value); setRewriteModel(MODELS[e.target.value]?.[0]?.value || ''); }}
                      className="w-full border rounded px-2 py-1.5 text-xs bg-white"
                    >
                      {PROVIDERS.filter(p => p.value !== 'huggingface').map(p => (
                        <option key={p.value} value={p.value}>{p.label}</option>
                      ))}
                    </select>
                    <select
                      value={rewriteModel}
                      onChange={(e) => setRewriteModel(e.target.value)}
                      className="w-full border rounded px-2 py-1.5 text-xs bg-white"
                    >
                      {(MODELS[rewriteProvider] || []).map(m => (
                        <option key={m.value} value={m.value}>{m.label}</option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Query */}
          <div className="bg-white rounded-lg shadow p-4 space-y-3">
            <h2 className="font-semibold text-gray-700">提问</h2>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入问题... (Ctrl+Enter 发送)"
              rows={4}
              className="w-full border rounded px-3 py-2 text-sm resize-none focus:outline-none focus:border-blue-400"
            />
            <button
              onClick={handleAsk}
              disabled={!query.trim() || loading}
              className={`w-full py-2.5 rounded-lg font-semibold text-white transition-colors ${
                !query.trim() || loading
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-blue-500 hover:bg-blue-600 active:bg-blue-700'
              }`}
            >
              {loading ? '检索生成中...' : '发送问题'}
            </button>
          </div>
        </div>

        {/* Right: Results */}
        <div className="flex-1 flex flex-col min-h-0 min-w-0">
          {error && (
            <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
              {error}
            </div>
          )}

          {/* Tabs */}
          <div className="flex gap-1 mb-3">
            <button
              onClick={() => setActiveTab('response')}
              className={`px-4 py-2 text-sm rounded-t font-medium transition-colors ${
                activeTab === 'response'
                  ? 'bg-white text-blue-600 shadow'
                  : 'bg-gray-200 text-gray-500 hover:bg-gray-300'
              }`}
            >
              生成回答
              {response && <span className="ml-1 text-xs text-green-500">●</span>}
            </button>
            <button
              onClick={() => setActiveTab('sources')}
              className={`px-4 py-2 text-sm rounded-t font-medium transition-colors ${
                activeTab === 'sources'
                  ? 'bg-white text-blue-600 shadow'
                  : 'bg-gray-200 text-gray-500 hover:bg-gray-300'
              }`}
            >
              检索来源
              {searchResults.length > 0 && (
                <span className="ml-1 text-xs bg-blue-100 text-blue-600 px-1.5 py-0.5 rounded-full">
                  {searchResults.length}
                </span>
              )}
            </button>
          </div>

          <div className="flex-1 bg-white rounded-lg shadow p-5 overflow-y-auto min-h-0">
            {loading && (
              <div className="flex items-center gap-2 text-gray-400">
                <div className="animate-spin w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full" />
                正在检索知识库并生成回答...
              </div>
            )}

            {!loading && activeTab === 'response' && (
              <>
                {response ? (
                  <div className="prose prose-sm max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{response}</ReactMarkdown>
                  </div>
                ) : (
                  <div className="text-gray-400 text-sm">
                    {error ? '' : '回答将在这里显示'}
                  </div>
                )}
              </>
            )}

            {!loading && activeTab === 'sources' && (
              <>
                {searchResults.length === 0 ? (
                  <p className="text-gray-400 text-sm">暂无检索结果</p>
                ) : (
                  <div className="space-y-4">
                    {searchResults.map((result, i) => (
                      <div key={i} className="border border-gray-200 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-semibold text-gray-500">
                            来源 {i + 1}
                            {result.metadata?.source && (
                              <span className="ml-2 text-blue-500">· {result.metadata.source}</span>
                            )}
                            {result.metadata?.page && (
                              <span className="ml-1 text-gray-400">第 {result.metadata.page} 页</span>
                            )}
                          </span>
                          <span
                            className={`text-xs font-mono px-2 py-0.5 rounded-full ${
                              result.score >= 0.8
                                ? 'bg-green-100 text-green-700'
                                : result.score >= 0.6
                                ? 'bg-yellow-100 text-yellow-700'
                                : 'bg-gray-100 text-gray-600'
                            }`}
                          >
                            {(result.score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">{result.text}</p>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
