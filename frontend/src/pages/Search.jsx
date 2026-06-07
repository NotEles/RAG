// src/pages/Search.jsx
import React, { useState, useEffect } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const POST_LLM_MODELS = {
  ollama: [
    { value: 'qwen2.5:3b', label: 'Qwen2.5 3B（Ollama 本地）' },
    { value: 'qwen2.5:1.5b', label: 'Qwen2.5 1.5B（Ollama 本地，快速）' },
    { value: 'zephyr:latest', label: 'Zephyr（Ollama 本地）' },
    { value: 'gpt-oss:20b', label: 'GPT OSS 20B（Ollama 本地）' },
  ],
  deepseek: [
    { value: 'deepseek-v3', label: 'DeepSeek V3' },
    { value: 'deepseek-r1', label: 'DeepSeek R1' },
  ],
  openai: [
    { value: 'gpt-4o-mini', label: 'GPT-4o mini' },
    { value: 'gpt-4o', label: 'GPT-4o' },
  ],
  aliyun: [
    { value: 'qwen-turbo', label: 'Qwen Turbo' },
    { value: 'qwen-plus', label: 'Qwen Plus' },
  ],
};

const Search = () => {
  const [query, setQuery] = useState('');
  const [collection, setCollection] = useState('');
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [topK, setTopK] = useState(3);
  const [threshold, setThreshold] = useState(0.7);
  const [collections, setCollections] = useState([]);
  const [providers, setProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('chroma');
  const [wordCountThreshold, setWordCountThreshold] = useState(100);
  const [saveResults, setSaveResults] = useState(false);
  const [status, setStatus] = useState('');

  // ── 查询优化 ──
  const [queryStrategies, setQueryStrategies] = useState([]);
  const [strategyMeta, setStrategyMeta] = useState([]);
  const [rewriteProvider, setRewriteProvider] = useState('deepseek');
  const [rewriteModel, setRewriteModel] = useState('deepseek-v3');
  const [showQueryOpt, setShowQueryOpt] = useState(false);

  // ── 检索后优化 ──
  const [postprocessEnabled, setPostprocessEnabled] = useState(true);
  const [postStrategies, setPostStrategies] = useState(['deduplicate', 'rerank', 'compress', 'context_pack']);
  const [showPostOpt, setShowPostOpt] = useState(false);
  const [postprocessFetchK, setPostprocessFetchK] = useState(20);
  const [rerankTopK, setRerankTopK] = useState(5);
  const [rerankMethod, setRerankMethod] = useState('llm');
  const [rerankModel, setRerankModel] = useState('BAAI/bge-reranker-base');
  const [compressMethod, setCompressMethod] = useState('llm');
  const [maxContextChars, setMaxContextChars] = useState(8000);
  const [maxContextTokens, setMaxContextTokens] = useState(3000);
  const [mmrLambda, setMmrLambda] = useState(0.7);
  const [postLlmProvider, setPostLlmProvider] = useState('ollama');
  const [postLlmModel, setPostLlmModel] = useState('qwen2.5:3b');
  const [postLlmApiKey, setPostLlmApiKey] = useState('');

  // 加载向量数据库providers和collections
  useEffect(() => {
    const fetchData = async () => {
      try {
        // 获取providers列表
        const providersResponse = await fetch(`${apiBaseUrl}/providers`);
        const providersData = await providersResponse.json();
        setProviders(providersData.providers);

        // 获取collections列表
        const collectionsResponse = await fetch(`${apiBaseUrl}/collections?provider=${selectedProvider}`);
        const collectionsData = await collectionsResponse.json();
        setCollections(collectionsData.collections);

        // 获取查询优化策略
        try {
          const strategiesRes = await fetch(`${apiBaseUrl}/query-strategies`);
          const strategiesData = await strategiesRes.json();
          if (strategiesData.strategies) {
            setStrategyMeta(strategiesData.strategies);
          }
        } catch {}
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [selectedProvider]);

  const handleSearch = async () => {
    if (!query || !collection) {
      setStatus('请选择集合并输入搜索内容');
      return;
    }

    setIsSearching(true);
    setStatus('');
    try {
      const searchParams = {
        query,
        collection_id: collection,
        top_k: topK,
        threshold,
        word_count_threshold: wordCountThreshold,
        save_results: saveResults,
        query_strategies: queryStrategies,
        rewrite_model_provider: rewriteProvider,
        rewrite_model_name: rewriteModel,
        postprocess_enabled: postprocessEnabled,
        postprocess_strategies: postStrategies,
        postprocess_fetch_k: postprocessFetchK,
        rerank_method: rerankMethod,
        rerank_model: rerankModel,
        rerank_top_k: rerankTopK,
        compress_method: postStrategies.includes('compress') ? compressMethod : 'none',
        max_context_chars: maxContextChars,
        max_context_tokens: maxContextTokens,
        mmr_lambda: mmrLambda,
        postprocess_llm_provider: postLlmProvider,
        postprocess_llm_model: postLlmModel,
        postprocess_api_key: postLlmProvider === 'ollama' ? null : postLlmApiKey || null,
      };
      
      console.log('发送搜索请求:', searchParams);

      const response = await fetch(`${apiBaseUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(searchParams),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('搜索响应:', data);

      const responseResults = Array.isArray(data.results) ? data.results : data.results?.results;
      if (responseResults && responseResults.length > 0) {
        setResults(responseResults);
        if (saveResults && data.saved_filepath) {
          setStatus(`搜索完成！结果已保存至: ${data.saved_filepath}`);
        } else {
          setStatus('搜索完成！');
        }
      } else {
        setResults([]);
        setStatus('未找到匹配的结果');
      }
    } catch (error) {
      console.error('搜索错误:', error);
      setStatus(`搜索出错: ${error.message}`);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  // 添加保存结果的函数
  const handleSaveResults = async () => {
    if (!results.length) {
      setStatus('没有可保存的搜索结果');
      return;
    }

    try {
      const saveParams = {
        query,
        collection_id: collection,
        results: results
      };

      console.log('发送保存请求:', saveParams);
      
      const response = await fetch(`${apiBaseUrl}/save-search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(saveParams),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setStatus(`结果已保存至: ${data.saved_filepath}`);
    } catch (error) {
      console.error('保存错误:', error);
      setStatus(`保存失败: ${error.message}`);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6"> 检索增强生成工具 </h1>
      <hr />
      <h2 className="text-2xl font-bold mb-6">相似性检索</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Search Controls */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm overflow-y-auto max-h-[calc(100vh-160px)]">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">用户查询</label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your search query..."
                  className="block w-full p-2 border rounded h-32 resize-none"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">向量库</label>
                <select
                  value={selectedProvider}
                  onChange={(e) => setSelectedProvider(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  {providers.map(provider => (
                    <option key={provider.id} value={provider.id}>
                      {provider.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">集合</label>
                <select
                  value={collection}
                  onChange={(e) => setCollection(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  <option value="">Choose a collection...</option>
                  {collections.map(coll => (
                    <option key={coll.id} value={coll.id}>
                      {coll.name} ({coll.count} documents)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">前K个检索结果</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  min="1"
                  max="10"
                  className="block w-full p-2 border rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  相似性阈值: {threshold}
                </label>
                <input
                  type="range"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  min="0"
                  max="1"
                  step="0.1"
                  className="block w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  包含最少单词个数: {wordCountThreshold}
                </label>
                <input
                  type="range"
                  value={wordCountThreshold}
                  onChange={(e) => setWordCountThreshold(parseInt(e.target.value))}
                  min="0"
                  max="500"
                  step="10"
                  className="block w-full"
                />
              </div>

              <div className="mt-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={saveResults}
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      console.log('Save Results changed to:', newValue);
                      setSaveResults(newValue);
                    }}
                    className="form-checkbox h-4 w-4 text-blue-600"
                  />
                  <span className="text-sm font-medium">保存检索结果</span>
                </label>
              </div>

              {/* ── 查询优化 ── */}
              <div className="border-t pt-3 mt-2">
                <button
                  onClick={() => setShowQueryOpt(!showQueryOpt)}
                  className="w-full flex items-center justify-between text-sm font-medium text-gray-700"
                >
                  <span>🔍 查询优化</span>
                  <span className={`text-xs text-gray-400 ${queryStrategies.length > 0 ? 'text-blue-500' : ''}`}>
                    {queryStrategies.length > 0 ? `已开启 ${queryStrategies.length} 项` : '关闭'}
                  </span>
                </button>
                {showQueryOpt && (
                  <div className="mt-2 space-y-1.5">
                    {strategyMeta.map((s) => (
                      <label
                        key={s.value}
                        className={`flex items-center gap-2 p-1.5 rounded text-xs cursor-pointer ${
                          queryStrategies.includes(s.value) ? 'bg-blue-50 text-blue-700' : 'text-gray-600'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={queryStrategies.includes(s.value)}
                          onChange={() => setQueryStrategies(prev =>
                            prev.includes(s.value) ? prev.filter(v => v !== s.value) : [...prev, s.value]
                          )}
                          className="accent-blue-500"
                        />
                        {s.label}
                        {s.requires_llm && <span className="text-[10px] text-amber-500">(LLM)</span>}
                      </label>
                    ))}
                    {queryStrategies.some(s => strategyMeta.find(m => m.value === s)?.requires_llm) && (
                      <div className="pt-2 space-y-1">
                        <select
                          value={rewriteProvider}
                          onChange={(e) => setRewriteProvider(e.target.value)}
                          className="w-full border rounded px-1.5 py-1 text-xs bg-white"
                        >
                          <option value="deepseek">DeepSeek</option>
                          <option value="openai">OpenAI</option>
                          <option value="aliyun">阿里云百炼</option>
                        </select>
                        <select
                          value={rewriteModel}
                          onChange={(e) => setRewriteModel(e.target.value)}
                          className="w-full border rounded px-1.5 py-1 text-xs bg-white"
                        >
                          {rewriteProvider === 'deepseek' && (
                            <>
                              <option value="deepseek-v3">DeepSeek V3</option>
                              <option value="deepseek-r1">DeepSeek R1</option>
                            </>
                          )}
                          {rewriteProvider === 'openai' && (
                            <>
                              <option value="gpt-4o-mini">GPT-4o mini</option>
                              <option value="gpt-4o">GPT-4o</option>
                            </>
                          )}
                          {rewriteProvider === 'aliyun' && (
                            <>
                              <option value="qwen-turbo">Qwen Turbo</option>
                              <option value="qwen-plus">Qwen Plus</option>
                            </>
                          )}
                        </select>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* ── 检索后优化 ── */}
              <div className="border-t pt-3 mt-2">
                <button
                  onClick={() => setShowPostOpt(!showPostOpt)}
                  className="w-full flex items-center justify-between text-sm font-medium text-gray-700"
                >
                  <span>检索后优化</span>
                  <span className={`text-xs ${postprocessEnabled ? 'text-blue-500' : 'text-gray-400'}`}>
                    {postprocessEnabled ? `已开启 ${postStrategies.length} 项` : '关闭'}
                  </span>
                </button>
                {showPostOpt && (
                  <div className="mt-2 space-y-2">
                    <label className="flex items-center gap-2 text-xs cursor-pointer">
                      <input
                        type="checkbox"
                        checked={postprocessEnabled}
                        onChange={(e) => setPostprocessEnabled(e.target.checked)}
                        className="accent-blue-500"
                      />
                      启用检索后优化
                    </label>
                    {[
                      ['deduplicate', '去重'],
                      ['rerank', '重排'],
                      ['compress', '压缩'],
                      ['diversify', '多样性'],
                      ['context_pack', '打包'],
                    ].map(([value, label]) => (
                      <label
                        key={value}
                        className={`flex items-center gap-2 p-1.5 rounded text-xs cursor-pointer ${
                          postStrategies.includes(value) ? 'bg-blue-50 text-blue-700' : 'text-gray-600'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={postStrategies.includes(value)}
                          onChange={() => setPostStrategies(prev =>
                            prev.includes(value) ? prev.filter(v => v !== value) : [...prev, value]
                          )}
                          className="accent-blue-500"
                        />
                        {label}
                      </label>
                    ))}
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        type="number"
                        min="1"
                        max="50"
                        value={postprocessFetchK}
                        onChange={(e) => setPostprocessFetchK(parseInt(e.target.value) || 1)}
                        className="w-full border rounded px-1.5 py-1 text-xs"
                        title="候选数"
                      />
                      <input
                        type="number"
                        min="1"
                        max="20"
                        value={rerankTopK}
                        onChange={(e) => setRerankTopK(parseInt(e.target.value) || 1)}
                        className="w-full border rounded px-1.5 py-1 text-xs"
                        title="重排后数量"
                      />
                    </div>
                    <label className="block text-xs text-gray-500">
                      重排方式
                      <select
                        value={rerankMethod}
                        onChange={(e) => setRerankMethod(e.target.value)}
                        className="mt-1 w-full border rounded px-1.5 py-1 text-xs bg-white"
                      >
                        <option value="cross_encoder">本地 CrossEncoder 重排</option>
                        <option value="llm">LLM 重排</option>
                      </select>
                    </label>
                    {rerankMethod === 'cross_encoder' && (
                      <input
                        value={rerankModel}
                        onChange={(e) => setRerankModel(e.target.value)}
                        className="w-full border rounded px-1.5 py-1 text-xs font-mono"
                      />
                    )}
                    <label className="block text-xs text-gray-500">
                      压缩方式
                      <select
                        value={compressMethod}
                        onChange={(e) => setCompressMethod(e.target.value)}
                        className="mt-1 w-full border rounded px-1.5 py-1 text-xs bg-white"
                      >
                        <option value="extractive">抽取式压缩</option>
                        <option value="llm">LLM 压缩</option>
                      </select>
                    </label>
                    {(rerankMethod === 'llm' || compressMethod === 'llm') && (
                      <div className="grid grid-cols-2 gap-2">
                        <select
                          value={postLlmProvider}
                          onChange={(e) => {
                            setPostLlmProvider(e.target.value);
                            setPostLlmModel(POST_LLM_MODELS[e.target.value][0].value);
                          }}
                          className="w-full border rounded px-1.5 py-1 text-xs bg-white"
                        >
                          <option value="ollama">Ollama 本地</option>
                          <option value="deepseek">DeepSeek</option>
                          <option value="openai">OpenAI</option>
                          <option value="aliyun">阿里云百炼</option>
                        </select>
                        <select
                          value={postLlmModel}
                          onChange={(e) => setPostLlmModel(e.target.value)}
                          className="w-full border rounded px-1.5 py-1 text-xs bg-white"
                        >
                          {(POST_LLM_MODELS[postLlmProvider] || []).map(m => (
                            <option key={m.value} value={m.value}>{m.label}</option>
                          ))}
                        </select>
                        {postLlmProvider !== 'ollama' && (
                          <input
                            type="password"
                            value={postLlmApiKey}
                            onChange={(e) => setPostLlmApiKey(e.target.value)}
                            placeholder="后处理 API Key，可留空使用环境变量"
                            className="col-span-2 w-full border rounded px-1.5 py-1 text-xs"
                          />
                        )}
                      </div>
                    )}
                    <label className="block text-xs text-gray-500">
                      MMR 相关性权重: {mmrLambda.toFixed(2)}
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={mmrLambda}
                        onChange={(e) => setMmrLambda(parseFloat(e.target.value))}
                        className="block w-full"
                      />
                    </label>
                    <label className="block text-xs text-gray-500">
                      字符预算: {maxContextChars}
                      <input
                        type="range"
                        min="1000"
                        max="20000"
                        step="1000"
                        value={maxContextChars}
                        onChange={(e) => setMaxContextChars(parseInt(e.target.value))}
                        className="block w-full"
                      />
                    </label>
                    <label className="block text-xs text-gray-500">
                      Token 预算: {maxContextTokens}
                      <input
                        type="range"
                        min="500"
                        max="12000"
                        step="500"
                        value={maxContextTokens}
                        onChange={(e) => setMaxContextTokens(parseInt(e.target.value))}
                        className="block w-full"
                      />
                    </label>
                  </div>
                )}
              </div>

              <button
                onClick={() => {
                  console.log('Search clicked with saveResults:', saveResults);
                  handleSearch();
                }}
                disabled={isSearching}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
              >
                {isSearching ? '检索过程中...' : '检索'}
              </button>
            </div>
          </div>

          {status && (
            <div className={`p-4 rounded-lg ${
              status.includes('错误') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
            }`}>
              {status}
            </div>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className="col-span-9 border rounded-lg bg-white shadow-sm">
          {results.length > 0 ? (
            <div className="p-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-semibold">Search Results</h3>
                <button
                  onClick={handleSaveResults}
                  className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  保存搜索结果
                </button>
              </div>
              <div className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
                {results.map((result, idx) => (
                  <div key={idx} className="p-4 border rounded bg-gray-50">
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-sm text-gray-500">
                        Match Score: {(result.score * 100).toFixed(1)}%
                      </span>
                      <div className="text-sm text-gray-500">
                        <div>Source: {result.metadata.source}</div>
                        <div>Page: {result.metadata.page}</div>
                        <div>Chunk: {result.metadata.chunk}</div>
                      </div>
                    </div>
                    {result.metadata?.postprocess_reason && (
                      <div className="mb-2 flex flex-wrap gap-1 text-xs text-gray-500">
                        <span className="px-1.5 py-0.5 rounded bg-white border">{result.metadata.postprocess_reason}</span>
                        {result.metadata?.original_score != null && (
                          <span className="px-1.5 py-0.5 rounded bg-white border">
                            Original {(result.metadata.original_score * 100).toFixed(1)}%
                          </span>
                        )}
                        {result.metadata?.rerank_score != null && (
                          <span className="px-1.5 py-0.5 rounded bg-white border">
                            Rerank {(result.metadata.rerank_score * 100).toFixed(1)}%
                          </span>
                        )}
                        {result.metadata?.compressed && (
                          <span className="px-1.5 py-0.5 rounded bg-white border">
                            Compressed {result.metadata.original_length}→{result.metadata.compressed_length}
                          </span>
                        )}
                        {result.metadata?.packed_tokens != null && (
                          <span className="px-1.5 py-0.5 rounded bg-white border">
                            {result.metadata.packed_tokens} tokens
                          </span>
                        )}
                      </div>
                    )}
                    <p className="text-sm whitespace-pre-wrap">{result.text}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <RandomImage message="Search results will appear here" />
          )}
        </div>
      </div>
    </div>
  );
};

export default Search;
