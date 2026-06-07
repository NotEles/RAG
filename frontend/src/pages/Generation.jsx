import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const MarkdownViewer = ({ markdownText }) => {
  return (
    <div className="markdown-container">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdownText}</ReactMarkdown>
    </div>
  );
};

const POST_LLM_MODELS = {
  ollama: ['qwen2.5:3b', 'qwen2.5:1.5b', 'zephyr:latest', 'gpt-oss:20b'],
  deepseek: ['deepseek-v3', 'deepseek-r1'],
  openai: ['gpt-4o-mini', 'gpt-4o'],
  aliyun: ['qwen-turbo', 'qwen-plus'],
};

const Generation = () => {
  const location = useLocation();
  const [provider, setProvider] = useState('');
  const [modelName, setModelName] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [models, setModels] = useState({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [response, setResponse] = useState('');
  const [status, setStatus] = useState('');
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [searchFiles, setSearchFiles] = useState([]);
  const [showReasoning, setShowReasoning] = useState(true);
  const [loadModel, setLoadModel] = useState(false);
  const [taskType, setTaskType] = useState('auto');
  const [taskTypes, setTaskTypes] = useState([]);
  const [postprocessEnabled, setPostprocessEnabled] = useState(false);
  const [postStrategies, setPostStrategies] = useState(['deduplicate', 'rerank', 'compress', 'context_pack']);
  const [showPostOpt, setShowPostOpt] = useState(false);
  const [rerankMethod, setRerankMethod] = useState('llm');
  const [rerankModel, setRerankModel] = useState('BAAI/bge-reranker-base');
  const [rerankTopK, setRerankTopK] = useState(5);
  const [compressMethod, setCompressMethod] = useState('llm');
  const [maxContextChars, setMaxContextChars] = useState(8000);
  const [maxContextTokens, setMaxContextTokens] = useState(3000);
  const [mmrLambda, setMmrLambda] = useState(0.7);
  const [postLlmProvider, setPostLlmProvider] = useState('ollama');
  const [postLlmModel, setPostLlmModel] = useState('qwen2.5:3b');
  const [postLlmApiKey, setPostLlmApiKey] = useState('');
  const [forcePostprocess, setForcePostprocess] = useState(false);
  const hasPostProcessedResults = searchResults.some(result => result.metadata?.postprocess_reason);
  const allPostProcessedResults = searchResults.length > 0 && searchResults.every(result => result.metadata?.postprocess_reason);
  const shouldPostprocess = postprocessEnabled && (forcePostprocess || !allPostProcessedResults);

  // 加载可用模型列表和搜索结果文件列表
  useEffect(() => {
    const fetchData = async () => {
      try {
        // 获取模型列表
        const modelsResponse = await fetch(`${apiBaseUrl}/generation/models`);
        const modelsData = await modelsResponse.json();
        setModels(modelsData.models);

        // 获取搜索结果文件列表
        const filesResponse = await fetch(`${apiBaseUrl}/search-results`);
        const filesData = await filesResponse.json();
        setSearchFiles(filesData.files);

        // 获取任务类型列表
        const taskTypesResponse = await fetch(`${apiBaseUrl}/task-types`);
        const taskTypesData = await taskTypesResponse.json();
        setTaskTypes(taskTypesData.task_types || [{ value: 'auto', label: '自动识别' }]);
      } catch (error) {
        console.error('Error fetching data:', error);
        setStatus('获取数据失败');
      }
    };

    fetchData();
  }, []);

  // 加载选中的搜索结果文件内容
  useEffect(() => {
    const loadSearchResults = async () => {
      if (!selectedFile) {
        setQuery('');
        setSearchResults([]);
        return;
      }

      try {
        const response = await fetch(`${apiBaseUrl}/search-results/${selectedFile}`);
        const data = await response.json();
        setQuery(data.query);
        setSearchResults(data.results);
      } catch (error) {
        console.error('Error loading search results:', error);
        setStatus('加载搜索结果失败');
      }
    };

    loadSearchResults();
  }, [selectedFile]);

  // 如果从搜索页面跳转过来，获取搜索结果
  useEffect(() => {
    if (location.state) {
      const { query: searchQuery, results } = location.state;
      if (searchQuery) setQuery(searchQuery);
      if (results) setSearchResults(results);
    }
  }, [location]);

  const handleGenerate = async () => {
    if (!provider || !modelName) {
      setStatus('请选择生成模型');
      return;
    }

    if (!query /*|| searchResults.length === 0 */) {
      setStatus('请输入问题并确保有搜索结果');
      return;
    }

    setIsGenerating(true);
    setStatus('');
    try {
      const response = await fetch(`${apiBaseUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          provider,
          model_name: modelName,
          search_results: searchResults,
          load_model: loadModel,
          api_key: apiKey || null,
          show_reasoning: showReasoning,
          task_type: taskType === 'auto' ? null : taskType,
          postprocess_enabled: shouldPostprocess,
          postprocess_strategies: postStrategies,
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
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResponse(data.response);
      if (data.search_results) {
        setSearchResults(data.search_results);
      }
      setLoadModel(false);
      const detectedTask = data.detected_task || taskType;
      setStatus(`生成完成！任务类型: ${detectedTask} 结果已保存至: ${data.saved_filepath}`);
    } catch (error) {
      console.error('Generation error:', error);
      setStatus(`生成失败: ${error.message}`);
    } finally {
      setIsGenerating(false);
      setLoadModel(false);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6"> 检索增强生成工具 </h1>
      <hr />
      <h2 className="text-2xl font-bold mb-6">响应生成</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Generation Controls */}
        <div className="col-span-4 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div className="space-y-4">
              <div>
                    <label className="block text-sm font-medium mb-1">提问</label>
                    <textarea
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Enter your question..."
                      className="block w-full p-2 border rounded h-32 resize-none"
                    />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">检索文档（可选）</label>
                <select
                  value={selectedFile}
                  onChange={(e) => setSelectedFile(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  <option value="">Select search results file...</option>
                  {searchFiles.map(file => (
                    <option key={file.id} value={file.id}>
                      {file.name}
                    </option>
                  ))}
                </select>
              </div>

              {/*selectedFile && */ (
                <>
                  <div>
                    <label className="block text-sm font-medium mb-1">任务类型</label>
                    <select
                      value={taskType}
                      onChange={(e) => setTaskType(e.target.value)}
                      className="block w-full p-2 border rounded"
                    >
                      {taskTypes.map(t => (
                        <option key={t.value} value={t.value}>{t.label}</option>
                      ))}
                    </select>
                    {taskType === 'auto' && (
                      <p className="text-xs text-gray-400 mt-1">
                        系统将自动识别问题类型并选择最优 prompt 和参数
                      </p>
                    )}
                    {taskType !== 'auto' && taskTypes.find(t => t.value === taskType) && (
                      <p className="text-xs text-gray-400 mt-1">
                        {taskTypes.find(t => t.value === taskType).description}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1">生成模型提供方</label>
                    <select
                      value={provider}
                      onChange={(e) => setProvider(e.target.value)}
                      className="block w-full p-2 border rounded"
                    >
                      <option value="">Select provider...</option>
                      {Object.keys(models).map(p => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  </div>

                  {provider && (
                    <div>
                      <label className="block text-sm font-medium mb-1">生成模型</label>
                      <select
                        value={modelName}
                        onChange={(e) => {setModelName(e.target.value); setLoadModel(true)}}
                        className="block w-full p-2 border rounded"
                      >
                        <option value="">Select model...</option>
                        {Object.entries(models[provider] || {}).map(([id, name]) => (
                          <option key={id} value={id}>
                            {id === 'deepseek-v3' ? 'DeepSeek V3' :
                             id === 'deepseek-r1' ? 'DeepSeek R1' :
                             name}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}

                  {(provider === 'openai' || provider === 'deepseek') && (
                    <div>
                      <label className="block text-sm font-medium mb-1">API Key</label>
                      <input
                        type="password"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        placeholder="Enter your API key..."
                        className="block w-full p-2 border rounded"
                      />
                    </div>
                  )}

                  {provider === 'deepseek' && modelName === 'deepseek-r1' && (
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="showReasoning"
                        checked={showReasoning}
                        onChange={(e) => setShowReasoning(e.target.checked)}
                        className="rounded border-gray-300 text-green-500 focus:ring-green-500"
                      />
                      <label htmlFor="showReasoning" className="text-sm font-medium">
                        显示思维链过程
                      </label>
                    </div>
                  )}

                  <div className="border-t pt-3">
                    <button
                      onClick={() => setShowPostOpt(!showPostOpt)}
                      className="w-full flex items-center justify-between text-sm font-medium text-gray-700"
                    >
                      <span>检索后优化</span>
                      <span className={`text-xs ${shouldPostprocess ? 'text-blue-500' : 'text-gray-400'}`}>
                        {allPostProcessedResults && !forcePostprocess ? '已优化，跳过重复处理' : postprocessEnabled ? `已开启 ${postStrategies.length} 项` : '关闭'}
                      </span>
                    </button>
                    {showPostOpt && (
                      <div className="mt-2 space-y-2">
                        <label className="flex items-center gap-2 text-xs cursor-pointer">
                          <input
                            type="checkbox"
                            checked={postprocessEnabled}
                            onChange={(e) => setPostprocessEnabled(e.target.checked)}
                            className="accent-green-500"
                          />
                          启用检索后优化
                        </label>
                        {hasPostProcessedResults && (
                          <label className="flex items-center gap-2 text-xs cursor-pointer">
                            <input
                              type="checkbox"
                              checked={forcePostprocess}
                              onChange={(e) => setForcePostprocess(e.target.checked)}
                              className="accent-green-500"
                            />
                            重新优化已有后处理结果
                          </label>
                        )}
                        {[
                          ['deduplicate', '去重'],
                          ['rerank', '重排'],
                          ['compress', '压缩'],
                          ['diversify', 'MMR 多样性'],
                          ['context_pack', 'Token 打包'],
                        ].map(([value, label]) => (
                          <label
                            key={value}
                            className={`flex items-center gap-2 p-1.5 rounded text-xs cursor-pointer ${
                              postStrategies.includes(value) ? 'bg-green-50 text-green-700' : 'text-gray-600'
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={postStrategies.includes(value)}
                              onChange={() => setPostStrategies(prev =>
                                prev.includes(value) ? prev.filter(v => v !== value) : [...prev, value]
                              )}
                              className="accent-green-500"
                            />
                            {label}
                          </label>
                        ))}
                        <div className="grid grid-cols-2 gap-2">
                          <input
                            type="number"
                            min="1"
                            max="20"
                            value={rerankTopK}
                            onChange={(e) => setRerankTopK(parseInt(e.target.value) || 1)}
                            className="w-full border rounded px-1.5 py-1 text-xs"
                            title="重排后数量"
                          />
                          <label className="block text-xs text-gray-500">
                            重排方式
                            <select
                              value={rerankMethod}
                              onChange={(e) => setRerankMethod(e.target.value)}
                              className="mt-1 w-full border rounded px-1.5 py-1 text-xs bg-white"
                            >
                              <option value="cross_encoder">本地重排</option>
                              <option value="llm">LLM 重排</option>
                            </select>
                          </label>
                        </div>
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
                                setPostLlmModel(POST_LLM_MODELS[e.target.value][0]);
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
                              {POST_LLM_MODELS[postLlmProvider].map(m => (
                                <option key={m} value={m}>{m}</option>
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
                    onClick={handleGenerate}
                    disabled={isGenerating}
                    className="w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-green-300"
                  >
                    {isGenerating ? '生成回答中...' : '生成回答'}
                  </button>

                  {status && (
                    <div className={`p-4 rounded-lg ${
                      status.includes('失败') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                    }`}>
                      {status}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Context and Response */}
        <div className="col-span-8">
          {selectedFile ? (
            <>
              {/* Search Results Context */}
              <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm">
                <h3 className="text-xl font-semibold mb-4">检索的上下文</h3>
                <div className="space-y-4 max-h-[300px] overflow-y-auto">
                  {searchResults.map((result, idx) => (
                    <div key={idx} className="p-4 border rounded bg-gray-50">
                      <div className="flex justify-between items-start mb-2">
                        <span className="font-medium text-sm text-gray-500">
                          Match Score: {(result.score * 100).toFixed(1)}%
                        </span>
                        <div className="text-sm text-gray-500">
                          <div>Source: {result.metadata.source}</div>
                          <div>Page: {result.metadata.page}</div>
                        </div>
                      </div>
                      {result.metadata?.postprocess_reason && (
                        <div className="mb-2 flex flex-wrap gap-1 text-xs text-gray-500">
                          <span className="px-1.5 py-0.5 rounded bg-white border">{result.metadata.postprocess_reason}</span>
                          {result.metadata?.rerank_score != null && (
                            <span className="px-1.5 py-0.5 rounded bg-white border">
                              Rerank {(result.metadata.rerank_score * 100).toFixed(1)}%
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
           </>
          ) : (
            <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm">
                <h3 className="text-xl font-semibold mb-4">无检索上下文</h3>
            </div>
          )}
              {/* Generated Response */}
              {response && (
                <div className="p-4 border rounded-lg bg-white shadow-sm">
                  <h3 className="text-xl font-semibold mb-4">生成的回答</h3>
                  <div className="p-4 border rounded bg-gray-50">
                    <p className="whitespace-pre-wrap"><MarkdownViewer markdownText={response} /></p>
                  </div>
                </div>
              )}
        </div>
      </div>
    </div>
  );
};

export default Generation;
