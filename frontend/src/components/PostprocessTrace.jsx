import React from 'react';

const STEP_LABELS = {
  normalize: '标准化',
  deduplicate: '去重',
  rerank: '重排',
  final_threshold: '阈值',
  compress: '压缩',
  context_pack: '打包',
  final_top_k: '截断',
  diversify: '多样性',
};

const TraceBadge = ({ children, tone = 'gray' }) => {
  const tones = {
    gray: 'bg-gray-100 text-gray-600 border-gray-200',
    blue: 'bg-blue-50 text-blue-700 border-blue-100',
    amber: 'bg-amber-50 text-amber-700 border-amber-100',
    green: 'bg-green-50 text-green-700 border-green-100',
    red: 'bg-red-50 text-red-700 border-red-100',
  };
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[11px] ${tones[tone] || tones.gray}`}>
      {children}
    </span>
  );
};

const PostprocessTrace = ({ trace }) => {
  if (!Array.isArray(trace) || trace.length === 0) return null;

  return (
    <div className="mb-4 rounded border border-gray-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <h4 className="text-sm font-semibold text-gray-700">检索后优化 Trace</h4>
        <span className="text-xs text-gray-400">{trace.length} 步</span>
      </div>
      <div className="space-y-2">
        {trace.map((step, idx) => (
          <div key={`${step.step}-${idx}`} className="rounded border border-gray-100 bg-gray-50 px-2.5 py-2">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-semibold text-gray-700">
                {idx + 1}. {STEP_LABELS[step.step] || step.step}
              </span>
              <TraceBadge>{step.input_count ?? 0} → {step.output_count ?? 0}</TraceBadge>
              {step.elapsed_ms != null && <TraceBadge>{step.elapsed_ms} ms</TraceBadge>}
              {step.method && <TraceBadge tone="blue">{step.method}</TraceBadge>}
              {step.fallback && <TraceBadge tone="amber">降级 {step.fallback_from} → {step.fallback_method}</TraceBadge>}
              {step.llm_compress_top_n != null && <TraceBadge tone="blue">LLM 前 {step.llm_compress_top_n} 条</TraceBadge>}
              {step.llm_compressed > 0 && <TraceBadge tone="green">压缩 {step.llm_compressed}</TraceBadge>}
              {step.llm_skipped_rank > 0 && <TraceBadge>限额跳过 {step.llm_skipped_rank}</TraceBadge>}
              {step.llm_skipped_short > 0 && <TraceBadge>短文本跳过 {step.llm_skipped_short}</TraceBadge>}
              {step.llm_rejected > 0 && <TraceBadge tone="amber">拒绝 {step.llm_rejected}</TraceBadge>}
              {step.llm_dropped > 0 && <TraceBadge tone="red">删除 {step.llm_dropped}</TraceBadge>}
              {step.llm_failed > 0 && <TraceBadge tone="red">失败 {step.llm_failed}</TraceBadge>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PostprocessTrace;
